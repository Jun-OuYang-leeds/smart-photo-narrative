#!/usr/bin/env python3
"""HPC 推理：读 story_requests.jsonl → Qwen3.5-27B 纯文本生成 → 写 story_outputs.jsonl。

设计原则：HPC 只做"笨推理"。本脚本**不 import 仓库任何模块**，只依赖
transformers + torch + 标准库（与 scripts/cloud/generate_scene_graph_jsonl.py 的自包含原则一致）。
所有接地校验 / 修复 / 写库都在本地导入阶段完成。

输入 JSONL 每行：{request_id, system, user, schema, gen_params{temperature,max_new_tokens}, ...}
输出 JSONL 每行：{request_id, raw_output, elapsed, engine, ok[, error]}

checkpoint 续跑：跳过 story_outputs.jsonl 中已有的 request_id；失败行 ok=false，下次会重试。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """防御性去除 Qwen3 思考块（正常应已用 enable_thinking=False 关闭）。"""
    return _THINK_RE.sub("", text).strip()


def load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    ids: set[str] = set()
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    ids.add(rec["request_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def render_prompt(processor, system: str, user: str) -> str:
    """应用 chat template。优先关闭 thinking；处理器不支持则回退。"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        # 该 processor 的 chat template 不接受 enable_thinking 参数
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def generate_one(model, processor, device, system: str, user: str, gen_params: dict) -> str:
    import torch

    prompt = render_prompt(processor, system, user)
    inputs = processor(text=prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(gen_params.get("max_new_tokens", 2048)),
            do_sample=True,
            temperature=float(gen_params.get("temperature", 0.25)),
            top_p=0.9,
        )
    new_ids = output_ids[0][input_len:]
    raw = processor.decode(new_ids, skip_special_tokens=True)
    return strip_think(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="HPC 上 Qwen3.5-27B 故事生成（纯文本）")
    parser.add_argument("--input", required=True, help="story_requests.jsonl 路径")
    parser.add_argument("--output", required=True, help="story_outputs.jsonl 路径（追加，带 checkpoint）")
    parser.add_argument("--model-path", required=True, help="本地模型目录 Qwen3.5-27B")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条（0=全部），用于冒烟")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 读请求 ----
    requests = []
    with in_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                requests.append(json.loads(line))
    if args.limit > 0:
        requests = requests[: args.limit]
    print(f"[infer] 读入 {len(requests)} 条请求", flush=True)

    done = load_done(out_path)
    pending = [r for r in requests if r["request_id"] not in done]
    print(f"[infer] 已完成 {len(done)}，待处理 {len(pending)}", flush=True)
    if not pending:
        print("[infer] 无待处理请求，退出", flush=True)
        return 0

    # ---- 加载模型（offline）----
    t0 = time.time()
    print(f"[infer] 加载 processor: {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"[infer] 加载模型 dtype={args.dtype} device_map=auto …", flush=True)
    model = AutoModelForMultimodalLM.from_pretrained(args.model_path, dtype=dtype, device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    print(f"[infer] 模型就绪，加载耗时 {time.time()-t0:.1f}s，主设备 {device}", flush=True)
    print(f"[infer] 显存：{torch.cuda.memory_allocated()/1e9:.1f}GB allocated", flush=True)

    # ---- 逐条推理 ----
    ok_count = 0
    err_count = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for i, req in enumerate(pending, 1):
            rid = req["request_id"]
            t1 = time.time()
            try:
                raw = generate_one(model, processor, device, req["system"], req["user"], req["gen_params"])
                rec = {
                    "request_id": rid,
                    "raw_output": raw,
                    "elapsed": round(time.time() - t1, 2),
                    "engine": "transformers",
                    "ok": bool(raw.strip()),
                }
                if not rec["ok"]:
                    rec["error"] = "empty_output"
            except Exception as exc:  # noqa: BLE001
                rec = {
                    "request_id": rid,
                    "raw_output": "",
                    "elapsed": round(time.time() - t1, 2),
                    "engine": "transformers",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            if rec["ok"]:
                ok_count += 1
                preview = rec["raw_output"].replace("\n", " ")[:80]
                print(f"[infer] {i}/{len(pending)} {rid} OK {rec['elapsed']}s | {preview}", flush=True)
            else:
                err_count += 1
                print(f"[infer] {i}/{len(pending)} {rid} FAIL {rec['elapsed']}s | {rec.get('error')}", flush=True)

    print(f"[infer] 全部完成：成功 {ok_count}，失败 {err_count} → {out_path}", flush=True)
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
