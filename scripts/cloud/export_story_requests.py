#!/usr/bin/env python3
"""本地导出：把故事请求 (Story Request) 导成 ``story_requests.jsonl``，供 HPC 离线推理。

本脚本**不调用任何 LLM**。它只做三件事：
  1. 用 ``ContextAggregator`` 聚合证据 (caption + Qwen 三元组 + EXIF + 情绪)；
  2. 用 ``PromptBuilder`` 构造 (system, user)，并拼上 ``DashScopeGenerator._schema_instruction``
     —— 使 HPC 拿到的 prompt 与 DashScope 远程后端**逐字一致**；
  3. 写成 JSONL，每行一个请求。

``request_id = sha1(sorted(photo_ids))``，保证 export → upload → inference → import 全链路幂等、可断点续跑。

运行环境：与 ``app.py`` 相同的本地 conda 环境（需要 Chroma + CLIP，不需要 LLM）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    REMOTE_LLM_NUM_PREDICT,
    STORY_DEFAULT_TEMPERATURE,
    STORY_NARRATIVE_MAX_PARAGRAPHS,
)
from story_agent import (
    ContextAggregator,
    DashScopeGenerator,
    PhotographerMoodPromptBuilder,
    PromptBuilder,
    StoryGenerator,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_context(agg: ContextAggregator, args: argparse.Namespace):
    """按 --date / --event / --photo-ids 三选一聚合证据，返回 (context, scope_kind)。"""
    if args.date:
        return agg.aggregate_by_date(args.date, narrator_role=args.narrator_role), "date"
    if args.event:
        return agg.aggregate_by_event(args.event, narrator_role=args.narrator_role), "event"
    if args.photo_ids:
        context = agg.aggregate_by_photo_ids(
            args.photo_ids,
            label=args.label or "Selected photos",
            source_kind="basket",
            narrator_role=args.narrator_role,
        )
        return context, "photos"
    raise SystemExit("必须指定 --date / --event / --photo-ids 之一")


def select_builder(mode: str):
    """与 generate_story_with_context (story_agent.py:2198) 完全一致的 builder 选择。"""
    return PhotographerMoodPromptBuilder if mode == "creative" else PromptBuilder


def export_one(context, scope_kind: str, mode: str, language: str) -> dict:
    """把一个 StoryContext 转成一条 JSONL 请求。复刻远程后端的 prompt 构造。"""
    # 复刻 generate_story_with_context 的预处理：证据组超过段落上限时先合并
    if len(context.groups) > STORY_NARRATIVE_MAX_PARAGRAPHS:
        reducer = StoryGenerator()  # 仅借用 _reduce_groups，不触发任何 LLM 调用
        context = replace(
            context,
            groups=reducer._reduce_groups(context.groups, STORY_NARRATIVE_MAX_PARAGRAPHS),
        )

    builder = select_builder(mode)
    system, user = builder.build_story_prompt(context, mode=mode, language=language)
    schema = builder.output_schema(context, language=language)
    # 关键：远程后端最终 system = system + schema 契约 (story_agent.py:1870)，这里原样复刻
    system_full = system + DashScopeGenerator._schema_instruction(schema)

    photo_ids = [item.photo_id for item in context.evidence]
    request_id = hashlib.sha1("|".join(sorted(photo_ids)).encode("utf-8")).hexdigest()[:16]
    prompt_hash = hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest()

    return {
        "request_id": request_id,
        "scope": {
            "kind": scope_kind,
            "source_kind": context.source_kind,
            "event_id": context.event_id,
            "label": context.label,
            "narrator_role": context.narrator_role,
            "photo_ids": photo_ids,
        },
        "system": system_full,
        "user": user,
        "schema": schema,
        "gen_params": {
            "temperature": STORY_DEFAULT_TEMPERATURE,
            "max_new_tokens": REMOTE_LLM_NUM_PREDICT,
        },
        "prompt_hash": prompt_hash,
        "prompt_version": builder.PROMPT_VERSION,
        "mode": mode,
        "language": language,
        "n_groups": len(context.groups),
    }


def load_existing_ids(out_path: Path) -> set[str]:
    """读取已存在的 request_id，用于追加时去重。"""
    if not out_path.exists():
        return set()
    ids: set[str] = set()
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["request_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="导出故事请求到 JSONL（不调用 LLM）")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--date", help="单日日记，如 2025-12-19")
    scope.add_argument("--event", help="单个 event_id")
    scope.add_argument("--photo-ids", nargs="+", help="若干 photo_id（空格分隔）")
    parser.add_argument("--label", help="--photo-ids 时的标签（可选）")
    parser.add_argument("--mode", choices=["faithful", "creative"], default="faithful")
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument("--narrator-role", default="observer", choices=["observer", "confirmed_subject"])
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "story_requests.jsonl",
        help="输出 JSONL 路径（默认仓库 outputs/story_requests.jsonl，追加模式）",
    )
    args = parser.parse_args()

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] 初始化 ContextAggregator（加载 Chroma + CLIP）……", file=sys.stderr)
    agg = ContextAggregator()

    # 支持多日期批处理：--date 可传 "2025-12-19,2025-12-20" 或 @dates.txt
    date_tokens: list[str] = []
    if args.date:
        if args.date.startswith("@"):
            date_tokens = [
                line.strip()
                for line in Path(args.date[1:]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            date_tokens = [d.strip() for d in args.date.split(",") if d.strip()]

    scopes: list[tuple[str, str]] = []  # (key, kind)
    if date_tokens:
        scopes = [(d, "date") for d in date_tokens]
    elif args.event:
        scopes = [(args.event, "event")]
    elif args.photo_ids:
        scopes = [("photos", "photos")]

    existing = load_existing_ids(out_path)
    written = 0
    skipped = 0

    with out_path.open("a", encoding="utf-8") as fh:
        for key, kind in scopes:
            # 临时把当前 scope 塞回 args，复用 build_context
            args.date = key if kind == "date" else None
            try:
                context, used_kind = build_context(agg, args) if kind != "date" else (
                    agg.aggregate_by_date(key, narrator_role=args.narrator_role), "date",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[export] 跳过 {kind}={key}：聚合失败 {type(exc).__name__}: {exc}", file=sys.stderr)
                continue

            if not context.evidence or not context.groups:
                print(f"[export] 跳过 {kind}={key}：无可用照片/证据", file=sys.stderr)
                skipped += 1
                continue

            record = export_one(context, used_kind, args.mode, args.language)
            if record["request_id"] in existing:
                print(f"[export] 跳过 {kind}={key}：request_id 已存在 ({record['request_id']})", file=sys.stderr)
                skipped += 1
                continue

            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            existing.add(record["request_id"])
            written += 1
            print(
                f"[export] 写入 {kind}={key} → request_id={record['request_id']} "
                f"photos={len(record['scope']['photo_ids'])} groups={record['n_groups']}",
                file=sys.stderr,
            )

    print(
        f"[export] 完成：写入 {written} 条，跳过 {skipped} 条 → {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
