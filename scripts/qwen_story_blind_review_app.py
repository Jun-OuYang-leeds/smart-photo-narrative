"""Private Streamlit page for the available Qwen-only dual-track Story comparisons."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = Path(os.environ.get(
    "QWEN_STORY_BLIND_PACKET",
    ROOT / "evaluation" / "private" / "story_qwen_dual_blind_packet_v1.json",
))
RESPONSE_PATH = Path(os.environ.get(
    "QWEN_STORY_BLIND_RESPONSES",
    ROOT / "evaluation" / "private" / "story_qwen_dual_blind_responses_v1.json",
))
PHOTO_ROOT = Path(os.environ.get("QWEN_STORY_PHOTO_ROOT", ROOT / "photos"))
CHOICES = {"未选择": "", "A 更好": "A", "B 更好": "B", "平局": "tie"}
DIMENSION_LABELS = {
    "coherence": "连贯性 / Coherence",
    "informativeness": "信息量 / Informativeness",
    "evidence_consistency": "证据一致性 / Evidence consistency",
    "personalization": "个性化 / Personalization",
    "trustworthiness": "可信度 / Trustworthiness",
}


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _blank_response(case: dict) -> dict:
    return {
        "case_token": case["case_token"], "choice": "", "reason": "",
        "scores": {
            side: {dimension: 3 for dimension in case["dimensions"]}
            for side in ("A", "B")
        },
    }


if not PACKET_PATH.exists():
    st.set_page_config(page_title="Qwen Story 盲评", layout="wide")
    st.error("尚未生成新版 Qwen Story 盲评数据。")
    st.stop()

packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
cases = list(packet.get("cases", []))
tokens = [str(case["case_token"]) for case in cases]
if RESPONSE_PATH.exists():
    saved = json.loads(RESPONSE_PATH.read_text(encoding="utf-8"))
else:
    saved = {"protocol_version": "story-qwen-dual-blind-responses-v1", "status": "in_progress", "responses": []}
by_token = {str(item["case_token"]): item for item in saved.get("responses", [])}
saved["responses"] = [by_token.get(str(case["case_token"]), _blank_response(case)) for case in cases]
by_token = {str(item["case_token"]): item for item in saved["responses"]}
locked = saved.get("status") == "locked"

st.set_page_config(page_title="Qwen Story 双轨盲评", layout="wide")
st.title("Qwen3:4B Story 双轨盲评")
case_count = len(cases)
track_counts = packet.get("track_counts", {})
st.caption(
    f"共{case_count}组：{track_counts.get('faithful', 0)}组忠实叙事、"
    f"{track_counts.get('creative', 0)}组创意叙事。只比较可见故事；"
    "变体、校验、修复和回退信息在全部提交前保持隐藏。"
)
if packet.get("status") == "incomplete":
    st.warning(
        "预注册目标为24组，但冻结的6个Creative后备案例用尽后仍只有11个有效配对。"
        "本页面保留全部12组Faithful和可用的11组Creative，不再事后挑选新案例。"
    )

reverse_choices = {value: label for label, value in CHOICES.items()}
for case in cases:
    token = str(case["case_token"])
    previous = by_token[token]
    st.session_state.setdefault(f"choice_{token}", reverse_choices.get(previous.get("choice", ""), "未选择"))
    st.session_state.setdefault(f"reason_{token}", str(previous.get("reason") or ""))
    for side in ("A", "B"):
        for dimension in case["dimensions"]:
            value = int(previous.get("scores", {}).get(side, {}).get(dimension, 3))
            st.session_state.setdefault(f"score_{token}_{side}_{dimension}", value)

selected_count = sum(CHOICES[st.session_state[f"choice_{token}"]] in {"A", "B", "tie"} for token in tokens)
st.progress(selected_count / max(1, len(cases)), text=f"已评价 {selected_count}/{len(cases)}")

for case in cases:
    token = str(case["case_token"])
    track_name = "忠实叙事" if case["track"] == "faithful" else "创意叙事"
    with st.expander(
        f"案例 {case['review_index']} · {track_name} · {case['source_kind']} · {case['language']}",
        expanded=False,
    ):
        images = [PHOTO_ROOT / value for value in case.get("photo_paths", []) if (PHOTO_ROOT / value).exists()]
        if images:
            st.image([str(path) for path in images], width=180)
        else:
            st.warning("原图不可访问，请不要在看不到证据时评分。")
        columns = st.columns(2)
        for column, side, key in zip(columns, ("A", "B"), ("story_a", "story_b")):
            with column:
                st.subheader(f"Story {side}")
                st.markdown(f"### {case[key]['title']}")
                for paragraph in case[key].get("paragraphs", []):
                    st.write(paragraph)
                st.caption("请为这一侧单独评分")
                for dimension in case["dimensions"]:
                    st.slider(
                        DIMENSION_LABELS[dimension], 1, 5,
                        key=f"score_{token}_{side}_{dimension}", disabled=locked,
                    )
        st.radio("总体选择", list(CHOICES), horizontal=True, key=f"choice_{token}", disabled=locked)
        st.text_input("简短原因（可选）", key=f"reason_{token}", disabled=locked)


def _current() -> list[dict]:
    result = []
    for case in cases:
        token = str(case["case_token"])
        result.append({
            "case_token": token,
            "choice": CHOICES[st.session_state[f"choice_{token}"]],
            "reason": st.session_state[f"reason_{token}"].strip(),
            "scores": {
                side: {
                    dimension: int(st.session_state[f"score_{token}_{side}_{dimension}"])
                    for dimension in case["dimensions"]
                }
                for side in ("A", "B")
            },
        })
    return result


if locked:
    st.success(f"{case_count}组盲评已经锁定。页面不会显示A/B映射；请运行独立揭盲命令。")
else:
    left, right = st.columns(2)
    with left:
        if st.button("保存进度", use_container_width=True):
            saved["responses"] = _current()
            saved["status"] = "in_progress"
            _atomic_write(RESPONSE_PATH, saved)
            st.success("进度已保存。")
    complete = all(CHOICES[st.session_state[f"choice_{token}"]] in {"A", "B", "tie"} for token in tokens)
    with right:
        if st.button(f"完成并锁定{case_count}组", type="primary", use_container_width=True, disabled=not complete):
            saved["responses"] = _current()
            saved["status"] = "locked"
            _atomic_write(RESPONSE_PATH, saved)
            st.rerun()
