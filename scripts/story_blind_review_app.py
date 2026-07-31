"""Standalone private Streamlit UI for the 12 N0-vs-N3 blind choices."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = Path(os.environ.get(
    "STORY_BLIND_PACKET", ROOT / "evaluation" / "private" / "story_n0_n3_blind_packet.json",
))
RESPONSE_PATH = Path(os.environ.get(
    "STORY_BLIND_RESPONSES", ROOT / "evaluation" / "private" / "story_n0_n3_blind_responses.json",
))
PHOTO_ROOT = Path(os.environ.get("STORY_BLIND_PHOTO_ROOT", ROOT / "photos"))
CHOICES = {"未选择": "", "A 更好": "A", "B 更好": "B", "平局": "tie"}


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_responses(tokens: list[str]) -> dict:
    if RESPONSE_PATH.exists():
        payload = json.loads(RESPONSE_PATH.read_text(encoding="utf-8"))
    else:
        payload = {"protocol_version": "story-blind-responses-v1", "status": "in_progress", "responses": []}
    by_token = {str(item["case_token"]): item for item in payload.get("responses", [])}
    payload["responses"] = [
        by_token.get(token, {"case_token": token, "choice": "", "reason": ""}) for token in tokens
    ]
    return payload


def _current_responses(cases: list[dict]) -> list[dict]:
    return [{
        "case_token": case["case_token"],
        "choice": CHOICES[st.session_state[f"choice_{case['case_token']}"]],
        "reason": st.session_state[f"reason_{case['case_token']}"].strip(),
    } for case in cases]


st.set_page_config(page_title="Story N0/N3 盲评", layout="wide")
st.title("Story A/B 盲评")
st.caption("每组查看相同照片，只比较标题和叙事正文。A/B 的实验变体在完成 12 组之前不会显示。")

if not PACKET_PATH.exists():
    st.error("尚未生成盲评数据。请先完成 12×4 正式生成。")
    st.stop()

packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
cases = list(packet.get("cases", []))
tokens = [str(case["case_token"]) for case in cases]
saved = _load_responses(tokens)
saved_by_token = {str(item["case_token"]): item for item in saved["responses"]}
locked = saved.get("status") == "locked"

for case in cases:
    token = str(case["case_token"])
    previous = saved_by_token[token]
    reverse_choices = {value: label for label, value in CHOICES.items()}
    st.session_state.setdefault(f"choice_{token}", reverse_choices.get(previous.get("choice", ""), "未选择"))
    st.session_state.setdefault(f"reason_{token}", str(previous.get("reason") or ""))

selected_count = sum(
    CHOICES[st.session_state[f"choice_{token}"]] in {"A", "B", "tie"} for token in tokens
)
st.progress(selected_count / max(1, len(cases)), text=f"已选择 {selected_count}/{len(cases)}")

for case in cases:
    token = str(case["case_token"])
    with st.expander(f"案例 {case['review_index']} · {case['source_kind']} · {case['language']}", expanded=False):
        valid_images = [PHOTO_ROOT / value for value in case.get("photo_paths", []) if (PHOTO_ROOT / value).exists()]
        if valid_images:
            st.image([str(path) for path in valid_images], width=180)
        else:
            st.warning("此案例的原图文件当前不可访问；请不要在看不到证据时评分。")
        left, right = st.columns(2)
        with left:
            st.subheader("Story A")
            st.markdown(f"### {case['story_a']['title']}")
            for paragraph in case["story_a"].get("paragraphs", []):
                st.write(paragraph)
        with right:
            st.subheader("Story B")
            st.markdown(f"### {case['story_b']['title']}")
            for paragraph in case["story_b"].get("paragraphs", []):
                st.write(paragraph)
        st.radio(
            "你的选择", list(CHOICES), horizontal=True, key=f"choice_{token}", disabled=locked,
        )
        st.text_input("简短原因（可选）", key=f"reason_{token}", disabled=locked)

if locked:
    st.success("12 组盲评已锁定。请运行独立分析脚本揭盲；本页面不会显示 A/B 映射。")
else:
    left, right = st.columns(2)
    with left:
        if st.button("保存进度", use_container_width=True):
            saved["responses"] = _current_responses(cases)
            saved["status"] = "in_progress"
            _atomic_write(RESPONSE_PATH, saved)
            st.success("进度已保存，可以关闭页面后继续。")
    complete = all(CHOICES[st.session_state[f"choice_{token}"]] in {"A", "B", "tie"} for token in tokens)
    with right:
        if st.button("完成并锁定 12 组", type="primary", use_container_width=True, disabled=not complete):
            saved["responses"] = _current_responses(cases)
            saved["status"] = "locked"
            _atomic_write(RESPONSE_PATH, saved)
            st.rerun()
