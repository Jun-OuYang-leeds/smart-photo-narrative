"""Standalone private Streamlit page for the single M0/M1 blind review."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_DB_PATH, PHOTOS_DIR  # noqa: E402
from mood_evaluation import BLIND_PACKET_PATH, BLIND_RESPONSES_PATH, atomic_write_json  # noqa: E402
from storage import PhotoStorage  # noqa: E402


PACKET_PATH = Path(os.getenv("MOOD_BLIND_PACKET", str(BLIND_PACKET_PATH)))
RESPONSES_PATH = Path(os.getenv("MOOD_BLIND_RESPONSES", str(BLIND_RESPONSES_PATH)))
DATABASE_PATH = Path(os.getenv("MOOD_BLIND_DATABASE", str(APP_DB_PATH)))
PHOTO_ROOT = Path(os.getenv("MOOD_BLIND_PHOTOS_DIR", str(PHOTOS_DIR)))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Mood Story Blind Review", layout="wide")
    st.title("Photographer Mood Story A/B Blind Review")
    st.caption(
        "Compare only the visible narratives. Variant identity, validation details and mood condition remain hidden."
    )
    if not PACKET_PATH.exists():
        st.error("Blind packet not found. Run scripts/run_mood_story_experiment.py first.")
        return
    packet = _load(PACKET_PATH)
    existing = _load(RESPONSES_PATH) if RESPONSES_PATH.exists() else {}
    locked = bool(existing.get("locked"))

    storage = PhotoStorage(DATABASE_PATH)
    details = [storage.get_photo_detail(photo_id) for photo_id in packet["photo_ids"]]
    st.subheader("Evidence photos / 证据照片")
    columns = st.columns(len(details))
    for column, detail in zip(columns, details):
        if detail is None:
            column.warning("Missing photo")
            continue
        relative_path = str(detail["photo"]["relative_path"])
        path = PHOTO_ROOT / relative_path
        if path.is_file():
            column.image(str(path), width="stretch")
        column.caption(Path(relative_path).name)

    story_columns = st.columns(2)
    for column, display in zip(story_columns, ("A", "B")):
        story = packet["stories"][display]
        column.markdown(f"## Story {display}")
        column.markdown(f"### {story['title']}")
        column.write(story["content"])

    if locked:
        st.success("Review locked / 盲评已锁定")
        st.write(f"Preference: {existing['preference']}")
        st.json(existing.get("scores", {}))
        return

    with st.form("mood_blind_review"):
        preference = st.radio(
            "Overall preference / 总体偏好",
            ["A", "B", "Tie"],
            index=None,
            horizontal=True,
        )
        scores: dict[str, dict[str, int]] = {"A": {}, "B": {}}
        for display in ("A", "B"):
            st.markdown(f"**Story {display} scores (1–5)**")
            score_columns = st.columns(3)
            scores[display]["coherence"] = score_columns[0].slider(
                f"{display} coherence / 连贯性", 1, 5, 3
            )
            scores[display]["personalization"] = score_columns[1].slider(
                f"{display} personalization / 个性化", 1, 5, 3
            )
            scores[display]["trustworthiness"] = score_columns[2].slider(
                f"{display} trustworthiness / 可信度", 1, 5, 3
            )
        reason = st.text_area("Optional reason / 可选原因")
        submitted = st.form_submit_button("Submit and lock / 提交并锁定", type="primary")
    if submitted:
        if preference is None:
            st.error("Choose A, B or Tie before submitting.")
            return
        response = {
            "protocol_version": "mood-blind-response-v3",
            "case_id": packet["case_id"],
            "locked": True,
            "preference": preference,
            "scores": scores,
            "reason": reason.strip(),
        }
        atomic_write_json(RESPONSES_PATH, response)
        st.rerun()


if __name__ == "__main__":
    main()
