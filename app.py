"""Smart Photo Narrative 2.0 Streamlit application.

CLIP and BLIP remain the original local visual foundation.  This UI adds the
independent BM25/Scene Graph channels, reliable temporal retrieval, event
organization, and evidence-cited storytelling around them.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import streamlit as st
from PIL import Image

from config import (
    APP_DB_PATH,
    DEVICE,
    IMAGES_PER_PAGE,
    MOOD_EXPERIMENT_ANNOTATION_SET,
    MOOD_EXPERIMENT_EVENT_ID,
    OLLAMA_MODEL,
    OLLAMA_THINK,
    PHOTOS_DIR,
    REMOTE_LLM_BACKENDS,
    REMOTE_LLM_BASE_URL,
    REMOTE_LLM_DEFAULT_BACKEND,
    REMOTE_LLM_MODEL,
    SEARCH_TOP_K,
    STORY_DEFAULT_TEMPERATURE,
    STORY_PRODUCTION_NUM_PREDICT,
)
from data_ingestion import scan_photos_directory
from event_service import EventService
from model_pipeline import get_vision_pipeline
from retrieval_engine import SearchFilters, SearchResponse, SearchResult, get_multimodal_retriever
from scene_graph_io import export_scene_graph_batch
from scene_graph_service import apply_indexable_scene_graphs
from story_agent import GeneratedStory, StoryContext, get_story_generator
from storage import MOOD_LABELS


st.set_page_config(page_title="Smart Photo Narrative", page_icon="📸", layout="wide")


@st.cache_resource
def retriever():
    return get_multimodal_retriever()


@st.cache_resource
def vision_pipeline():
    return get_vision_pipeline()


@st.cache_resource
def story_generator(backend: str = REMOTE_LLM_DEFAULT_BACKEND):
    return get_story_generator(backend=backend)


@st.cache_resource
def event_service():
    engine = retriever()
    return EventService(engine.storage, engine.vector_store, engine)


def _basket() -> list[str]:
    return st.session_state.setdefault("story_basket", [])


def _add_to_basket(photo_ids: Iterable[str]) -> None:
    basket = _basket()
    for photo_id in photo_ids:
        if photo_id not in basket:
            basket.append(photo_id)


def _confidence_label(value: float) -> str:
    if value >= 0.9:
        return "high"
    if value >= 0.6:
        return "medium"
    return "low"


@st.dialog("Photo details / 照片详情", width="large")
def show_photo_detail(photo_id: str) -> None:
    detail = retriever().storage.get_photo_detail(photo_id)
    if detail is None:
        st.error("Photo record not found / 找不到照片记录")
        return

    photo = detail["photo"]
    relative_path = str(photo["relative_path"])
    st.subheader(Path(relative_path).name)
    image_column, information_column = st.columns([3, 2])
    with image_column:
        image_path = PHOTOS_DIR / relative_path
        if image_path.is_file():
            st.image(str(image_path), width="stretch")
        else:
            st.warning("Photo file is missing / 照片文件不存在")
    with information_column:
        st.markdown("**Basic information / 基础信息**")
        captured_at = str(photo.get("captured_at") or "")
        if captured_at:
            st.write(f"📅 {captured_at[:16]}")
            confidence = float(photo.get("timestamp_confidence") or 0.0)
            st.caption(
                f"{photo.get('timestamp_source') or 'unknown'} · "
                f"{_confidence_label(confidence)} ({confidence:.2f})"
            )
        else:
            st.write("📅 Unknown capture time / 拍摄时间未知")
        location = str(photo.get("location") or "").strip()
        st.write(f"📍 {location}" if location else "📍 Unknown location / 地点未知")
        width = photo.get("image_width")
        height = photo.get("image_height")
        if width and height:
            st.write(f"🖼️ {width} × {height}")
        mood = detail.get("mood")
        if mood:
            mood_labels = {
                "neutral": "中性", "calm": "平静", "happy": "开心",
                "excited": "兴奋", "tense": "紧张", "sad": "难过",
            }
            st.write(
                "💭 Photographer mood / 拍摄者心情: "
                f"{mood_labels.get(str(mood['mood_label']), mood['mood_label'])} "
                f"({mood['mood_label']})"
            )
            st.caption("Manual · user-confirmed · photographer / 人工确认 · 拍摄者")
        else:
            st.write("💭 Photographer mood / 拍摄者心情: Not annotated / 未标注")

        caption = detail["caption"]
        if caption and str(caption.get("caption") or "").strip():
            caption_model = str(caption.get("model_name") or "").strip()
            normalized_model = caption_model.casefold()
            if "blip2" in normalized_model:
                caption_heading = "BLIP2 Caption / BLIP2 描述"
            elif "blip" in normalized_model:
                caption_heading = "BLIP Caption / BLIP 描述"
            else:
                caption_heading = "Primary Caption / 当前主要描述"
            st.markdown(f"**{caption_heading}**")
            if caption_model:
                st.caption(f"Primary model / 当前模型: `{caption_model}`")
            st.write(str(caption["caption"]))
        else:
            st.markdown("**Primary Caption / 当前主要描述**")
            st.info("No primary caption available / 暂无主要描述")

    st.markdown("**Qwen Scene Graph / Qwen 三元组**")
    triples = detail["triples"]
    if triples:
        st.caption(f"{len(triples)} triples / {len(triples)} 个三元组")
        st.dataframe(
            [
                {
                    "Subject / 主体": triple["subject"],
                    "Relation / 关系": triple["relation"],
                    "Object / 客体": triple["object"],
                }
                for triple in triples
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No Qwen Scene Graph available / 暂无 Qwen Scene Graph")


def display_results(
    results: Sequence[SearchResult],
    *,
    columns: int = 4,
    show_score: bool = False,
    selectable: bool = False,
    key_prefix: str = "result",
) -> None:
    if not results:
        st.info("No photos found / 没有找到照片")
        return
    for offset in range(0, len(results), columns):
        row = st.columns(columns)
        for column, result in zip(row, results[offset:offset + columns]):
            with column:
                path = Path(result.image_path)
                if path.is_file():
                    st.image(str(path), width="stretch")
                else:
                    st.warning(f"Missing: {result.id}")
                if show_score:
                    st.caption(f"RRF contribution: {result.score:.3f}")
                if result.datetime:
                    st.caption(f"📅 {result.datetime[:16]} · {_confidence_label(result.timestamp_confidence)}")
                if result.location:
                    st.caption(f"📍 {result.location.split(',')[0]}")
                if result.matched_modalities:
                    st.caption("Matched: " + " · ".join(result.matched_modalities))
                if result.explanation:
                    with st.expander("Why this photo? / 命中原因"):
                        st.write(result.explanation)
                        if result.matched_caption_terms:
                            st.write("Caption terms:", result.matched_caption_terms)
                        if result.filters_applied:
                            st.write("Pre-filters:", result.filters_applied)
                if selectable:
                    already = result.photo_id in _basket()
                    detail_column, story_column = st.columns(2)
                    if detail_column.button(
                        "View details / 查看详情",
                        key=f"detail_{key_prefix}_{result.photo_id}",
                        icon="🔎",
                        width="stretch",
                    ):
                        show_photo_detail(result.photo_id)
                    if story_column.button(
                        "✓ In story" if already else "+ Add to story",
                        key=f"story_{key_prefix}_{result.photo_id}",
                        disabled=already,
                        width="stretch",
                    ):
                        _add_to_basket([result.photo_id])
                        st.rerun()
                elif st.button(
                    "View details / 查看详情",
                    key=f"detail_{key_prefix}_{result.photo_id}",
                    icon="🔎",
                    width="stretch",
                ):
                    show_photo_detail(result.photo_id)


def _database_metrics() -> dict[str, int]:
    storage = retriever().storage
    with storage.transaction(write=False) as connection:
        return {
            "photos": connection.execute("SELECT COUNT(*) FROM photos").fetchone()[0],
            "captions": connection.execute("SELECT COUNT(DISTINCT photo_id) FROM captions WHERE status='ok'").fetchone()[0],
            "scene_graphs": connection.execute("SELECT COUNT(DISTINCT photo_id) FROM scene_graphs WHERE status='ok'").fetchone()[0],
            "events": connection.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "failures": connection.execute("SELECT COUNT(*) FROM index_failures").fetchone()[0],
        }


def _run_index(force: bool) -> None:
    paths = scan_photos_directory()
    progress = st.progress(0.0, text="Preparing index...")
    status_line = st.empty()

    def callback(payload):
        status_line.caption(json.dumps(dict(payload), ensure_ascii=False))
        indexed = int(payload.get("indexed_count", 0))
        if paths:
            progress.progress(min(1.0, indexed / len(paths)), text=f"{payload.get('phase')}: {indexed}/{len(paths)}")

    with st.spinner("CLIP → BLIP → SQLite/Chroma indexing..."):
        count = vision_pipeline().index_paths(paths, force=force, show_progress=False, progress_callback=callback)
    summary = vision_pipeline().last_index_summary
    progress.progress(1.0, text="Index run finished")
    st.success(f"Indexed/updated {count} photos · status={summary.status}")
    st.json(asdict(summary))


def page_library() -> None:
    st.header("📷 Library & Index / 照片库与索引")
    metrics = _database_metrics()
    cols = st.columns(5)
    for column, (label, value) in zip(cols, metrics.items()):
        column.metric(label.replace("_", " ").title(), value)

    with st.expander("Index controls / 索引控制", expanded=metrics["photos"] == 0):
        st.write("Incremental mode hashes files, skips completed CLIP/BLIP work, and resumes failed stages.")
        left, right = st.columns(2)
        if left.button("Scan / resume index", type="primary", width="stretch"):
            _run_index(force=False)
        if right.button("Force reproducible rebuild", width="stretch"):
            _run_index(force=True)
        st.caption(f"SQLite: {APP_DB_PATH}")

    ids = retriever().storage.metadata_eligible_ids()
    if not ids:
        st.info("The v2 index is empty. Click Scan / resume index; the legacy Chroma backup remains untouched.")
        return
    page_count = max(1, (len(ids) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE)
    page = st.number_input("Page", 1, page_count, 1) - 1
    page_ids = ids[page * IMAGES_PER_PAGE:(page + 1) * IMAGES_PER_PAGE]
    display_results(retriever().get_by_ids(page_ids), columns=5, selectable=True, key_prefix=f"gallery_{page}")


def _render_search_response(response: SearchResponse) -> None:
    st.caption(
        f"Parser: {response.query.parser_source} · type: {response.query.query_type} · "
        f"eligible before recall: {response.eligible_count} · channels: {', '.join(response.channels_used)}"
    )
    if response.query.warnings:
        st.warning("; ".join(response.query.warnings))
    if response.temporal_pairs:
        st.subheader(f"Before–after pairs ({len(response.temporal_pairs)})")
        for index, pair in enumerate(response.temporal_pairs, 1):
            st.markdown(f"**Pair {index} · {pair.delta_minutes:.0f} minutes · score {pair.score:.3f}**")
            before_col, arrow_col, after_col = st.columns([4, 1, 4])
            with before_col:
                st.caption("Before")
                display_results([pair.before], columns=1, selectable=True, key_prefix=f"before_{index}")
            with arrow_col:
                st.markdown("### →")
            with after_col:
                st.caption("After")
                display_results([pair.after], columns=1, selectable=True, key_prefix=f"after_{index}")
        return
    display_results(response.results, columns=4, show_score=True, selectable=True, key_prefix="search")


def page_search() -> None:
    st.header("🔍 Multimodal Search / 多模态检索")
    metrics = _database_metrics()
    if not metrics["photos"]:
        st.info("Index photos first.")
        return
    if not metrics["scene_graphs"]:
        st.info("Scene Graph is currently missing for personal photos. CLIP + Caption + Metadata remain active.")

    with st.form("search_form"):
        query = st.text_input(
            "Natural-language query",
            placeholder="一个人在桌边使用电脑 / waiting at an airport before boarding within 2 hours",
        )
        uploaded = st.file_uploader("Optional query image / 可选查询图片", type=["jpg", "jpeg", "png"])
        mode = st.selectbox(
            "Channels",
            ["hybrid", "vector", "keyword", "scene_graph"],
            format_func=lambda value: {
                "hybrid": "CLIP + Caption BM25 + Scene Graph",
                "vector": "CLIP only (A0)",
                "keyword": "Caption BM25 only",
                "scene_graph": "Scene Graph only",
            }[value],
        )
        top_k = st.slider("Top K", 1, 50, SEARCH_TOP_K)
        with st.expander("Metadata filters"):
            start, end = st.columns(2)
            start_date = start.date_input("Start date", value=None)
            end_date = end.date_input("End date", value=None)
            tags = st.multiselect("Tags", retriever().get_available_tags())
            location = st.text_input("Location contains")
            reliable_time_only = st.checkbox("Reliable time only (exclude mtime)", value=False)
            use_ollama_parser = st.checkbox("Use Ollama for complex query parsing", value=False)
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        filters = SearchFilters(
            start_date=str(start_date) if start_date else None,
            end_date=str(end_date) if end_date else None,
            tags=tuple(tags),
            location=location.strip() or None,
            min_timestamp_confidence=0.6 if reliable_time_only else None,
        )
        with st.spinner("Searching independent channels and fusing ranks..."):
            if uploaded is not None:
                with Image.open(uploaded) as source:
                    response = retriever().search_by_image(source.convert("RGB"), top_k=top_k, filters=filters)
            elif query.strip():
                channels = {
                    "hybrid": ("clip", "caption", "scene_graph"),
                    "vector": ("clip",),
                    "keyword": ("caption",),
                    "scene_graph": ("scene_graph",),
                }[mode]
                response = retriever().search(
                    query.strip(), top_k=top_k, filters=filters,
                    enabled_channels=channels, use_ollama_parser=use_ollama_parser,
                )
            else:
                response = None
                st.warning("Enter text or upload an image.")
        if response is not None:
            st.session_state.search_response = response
    if "search_response" in st.session_state:
        _render_search_response(st.session_state.search_response)


def page_events() -> None:
    st.header("🗓️ Event Organization / 事件组织")
    st.write("Reliable EXIF/filename times are segmented using time gaps, adjacent CLIP change, and local GPS distance.")
    if st.button("Rebuild automatic events", type="primary"):
        with st.spinner("Organizing events..."):
            organization = event_service().organize(persist=True)
        st.success(f"Created {len(organization.events)} events; {len(organization.unassigned_photo_ids)} low-confidence photos need confirmation.")

    events = event_service().list_events()
    if not events:
        st.info("No events yet. Build the v2 photo index, then click Rebuild automatic events.")
        return
    selected_date = st.selectbox("Date", sorted({event.date_local for event in events}, reverse=True))
    for event in [item for item in events if item.date_local == selected_date]:
        with st.expander(f"{event.title} · {len(event.photo_ids)} photos", expanded=True):
            st.caption(f"{event.start_at[11:16]}–{event.end_at[11:16]} · {event.method}")
            display_results(
                retriever().get_by_ids(event.representative_ids or event.photo_ids),
                columns=4,
                selectable=True,
                key_prefix=f"event_{event.event_id}",
            )
            if st.button("Add complete event to story", key=f"add_event_{event.event_id}"):
                _add_to_basket(event.photo_ids)
                st.rerun()


def _story_source() -> tuple[list[str], str, str, str | None]:
    source = st.radio("Story source", ["Story basket", "Date", "Event"], horizontal=True)
    if source == "Story basket":
        return list(_basket()), "Selected photos", "basket", None
    if source == "Date":
        dates = retriever().get_available_dates()
        if not dates:
            return [], "Date", "date", None
        date = st.selectbox("Date", dates)
        return retriever().storage.metadata_eligible_ids(date_from=date, date_to=date), date, "date", None
    events = event_service().list_events()
    if not events:
        return [], "Event", "event", None
    selected = st.selectbox("Event", events, format_func=lambda item: item.title)
    return list(selected.photo_ids), selected.title, "event", selected.event_id


def _edit_basket() -> None:
    basket = _basket()
    if not basket:
        st.info("Add photos from Search, Library, or Events.")
        return
    docs = retriever()._documents(basket)
    for index, photo_id in enumerate(list(basket)):
        cols = st.columns([1, 5, 1, 1, 1])
        path = PHOTOS_DIR / docs[photo_id]["relative_path"]
        cols[0].image(str(path), width="stretch")
        cols[1].write(docs[photo_id]["caption"] or docs[photo_id]["relative_path"])
        if cols[2].button("↑", key=f"up_{photo_id}", disabled=index == 0):
            basket[index - 1], basket[index] = basket[index], basket[index - 1]
            st.rerun()
        if cols[3].button("↓", key=f"down_{photo_id}", disabled=index == len(basket) - 1):
            basket[index + 1], basket[index] = basket[index], basket[index + 1]
            st.rerun()
        if cols[4].button("×", key=f"remove_{photo_id}"):
            basket.remove(photo_id)
            st.rerun()


def _event_summary_text(story: GeneratedStory) -> str:
    summary = story.event_summary or {}
    date_value = str(summary.get("date") or "")
    time_value = str(summary.get("time_range") or "")
    locations = [str(value) for value in summary.get("locations", []) if str(value).strip()]
    parts = [date_value or ("时间未知" if story.language.startswith("zh") else "Time unavailable")]
    if time_value:
        parts.append(time_value)
    if locations:
        parts.append(" / ".join(locations))
    return " · ".join(parts)


def _story_markdown(story: GeneratedStory) -> str:
    if story.opening or story.closing:
        sections = [f"# {story.title}", f"_{_event_summary_text(story)}_"]
        sections.extend(str(block["text"]) for block in story.narrative_blocks if block.get("text"))
        audit = [
            "<details>",
            "<summary>故事依据与校验 / Story grounding audit</summary>",
            "",
            f"- Model: {story.model}",
            f"- Prompt: {story.prompt_version}",
            f"- Status: {story.status}",
            f"- Narrator role: {story.narrator_role}",
        ]
        for block in story.narrative_blocks:
            if block.get("kind") == "grounded_fact":
                audit.append(
                    f"- Grounded {block.get('group_id')}: Evidence {', '.join(block.get('evidence_ids', []))}"
                )
            elif block.get("kind") == "verified_photographer_mood":
                audit.append(
                    f"- Verified photographer mood after {block.get('group_id')}: "
                    f"Evidence {block.get('evidence_id')} = {block.get('mood_label')} (manual)"
                )
            else:
                audit.append(f"- {block.get('kind')}: non-photo creative narration")
        if story.use_photographer_mood:
            audit.append("- Photographer mood metadata: enabled; manual, user-confirmed photographer state")
            for mood in story.photographer_moods:
                audit.append(
                    f"  - Evidence {mood.get('evidence_id')}: {mood.get('mood_label')} "
                    f"(source={mood.get('source')}, subject=photographer)"
                )
            for claim in story.mood_claims:
                audit.append(
                    f"  - Mood claim in {claim.get('block_kind')}: {', '.join(claim.get('mood_labels', []))}"
                )
        if story.validation_codes:
            audit.append("- Validation history: " + " | ".join(story.validation_codes))
        audit.extend(["", "</details>"])
        sections.append("\n".join(audit))
        return "\n\n".join(sections)

    sections = [f"# {story.title}"]
    for index, paragraph in enumerate(story.paragraphs):
        sections.append(f"{paragraph.text}\n\n_Evidence: {', '.join(paragraph.evidence_ids)}_")
        if index < len(story.creative_transitions):
            sections.append(
                "> ✨ 创意过渡——非照片事实\n>\n> "
                + story.creative_transitions[index].replace("\n", "\n> ")
            )
    for transition in story.creative_transitions[len(story.paragraphs):]:
        sections.append("> ✨ 创意过渡——非照片事实\n>\n> " + transition.replace("\n", "\n> "))
    return "\n\n".join(sections)


def _humanize_story_failure(story: GeneratedStory) -> list[str]:
    """Turn per-attempt diagnostics / validation codes into readable bilingual reasons."""

    def describe(code: str) -> str:
        head = code.split(":", 1)[0]
        if head.startswith("E_LENGTH_TRUNCATED"):
            return ("生成超出长度上限，输出被截断为无效 JSON "
                    "/ exceeded the generation length limit and was truncated to invalid JSON")
        if head.startswith("E_GENERATION"):
            detail = code.split(":", 2)[-1].strip()
            if "RuntimeError" in code or "Ollama" in code:
                return "Ollama/模型不可用 / Ollama or the model was unavailable"
            return f"模型输出不是合法 JSON / model output was not valid JSON（{detail}）"
        mapping = {
            "E_UNSUPPORTED_CLAIM": "含无证据支持的推测/情绪/因果词 / unsupported speculation, emotion or causal term",
            "E_FIRST_PERSON_IN_FACT": "事实段落出现第一人称 / first person used in a factual paragraph",
            "E_CREATIVE_IN_FAITHFUL": "忠实模式不应包含创意过渡 / creative transition present in faithful mode",
            "E_CREATIVE_TRANSITION_COUNT": "创意过渡数量不符 / wrong number of creative transitions",
            "E_GROUP_PARAGRAPH_COUNT": "段落数与证据组数不符 / paragraph count does not match evidence groups",
            "E_GROUP_COVERAGE": "有证据组未被覆盖 / some evidence groups are uncovered",
            "E_CITATION_MISSING": "段落缺少证据引用 / a paragraph has no evidence citations",
            "E_CITATION_UNKNOWN": "引用了不存在的证据 ID / cited an unknown evidence ID",
            "E_CONFLICT_IN_BODY": "正文使用了冲突/不确定的观察 / a conflicted observation appears in the body",
            "E_LANGUAGE_ZH": "中文占比不足 / Chinese-character ratio too low",
            "E_LANGUAGE_EN": "英文中混入过多中文 / too many Chinese characters for English output",
            "E_PARAGRAPH_DUPLICATE": "段落之间过于重复 / paragraphs are too similar",
            "E_TITLE_MISSING": "缺少标题 / title is missing",
        }
        return mapping.get(head, code)

    lines: list[str] = []
    for diag in story.attempt_diagnostics:
        phase = diag.get("phase")
        if phase == "ok":
            continue
        attempt = diag.get("attempt")
        codes = diag.get("codes") or []
        reasons = "；".join(dict.fromkeys(describe(str(c)) for c in codes)) or "未知原因 / unknown"
        lines.append(f"第 {attempt} 次 / attempt {attempt}: {reasons}")
    if not lines:
        for code in story.validation_codes or list(dict.fromkeys(story.warnings)):
            lines.append(describe(str(code)))
    return lines


def _render_story(story: GeneratedStory) -> None:
    if story.status == "ok":
        st.success("✅ Qwen model output accepted / 模型输出已通过校验")
    elif story.status == "repaired":
        st.warning("🛠️ Qwen output accepted after one repair / 模型输出经一次修复后通过")
    elif story.status in {"error", "fallback"}:
        st.error("❌ 故事生成失败：两次尝试均未通过校验，未展示任何占位文本 / "
                 "Story generation failed: both attempts were rejected, no placeholder text is shown")
        if story.title:
            st.markdown(f"### {story.title}")
        reasons = _humanize_story_failure(story)
        if reasons:
            with st.expander("失败原因（逐次）/ Failure reasons (per attempt)", expanded=True):
                for line in reasons:
                    st.write(f"- {line}")
        st.caption(
            "可点“重新生成”重试；忠实模式反复失败通常是模型输出了无证据支持的推断，创意模式多为长度截断。 / "
            "Use Retry; repeated faithful failures usually mean unsupported inference, creative failures usually mean truncation."
        )
        st.caption(f"Model: {story.model} · mode: {story.mode} · photos: {story.photo_count} · status: {story.status}")
        return

    st.markdown(f"## {story.title}")
    if story.opening or story.closing:
        st.caption(_event_summary_text(story))
        for block in story.narrative_blocks:
            if block.get("text"):
                st.markdown(str(block["text"]))
        with st.expander("故事依据与校验 / Story grounding audit", expanded=False):
            st.caption(
                f"Model: {story.model} · Prompt: {story.prompt_version} · mode: {story.mode} · "
                f"narrator: {story.narrator_role} · photos: {story.photo_count} · status: {story.status}"
            )
            if story.use_photographer_mood:
                st.markdown("**拍摄者心情 Metadata / Photographer mood metadata**")
                for mood in story.photographer_moods:
                    st.write(
                        f"- {mood.get('evidence_id')}: {mood.get('mood_label')} · "
                        "manual · user-confirmed · photographer"
                    )
                if story.mood_claims:
                    st.caption(
                        "Validated mood claims: "
                        + "; ".join(
                            f"{claim.get('block_kind')}={','.join(claim.get('mood_labels', []))}"
                            for claim in story.mood_claims
                        )
                    )
            for block in story.narrative_blocks:
                kind = str(block.get("kind") or "")
                if kind == "grounded_fact":
                    st.markdown(
                        f"**事实层 / Grounded fact · {block.get('group_id')}**  \n"
                        f"Evidence: {', '.join(block.get('evidence_ids', []))}"
                    )
                elif kind == "verified_photographer_mood":
                    st.markdown(
                        "**人工确认拍摄者心情 / Verified photographer mood**  \n"
                        f"Evidence: {block.get('evidence_id')} · Mood: {block.get('mood_label')} · "
                        "subject: photographer"
                    )
                else:
                    st.markdown(f"**创意层 / Non-photo creative · {kind}**")
                st.write(block.get("text"))
            if story.uncertain_observations:
                st.warning("Conflicting model observations excluded from the grounded facts:")
                for observation in story.uncertain_observations:
                    st.write(f"- {observation.text} (Evidence: {', '.join(observation.evidence_ids)})")
            if story.unused_evidence_ids:
                st.caption("Unused evidence IDs: " + ", ".join(story.unused_evidence_ids))
            if story.warnings:
                st.markdown("**Validation history / 校验历史**")
                for warning in story.warnings:
                    st.write(f"- {warning}")
            photos = retriever().get_by_ids(story.photo_ids)
            display_results(photos, columns=5, key_prefix="story_audit_evidence")
        markdown = _story_markdown(story)
        left, right = st.columns(2)
        left.download_button("Download Markdown", markdown, file_name="grounded_story.md")
        right.download_button(
            "Download JSON",
            json.dumps({
                **story.to_dict(), "photo_ids": story.photo_ids, "warnings": story.warnings,
                "model": story.model, "status": story.status, "prompt_hash": story.prompt_hash,
            }, ensure_ascii=False, indent=2),
            file_name="grounded_story.json",
        )
        return

    for index, paragraph in enumerate(story.paragraphs, 1):
        st.markdown(paragraph.text)
        st.caption(f"{paragraph.group_id} · Evidence: " + ", ".join(paragraph.evidence_ids))
        transition_index = index - 1
        if transition_index < len(story.creative_transitions):
            st.info(
                "✨ **创意过渡——非照片事实 / Creative transition—not a photo fact**\n\n"
                + story.creative_transitions[transition_index]
            )
    if story.uncertain_observations:
        st.warning("Conflicting or uncertain model observations are excluded from the factual narrative.")
        for observation in story.uncertain_observations:
            st.write(f"- {observation.text} (Evidence: {', '.join(observation.evidence_ids)})")
    for transition in story.creative_transitions[len(story.paragraphs):]:
        st.info("✨ **创意过渡——非照片事实 / Creative transition—not a photo fact**\n\n" + transition)
    if story.unused_evidence_ids:
        st.caption("Unused evidence IDs: " + ", ".join(story.unused_evidence_ids))
    if story.warnings:
        with st.expander("Validation history / 校验历史"):
            st.write(story.warnings)
    st.caption(f"Model: {story.model} · mode: {story.mode} · photos: {story.photo_count} · status: {story.status}")
    photos = retriever().get_by_ids(story.photo_ids)
    display_results(photos, columns=5, key_prefix="story_evidence")
    markdown = _story_markdown(story)
    left, right = st.columns(2)
    left.download_button("Download Markdown", markdown, file_name="grounded_story.md")
    right.download_button(
        "Download JSON",
        json.dumps({
            **story.to_dict(), "photo_ids": story.photo_ids, "warnings": story.warnings,
            "model": story.model, "status": story.status, "prompt_hash": story.prompt_hash,
        }, ensure_ascii=False, indent=2),
        file_name="grounded_story.json",
    )


def _render_evidence_plan(context: StoryContext) -> None:
    with st.expander(
        f"Evidence-group preview / 证据分组预览 ({len(context.groups)} groups, {context.photo_count} photos)",
        expanded=True,
    ):
        for group in context.groups:
            start = group.start_at[11:16] if group.start_at and len(group.start_at) >= 16 else "?"
            end = group.end_at[11:16] if group.end_at and len(group.end_at) >= 16 else "?"
            st.markdown(
                f"**{group.group_id}** · {start}–{end} · {len(group.evidence_ids)} photos · "
                f"representative {group.representative_evidence_id}"
            )
            st.caption("Evidence: " + ", ".join(group.evidence_ids))
            if group.near_duplicate_evidence_ids:
                st.caption("Compressed near-duplicates: " + ", ".join(group.near_duplicate_evidence_ids))
            if group.locations:
                st.caption("Location: " + ", ".join(group.locations))
            for conflict in group.conflicts:
                st.warning(conflict)


_MOOD_LABEL_NAMES = {
    "": "未标注 / Not annotated",
    "neutral": "中性 / Neutral",
    "calm": "平静 / Calm",
    "happy": "开心 / Happy",
    "excited": "兴奋 / Excited",
    "tense": "紧张 / Tense",
    "sad": "难过 / Sad",
}


def _mood_experiment_photo_ids() -> list[str]:
    return retriever().storage.event_photo_ids(MOOD_EXPERIMENT_EVENT_ID)


def _is_mood_experiment_selection(photo_ids: Sequence[str]) -> bool:
    target = _mood_experiment_photo_ids()
    return bool(target) and set(photo_ids) == set(target) and len(photo_ids) == len(target)


def _render_mood_editor(photo_ids: Sequence[str]) -> None:
    """Edit only the frozen five-photo mood case; save the group atomically."""

    storage = retriever().storage
    moods = storage.get_photo_moods(photo_ids)
    docs = retriever()._documents(photo_ids)
    selections: dict[str, str] = {}
    with st.expander("拍摄者心情 Metadata（实验） / Photographer mood metadata", expanded=True):
        st.info(
            "这里记录的是按下快门时拍摄者本人回忆并确认的心情，不是照片中人物的心情，也不是心率推断。"
        )
        for photo_id in photo_ids:
            doc = docs.get(photo_id, {})
            current = moods.get(photo_id)
            columns = st.columns([1, 3, 2])
            image_path = PHOTOS_DIR / str(doc.get("relative_path") or "")
            if image_path.is_file():
                columns[0].image(str(image_path), width="stretch")
            columns[1].write(Path(str(doc.get("relative_path") or photo_id)).name)
            selections[photo_id] = columns[2].selectbox(
                "Mood / 心情",
                ["", *MOOD_LABELS],
                index=["", *MOOD_LABELS].index(current.mood_label if current is not None else ""),
                format_func=lambda value: _MOOD_LABEL_NAMES[value],
                key=f"photo_mood_{photo_id}",
                label_visibility="collapsed",
            )
        annotated = sum(bool(value) for value in selections.values())
        st.caption(f"{annotated}/{len(photo_ids)} photos annotated / 已标注")
        if st.button("保存整组心情 / Save mood set", key="save_mood_set"):
            storage.save_photo_moods(
                {photo_id: label or None for photo_id, label in selections.items()},
                annotation_set=MOOD_EXPERIMENT_ANNOTATION_SET,
            )
            st.success("Photographer mood metadata saved atomically / 整组心情已保存")
            st.rerun()


def page_story() -> None:
    st.header("📖 Evidence-grounded Story / 证据约束叙事")
    with st.expander(f"Story basket ({len(_basket())})", expanded=True):
        _edit_basket()
    photo_ids, label, source_kind, event_id = _story_source()
    backend = st.selectbox(
        "Story backend / 故事后端",
        list(REMOTE_LLM_BACKENDS),
        index=REMOTE_LLM_BACKENDS.index(REMOTE_LLM_DEFAULT_BACKEND) if REMOTE_LLM_DEFAULT_BACKEND in REMOTE_LLM_BACKENDS else 0,
        format_func=lambda x: "Remote API (DashScope) / 远程 API" if x == "remote" else "Local Ollama / 本地",
        help=(
            "Local uses the on-device Ollama qwen model. Remote calls the DashScope OpenAI-compatible "
            f"endpoint ({REMOTE_LLM_MODEL}); the API key is read only from the DASHSCOPE_API_KEY env var and is "
            "never stored in the UI. Remote failures raise an auditable error and do not fall back to local."
        ),
    )
    st.session_state["story_backend"] = backend
    if backend == "remote":
        st.caption(
            f"Remote model `{REMOTE_LLM_MODEL}` · endpoint `{REMOTE_LLM_BASE_URL}`. "
            "Make sure DASHSCOPE_API_KEY is set and {WorkspaceId} in the endpoint has been replaced."
        )
    left, middle = st.columns(2)
    mode = left.selectbox("Mode", ["faithful", "creative"], format_func=lambda x: "Faithful / 忠实" if x == "faithful" else "Creative / 创意")
    language = middle.selectbox("Language", ["zh", "en"], format_func=lambda x: "中文" if x == "zh" else "English")
    st.caption(
        "Faithful is neutral and checkable: no first person, speculation, identity, emotion, purpose or causal claims. "
        "Creative is a continuous first-person memoir; factual and imaginative layers remain separated in the audit. "
        f"Decoding temperature is fixed at {STORY_DEFAULT_TEMPERATURE}, the same value as the frozen evaluations; the "
        "mode—not a randomness slider—is the user-facing control. / 解码温度固定为 "
        f"{STORY_DEFAULT_TEMPERATURE}（与冻结实验一致），用户可调的是模式而非随机性。"
    )
    narrator_is_subject = False
    use_photographer_mood = False
    if mode == "creative":
        narrator_is_subject = st.checkbox(
            "照片中的主要人物是我 / The principal photographed person is me",
            value=False,
            help=(
                "Only enable this when you personally confirm the identity. It permits evidenced actions to use ‘I’, "
                "but does not confirm emotion, purpose, relationships, or causes."
            ),
        )
        if _is_mood_experiment_selection(photo_ids):
            _render_mood_editor(photo_ids)
            annotated_count = len(retriever().storage.get_photo_moods(photo_ids))
            use_photographer_mood = st.checkbox(
                "Use photographer mood metadata / 使用拍摄者心情 Metadata",
                value=False,
                disabled=annotated_count == 0,
                help=(
                    "Mood is used only by Creative v6 and refers to the photographer at capture time. "
                    "Faithful Story and retrieval always ignore it."
                ),
            )
            if 0 < annotated_count < len(photo_ids):
                st.warning(
                    f"Only {annotated_count}/{len(photo_ids)} photos have mood labels. "
                    "The formal M0/M1 case requires all five."
                )
        elif photo_ids:
            st.caption(
                "Mood annotation is limited to the frozen 2026-07-07 Event 1 five-photo experiment."
            )
    verified_context = st.text_area(
        "Verified context / 用户确认背景（可选）",
        placeholder="Only enter facts you personally confirm, for example: 'This is my graduation ceremony.'",
        help="Only this field may support personal identity, relationship, emotion or purpose. BLIP/Qwen/CLIP remain model observations.",
    )
    context = story_generator(backend).aggregator.aggregate_by_photo_ids(
        photo_ids, label=label, event_id=event_id, verified_context=verified_context,
        source_kind=source_kind,
        narrator_role="confirmed_subject" if narrator_is_subject else "observer",
        use_photographer_mood=use_photographer_mood if mode == "creative" else False,
    )
    if context.evidence:
        _render_evidence_plan(context)
    if st.button("Generate grounded story", type="primary", disabled=not photo_ids):
        with st.spinner("Generating and validating citations..."):
            story = story_generator(backend).generate_story_with_context(
                context, temperature=STORY_DEFAULT_TEMPERATURE, mode=mode, language=language,
                allow_deterministic_fallback=False, num_predict=STORY_PRODUCTION_NUM_PREDICT,
            )
        st.session_state.generated_story = story
    if "generated_story" in st.session_state:
        generated_story = st.session_state.generated_story
        _render_story(generated_story)
        if generated_story.status == "error":
            if st.button(
                "重新生成 / Retry story",
                type="primary",
                disabled=not photo_ids,
                key="retry_story",
            ):
                with st.spinner("Regenerating and validating the grounded story..."):
                    st.session_state.generated_story = story_generator(backend).generate_story_with_context(
                        context, temperature=STORY_DEFAULT_TEMPERATURE, mode=generated_story.mode, language=language,
                        allow_deterministic_fallback=False, num_predict=STORY_PRODUCTION_NUM_PREDICT,
                    )
                st.rerun()


def page_scene_graph() -> None:
    st.header("☁️ Qwen Scene Graph Workflow")
    st.warning("The existing 10,056-row JSONL is Geograph-only. It has 67 filename collisions but 0 content-hash matches with this album.")
    st.code(
        "python scripts/export_scene_graph_batch.py --album-root photos --export-dir scene_graph_exports/personal_v1 --dataset-id personal_v1\n"
        "# upload the generated batch to AIRE and follow scripts/cloud/README_scene_graph_aire.md\n"
        "python scripts/import_scene_graph_results.py ...\n"
        "python scripts/index_scene_graph_results.py audited_indexable.jsonl --dry-run",
        language="powershell",
    )
    dataset_id = st.text_input("Dataset ID", "personal_v1")
    if st.button("Export verified personal-photo batch"):
        output = Path(__file__).parent / "scene_graph_exports" / dataset_id
        report = export_scene_graph_batch(PHOTOS_DIR, output, dataset_id=dataset_id)
        st.success(f"Exported {report.exported} photos to {output}")
        st.json(report.to_dict())

    uploaded = st.file_uploader("Audited indexable JSONL", type=["jsonl"], key="sg_indexable")
    if uploaded is not None:
        dry_run = st.checkbox("Dry run", value=True, key="sg_dry")
        if st.button("Validate / index Scene Graph"):
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as handle:
                handle.write(uploaded.getvalue())
                temp_path = Path(handle.name)
            try:
                summary = apply_indexable_scene_graphs(temp_path, dry_run=dry_run)
                st.json(asdict(summary))
            finally:
                temp_path.unlink(missing_ok=True)


def page_diagnostics() -> None:
    st.header("🧪 Diagnostics & Experiment State")
    st.json(_database_metrics())
    try:
        st.json(retriever().vector_store.counts())
    except Exception as exc:
        st.error(f"Vector index: {type(exc).__name__}: {exc}")
    storage = retriever().storage
    with storage.transaction(write=False) as connection:
        runs = [dict(row) for row in connection.execute("SELECT * FROM index_runs ORDER BY started_at DESC LIMIT 10")]
        failures = [dict(row) for row in connection.execute("SELECT * FROM index_failures ORDER BY failure_id DESC LIMIT 50")]
    st.subheader("Recent index runs")
    st.dataframe(runs, width="stretch")
    st.subheader("Recorded failures")
    st.dataframe(failures, width="stretch")


def main() -> None:
    with st.sidebar:
        st.title("📸 Smart Photo Narrative")
        page = st.radio(
            "Navigation",
            ["📷 Library", "🔍 Search", "🗓️ Events", "📖 Story", "☁️ Scene Graph", "🧪 Diagnostics"],
            label_visibility="collapsed",
        )
        st.divider()
        st.metric("Story basket", len(_basket()))
        active_backend = st.session_state.get("story_backend", REMOTE_LLM_DEFAULT_BACKEND)
        if active_backend == "remote":
            st.caption(f"Backend: `remote` ({REMOTE_LLM_MODEL})")
        else:
            thinking_status = "on" if OLLAMA_THINK else "off"
            st.caption(f"Backend: `local` ({OLLAMA_MODEL}) · thinking `{thinking_status}` (optional)")
        st.caption(f"Device: `{DEVICE}`")
        st.caption(f"CLIP + BLIP: `torchtest`")

    {
        "📷 Library": page_library,
        "🔍 Search": page_search,
        "🗓️ Events": page_events,
        "📖 Story": page_story,
        "☁️ Scene Graph": page_scene_graph,
        "🧪 Diagnostics": page_diagnostics,
    }[page]()


if __name__ == "__main__":
    main()
