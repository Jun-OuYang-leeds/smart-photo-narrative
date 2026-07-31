"""Conservative evidence audit for the Qwen dual-track Story experiment.

This is a reproducible agent evidence audit, not a human multi-rater study and
not a second model judgment.  Explicitly cited units are checked against their
cited evidence; uncited legacy/title/transition units are checked against all
evidence frozen for that case.  Cross-lingual or partial matches are labelled
``uncertain`` rather than being forced to supported.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_story_evaluation import (  # noqa: E402
    CLAIM_AUDIT_PATH,
    CREATIVE_RESULTS_PATH,
    FAITHFUL_RESULTS_PATH,
    build_claim_audit_packet,
    default_context_factory,
    load_creative_case_manifest,
    load_faithful_cases,
)


EN_TOKEN = re.compile(r"[a-z0-9]+", re.I)
ZH_CHAR = re.compile(r"[\u4e00-\u9fff]")
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have",
    "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "the", "their",
    "there", "this", "to", "was", "we", "were", "with", "photo", "photos", "image", "images",
    "picture", "pictures", "scene", "scenes", "view", "visible", "see", "saw", "notice", "noticed",
    "observe", "observed", "remember", "recall", "moment", "day", "later", "then", "around",
}
ALIASES = {
    "people": "person", "persons": "person", "men": "man", "women": "woman", "children": "child",
    "sitting": "sit", "seated": "sit", "standing": "stand", "walking": "walk", "holding": "hold",
    "held": "hold", "wearing": "wear", "looking": "look", "typing": "type", "playing": "play",
    "buildings": "building", "cars": "car", "trees": "tree", "boats": "boat", "balloons": "balloon",
}
ZH_CONCEPTS = {
    "人物": "person", "人": "person", "男人": "man", "男子": "man", "女人": "woman", "女子": "woman",
    "孩子": "child", "猫": "cat", "狗": "dog", "鸟": "bird", "马": "horse", "鱼": "fish",
    "桌": "table", "椅": "chair", "电脑": "computer", "笔记本": "laptop", "键盘": "keyboard",
    "手机": "phone", "屏幕": "screen", "电视": "television", "书": "book", "杯": "cup",
    "瓶": "bottle", "盘": "plate", "食物": "food", "车": "car", "自行车": "bicycle",
    "船": "boat", "飞机": "airplane", "热气球": "balloon", "气球": "balloon", "桥": "bridge",
    "河": "river", "湖": "lake", "海": "sea", "山": "mountain", "树": "tree", "花": "flower",
    "草": "grass", "天空": "sky", "云": "cloud", "太阳": "sun", "建筑": "building",
    "房间": "room", "室内": "indoor", "室外": "outdoor", "厨房": "kitchen", "浴室": "bathroom",
    "街": "street", "道路": "road", "公园": "park", "泳池": "pool", "游泳池": "pool",
    "窗": "window", "门": "door", "灯": "light", "镜": "mirror", "床": "bed", "地板": "floor",
    "坐": "sit", "站": "stand", "走": "walk", "跑": "run", "拿": "hold", "握": "hold",
    "穿": "wear", "看": "look", "吃": "eat", "喝": "drink", "游泳": "swim", "打字": "type",
    "玩": "play", "拍摄": "camera", "拍照": "camera", "说话": "talk", "指": "point",
    "红": "red", "蓝": "blue", "绿": "green", "白": "white", "黑": "black", "黄": "yellow",
}
UNSUPPORTED = re.compile(
    r"\b(?:friend|family|mother|father|wife|husband|daughter|son|colleague|felt|feeling|happy|sad|"
    r"excited|nervous|calm|relaxed|proud|grateful|wanted|decided|planned|intended|working|studying|"
    r"waiting|preparing|because|therefore|probably|possibly|likely|perhaps|maybe|might|purpose|"
    r"suggest|suggests|suggested|hint|hinted|imply|implies|mystery|childhood|under the ice|unseen current|"
    r"favorite|favourite|delicious|smell|aroma|sound|laughter)\b|"
    r"朋友|家人|母亲|父亲|妻子|丈夫|女儿|儿子|同事|感到|觉得|开心|难过|兴奋|紧张|平静|"
    r"放松|骄傲|感谢|想要|决定|计划|打算|工作|学习|等待|准备|因为|所以|可能|也许|似乎|"
    r"大概|目的|最爱|美味|气味|香味|声音|笑声|小时候|暗流|冰层下",
    re.I,
)
UNSUPPORTED_FIRST_PERSON_ACTION = re.compile(
    r"\bi\s+(?:am|was|sat|stood|walked|ran|held|wore|swam|ate|drank|worked|studied|played|"
    r"entered|left|arrived|travelled|traveled|waited|prepared)\b|"
    r"我(?:正?在)?(?:坐|站|走|跑|拿|穿|游泳|吃|喝|工作|学习|玩|进入|离开|到达|等待|准备)",
    re.I,
)
FAITHFUL_FIRST_PERSON = re.compile(r"\b(?:i|me|my|we|us|our)\b|我|我们|我的|我们的", re.I)
FIGURATIVE = re.compile(
    r"\b(?:as if|as though|like a dream|seemed to whisper|time stood still|chapter of life)\b|"
    r"仿佛|好像一场梦|像梦一样|像[^，。！？]{0,24}一样|时间静止|岁月定格|新的篇章",
    re.I,
)
META_LEAK = re.compile(
    r"\b(?:the user|user wants|we need|we must|we should|let me|i should|system prompt|instructions?|"
    r"response|diary entry|reliable_time|verified_context|title:|start with|do not use|"
    r"let's (?:adjust|write|craft|create)|must (?:not|be|include|use|write))\b|"
    r"用户|提示词|系统提示|我应该|我们需要|不能说|需要写|输出格式|字段|回忆录中",
    re.I,
)
FALLBACK_PHRASES = (
    "direct visual record", "directly checkable", "documents visible", "observable visual",
    "直接核查", "直接观察", "可见的人物", "可观察的环境", "画面中可见",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stem(token: str) -> str:
    token = token.casefold()
    if token in ALIASES:
        return ALIASES[token]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _concepts(text: str) -> set[str]:
    values = {
        _stem(token) for token in EN_TOKEN.findall(text or "")
        if token.casefold() not in STOP and (token.isdigit() or len(token) > 1)
    }
    if ZH_CHAR.search(text or ""):
        values.update(concept for phrase, concept in ZH_CONCEPTS.items() if phrase in text)
        values.update(re.findall(r"\d+", text))
    return values


def _evidence_text(item: Mapping[str, Any]) -> str:
    values: list[str] = []
    for evidence in item.get("evidence", []):
        values.extend((
            str(evidence.get("caption") or ""),
            " ".join(str(value) for value in evidence.get("scene_graph_triples", [])),
            " ".join(str(value) for value in evidence.get("tags", [])),
            str(evidence.get("timestamp") or ""),
            str(evidence.get("location") or ""),
        ))
    return " ".join(values)


def _classify(item: Mapping[str, Any], *, fallback: bool) -> tuple[str, str]:
    claim = str(item.get("claim") or "").strip()
    field = str(item.get("field") or "")
    variant = str(item.get("variant_id") or "")
    lowered = claim.casefold()
    creative = variant.startswith("QC")
    if META_LEAK.search(claim):
        return "non_factual", "The unit leaks prompt planning, instructions or output-format commentary rather than making a story claim."
    if creative and field.startswith("transition_"):
        return "non_factual", "The registered Creative contract marks this transition as imaginative, not a photo fact."
    if creative and field in {"title", "opening", "closing"} and FIGURATIVE.search(claim):
        return "non_factual", "The unit is explicitly figurative Creative framing rather than a checkable photo claim."
    if fallback and any(phrase in lowered for phrase in FALLBACK_PHRASES):
        return "supported", "The deterministic fallback makes only a generic claim about the existence of the cited visible evidence group."
    if UNSUPPORTED.search(claim):
        return "unsupported", "The unit asserts identity, relationship, emotion, purpose, cause, sensory detail or speculation absent from verified context."
    if creative and UNSUPPORTED_FIRST_PERSON_ACTION.search(claim):
        return "unsupported", "Observer mode permits seeing/noticing but does not verify that the narrator performed the photographed action."
    if not creative and FAITHFUL_FIRST_PERSON.search(claim):
        return "unsupported", "Faithful mode has no verified first-person identity for this claim."
    if FIGURATIVE.search(claim):
        return (
            ("non_factual", "The text is an explicitly figurative Creative expression, not a checkable proposition.")
            if creative else
            ("unsupported", "Faithful text adds a figurative interpretation that is not licensed by the evidence.")
        )
    claim_concepts = _concepts(claim)
    evidence_concepts = _concepts(_evidence_text(item))
    if not claim_concepts:
        return "non_factual", "No independently checkable content concept remains after observer/discourse normalization."
    overlap = claim_concepts & evidence_concepts
    coverage = len(overlap) / len(claim_concepts)
    if coverage >= 0.60 or (len(claim_concepts) <= 3 and coverage >= 0.50):
        return "supported", f"Frozen observations/metadata license {len(overlap)}/{len(claim_concepts)} normalized content concepts."
    if overlap:
        return "uncertain", f"Only {len(overlap)}/{len(claim_concepts)} normalized concepts are licensed; the unit mixes matched and unmatched detail."
    if ZH_CHAR.search(claim):
        return "uncertain", "No glossary-level cross-lingual match was found; conservative audit avoids treating translation absence as factual contradiction."
    return "unsupported", "No material content concept is licensed by the frozen observations or reliable metadata."


def _contexts_and_records() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    faithful_cases = load_faithful_cases()
    creative_main, creative_reserve, _ = load_creative_case_manifest()
    records = [*_read_jsonl(FAITHFUL_RESULTS_PATH), *_read_jsonl(CREATIVE_RESULTS_PATH)]
    used = {str(record["case_id"]) for record in records}
    contexts = {
        case.case_id: default_context_factory(case, creative=False)
        for case in faithful_cases if case.case_id in used
    }
    contexts.update({
        case.case_id: default_context_factory(case, creative=True)
        for case in [*creative_main, *creative_reserve] if case.case_id in used
    })
    return contexts, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild claim units with the current evidence-selection rules")
    parser.add_argument("--show-status", choices=("supported", "unsupported", "uncertain", "non_factual"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    contexts, records = _contexts_and_records()
    if args.rebuild or not CLAIM_AUDIT_PATH.exists():
        audit = build_claim_audit_packet(records, contexts)
    else:
        audit = json.loads(CLAIM_AUDIT_PATH.read_text(encoding="utf-8"))
    by_record = {(str(record["case_id"]), str(record["variant_id"])): record for record in records}
    counts: Counter[str] = Counter()
    by_variant: dict[str, Counter[str]] = {}
    examples: list[dict[str, Any]] = []
    for item in audit["claims"]:
        record = by_record[(str(item["case_id"]), str(item["variant_id"]))]
        status, reason = _classify(item, fallback=bool(record.get("fallback")))
        item["status"] = status
        item["reason"] = reason
        item["source"] = "agent_evidence_audit"
        counts[status] += 1
        by_variant.setdefault(str(item["variant_id"]), Counter())[status] += 1
        if args.show_status == status and len(examples) < max(0, args.limit):
            examples.append({
                "audit_id": item["audit_id"], "variant_id": item["variant_id"],
                "field": item["field"], "claim": item["claim"], "reason": reason,
            })
    audit["audit_metadata"] = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "agent_evidence_audit",
        "method": "conservative deterministic lexical/concept triage using the frozen per-claim evidence, reviewed as an agent rubric",
        "limitations": (
            "This is not human multi-rater annotation or visual ground truth. BLIP/Qwen/CLIP observations may be wrong. "
            "Chinese claims without a glossary-level semantic match are marked uncertain rather than unsupported."
        ),
        "status_counts": dict(counts),
        "variant_status_counts": {key: dict(value) for key, value in by_variant.items()},
    }
    temporary = CLAIM_AUDIT_PATH.with_suffix(CLAIM_AUDIT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(CLAIM_AUDIT_PATH)
    print(json.dumps({
        "claim_count": len(audit["claims"]), "status_counts": dict(counts),
        "variant_status_counts": {key: dict(value) for key, value in by_variant.items()},
        "examples": examples,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
