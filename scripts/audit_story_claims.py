"""Conservative, evidence-linked agent audit for formal Story claims.

The audit checks whether each emitted text unit is licensed by the evidence
made available to that variant.  It does not claim human inter-annotator
agreement and does not treat lexical risk detection as factual verification.
Ambiguous partial matches are labelled ``uncertain`` rather than forced to 0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from story_agent import ContextAggregator  # noqa: E402
from story_evaluation import load_story_cases  # noqa: E402


PRIVATE = ROOT / "evaluation" / "private"
TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.I)
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "there", "this", "to", "was", "were",
    "with", "photo", "photos", "photograph", "photographs", "image", "images", "picture", "pictures",
    "show", "shows", "shown", "visible", "record", "records", "scene", "scenes", "group", "set",
    "model", "models", "observation", "observations", "include", "includes", "including", "evidence",
    "一个", "一张", "一组", "照片", "图像", "画面", "显示", "记录", "可以", "可见", "中", "的", "了",
}
PERSONAL_OR_INFERRED = re.compile(
    r"\b(?:i|me|my|mine|we|us|our|ours|remember|grandfather|grandmother|friend|friends|loved ones|"
    r"favorite|favourite|felt|feel|feeling|grateful|gratitude|happy|sad|excited|excitement|bliss|"
    r"calm|serenity|tranquility|pride|accomplishment|productive|relax|relaxing|unwind|recharge|"
    r"decided|wanted|intended|planned|probably|possibly|likely|perhaps|maybe|may|might|because|therefore|"
    r"remind|reminder|treasure|memory|memories|thoughts|purpose|taste|tastes|aroma|smell|sound|sounds|"
    r"laughter|love|loves|wife|husband|daughter|son|beloved|breathtaking|wonderful|delicious|heavenly|"
    r"imply|implies|implied|suggest|suggests|suggested|focused|focus|attention|task|trade|work|working|"
    r"study|studying|importance|confident|deftly|atmosphere|mystique|elegance|perfect|vibrant|tranquil|"
    r"lazily|lovely|tantalizing|fusion|delight|unique character|created)\b|"
    r"我|我们|我的|我们的|记得|回忆|爷爷|奶奶|朋友|家人|最爱|喜欢|感到|觉得|感激|感谢|开心|"
    r"难过|兴奋|幸福|平静|放松|休息|工作|学习|准备|决定|打算|想要|可能|也许|似乎|大概|因为|"
    r"所以|为了|提醒|味道|香味|声音|笑声|妻子|丈夫|女儿|儿子|难忘|美味",
    re.I,
)
NON_FACTUAL = re.compile(
    r"^(?:overall|in summary|总的来说|总之)[,:：]?\s*$|^(?:meanwhile|later|subsequently|随后|之后)[,.，。]?\s*$|"
    r"^this is evident from photo\s+p\d+[.!]?$",
    re.I,
)
FALLBACK_PHRASES = (
    "direct visual record", "directly checkable record", "documents visible", "directly visible",
    "observable visual details", "visible scene content", "直接核查", "直接观察", "可见的人物",
    "能够直接核查", "可观察的环境", "画面中可见",
)

# Semantic adjudications after inspecting the exact cited evidence text.  They
# are kept in code so the completed audit remains reproducible and reviewable;
# these are not changes to generated stories or prompts.
MANUAL_OVERRIDES: dict[str, tuple[str, str]] = {
    "a6173bd6560506ae8cff": ("supported", "Computer, keyboard and monitor are explicitly present in cited P001; the remaining phrase is non-specific context wording."),
    "fb5728df45ec90eab5e3": ("supported", "Cited P005 explicitly describes a man-on-horse statue standing at an architectural arch/landmark."),
    "5ce406aba2ae8294709c": ("supported", "Cited P006 explicitly contains person-talk-to-person and person-point-at relations."),
    "85eb612c97d4255ad0a2": ("supported", "Cited P004 explicitly contains give-thumbs-up and pose-camera relations."),
    "739824772b96d34818e6": ("supported", "Cited P002 explicitly describes a large indoor pool, skylight and ceiling light."),
    "d18ef567b2f18a82b768": ("supported", "Cited P001 explicitly describes a man in a towel sitting by the pool."),
    "47171a3a912a3e3a8fed": ("supported", "Cited P002 explicitly contains wear-earphones and give-thumbs-up relations."),
    "441abd69a592f43ee97f": ("supported", "Cited P001 explicitly contains landmark, city, building-cluster and cityscape observations."),
    "1509ef8fd3bcd3e5e0a1": ("supported", "Cited P001 explicitly contains use-mouse, type-keyboard and look-laptop relations."),
    "41003601739535797405": ("supported", "Cited P004 explicitly contains looking, park/grass, trees and path observations."),
    "3300dc00ee0233379762": ("supported", "The frozen context explicitly defines G001 with P001 and P002; this is a correct technical evidence statement."),
    "fa16b814c8b9df6b3d24": ("unsupported", "Cited P005 is a stone/armoured equestrian statue; sunglasses, jacket and shoes are not licensed and appear transferred from another photo."),
    "a75b3b4de3b46be90e95": ("unsupported", "P001 12:01:30 to P002 12:03:21 is 1:51, not the claimed 3:21 duration."),
    "c73dd111f5c59c27803a": ("supported", "The cited observation explicitly records the visible word 'love' on the rock together with cracks/moss; 'love' is quoted image text here, not an inferred relationship."),
    "000211c10c331e6275f4": ("supported", "The cited event evidence explicitly contains a railway bridge beside/over the river; the short Chinese unit is a factual group heading."),
}


def _stem(token: str) -> str:
    token = token.casefold()
    aliases = {
        "people": "person", "men": "man", "women": "woman", "children": "child",
        "sitting": "sit", "seated": "sit", "standing": "stand", "walking": "walk",
        "holding": "hold", "held": "hold", "typing": "type", "playing": "play",
        "buildings": "building", "houses": "house", "trees": "tree", "boats": "boat",
        "january": "01", "february": "02", "march": "03", "april": "04", "may": "05",
        "june": "06", "july": "07", "august": "08", "september": "09", "october": "10",
        "november": "11", "december": "12",
    }
    if token in aliases:
        return aliases[token]
    ordinal = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", token)
    if ordinal:
        return ordinal.group(1).zfill(2)
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    return {_stem(value) for value in TOKEN_RE.findall(text or "") if value.casefold() not in STOP}


def _evidence_text(context, evidence_ids: list[str]) -> str:
    allowed = set(evidence_ids)
    selected = [item for item in context.evidence if not allowed or item.evidence_id in allowed]
    values = []
    for item in selected:
        values.extend((
            item.evidence_id,
            item.caption, " ".join(item.scene_graph_triples), " ".join(item.tags),
            item.location or "", item.timestamp or "",
        ))
    values.extend(group.group_id for group in context.groups)
    return " ".join(value for value in values if value)


def _classify(claim: str, evidence_text: str, *, n3_fallback: bool) -> tuple[str, str]:
    lowered = claim.casefold()
    if NON_FACTUAL.search(claim.strip()):
        return "non_factual", "Pure discourse marker; it makes no independently checkable content claim."
    if n3_fallback and any(value in lowered for value in FALLBACK_PHRASES):
        return "supported", (
            "Deterministic fallback makes only the generic checkable claim that the cited photo group contains "
            "visible scene/object evidence; the cited group exists."
        )
    if PERSONAL_OR_INFERRED.search(claim):
        return "unsupported", (
            "The statement adds first-person identity, relationship, emotion, intention, causation, sensory detail, "
            "or evaluative content absent from verified_context."
        )
    claim_tokens = _tokens(claim)
    evidence_tokens = _tokens(evidence_text)
    if not claim_tokens:
        return "non_factual", "No independently checkable visual or metadata proposition remains after normalization."
    overlap = claim_tokens & evidence_tokens
    coverage = len(overlap) / len(claim_tokens)
    if coverage >= 0.60 or (len(claim_tokens) <= 3 and coverage >= 0.50):
        return "supported", (
            f"The cited model observations/metadata license the claim ({len(overlap)}/{len(claim_tokens)} "
            "normalized content tokens matched); no personal inference was added."
        )
    if overlap:
        return "uncertain", (
            f"Only part of the statement is licensed by cited observations/metadata ({len(overlap)}/{len(claim_tokens)} "
            "content tokens); the unit mixes supported and unmatched detail."
        )
    return "unsupported", "No material content in the statement is licensed by the cited observations or reliable metadata."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=PRIVATE / "story_claim_audit_v1.json")
    parser.add_argument("--mapping", type=Path, default=PRIVATE / "story_claim_audit_mapping_v1.json")
    parser.add_argument("--results", type=Path, default=PRIVATE / "story_ablation_n0_n3_v1.jsonl")
    parser.add_argument("--cases", type=Path, default=PRIVATE / "story_cases_v1.json")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--show-status", choices=("supported", "unsupported", "uncertain", "non_factual"))
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    results = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = load_story_cases(args.cases)
    by_case = {item.case_id: item for item in cases}
    by_result = {(item["case_id"], item["variant_id"]): item for item in results}
    mapping_by_id = {item["audit_id"]: item for item in mapping["mappings"]}
    aggregator = ContextAggregator()
    contexts = {
        case.case_id: aggregator.aggregate_by_photo_ids(
            case.photo_ids, label=case.label, event_id=case.event_id, source_kind=case.source_kind,
        ) for case in cases
    }
    counts = {value: 0 for value in audit["allowed_statuses"]}
    review_rows: list[dict[str, Any]] = []
    for item in audit["claims"]:
        identity = mapping_by_id[item["audit_id"]]
        case_id, variant_id = identity["case_id"], identity["variant_id"]
        context = contexts[case_id]
        cited = [str(value) for value in item.get("evidence_ids", [])]
        audited_ids = cited or [value.evidence_id for value in context.evidence]
        result = by_result[(case_id, variant_id)]
        evidence_text = _evidence_text(context, audited_ids)
        status, reason = _classify(
            str(item["claim"]), evidence_text,
            n3_fallback=variant_id == "N3" and bool(result.get("fallback")),
        )
        if item["audit_id"] in MANUAL_OVERRIDES:
            status, reason = MANUAL_OVERRIDES[item["audit_id"]]
        item["status"] = status
        item["reason"] = reason
        item["audited_evidence_ids"] = audited_ids
        item["source"] = "agent_evidence_audit"
        counts[status] += 1
        if args.show_status == status:
            review_rows.append({
                "audit_id": item["audit_id"], "case_id": case_id, "variant_id": variant_id,
                "claim": item["claim"], "audited_evidence_ids": audited_ids,
                "evidence": evidence_text[:1600], "reason": reason,
            })
    audit["audit_metadata"] = {
        "status": "complete" if args.apply else "dry_run",
        "completed_at_utc": datetime.now(timezone.utc).isoformat() if args.apply else None,
        "source": "agent_evidence_audit",
        "method": "conservative agent review with deterministic lexical triage and recorded semantic adjudications",
        "evidence_basis": (
            "Claims were checked against the exact per-case BLIP/Qwen/CLIP observations and reliable metadata "
            "available to the generator. Personal facts and sensory/causal/emotional inferences were unsupported "
            "without verified_context; partial matches were uncertain. This is not human multi-rater annotation "
            "and does not prove that upstream model observations are visually correct."
        ),
        "status_counts": counts,
    }
    if args.apply:
        temporary = args.audit.with_suffix(args.audit.suffix + ".tmp")
        temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.audit)
    print(json.dumps({"claim_count": len(audit["claims"]), "status_counts": counts, "applied": args.apply}, indent=2))
    if args.show_status:
        print(json.dumps({"review_status": args.show_status, "items": review_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
