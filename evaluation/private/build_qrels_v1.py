"""Private query definitions authored from original contact sheets and EXIF."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CATALOG = HERE / "contact_sheets" / "catalog.json"
OUTPUT = HERE / "qrels_v1.json"


# Six English and two Chinese queries per category. The first two entries in
# every category are dev (one EN, one ZH); the remaining six are test.
QUERY_SPECS = {
    "scene": [
        ("en", "an indoor swimming pool with marked lanes", {9: 2, 8: 1}),
        ("zh", "夜晚城市中的彩色高塔", {15: 2}),
        ("en", "a grassy park overlooking a stone railway viaduct", {28: 2}),
        ("en", "a dark sky with aurora over the horizon", {198: 2, 199: 1, 200: 1}),
        ("en", "a snowy volcanic field with small horses", {174: 2, 175: 2}),
        ("en", "a canal lined with buildings and boats", {260: 2, 261: 2, 262: 2, 263: 2, 264: 2}),
        ("en", "a coastal landscape with white chalk cliffs", {282: 2, 283: 2, 284: 2, 285: 1, 286: 2, 287: 2}),
        ("zh", "日出时天空中有许多热气球的山谷", {297: 1, 298: 2, 299: 2, 300: 2, 301: 2, 302: 2, 303: 1}),
    ],
    "object_attribute": [
        ("en", "an orange reusable bottle held in one hand", {49: 2}),
        ("zh", "装着黑莓的白碗和两片面包", {24: 2}),
        ("en", "two purple energy drink cans held outdoors", {73: 2}),
        ("en", "a slice of green matcha cake on a white plate", {214: 2}),
        ("en", "black over-ear headphones beside a small circuit board", {277: 2, 278: 2}),
        ("en", "a white cow with black patches in a field", {98: 2}),
        ("en", "a blue bowl filled with blueberries", {116: 2}),
        ("zh", "一只白色鹭鸟站在水边", {266: 2, 267: 2}),
    ],
    "relation_semantic": [
        ("en", "a person photographing a landscape with a phone", {152: 2, 362: 1}),
        ("zh", "一只手拿着饮料罐", {73: 2, 111: 2}),
        ("en", "a person standing beneath a stone railway arch", {119: 2}),
        ("en", "someone posing beside the Arc de Triomphe", {127: 2, 129: 2}),
        ("en", "people hiking through a rocky valley", {94: 2, 96: 2}),
        ("en", "a person looking across a town from a hilltop", {30: 2}),
        ("en", "a person holding a bouquet of pink flowers", {217: 2}),
        ("zh", "一个人坐在湖边的椅子上", {153: 2}),
    ],
    "relation_exact": [
        ("en", "a hand holding a blue drink above a table", {202: 2}),
        ("zh", "一只勺子伸进装有蓝莓的碗里", {116: 2}),
        ("en", "a person standing directly in front of the Arc de Triomphe", {127: 2, 129: 2}),
        ("en", "a smartphone lying on a wooden table in front of a bed", {330: 2, 331: 2}),
        ("en", "a laptop on a desk beside a plate of food", {357: 2, 358: 2, 359: 2, 360: 2}),
        ("en", "two seagulls standing next to each other on sand", {239: 2}),
        ("en", "a person holding a phone above an open laptop", {357: 2, 374: 2}),
        ("zh", "一个人坐在公园长椅上", {29: 2, 77: 2}),
    ],
    "caption_lexical": [
        ("en", "Sudoku puzzle printed in a newspaper", {1: 2}),
        ("zh", "货架上标价三英镑的商品", {92: 2}),
        ("en", "Lionel Messi kicking a football", {2: 2}),
        ("en", "Jet2holidays airport advertisement", {18: 2}),
        ("en", "Venezia Santa Lucia station sign", {259: 2}),
        ("en", "Budweiser beer can beside a laptop", {111: 2, 112: 2}),
        ("en", "Edinburgh illuminated sign", {159: 2}),
        ("zh", "屏幕上的贪吃蛇游戏", {276: 2}),
    ],
    "metadata": [
        ("en", "photos reliably dated 13 August 2025", "date", "2025-08-13"),
        ("zh", "在Knaresborough拍摄的照片", "location", "Knaresborough"),
        ("en", "photos taken in Paris", "location", "Paris"),
        ("en", "photos taken near Hoefn in Iceland", "location", "Hoefn"),
        ("en", "photos reliably dated 30 March 2026", "date", "2026-03-30"),
        ("en", "photos taken in Venice", "location", "Venice"),
        ("en", "photos reliably dated 10 July 2026", "date", "2026-07-10"),
        ("zh", "可靠拍摄于2026年7月12日的照片", "date", "2026-07-12"),
    ],
}


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_ordinal = {int(record["ordinal"]): record for record in catalog}
    records = []
    for category, specs in QUERY_SPECS.items():
        for index, spec in enumerate(specs, 1):
            language, query, *definition = spec
            split = "dev" if index <= 2 else "test"
            filters = {}
            if category == "metadata":
                kind, value = definition
                if kind == "date":
                    filters = {"start_date": value, "end_date": value, "min_timestamp_confidence": 0.6}
                    selected = [record for record in catalog if record["date_local"] == value and float(record["timestamp_confidence"]) >= 0.6]
                else:
                    filters = {"location": value, "min_timestamp_confidence": 0.6}
                    selected = [record for record in catalog if value.casefold() in str(record.get("location") or "").casefold() and float(record["timestamp_confidence"]) >= 0.6]
                relevance = {record["photo_id"]: 2 for record in selected}
            else:
                ordinal_relevance = definition[0]
                relevance = {by_ordinal[int(ordinal)]["photo_id"]: grade for ordinal, grade in ordinal_relevance.items()}
            query_id = f"{category}_{language}_{split}_{index:02d}"
            records.append({
                "query_id": query_id, "query": query, "task_type": "photo",
                "category": category, "language": language, "split": split,
                "judgment_source": "agent_pass1", "filters": filters,
                "relevance": relevance,
                "notes": "pass 1: authored from original-image contact sheets and reliable EXIF before rankings",
            })
    OUTPUT.write_text(json.dumps({"queries": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} queries to {OUTPUT}")


if __name__ == "__main__":
    main()
