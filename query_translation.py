"""Small deterministic Chinese-to-English fallback for the English CLIP model."""

from __future__ import annotations

import re


PHRASES = {
    "一个人": "a person",
    "人物": "person people portrait",
    "使用": "using",
    "拿着": "holding",
    "手持": "holding",
    "旁边": "beside next to",
    "桌边": "beside a table",
    "桌子": "table",
    "电脑": "computer laptop",
    "手机": "smartphone mobile phone",
    "咖啡店": "coffee shop cafe",
    "咖啡": "coffee espresso latte",
    "火车": "train railway",
    "飞机": "airplane aircraft",
    "机场": "airport",
    "海滩": "beach ocean sea sand waves",
    "日落": "sunset golden hour",
    "旅行": "travel trip vacation",
    "朋友": "friends group of people",
    "家庭": "family parents children",
    "儿童": "child kid",
    "猫": "cat kitten",
    "狗": "dog puppy",
    "风景": "landscape scenery nature",
    "森林": "forest woods trees",
    "山": "mountain hill",
    "城市": "city urban",
    "街道": "street road",
    "室内": "indoor room",
    "美食": "food meal cuisine",
    "甜点": "dessert cake pastry",
    "汽车": "car vehicle",
    "自行车": "bicycle bike",
    "照片": "photo",
}


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def translate_query_to_english(text: str) -> str:
    """Replace known phrases longest-first; preserve unknown text transparently."""
    if not contains_chinese(text):
        return text.strip()
    remaining = text
    translated: list[str] = []
    for phrase in sorted(PHRASES, key=len, reverse=True):
        if phrase in remaining:
            translated.append(PHRASES[phrase])
            remaining = remaining.replace(phrase, " ")
    residue = re.sub(r"[，。！？、；：的在我了和与一张这些寻找找看]", " ", remaining)
    residue = " ".join(residue.split())
    if residue:
        translated.append(residue)
    return " ".join(translated).strip() or text.strip()

