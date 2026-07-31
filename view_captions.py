"""
查看 BLIP 生成的原始 caption
"""
from app import get_cached_chroma_manager
import json

def view_captions(limit=20):
    chroma = get_cached_chroma_manager()
    collection = chroma.get_collection()

    results = collection.get(include=["metadatas"], limit=limit)

    print(f"{'图片 ID':<50} {'BLIP Caption':<60} {'Tags'}")
    print("=" * 150)

    for id_, meta in zip(results["ids"], results["metadatas"]):
        caption = meta.get('caption', 'N/A')
        tags_str = meta.get('tags', '[]')
        try:
            tags = json.loads(tags_str)
            tags_display = ', '.join(tags[:3])
        except:
            tags_display = tags_str
        print(f"{id_:<50} {caption:<60} {tags_display}")

if __name__ == "__main__":
    view_captions()
