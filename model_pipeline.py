"""
Smart Photo Narrative - Vision Engine & Vector Index Module
====================
Features:
1. CLIP model wrapper (batch inference, vector extraction)
2. BLIP image captioning
3. ChromaDB vector database management
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import json

from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration
)
import chromadb
from chromadb.config import Settings
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from config import (
    DEVICE,
    PHOTOS_DIR,
    CLIP_MODEL_NAME,
    CLIP_EMBEDDING_DIM,
    BLIP_MODEL_NAME,
    BATCH_SIZE,
    BLIP_BATCH_SIZE,
    MIN_GPU_BATCH_SIZE,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    APP_DB_PATH,
)
from data_ingestion import PhotoMetadata, load_image


def _from_pretrained_offline_first(factory, model_name: str, *, model_weights: bool = False, **kwargs):
    """Use a complete local Hugging Face cache before attempting the network.

    The project is commonly run on AIRE or behind a restricted network.  Older
    caches may contain ``pytorch_model.bin`` while newer Transformers versions
    probe for ``model.safetensors`` online first, adding long retries even though
    usable weights are already present.  Prefer the cached binary checkpoint,
    then retain the normal online path for a genuine first-time installation.
    """

    local_kwargs = dict(kwargs)
    local_kwargs["local_files_only"] = True
    if model_weights:
        local_kwargs["use_safetensors"] = False
    try:
        return factory.from_pretrained(model_name, **local_kwargs)
    except OSError as local_error:
        try:
            return factory.from_pretrained(model_name, **kwargs)
        except Exception as remote_error:
            raise RuntimeError(
                f"Model assets for {model_name!r} are neither complete in the local "
                "Hugging Face cache nor downloadable. Connect once to populate the "
                "cache, then retry offline."
            ) from remote_error


# ==================== Data Structures ====================
@dataclass
class ImageFeatures:
    """Image feature data"""
    file_path: str
    embedding: np.ndarray          # CLIP 512-dim vector
    caption: str                   # BLIP caption
    # Legacy fields kept for API stability; CLIP zero-shot tags were removed so
    # they are always empty. Indexed data uses the CLIP_INDEX_MARKER_TAG
    # sentinel (see indexing_service), not these values.
    tags: List[str] = None
    tag_scores: List[float] = None

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []
        if self.tag_scores is None:
            self.tag_scores = []


# ==================== CLIP Model Manager ====================
class CLIPModelManager:
    """
    CLIP model manager.
    Handles image encoding.
    """

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.model_name = CLIP_MODEL_NAME
        self.model_version = "transformers"
        self.model = None
        self.processor = None

    def load_model(self):
        """Load the CLIP model"""
        if self.model is None:
            print(f"[CLIP] Loading model: {CLIP_MODEL_NAME}")
            self.model = _from_pretrained_offline_first(
                CLIPModel, CLIP_MODEL_NAME, model_weights=True
            ).to(self.device)
            self.processor = _from_pretrained_offline_first(
                CLIPProcessor, CLIP_MODEL_NAME, use_fast=False
            )
            self.model.eval()
            print(f"[CLIP] Model loaded, device: {self.device}")

    def unload_model(self):
        """Release model memory between indexing phases on 4 GB GPUs."""
        self.model = None
        self.processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_images_batch(
        self,
        images: List[Image.Image]
    ) -> np.ndarray:
        """
        Batch-encode images into vectors.
        Returns: (N, 512) numpy array
        """
        self.load_model()

        # Process images
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Get image features
        outputs = self.model.get_image_features(**inputs)
        # Compatible with newer transformers versions: may be object or tensor
        if hasattr(outputs, 'pooler_output'):
            image_features = outputs.pooler_output
        elif hasattr(outputs, 'image_embeds'):
            image_features = outputs.image_embeds
        else:
            image_features = outputs
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text query into a vector for retrieval.
        """
        self.load_model()

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.model.get_text_features(**inputs)
        # Compatible with newer transformers versions: may be object or tensor
        if hasattr(outputs, 'pooler_output'):
            text_features = outputs.pooler_output
        elif hasattr(outputs, 'text_embeds'):
            text_features = outputs.text_embeds
        else:
            text_features = outputs
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()[0]

    @torch.no_grad()
    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Encode Scene Graph triples efficiently with the same CLIP space."""
        if not texts:
            return np.empty((0, CLIP_EMBEDDING_DIM), dtype=np.float32)
        self.load_model()
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        elif hasattr(outputs, "text_embeds"):
            features = outputs.text_embeds
        else:
            features = outputs
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


# ==================== BLIP Model Manager ====================
class BLIPModelManager:
    """
    BLIP model manager.
    Handles image caption generation.
    """

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.model_name = BLIP_MODEL_NAME
        self.model_version = "transformers"
        self.model = None
        self.processor = None

    def load_model(self):
        """Load the BLIP model"""
        if self.model is None:
            print(f"[BLIP] Loading model: {BLIP_MODEL_NAME}")
            self.model = _from_pretrained_offline_first(
                BlipForConditionalGeneration, BLIP_MODEL_NAME, model_weights=True
            ).to(self.device)
            self.processor = _from_pretrained_offline_first(
                BlipProcessor, BLIP_MODEL_NAME, use_fast=False
            )
            self.model.eval()
            print(f"[BLIP] Model loaded, device: {self.device}")

    def unload_model(self):
        self.model = None
        self.processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _validate_caption(self, caption: str) -> bool:
        """
        Validate whether a caption is usable.
        Filters repetitive output and invalid captions.
        """
        if not caption or len(caption.strip()) < 3:
            return False

        # Check for repeated words (e.g. "mess mess mess...")
        words = caption.lower().split()
        if len(words) >= 3:
            # If >60% is the same word, treat as invalid output
            unique_words = set(words)
            for word in unique_words:
                if len(word) >= 3:  # ignore short words
                    count = words.count(word)
                    if count / len(words) > 0.6:
                        return False

        # Check for repeating character patterns (e.g. "aaa...")
        if len(caption) >= 10:
            for pattern_len in range(3, min(20, len(caption) // 3)):
                pattern = caption[:pattern_len]
                if pattern * 3 in caption:
                    return False

        return True

    def _generate_caption_with_retry(self, image: Image.Image, max_retries: int = 3) -> str:
        """
        Caption generation with retry.
        If the generated caption is invalid, retries with different parameters.
        """
        self.load_model()

        # Try different generation parameters
        params_list = [
            {"max_length": 30, "num_beams": 1, "do_sample": False},
            {"max_length": 50, "num_beams": 3, "do_sample": False},
            {"max_length": 40, "num_beams": 1, "do_sample": True, "temperature": 0.7},
        ]

        for i, params in enumerate(params_list[:max_retries]):
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

            output = self.model.generate(**inputs, **params)
            caption = self.processor.decode(output[0], skip_special_tokens=True)

            if self._validate_caption(caption):
                if i > 0:
                    print(f"[BLIP] Generated valid caption after {i} retries")
                return caption

        # All attempts failed, return fallback
        return "an image"

    @torch.no_grad()
    def generate_captions_batch(
        self,
        images: List[Image.Image]
    ) -> List[str]:
        """
        Batch-generate image captions.
        Returns a list of English captions with validity checks.
        """
        self.load_model()

        if not images:
            return []

        # BLIP supports real batched generation.  The old implementation
        # accepted a list but invoked the model once per image.
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        output = self.model.generate(
            **inputs,
            max_length=30,
            num_beams=1,
            do_sample=False,
        )
        captions = [value.strip() for value in self.processor.batch_decode(output, skip_special_tokens=True)]
        for index, caption in enumerate(captions):
            if not self._validate_caption(caption):
                captions[index] = self._generate_caption_with_retry(images[index])
        return captions


# ==================== ChromaDB Manager ====================
class ChromaDBManager:
    """
    ChromaDB vector database manager.
    """

    def __init__(self, persist_dir: Path = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None

    def get_client(self) -> chromadb.Client:
        """Get the ChromaDB client"""
        if self.client is None:
            print(f"[ChromaDB] Initializing persistent client: {self.persist_dir}")
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self.client

    def get_collection(self) -> chromadb.Collection:
        """Get or create the collection"""
        try:
            if self.collection is None:
                client = self.get_client()
                self.collection = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={
                        "description": "Photo vector index",
                        "hnsw:space": "cosine"  # cosine distance: 1-distance = similarity
                    }
                )
                print(f"[ChromaDB] Collection '{COLLECTION_NAME}' ready, records: {self.collection.count()}")
            return self.collection
        except Exception as e:
            # Collection reference may be stale; reinitialize
            print(f"[ChromaDB] Collection reference invalid, reinitializing: {e}")
            self.collection = None
            self.client = None
            return self.get_collection()

    def add_photos(
        self,
        features_list: List[ImageFeatures],
        metadata_list: List[PhotoMetadata]
    ):
        """
        Batch-add photos to the vector database.
        """
        collection = self.get_collection()

        if not features_list:
            print("[Warning] No photos to add")
            return

        ids = [f.file_path for f in features_list]
        embeddings = [f.embedding.tolist() for f in features_list]

        # Build metadata
        metadatas = []
        for features, meta in zip(features_list, metadata_list):
            metadatas.append({
                "datetime": meta.datetime_original or "",
                "location": meta.location or "",
                "tags": json.dumps(features.tags, ensure_ascii=False),
                "tag_scores": json.dumps(features.tag_scores),
                "caption": features.caption,
                "image_width": meta.image_width,
                "image_height": meta.image_height,
                "file_size": meta.file_size
            })

        # Batch add
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=[f.caption for f in features_list]  # used for keyword search
        )

        print(f"[ChromaDB] Added {len(ids)} records")

    def query_by_text(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        where_filter: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Vector similarity search.
        """
        try:
            collection = self.get_collection()
        except Exception as e:
            print(f"[ChromaDB] query_by_text failed, retrying: {e}")
            self.collection = None
            self.client = None
            collection = self.get_collection()

        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["metadatas", "distances", "documents"]
        }

        if where_filter:
            query_params["where"] = where_filter

        if where_document:
            query_params["where_document"] = where_document

        results = collection.query(**query_params)
        return results

    def query_by_document(
        self,
        query_keyword: str,
        top_k: int = 10,
        where_filter: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Keyword search on document (caption) text using collection.get().
        Returns results without distance info (use _parse_raw_results to parse).
        """
        try:
            collection = self.get_collection()
        except Exception as e:
            print(f"[ChromaDB] query_by_document failed, retrying: {e}")
            self.collection = None
            self.client = None
            collection = self.get_collection()

        query_params = {
            "where_document": {"$contains": query_keyword},
            "include": ["metadatas", "documents"]
        }

        if where_filter:
            query_params["where"] = where_filter

        results = collection.get(**query_params)

        # Limit to top_k results
        if results and results.get("ids") and len(results["ids"]) > top_k:
            for key in ["ids", "metadatas", "documents"]:
                if key in results and results[key]:
                    results[key] = results[key][:top_k]

        return results

    def query_by_date(self, date_str: str) -> List[Dict]:
        """
        Query photos by date.
        date_str: "YYYY-MM-DD" or "YYYY-MM"

        Note: newer ChromaDB versions don't support string comparison,
        so filtering is done in Python.
        """
        try:
            collection = self.get_collection()
            all_results = collection.get(include=["metadatas", "documents"])
        except Exception as e:
            # Collection reference stale; reset and retry
            print(f"[ChromaDB] query_by_date failed, retrying: {e}")
            self.collection = None
            self.client = None
            collection = self.get_collection()
            all_results = collection.get(include=["metadatas", "documents"])

        # Filter photos matching the date
        filtered_ids = []
        filtered_metadatas = []
        filtered_documents = []

        for i, meta in enumerate(all_results.get("metadatas", [])):
            datetime_str = meta.get("datetime", "")
            # Match both YYYY-MM-DD and YYYY-MM formats
            if datetime_str and datetime_str.startswith(date_str):
                filtered_ids.append(all_results["ids"][i])
                filtered_metadatas.append(meta)
                filtered_documents.append(all_results["documents"][i] if all_results.get("documents") else "")

        results = {
            "ids": filtered_ids,
            "metadatas": filtered_metadatas,
            "documents": filtered_documents,
            "distances": [[]]
        }

        # Return raw dict format for use by _parse_raw_results
        return results

    def _format_results(self, results: Dict) -> List[Dict]:
        """Format query results"""
        formatted = []
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        for i, id_ in enumerate(ids):
            item = {
                "id": id_,
                "metadata": metadatas[i] if metadatas else {},
                "document": documents[i] if documents else "",
                "distance": distances[i] if i < len(distances) else 0
            }
            formatted.append(item)

        return formatted

    def get_all_ids(self) -> List[str]:
        """Get all indexed photo IDs"""
        try:
            collection = self.get_collection()
            results = collection.get(include=[])
            return results.get("ids", [])
        except Exception as e:
            # Collection reference may be stale; reset and retry
            print(f"[ChromaDB] Failed to get IDs, retrying: {e}")
            self.collection = None
            self.client = None
            collection = self.get_collection()
            results = collection.get(include=[])
            return results.get("ids", [])

    def delete_by_ids(self, ids: List[str]):
        """Delete records by ID"""
        if ids:
            collection = self.get_collection()
            collection.delete(ids=ids)
            print(f"[ChromaDB] Deleted {len(ids)} records")

    def get_count(self) -> int:
        """Get total record count"""
        try:
            return self.get_collection().count()
        except Exception as e:
            print(f"[ChromaDB] Failed to get count, retrying: {e}")
            self.collection = None
            self.client = None
            return self.get_collection().count()


# ==================== Versioned Chroma Compatibility Facade ====================
class ChromaDBManager:
    """Backward-compatible access to the v2 image-vector collection.

    Canonical captions and metadata live in SQLite; this facade intentionally
    exposes only vector-derived operations and fails fast on manifest mismatch.
    """

    def __init__(self, persist_dir: Path = CHROMA_PERSIST_DIR):
        from vector_store import ChromaVectorStore

        self.store = ChromaVectorStore(persist_dir)

    def get_client(self):
        return self.store.client

    def get_collection(self):
        return self.store.image_collection

    def get_all_ids(self) -> List[str]:
        return self.store.all_image_ids()

    def get_count(self) -> int:
        return self.store.counts()["image"]

    def delete_by_ids(self, ids: List[str]):
        self.store.delete_images(ids)

    def query_by_text(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        where_filter: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        del where_document
        # Legacy callers may still supply a Chroma-native filter.
        params: Dict[str, Any] = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, max(self.get_count(), 1)),
            "include": ["metadatas", "distances"],
        }
        if where_filter:
            params["where"] = where_filter
        return self.get_collection().query(**params)

    def close(self):
        self.store.close()


# ==================== Image Dataset (for DataLoader) ====================
class ImageDataset(Dataset):
    """
    Image dataset for batch processing.
    """

    def __init__(self, image_paths: List[Path]):
        self.image_paths = image_paths
        self.images = []
        self.valid_paths = []

        # Pre-load images
        for path in image_paths:
            img = load_image(path)
            if img is not None:
                self.images.append(img)
                self.valid_paths.append(path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], str(self.valid_paths[idx])


# ==================== Unified Vision Pipeline ====================
class VisionPipeline:
    """
    Unified vision processing pipeline.
    Integrates CLIP, BLIP, and ChromaDB.
    """

    def __init__(self):
        from indexing_service import IndexingService
        from storage import PhotoStorage
        from vector_store import ChromaVectorStore

        self.clip_manager = CLIPModelManager()
        self.blip_manager = BLIPModelManager()
        self.storage = PhotoStorage(APP_DB_PATH)
        self.vector_store = ChromaVectorStore(CHROMA_PERSIST_DIR)
        self.chroma_manager = ChromaDBManager()
        self.chroma_manager.store = self.vector_store
        self.indexer = IndexingService(
            storage=self.storage,
            clip_backend=self.clip_manager,
            caption_backend=self.blip_manager,
            vector_store=self.vector_store,
            clip_batch_size=BATCH_SIZE,
            caption_batch_size=BLIP_BATCH_SIZE,
            min_batch_size=MIN_GPU_BATCH_SIZE,
            clip_model_name=CLIP_MODEL_NAME,
            caption_model_name=BLIP_MODEL_NAME,
        )
        self.last_index_summary = None

    def process_photos_batch(
        self,
        metadata_list: List[PhotoMetadata],
        show_progress: bool = True
    ) -> List[ImageFeatures]:
        """
        Batch-process photos:
        1. CLIP vector extraction
        2. BLIP caption generation
        """
        # Load images
        images = []
        valid_metadata = []

        for meta in metadata_list:
            img = load_image(Path(meta.absolute_path))
            if img is not None:
                images.append(img)
                valid_metadata.append(meta)

        if not images:
            print("[Warning] No valid images to process")
            return []

        results = []

        # Process in batches
        total_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE
        iterator = range(total_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Feature extraction")

        for batch_idx in iterator:
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(images))

            batch_images = images[start:end]
            batch_meta = valid_metadata[start:end]

            # CLIP encoding
            embeddings = self.clip_manager.encode_images_batch(batch_images)

            # BLIP captioning
            captions = self.blip_manager.generate_captions_batch(batch_images)

            # Assemble results
            for i, meta in enumerate(batch_meta):
                features = ImageFeatures(
                    file_path=meta.file_path,
                    embedding=embeddings[i],
                    caption=captions[i]
                )
                results.append(features)

        return results

    def index_photos(
        self,
        metadata_list: List[PhotoMetadata],
        show_progress: bool = True
    ):
        """Compatibility entry point backed by the resumable staged indexer."""
        paths = [Path(meta.absolute_path) for meta in metadata_list]
        return self.index_paths(paths, show_progress=show_progress)

    def index_paths(
        self,
        image_paths: List[Path],
        *,
        force: bool = False,
        show_progress: bool = True,
        progress_callback=None,
    ) -> int:
        if progress_callback is not None:
            self.indexer.progress_callback = progress_callback
        elif show_progress:
            self.indexer.progress_callback = lambda payload: print(f"[Index] {payload}")
        else:
            self.indexer.progress_callback = None
        self.last_index_summary = self.indexer.index_photos(
            image_paths,
            PHOTOS_DIR,
            force=force,
        )
        return self.last_index_summary.indexed_count


# ==================== Global Instance ====================
_vision_pipeline = None


def get_vision_pipeline() -> VisionPipeline:
    """Get the vision pipeline singleton"""
    global _vision_pipeline
    if _vision_pipeline is None:
        _vision_pipeline = VisionPipeline()
    return _vision_pipeline


# ==================== Test Entry ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Smart Photo Narrative - Vision Engine Test")
    print("=" * 50)

    # Test CLIP model
    print("\n[Test] CLIP model...")
    clip = CLIPModelManager()
    clip.load_model()

    # Test text encoding
    text_embedding = clip.encode_text("a photo of a cat")
    print(f"Text embedding shape: {text_embedding.shape}")

    # Test BLIP model
    print("\n[Test] BLIP model...")
    blip = BLIPModelManager()
    blip.load_model()

    # Test ChromaDB
    print("\n[Test] ChromaDB...")
    chroma = ChromaDBManager()
    print(f"Record count: {chroma.get_count()}")
