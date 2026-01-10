"""Diagram index for visual similarity search."""

from pathlib import Path
from typing import Optional, Any
import numpy as np
from loguru import logger

from ultimate_rag.models.document import ImageBlock
from ultimate_rag.config import IndexingConfig


class DiagramIndex:
    """Index diagrams for visual similarity search."""

    def __init__(self, config: Optional[IndexingConfig] = None):
        self.config = config or IndexingConfig()
        self._clip_model = None
        self._clip_processor = None
        self._index = None
        self._id_map: dict[int, str] = {}
        self._metadata: dict[str, dict] = {}
        self._initialized = False

    def initialize(self):
        """Initialize CLIP model and FAISS index."""
        if self._initialized:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor

            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("CLIP model loaded for diagram indexing")

        except ImportError:
            logger.warning("transformers not available for CLIP")
        except Exception as e:
            logger.warning(f"CLIP loading failed: {e}")

        self._create_empty_index()
        self._initialized = True

    def _create_empty_index(self):
        """Create empty FAISS index for visual features."""
        try:
            import faiss

            dimension = 512
            self._index = faiss.IndexFlatIP(dimension)

        except ImportError:
            logger.warning("FAISS not available, using simple list-based search")
            self._features_list = []

    def _get_image_features(self, image: Any) -> Optional[np.ndarray]:
        """Extract visual features from image."""
        if not self._clip_model or not self._clip_processor:
            return None

        try:
            inputs = self._clip_processor(images=image, return_tensors="pt")
            features = self._clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.detach().numpy()[0]

        except Exception as e:
            logger.debug(f"Feature extraction failed: {e}")
            return None

    def add_diagram(self, diagram: ImageBlock, image: Optional[Any] = None):
        """Add a diagram to the index."""
        self.initialize()

        if image is None and diagram.image_bytes:
            from PIL import Image
            from io import BytesIO
            image = Image.open(BytesIO(diagram.image_bytes))

        if image is None:
            logger.warning(f"No image data for diagram {diagram.id}")
            return

        features = self._get_image_features(image)

        if features is not None:
            idx = len(self._id_map)
            self._id_map[idx] = diagram.id

            if self._index is not None:
                self._index.add(features.reshape(1, -1).astype('float32'))
            else:
                self._features_list.append(features)

            diagram.visual_features = features.tolist()

        self._metadata[diagram.id] = {
            "diagram_type": diagram.diagram_type,
            "caption": diagram.caption,
            "page": diagram.metadata.get("page"),
            "bbox": diagram.bbox.model_dump() if diagram.bbox else None
        }

    def add_diagrams(self, diagrams: list[ImageBlock], images: Optional[list[Any]] = None):
        """Add multiple diagrams to the index."""
        for i, diagram in enumerate(diagrams):
            image = images[i] if images and i < len(images) else None
            self.add_diagram(diagram, image)

    def search_by_image(
        self,
        query_image: Any,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search for similar diagrams using an image query."""
        self.initialize()

        if self._index is None or (hasattr(self._index, 'ntotal') and self._index.ntotal == 0):
            return []

        features = self._get_image_features(query_image)
        if features is None:
            return []

        return self._search_by_features(features, top_k)

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search for diagrams using text description."""
        self.initialize()

        if not self._clip_model or not self._clip_processor:
            return []

        if self._index is None or (hasattr(self._index, 'ntotal') and self._index.ntotal == 0):
            return []

        try:
            inputs = self._clip_processor(text=[query_text], return_tensors="pt", padding=True)
            features = self._clip_model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.detach().numpy()[0]

            return self._search_by_features(features, top_k)

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    def search_by_type(
        self,
        diagram_type: str,
        top_k: int = 20
    ) -> list[tuple[str, float]]:
        """Find diagrams of a specific type."""
        results = []

        for diagram_id, meta in self._metadata.items():
            if meta.get("diagram_type") == diagram_type:
                results.append((diagram_id, 1.0))

        return results[:top_k]

    def _search_by_features(
        self,
        features: np.ndarray,
        top_k: int
    ) -> list[tuple[str, float]]:
        """Search using pre-computed features."""
        if self._index is not None:
            features = features.reshape(1, -1).astype('float32')
            scores, indices = self._index.search(features, min(top_k, self._index.ntotal))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                diagram_id = self._id_map.get(idx)
                if diagram_id:
                    results.append((diagram_id, float(score)))

            return results

        else:
            results = []
            for i, stored_features in enumerate(self._features_list):
                sim = np.dot(features, stored_features)
                diagram_id = self._id_map.get(i)
                if diagram_id:
                    results.append((diagram_id, float(sim)))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def get_metadata(self, diagram_id: str) -> Optional[dict]:
        """Get metadata for a diagram."""
        return self._metadata.get(diagram_id)

    def save(self, path: Path):
        """Save index to disk."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            import faiss
            faiss.write_index(self._index, str(path.with_suffix('.faiss')))

        data = {
            "id_map": self._id_map,
            "metadata": self._metadata,
            "features_list": getattr(self, '_features_list', [])
        }
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved diagram index to {path}")

    def load(self, path: Path):
        """Load index from disk."""
        import pickle

        path = Path(path)

        try:
            import faiss
            faiss_path = path.with_suffix('.faiss')
            if faiss_path.exists():
                self._index = faiss.read_index(str(faiss_path))
        except Exception as e:
            logger.warning(f"Could not load FAISS index: {e}")

        pkl_path = path.with_suffix('.pkl')
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            self._id_map = data.get("id_map", {})
            self._metadata = data.get("metadata", {})
            self._features_list = data.get("features_list", [])

        self._initialized = True
        logger.info(f"Loaded diagram index from {path}")

    @property
    def size(self) -> int:
        """Get number of indexed diagrams."""
        return len(self._metadata)
