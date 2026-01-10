"""Diagram detection and extraction."""

import time
from pathlib import Path
from typing import Optional, Any
from loguru import logger

from ultimate_rag.extraction.base import BaseExtractor, ExtractionResult
from ultimate_rag.models.document import ImageBlock, BoundingBox, BlockType
from ultimate_rag.config import ExtractionConfig


class DiagramExtractor(BaseExtractor):
    """Extract and classify diagrams from documents."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        super().__init__(config)
        self.config = config or ExtractionConfig()
        self._model = None
        self._clip_model = None
        self._clip_processor = None

        self.diagram_classes = [
            'graph', 'chart', 'diagram', 'figure', 'plot',
            'circuit', 'flowchart', 'schematic', 'vector_field',
            'coordinate_system', 'geometry', 'molecular_structure',
            'free_body_diagram', 'waveform', 'spectrum'
        ]

    def _setup(self):
        """Initialize detection models."""
        try:
            from ultralytics import YOLO
            model_path = self.config.diagram_detection_model
            if not Path(model_path).exists():
                model_path = "yolov8n.pt"
            self._model = YOLO(model_path)
            logger.info("YOLO model loaded for diagram detection")
        except ImportError:
            logger.warning("ultralytics not available, using heuristic detection")
        except Exception as e:
            logger.warning(f"YOLO loading failed: {e}")

        try:
            from transformers import CLIPProcessor, CLIPModel
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("CLIP model loaded for diagram classification")
        except ImportError:
            logger.warning("transformers not available for CLIP")
        except Exception as e:
            logger.warning(f"CLIP loading failed: {e}")

    def extract(self, source: Any, **kwargs) -> ExtractionResult:
        """Extract diagrams from image or page."""
        self.initialize()
        start_time = time.time()

        try:
            if isinstance(source, (str, Path)):
                diagrams = self._extract_from_image(Path(source))
            elif isinstance(source, bytes):
                diagrams = self._extract_from_bytes(source)
            elif hasattr(source, 'image_blocks'):
                diagrams = self._extract_from_page(source)
            else:
                return ExtractionResult(
                    success=False,
                    error=f"Unsupported source type: {type(source)}"
                )

            processing_time = (time.time() - start_time) * 1000

            return ExtractionResult(
                success=True,
                data=diagrams,
                processing_time_ms=processing_time,
                metadata={"diagram_count": len(diagrams)}
            )

        except Exception as e:
            logger.error(f"Diagram extraction failed: {e}")
            return ExtractionResult(success=False, error=str(e))

    def _extract_from_image(self, image_path: Path) -> list[ImageBlock]:
        """Extract diagrams from an image file."""
        from PIL import Image

        img = Image.open(image_path)
        diagrams = []

        if self._model:
            results = self._model(str(image_path), verbose=False)

            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    if conf < self.config.diagram_confidence_threshold:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox = BoundingBox(x0=x1, y0=y1, x1=x2, y1=y2)

                    cropped = img.crop((x1, y1, x2, y2))
                    diagram_type = self._classify_diagram(cropped)

                    from io import BytesIO
                    buf = BytesIO()
                    cropped.save(buf, format='PNG')

                    diagram = ImageBlock(
                        id=f"DG_{image_path.stem}_{i}",
                        bbox=bbox,
                        block_type=BlockType.DIAGRAM,
                        image_bytes=buf.getvalue(),
                        diagram_type=diagram_type,
                        metadata={"confidence": conf}
                    )
                    diagrams.append(diagram)

        else:
            if self._is_likely_diagram(img):
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format='PNG')

                diagram = ImageBlock(
                    id=f"DG_{image_path.stem}_0",
                    bbox=BoundingBox(x0=0, y0=0, x1=img.width, y1=img.height),
                    block_type=BlockType.DIAGRAM,
                    image_bytes=buf.getvalue(),
                    diagram_type=self._classify_diagram(img)
                )
                diagrams.append(diagram)

        return diagrams

    def _extract_from_bytes(self, image_bytes: bytes) -> list[ImageBlock]:
        """Extract diagrams from image bytes."""
        from PIL import Image
        from io import BytesIO

        img = Image.open(BytesIO(image_bytes))

        if self._is_likely_diagram(img):
            diagram_type = self._classify_diagram(img)

            diagram = ImageBlock(
                id=f"DG_bytes_0",
                bbox=BoundingBox(x0=0, y0=0, x1=img.width, y1=img.height),
                block_type=BlockType.DIAGRAM,
                image_bytes=image_bytes,
                diagram_type=diagram_type
            )
            return [diagram]

        return []

    def _extract_from_page(self, page: Any) -> list[ImageBlock]:
        """Process image blocks from a page."""
        diagrams = []

        for img_block in page.image_blocks:
            if img_block.image_bytes:
                result = self._extract_from_bytes(img_block.image_bytes)
                if result:
                    for diagram in result:
                        diagram.id = f"{img_block.id}_diag"
                        diagram.bbox = img_block.bbox
                        diagrams.append(diagram)

        return diagrams

    def _is_likely_diagram(self, img: Any) -> bool:
        """Heuristic check if image is likely a diagram."""
        try:
            import numpy as np

            if img.mode != 'RGB':
                img = img.convert('RGB')

            arr = np.array(img)

            unique_colors = len(np.unique(arr.reshape(-1, 3), axis=0))
            is_limited_colors = unique_colors < 1000

            gray = np.mean(arr, axis=2)
            edge_ratio = np.sum(np.abs(np.diff(gray))) / gray.size
            has_clear_lines = edge_ratio > 10

            white_ratio = np.sum(arr > 240) / arr.size
            is_mostly_white = white_ratio > 0.5

            aspect_ratio = img.width / img.height
            reasonable_aspect = 0.3 < aspect_ratio < 3.0

            score = sum([
                is_limited_colors,
                has_clear_lines,
                is_mostly_white,
                reasonable_aspect
            ])

            return score >= 2

        except Exception as e:
            logger.debug(f"Diagram heuristic check failed: {e}")
            return True

    def _classify_diagram(self, img: Any) -> str:
        """Classify diagram type using CLIP or heuristics."""
        if self._clip_model and self._clip_processor:
            try:
                labels = [f"a {cls}" for cls in self.diagram_classes]

                inputs = self._clip_processor(
                    text=labels,
                    images=img,
                    return_tensors="pt",
                    padding=True
                )

                outputs = self._clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                best_idx = probs.argmax().item()

                return self.diagram_classes[best_idx]

            except Exception as e:
                logger.debug(f"CLIP classification failed: {e}")

        return self._heuristic_classify(img)

    def _heuristic_classify(self, img: Any) -> str:
        """Classify diagram using heuristics."""
        try:
            import numpy as np

            if img.mode != 'RGB':
                img = img.convert('RGB')

            arr = np.array(img)

            has_red = np.sum((arr[:, :, 0] > 200) & (arr[:, :, 1] < 100)) > 100
            has_blue = np.sum((arr[:, :, 2] > 200) & (arr[:, :, 0] < 100)) > 100
            has_green = np.sum((arr[:, :, 1] > 200) & (arr[:, :, 0] < 100)) > 100

            if has_red and has_blue:
                return "chart"

            gray = np.mean(arr, axis=2)
            edges = np.abs(np.diff(gray, axis=0)) + np.abs(np.diff(gray, axis=1)[:, :-1])
            edge_density = np.sum(edges > 50) / edges.size

            if edge_density > 0.1:
                return "diagram"

            return "figure"

        except Exception:
            return "unknown"

    def get_visual_features(self, img: Any) -> Optional[list[float]]:
        """Get visual embedding for image."""
        if not self._clip_model or not self._clip_processor:
            return None

        try:
            inputs = self._clip_processor(images=img, return_tensors="pt")
            features = self._clip_model.get_image_features(**inputs)
            return features[0].detach().numpy().tolist()

        except Exception as e:
            logger.debug(f"Visual feature extraction failed: {e}")
            return None
