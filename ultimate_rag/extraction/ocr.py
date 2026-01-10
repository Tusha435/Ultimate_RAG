"""OCR processing utilities."""

import time
from pathlib import Path
from typing import Optional, Any
from loguru import logger

from ultimate_rag.extraction.base import BaseExtractor, ExtractionResult
from ultimate_rag.config import ExtractionConfig


class OCRProcessor(BaseExtractor):
    """Handle OCR for scanned documents and images."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        super().__init__(config)
        self.config = config or ExtractionConfig()
        self._tesseract_available = False
        self._easyocr_reader = None

    def _setup(self):
        """Initialize OCR backends."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            logger.info("Tesseract OCR available")
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")

        try:
            import easyocr
            self._easyocr_reader = easyocr.Reader(
                [self.config.ocr_language[:2]],
                gpu=False
            )
            logger.info("EasyOCR available")
        except ImportError:
            logger.debug("EasyOCR not available")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")

    def extract(self, source: Any, **kwargs) -> ExtractionResult:
        """Extract text from image using OCR."""
        self.initialize()
        start_time = time.time()

        try:
            if isinstance(source, (str, Path)):
                text = self._ocr_file(Path(source))
            elif isinstance(source, bytes):
                text = self._ocr_bytes(source)
            elif hasattr(source, 'save'):
                text = self._ocr_pil_image(source)
            else:
                return ExtractionResult(
                    success=False,
                    error=f"Unsupported source type: {type(source)}"
                )

            processing_time = (time.time() - start_time) * 1000

            return ExtractionResult(
                success=True,
                data=text,
                processing_time_ms=processing_time,
                metadata={"char_count": len(text) if text else 0}
            )

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ExtractionResult(success=False, error=str(e))

    def _ocr_file(self, image_path: Path) -> str:
        """OCR a file."""
        from PIL import Image
        img = Image.open(image_path)
        return self._ocr_pil_image(img)

    def _ocr_bytes(self, image_bytes: bytes) -> str:
        """OCR image bytes."""
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes))
        return self._ocr_pil_image(img)

    def _ocr_pil_image(self, img: Any) -> str:
        """OCR a PIL Image."""
        if self._tesseract_available:
            return self._tesseract_ocr(img)
        elif self._easyocr_reader:
            return self._easyocr_ocr(img)
        else:
            raise RuntimeError("No OCR backend available")

    def _tesseract_ocr(self, img: Any) -> str:
        """Use Tesseract for OCR."""
        import pytesseract

        custom_config = r'--oem 3 --psm 6'

        text = pytesseract.image_to_string(
            img,
            lang=self.config.ocr_language,
            config=custom_config
        )

        return text.strip()

    def _easyocr_ocr(self, img: Any) -> str:
        """Use EasyOCR for OCR."""
        import numpy as np

        if hasattr(img, 'convert'):
            img_array = np.array(img.convert('RGB'))
        else:
            img_array = img

        results = self._easyocr_reader.readtext(img_array)

        texts = [result[1] for result in results]
        return '\n'.join(texts)

    def ocr_with_boxes(self, source: Any) -> list[dict]:
        """OCR with bounding box information."""
        self.initialize()

        try:
            from PIL import Image

            if isinstance(source, (str, Path)):
                img = Image.open(source)
            elif isinstance(source, bytes):
                from io import BytesIO
                img = Image.open(BytesIO(source))
            else:
                img = source

            if self._tesseract_available:
                return self._tesseract_ocr_boxes(img)
            elif self._easyocr_reader:
                return self._easyocr_ocr_boxes(img)
            else:
                raise RuntimeError("No OCR backend available")

        except Exception as e:
            logger.error(f"OCR with boxes failed: {e}")
            return []

    def _tesseract_ocr_boxes(self, img: Any) -> list[dict]:
        """Tesseract OCR with boxes."""
        import pytesseract

        data = pytesseract.image_to_data(
            img,
            lang=self.config.ocr_language,
            output_type=pytesseract.Output.DICT
        )

        results = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue

            results.append({
                'text': text,
                'confidence': data['conf'][i] / 100.0,
                'bbox': {
                    'x0': data['left'][i],
                    'y0': data['top'][i],
                    'x1': data['left'][i] + data['width'][i],
                    'y1': data['top'][i] + data['height'][i]
                }
            })

        return results

    def _easyocr_ocr_boxes(self, img: Any) -> list[dict]:
        """EasyOCR with boxes."""
        import numpy as np

        if hasattr(img, 'convert'):
            img_array = np.array(img.convert('RGB'))
        else:
            img_array = img

        results = self._easyocr_reader.readtext(img_array)

        return [
            {
                'text': result[1],
                'confidence': result[2],
                'bbox': {
                    'x0': min(p[0] for p in result[0]),
                    'y0': min(p[1] for p in result[0]),
                    'x1': max(p[0] for p in result[0]),
                    'y1': max(p[1] for p in result[0])
                }
            }
            for result in results
        ]

    def detect_text_regions(self, img: Any) -> list[dict]:
        """Detect regions containing text."""
        self.initialize()

        try:
            import cv2
            import numpy as np

            if hasattr(img, 'convert'):
                img_array = np.array(img.convert('RGB'))
            else:
                img_array = img

            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(thresh, kernel, iterations=3)

            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                if w < 20 or h < 10:
                    continue

                regions.append({
                    'bbox': {'x0': x, 'y0': y, 'x1': x + w, 'y1': y + h},
                    'area': w * h
                })

            regions.sort(key=lambda r: r['area'], reverse=True)
            return regions

        except ImportError:
            logger.warning("OpenCV not available for text region detection")
            return []
        except Exception as e:
            logger.error(f"Text region detection failed: {e}")
            return []
