"""
OCR Engine Module

Extracts text from manga panels.
Supports multiple OCR backends with fallback options.
"""

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

from interfaces.base_ocr_engine import BaseOCREngine, OCRResult, TextBox

logger = logging.getLogger(__name__)


class OCREngine(BaseOCREngine):
    """
    Concrete implementation of OCR engine.
    
    Supports multiple OCR backends:
    - Tesseract (via pytesseract)
    - EasyOCR
    - Placeholder mode (returns empty text)
    """
    
    SUPPORTED_LANGUAGES = ['en', 'ja', 'ko', 'zh-cn', 'zh-tw']
    
    def __init__(self):
        self._language = 'en'
        self._backend = self._detect_backend()
        self._ocr_instance = None
        self._initialize_backend()
    
    def _detect_backend(self) -> str:
        """Detect available OCR backend."""
        # Try EasyOCR first (better for manga)
        try:
            import easyocr
            logger.info("EasyOCR backend available")
            return 'easyocr'
        except ImportError:
            pass
        
        # Try Tesseract
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            logger.info("Tesseract backend available")
            return 'tesseract'
        except Exception:
            pass
        
        logger.warning("No OCR backend available. Using placeholder mode.")
        return 'placeholder'
    
    def _initialize_backend(self) -> None:
        """Initialize the OCR backend."""
        if self._backend == 'easyocr':
            try:
                import easyocr
                # Map language codes
                lang_map = {
                    'en': 'en',
                    'ja': 'ja',
                    'ko': 'ko',
                    'zh-cn': 'ch_sim',
                    'zh-tw': 'ch_tra'
                }
                lang = lang_map.get(self._language, 'en')
                # Auto-detect GPU — use CUDA if available, else CPU
                try:
                    import torch
                    use_gpu = torch.cuda.is_available()
                except ImportError:
                    use_gpu = False
                self._ocr_instance = easyocr.Reader([lang], gpu=use_gpu)
                logger.info(f"Initialized EasyOCR with language: {lang}, GPU: {use_gpu}")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self._backend = 'placeholder'
    
    def set_language(self, language: str) -> None:
        """Set the primary language for OCR."""
        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Language {language} not supported, defaulting to 'en'")
            language = 'en'
        
        if self._language != language:
            self._language = language
            self._initialize_backend()
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.SUPPORTED_LANGUAGES.copy()
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(contrast, h=10)
        
        # Threshold to binary (helps with manga text bubbles)
        _, binary = cv2.threshold(
            denoised, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return binary
    
    def extract_text(self, panel_image: np.ndarray, panel_index: int) -> OCRResult:
        """Extract text from a single panel."""
        logger.debug(f"Extracting text from panel {panel_index}")
        
        if self._backend == 'placeholder':
            return self._placeholder_extract(panel_index)
        
        # Preprocess
        processed = self.preprocess_for_ocr(panel_image)
        
        if self._backend == 'easyocr':
            return self._easyocr_extract(panel_image, panel_index)
        elif self._backend == 'tesseract':
            return self._tesseract_extract(processed, panel_index)
        else:
            return self._placeholder_extract(panel_index)
    
    def extract_text_batch(self, panels: List[Tuple[np.ndarray, int]]) -> List[OCRResult]:
        """Extract text from multiple panels."""
        results = []
        for panel_image, panel_index in panels:
            result = self.extract_text(panel_image, panel_index)
            results.append(result)
        return results
    
    def _easyocr_extract(self, image: np.ndarray, panel_index: int) -> OCRResult:
        """Extract text using EasyOCR."""
        try:
            # EasyOCR expects RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            results = self._ocr_instance.readtext(image)
            
            text_boxes = []
            texts = []
            
            for bbox, text, confidence in results:
                # Convert bbox format
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x = int(min(x_coords))
                y = int(min(y_coords))
                w = int(max(x_coords) - x)
                h = int(max(y_coords) - y)
                
                text_box = TextBox(
                    text=text,
                    x=x, y=y, width=w, height=h,
                    confidence=confidence,
                    language=self._language
                )
                text_boxes.append(text_box)
                texts.append(text)
            
            full_text = ' '.join(texts)
            avg_confidence = sum(t.confidence for t in text_boxes) / len(text_boxes) if text_boxes else 0.0
            
            logger.debug(f"Panel {panel_index}: Found {len(text_boxes)} text regions")
            
            return OCRResult(
                panel_index=panel_index,
                text_boxes=text_boxes,
                full_text=full_text,
                confidence=avg_confidence,
                language=self._language,
                metadata={'backend': 'easyocr'}
            )
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed for panel {panel_index}: {e}")
            return self._placeholder_extract(panel_index)
    
    def _tesseract_extract(self, image: np.ndarray, panel_index: int) -> OCRResult:
        """Extract text using Tesseract."""
        try:
            import pytesseract
            
            # Get detailed data
            data = pytesseract.image_to_data(
                image, 
                lang=self._language,
                output_type=pytesseract.Output.DICT
            )
            
            text_boxes = []
            texts = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text:
                    conf = float(data['conf'][i]) / 100.0
                    if conf > 0:  # Filter low confidence
                        text_box = TextBox(
                            text=text,
                            x=data['left'][i],
                            y=data['top'][i],
                            width=data['width'][i],
                            height=data['height'][i],
                            confidence=conf,
                            language=self._language
                        )
                        text_boxes.append(text_box)
                        texts.append(text)
            
            full_text = ' '.join(texts)
            avg_confidence = sum(t.confidence for t in text_boxes) / len(text_boxes) if text_boxes else 0.0
            
            logger.debug(f"Panel {panel_index}: Found {len(text_boxes)} text regions")
            
            return OCRResult(
                panel_index=panel_index,
                text_boxes=text_boxes,
                full_text=full_text,
                confidence=avg_confidence,
                language=self._language,
                metadata={'backend': 'tesseract'}
            )
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed for panel {panel_index}: {e}")
            return self._placeholder_extract(panel_index)
    
    def _placeholder_extract(self, panel_index: int) -> OCRResult:
        """Return placeholder OCR result."""
        logger.debug(f"Using placeholder OCR for panel {panel_index}")
        return OCRResult(
            panel_index=panel_index,
            text_boxes=[],
            full_text="",
            confidence=0.0,
            language=self._language,
            metadata={'backend': 'placeholder', 'note': 'No OCR backend available'}
        )
