"""
Base OCR Engine Interface

Defines the contract for extracting text from manga panels.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class TextBox:
    """Represents a detected text region."""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    language: str = "unknown"


@dataclass
class OCRResult:
    """Result of OCR for a single panel."""
    panel_index: int
    text_boxes: List[TextBox]
    full_text: str
    confidence: float
    language: str = "unknown"
    metadata: dict = field(default_factory=dict)


class BaseOCREngine(ABC):
    """
    Abstract base class for OCR engines.
    
    Implementations should extract text from manga panels,
    handling various languages and text orientations.
    """
    
    @abstractmethod
    def extract_text(self, panel_image: np.ndarray, panel_index: int) -> OCRResult:
        """
        Extract text from a single panel.
        
        Args:
            panel_image: The panel image as numpy array
            panel_index: Index of the panel
            
        Returns:
            OCRResult containing extracted text
        """
        pass
    
    @abstractmethod
    def extract_text_batch(self, panels: List[Tuple[np.ndarray, int]]) -> List[OCRResult]:
        """
        Extract text from multiple panels.
        
        Args:
            panels: List of (panel_image, panel_index) tuples
            
        Returns:
            List of OCRResult for each panel
        """
        pass
    
    @abstractmethod
    def set_language(self, language: str) -> None:
        """
        Set the primary language for OCR.
        
        Args:
            language: Language code (e.g., 'en', 'ja', 'ko')
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        pass
    
    @abstractmethod
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Raw panel image
            
        Returns:
            Preprocessed image optimized for OCR
        """
        pass
