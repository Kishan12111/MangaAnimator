"""
Base Panel Detector Interface

Defines the contract for detecting and extracting individual panels
from manga pages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class BoundingBox:
    """Represents a bounding box for a panel."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class Panel:
    """Represents a detected panel from a manga page."""
    index: int
    page_index: int
    image: np.ndarray
    bounding_box: BoundingBox
    confidence: float = 1.0
    reading_order: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class PanelDetectionResult:
    """Result of panel detection for a single page."""
    page_index: int
    panels: List[Panel]
    total_panels: int
    detection_confidence: float


class BasePanelDetector(ABC):
    """
    Abstract base class for panel detection.
    
    Implementations should detect individual panels within manga pages
    and determine their reading order.
    """
    
    @abstractmethod
    def detect(self, page_image: np.ndarray, page_index: int) -> PanelDetectionResult:
        """
        Detect panels in a single manga page.
        
        Args:
            page_image: The manga page image as numpy array
            page_index: Index of the page in the manga
            
        Returns:
            PanelDetectionResult containing all detected panels
        """
        pass
    
    @abstractmethod
    def detect_batch(self, pages: List[Tuple[np.ndarray, int]]) -> List[PanelDetectionResult]:
        """
        Detect panels in multiple pages.
        
        Args:
            pages: List of (page_image, page_index) tuples
            
        Returns:
            List of PanelDetectionResult for each page
        """
        pass
    
    @abstractmethod
    def set_detection_params(self, **kwargs) -> None:
        """
        Set detection parameters.
        
        Args:
            **kwargs: Implementation-specific parameters
        """
        pass
    
    @abstractmethod
    def determine_reading_order(self, panels: List[Panel], reading_direction: str = "rtl") -> List[Panel]:
        """
        Determine the reading order of detected panels.
        
        Args:
            panels: List of detected panels
            reading_direction: 'rtl' (right-to-left) or 'ltr' (left-to-right)
            
        Returns:
            Panels sorted by reading order
        """
        pass
