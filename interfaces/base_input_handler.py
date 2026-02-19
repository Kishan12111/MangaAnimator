"""
Base Input Handler Interface

Defines the contract for handling various manga input formats.
Implementations should handle ZIP, PDF, and folder inputs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np


@dataclass
class MangaPage:
    """Represents a single manga page."""
    index: int
    image: np.ndarray
    source_path: Optional[Path] = None
    metadata: Optional[dict] = None


@dataclass
class InputResult:
    """Result of input processing."""
    pages: List[MangaPage]
    source_type: str  # 'zip', 'pdf', 'folder'
    total_pages: int
    metadata: dict


class BaseInputHandler(ABC):
    """
    Abstract base class for manga input handlers.
    
    Implementations must handle different input formats and produce
    a standardized list of manga pages for the pipeline.
    """
    
    @abstractmethod
    def load(self, input_path: Path) -> InputResult:
        """
        Load manga from the given input path.
        
        Args:
            input_path: Path to ZIP file, PDF file, or folder containing images
            
        Returns:
            InputResult containing all loaded pages
            
        Raises:
            ValueError: If input format is not supported
            FileNotFoundError: If input path doesn't exist
        """
        pass
    
    @abstractmethod
    def validate(self, input_path: Path) -> bool:
        """
        Validate if the input can be processed.
        
        Args:
            input_path: Path to validate
            
        Returns:
            True if input is valid and can be processed
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.
        
        Returns:
            List of supported format extensions/types
        """
        pass
