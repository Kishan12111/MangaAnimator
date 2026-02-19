"""
Base Colorizer Interface

Defines the contract for colorizing black and white manga panels.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np


@dataclass
class ColorizationResult:
    """Result of colorization for a single panel."""
    panel_index: int
    original_image: np.ndarray
    colorized_image: np.ndarray
    confidence: float = 1.0
    model_used: str = "unknown"
    metadata: Dict[str, any] = field(default_factory=dict)


class BaseColorizer(ABC):
    """
    Abstract base class for manga colorization.
    
    Implementations should colorize black and white manga panels
    using AI models or other techniques.
    """
    
    @abstractmethod
    def colorize(self, panel_image: np.ndarray, panel_index: int) -> ColorizationResult:
        """
        Colorize a single panel.
        
        Args:
            panel_image: The B&W panel image as numpy array
            panel_index: Index of the panel
            
        Returns:
            ColorizationResult with colorized image
        """
        pass
    
    @abstractmethod
    def colorize_batch(self, panels: List[tuple[np.ndarray, int]]) -> List[ColorizationResult]:
        """
        Colorize multiple panels.
        
        Args:
            panels: List of (panel_image, panel_index) tuples
            
        Returns:
            List of ColorizationResult for each panel
        """
        pass
    
    @abstractmethod
    def set_model(self, model_name: str, **model_params) -> None:
        """
        Set the colorization model.
        
        Args:
            model_name: Name of the model to use
            **model_params: Additional model parameters
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available colorization models.
        
        Returns:
            List of available model names
        """
        pass
    
    @abstractmethod
    def is_already_colored(self, image: np.ndarray) -> bool:
        """
        Check if an image is already colored.
        
        Args:
            image: Image to check
            
        Returns:
            True if image appears to be already colored
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for colorization.
        
        Args:
            image: Raw panel image
            
        Returns:
            Preprocessed image ready for colorization
        """
        pass
    
    @abstractmethod
    def postprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Postprocess colorized image.
        
        Args:
            image: Raw colorized output
            
        Returns:
            Postprocessed final image
        """
        pass
