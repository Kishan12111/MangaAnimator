"""
Base Narrator Interface

Defines the contract for text-to-speech narration generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Union
import numpy as np


@dataclass
class NarrationConfig:
    """Configuration for narration generation."""
    voice: str = "default"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    language: str = "en"
    emotion: str = "neutral"


@dataclass
class NarrationResult:
    """Result of narration generation."""
    audio_data: np.ndarray
    sample_rate: int
    duration_seconds: float
    script: str
    segments: List[Dict[str, any]] = field(default_factory=list)  # Word/sentence timings
    output_path: Optional[Path] = None
    metadata: Dict[str, any] = field(default_factory=dict)


class BaseNarrator(ABC):
    """
    Abstract base class for TTS narration.
    
    Implementations should generate audio narration from text
    using TTS models or services.
    """
    
    @abstractmethod
    def generate(
        self,
        script: str,
        config: Optional[NarrationConfig] = None
    ) -> NarrationResult:
        """
        Generate narration audio from script.
        
        Args:
            script: The narration script text
            config: Optional narration configuration
            
        Returns:
            NarrationResult with audio data
        """
        pass
    
    @abstractmethod
    def generate_to_file(
        self,
        script: str,
        output_path: Path,
        config: Optional[NarrationConfig] = None
    ) -> NarrationResult:
        """
        Generate narration and save to file.
        
        Args:
            script: The narration script text
            output_path: Path to save audio file
            config: Optional narration configuration
            
        Returns:
            NarrationResult with output path
        """
        pass
    
    @abstractmethod
    def estimate_duration(self, script: str, speed: float = 1.0) -> float:
        """
        Estimate narration duration for a script.
        
        Args:
            script: The narration script
            speed: Speaking speed multiplier
            
        Returns:
            Estimated duration in seconds
        """
        pass
    
    @abstractmethod
    def set_model(self, model_name: str, **model_params) -> None:
        """
        Set the TTS model.
        
        Args:
            model_name: Name of the model to use
            **model_params: Additional model parameters
        """
        pass
    
    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voices.
        
        Returns:
            List of available voice names
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available TTS models.
        
        Returns:
            List of available model names
        """
        pass
    
    @abstractmethod
    def adjust_speed_for_duration(
        self,
        script: str,
        target_duration: float
    ) -> float:
        """
        Calculate speed needed to fit target duration.
        
        Args:
            script: The narration script
            target_duration: Target duration in seconds
            
        Returns:
            Speed multiplier to use
        """
        pass
