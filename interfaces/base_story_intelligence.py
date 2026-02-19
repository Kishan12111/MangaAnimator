"""
Base Story Intelligence Interface

Defines the contract for AI-powered story understanding.
This is the core intelligence module that analyzes extracted text
and determines the narrative structure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class NarrativeTone(Enum):
    """Possible tones for narration."""
    DRAMATIC = "dramatic"
    COMEDIC = "comedic"
    ACTION = "action"
    ROMANTIC = "romantic"
    MYSTERIOUS = "mysterious"
    HORROR = "horror"
    SLICE_OF_LIFE = "slice_of_life"
    NEUTRAL = "neutral"


@dataclass
class StoryIntelligenceInput:
    """Input for story intelligence processing."""
    panel_texts: List[str]  # Ordered list of text extracted from each panel
    panel_indices: List[int]  # Corresponding panel indices
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_duration_seconds: float = 60.0
    narration_style: str = "engaging"


@dataclass
class StoryIntelligenceOutput:
    """Output from story intelligence processing."""
    summary_script: str  # The narration script
    selected_panels: List[int]  # Indices of panels to include in video
    tone: str  # Detected/chosen tone
    key_events: List[str]  # List of key story events
    characters: List[str]  # Detected character names
    estimated_duration: float  # Estimated narration duration in seconds
    confidence: float  # Confidence in the analysis
    metadata: Dict[str, Any] = field(default_factory=dict)
    intro_hook: str = ""  # Short atmospheric intro line for context card
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary_script": self.summary_script,
            "selected_panels": self.selected_panels,
            "tone": self.tone,
            "key_events": self.key_events,
            "characters": self.characters,
            "estimated_duration": self.estimated_duration,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "intro_hook": self.intro_hook,
        }


class BaseStoryIntelligence(ABC):
    """
    Abstract base class for story intelligence engines.
    
    Implementations should use LLMs or other AI to understand
    the manga story and produce narration scripts with panel selections.
    """
    
    @abstractmethod
    def analyze(self, input_data: StoryIntelligenceInput) -> StoryIntelligenceOutput:
        """
        Analyze the story from extracted panel texts.
        
        Args:
            input_data: StoryIntelligenceInput containing panel texts and config
            
        Returns:
            StoryIntelligenceOutput with narration script and panel selections
        """
        pass
    
    @abstractmethod
    def generate_script(
        self, 
        panel_texts: List[str], 
        max_words: int = 150,
        style: str = "engaging"
    ) -> str:
        """
        Generate a narration script from panel texts.
        
        Args:
            panel_texts: Ordered list of panel texts
            max_words: Maximum word count for the script
            style: Narration style (engaging, dramatic, etc.)
            
        Returns:
            Generated narration script
        """
        pass
    
    @abstractmethod
    def select_key_panels(
        self,
        panel_texts: List[str],
        max_panels: int = 10,
        selection_strategy: str = "llm"
    ) -> List[int]:
        """
        Select the most important panels for the video.
        
        Args:
            panel_texts: Ordered list of panel texts
            max_panels: Maximum number of panels to select
            selection_strategy: 'llm', 'heuristic', or 'hybrid'
            
        Returns:
            List of selected panel indices
        """
        pass
    
    @abstractmethod
    def detect_tone(self, panel_texts: List[str]) -> NarrativeTone:
        """
        Detect the narrative tone of the manga.
        
        Args:
            panel_texts: Ordered list of panel texts
            
        Returns:
            Detected narrative tone
        """
        pass
    
    @abstractmethod
    def set_model(self, model_name: str, **model_params) -> None:
        """
        Set the underlying LLM model.
        
        Args:
            model_name: Name of the model to use
            **model_params: Additional model parameters
        """
        pass
    
    @abstractmethod
    def adjust_for_duration(
        self,
        script: str,
        target_duration: float,
        words_per_second: float = 2.5
    ) -> str:
        """
        Adjust script length to fit target duration.
        
        Args:
            script: Original narration script
            target_duration: Target duration in seconds
            words_per_second: Estimated speaking rate
            
        Returns:
            Adjusted script
        """
        pass
