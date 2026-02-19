"""
Base Panel Selector Interface

Defines the contract for panel selection strategies.
Supports LLM-based, heuristic, and hybrid approaches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
import numpy as np

from interfaces.base_panel_detector import Panel
from interfaces.base_ocr_engine import OCRResult


@dataclass
class PanelScore:
    """Score information for a panel."""
    panel_index: int
    total_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""


@dataclass 
class SelectionResult:
    """Result of panel selection."""
    selected_indices: List[int]
    panel_scores: List[PanelScore]
    selection_method: str
    metadata: Dict[str, any] = field(default_factory=dict)


class SelectionStrategy:
    """Enumeration of selection strategies."""
    LLM = "llm"
    HEURISTIC = "heuristic"
    HYBRID = "hybrid"


class BasePanelSelector(ABC):
    """
    Abstract base class for panel selection.
    
    Implementations should rank and select panels based on
    their importance to the story.
    """
    
    @abstractmethod
    def select(
        self,
        panels: List[Panel],
        ocr_results: List[OCRResult],
        max_panels: int,
        strategy: str = SelectionStrategy.HYBRID
    ) -> SelectionResult:
        """
        Select the most important panels.
        
        Args:
            panels: List of detected panels
            ocr_results: OCR results for each panel
            max_panels: Maximum number of panels to select
            strategy: Selection strategy to use
            
        Returns:
            SelectionResult with selected panel indices
        """
        pass
    
    @abstractmethod
    def score_panel_heuristic(
        self,
        panel: Panel,
        ocr_result: OCRResult
    ) -> PanelScore:
        """
        Score a panel using heuristic methods.
        
        Args:
            panel: The panel to score
            ocr_result: OCR result for the panel
            
        Returns:
            PanelScore with heuristic scoring
        """
        pass
    
    @abstractmethod
    def score_panel_llm(
        self,
        panel: Panel,
        ocr_result: OCRResult,
        context: Optional[str] = None
    ) -> PanelScore:
        """
        Score a panel using LLM-based analysis.
        
        Args:
            panel: The panel to score
            ocr_result: OCR result for the panel
            context: Optional story context
            
        Returns:
            PanelScore with LLM-based scoring
        """
        pass
    
    @abstractmethod
    def register_scoring_function(
        self,
        name: str,
        func: Callable[[Panel, OCRResult], float],
        weight: float = 1.0
    ) -> None:
        """
        Register a custom scoring function.
        
        Args:
            name: Name of the scoring function
            func: The scoring function
            weight: Weight for this scoring component
        """
        pass
    
    @abstractmethod
    def set_llm_selected_panels(self, panel_indices: List[int]) -> None:
        """
        Set panels selected by LLM for hybrid mode.
        
        Args:
            panel_indices: Indices selected by LLM
        """
        pass
