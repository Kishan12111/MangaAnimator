"""
Panel Selector Module

Selects the most important panels for video creation.
Supports LLM-based, heuristic, and hybrid selection strategies.
"""

import logging
from typing import List, Optional, Dict, Callable

import numpy as np

from interfaces.base_panel_selector import (
    BasePanelSelector,
    PanelScore,
    SelectionResult,
    SelectionStrategy
)
from interfaces.base_panel_detector import Panel
from interfaces.base_ocr_engine import OCRResult

logger = logging.getLogger(__name__)


class PanelSelector(BasePanelSelector):
    """
    Concrete implementation of panel selection.
    
    Combines multiple scoring approaches to select the best panels.
    """
    
    def __init__(self):
        self._scoring_functions: Dict[str, tuple[Callable, float]] = {}
        self._llm_selected_panels: List[int] = []
        self._register_default_scorers()
    
    def _register_default_scorers(self) -> None:
        """Register default scoring functions.
        
        Weights are balanced for manga where OCR text may be sparse.
        Visual features dominate since they're always available.
        """
        # Visual complexity scorer (edge density — works on all panels)
        self.register_scoring_function(
            "visual_complexity",
            self._score_visual_complexity,
            weight=0.25
        )
        
        # Face/character presence scorer (cascade classifier)
        self.register_scoring_function(
            "character_presence",
            self._score_character_presence,
            weight=0.25
        )
        
        # Panel size scorer (larger panels often more important)
        self.register_scoring_function(
            "panel_size",
            lambda p, o: min(p.bounding_box.area / (1000 * 1000), 1.0),
            weight=0.15
        )
        
        # Contrast/drama scorer (high contrast = dramatic moment)
        self.register_scoring_function(
            "contrast_drama",
            self._score_contrast_drama,
            weight=0.15
        )
        
        # Text content scorer (dialogue panels are important)
        self.register_scoring_function(
            "text_content",
            lambda p, o: min(len(o.full_text) / 50.0, 1.0),
            weight=0.10
        )
        
        # Action word scorer
        self.register_scoring_function(
            "action_words",
            self._score_action_words,
            weight=0.10
        )
    
    def _score_action_words(self, panel: Panel, ocr_result: OCRResult) -> float:
        """Score based on presence of action/emotion words."""
        action_words = {
            'stop', 'run', 'fight', 'die', 'kill', 'love', 'hate',
            'no', 'yes', 'help', 'wait', 'attack', 'defend', 'save',
            'what', 'why', 'how', 'impossible', 'amazing', 'incredible'
        }
        
        text_lower = ocr_result.full_text.lower()
        words = set(text_lower.split())
        
        matches = len(words.intersection(action_words))
        return min(matches / 3.0, 1.0)
    
    def _score_visual_complexity(self, panel: Panel, ocr_result: OCRResult) -> float:
        """Score based on visual complexity of the panel."""
        try:
            import cv2
            
            # Convert to grayscale
            if len(panel.image.shape) == 3:
                gray = cv2.cvtColor(panel.image, cv2.COLOR_RGB2GRAY)
            else:
                gray = panel.image
            
            # Calculate edge density as proxy for complexity
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Normalize to 0-1 range
            return min(edge_density * 10, 1.0)
        except Exception:
            return 0.5
    
    def _score_character_presence(self, panel: Panel, ocr_result: OCRResult) -> float:
        """Score based on presence of faces/characters in the panel.
        
        Uses OpenCV's anime face cascade and contour analysis
        to detect character presence without deep learning.
        """
        try:
            import cv2
            
            if len(panel.image.shape) == 3:
                gray = cv2.cvtColor(panel.image, cv2.COLOR_RGB2GRAY)
            else:
                gray = panel.image
            
            h, w = gray.shape[:2]
            score = 0.0
            
            # 1. Try anime/lbp face cascade (fast, works for manga)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Resize for faster detection
            scale = min(1.0, 400 / max(h, w))
            small = cv2.resize(gray, None, fx=scale, fy=scale)
            
            faces = face_cascade.detectMultiScale(
                small, scaleFactor=1.1, minNeighbors=3,
                minSize=(int(20 * scale), int(20 * scale))
            )
            
            if len(faces) > 0:
                # More faces = more character interaction = more important
                face_area_ratio = sum(fw * fh for _, _, fw, fh in faces) / (small.shape[0] * small.shape[1])
                score = min(0.5 + face_area_ratio * 5, 1.0)
            
            # 2. Check for large dark regions (eyes, hair) typical of manga characters
            if score < 0.3:
                # Look for circular/elliptical dark blobs (eyes)
                _, thresh = cv2.threshold(small, 60, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter for eye-like contours (small, roughly circular)
                eye_like = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 50 < area < 5000:
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.3:  # Somewhat circular
                                eye_like += 1
                
                if eye_like >= 2:  # At least a pair of eyes
                    score = max(score, 0.4 + min(eye_like / 10.0, 0.3))
            
            return score
            
        except Exception as e:
            logger.debug(f"Character detection failed: {e}")
            return 0.3  # Neutral score on failure
    
    def _score_contrast_drama(self, panel: Panel, ocr_result: OCRResult) -> float:
        """Score based on contrast/drama level of the panel.
        
        High contrast panels (deep blacks + bright whites) often represent
        dramatic moments, action scenes, or emotional climaxes in manga.
        """
        try:
            import cv2
            
            if len(panel.image.shape) == 3:
                gray = cv2.cvtColor(panel.image, cv2.COLOR_RGB2GRAY)
            else:
                gray = panel.image
            
            # 1. Standard deviation of pixel values (higher = more contrast)
            std_dev = np.std(gray.astype(np.float32)) / 128.0  # Normalize to ~0-1
            
            # 2. Check for dramatic black/white ratio
            black_ratio = np.sum(gray < 30) / gray.size
            white_ratio = np.sum(gray > 225) / gray.size
            drama_ratio = black_ratio * white_ratio * 20  # High when both present
            
            # 3. Local contrast (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = np.var(laplacian) / 5000.0  # Normalize
            
            # Combine scores
            score = min(std_dev * 0.4 + drama_ratio * 0.3 + lap_var * 0.3, 1.0)
            
            return score
            
        except Exception:
            return 0.3
    
    def register_scoring_function(
        self,
        name: str,
        func: Callable[[Panel, OCRResult], float],
        weight: float = 1.0
    ) -> None:
        """Register a custom scoring function."""
        self._scoring_functions[name] = (func, weight)
        logger.debug(f"Registered scoring function: {name} (weight={weight})")
    
    def set_llm_selected_panels(self, panel_indices: List[int]) -> None:
        """Set panels selected by LLM for hybrid mode."""
        self._llm_selected_panels = panel_indices
        logger.debug(f"Set LLM-selected panels: {panel_indices}")
    
    def select(
        self,
        panels: List[Panel],
        ocr_results: List[OCRResult],
        max_panels: int,
        strategy: str = SelectionStrategy.HYBRID
    ) -> SelectionResult:
        """Select the most important panels."""
        logger.info(f"Selecting panels using strategy: {strategy}")
        
        if not panels:
            return SelectionResult(
                selected_indices=[],
                panel_scores=[],
                selection_method=strategy
            )
        
        # Build OCR lookup by panel index
        ocr_lookup = {o.panel_index: o for o in ocr_results}
        
        # Score all panels
        panel_scores = []
        for panel in panels:
            ocr_result = ocr_lookup.get(
                panel.index,
                OCRResult(panel_index=panel.index, text_boxes=[], full_text="", confidence=0.0)
            )
            
            if strategy == SelectionStrategy.LLM:
                score = self.score_panel_llm(panel, ocr_result)
            elif strategy == SelectionStrategy.HEURISTIC:
                score = self.score_panel_heuristic(panel, ocr_result)
            else:  # HYBRID
                score = self._score_panel_hybrid(panel, ocr_result)
            
            panel_scores.append(score)
        
        # Sort by score and select top panels
        scored_panels = list(zip(panel_scores, panels))
        scored_panels.sort(key=lambda x: -x[0].total_score)
        
        # Select top panels while maintaining order
        top_indices = set()
        for score, panel in scored_panels[:max_panels]:
            top_indices.add(panel.index)
        
        # Return in original reading order
        selected_indices = [
            panel.index for panel in panels
            if panel.index in top_indices
        ]
        
        # Get scores for selected panels
        selected_scores = [
            score for score, panel in scored_panels
            if panel.index in selected_indices
        ]
        
        logger.info(f"Selected {len(selected_indices)} panels: {selected_indices}")
        
        return SelectionResult(
            selected_indices=selected_indices,
            panel_scores=panel_scores,
            selection_method=strategy,
            metadata={
                'total_panels': len(panels),
                'max_requested': max_panels
            }
        )
    
    def score_panel_heuristic(
        self,
        panel: Panel,
        ocr_result: OCRResult
    ) -> PanelScore:
        """Score a panel using heuristic methods."""
        component_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, (func, weight) in self._scoring_functions.items():
            try:
                score = func(panel, ocr_result)
                component_scores[name] = score
                weighted_sum += score * weight
                total_weight += weight
            except Exception as e:
                logger.warning(f"Scoring function {name} failed: {e}")
                component_scores[name] = 0.0
        
        total_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return PanelScore(
            panel_index=panel.index,
            total_score=total_score,
            component_scores=component_scores,
            reasoning="Heuristic scoring based on text, size, and visual features"
        )
    
    def score_panel_llm(
        self,
        panel: Panel,
        ocr_result: OCRResult,
        context: Optional[str] = None
    ) -> PanelScore:
        """Score a panel using LLM-based analysis."""
        # Check if panel is in LLM-selected list
        if panel.index in self._llm_selected_panels:
            return PanelScore(
                panel_index=panel.index,
                total_score=1.0,
                component_scores={'llm_selected': 1.0},
                reasoning="Selected by LLM as important"
            )
        else:
            return PanelScore(
                panel_index=panel.index,
                total_score=0.0,
                component_scores={'llm_selected': 0.0},
                reasoning="Not selected by LLM"
            )
    
    def _score_panel_hybrid(
        self,
        panel: Panel,
        ocr_result: OCRResult
    ) -> PanelScore:
        """Score using hybrid approach (LLM + heuristics)."""
        heuristic_score = self.score_panel_heuristic(panel, ocr_result)
        llm_score = self.score_panel_llm(panel, ocr_result)
        
        # Combine scores (LLM has higher weight)
        llm_weight = 0.6
        heuristic_weight = 0.4
        
        combined_score = (
            llm_score.total_score * llm_weight +
            heuristic_score.total_score * heuristic_weight
        )
        
        component_scores = heuristic_score.component_scores.copy()
        component_scores['llm_boost'] = llm_score.total_score
        
        return PanelScore(
            panel_index=panel.index,
            total_score=combined_score,
            component_scores=component_scores,
            reasoning=f"Hybrid: LLM={llm_score.total_score:.2f}, Heuristic={heuristic_score.total_score:.2f}"
        )
