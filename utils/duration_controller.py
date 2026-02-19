"""
Duration Controller

Manages video duration constraints by adjusting narration length,
panel count, and panel display duration.
"""

import logging
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DurationPlan:
    """Plan for achieving target video duration."""
    target_duration: float
    estimated_duration: float
    panel_count: int
    time_per_panel: float
    narration_word_count: int
    narration_speed: float
    adjustments_made: List[str]


class DurationController:
    """
    Controller for managing video duration constraints.
    
    Automatically adjusts various parameters to meet the target
    video duration while maintaining quality.
    """
    
    WORDS_PER_SECOND = 2.5  # Average speaking rate
    MIN_TIME_PER_PANEL = 2.0  # Minimum seconds to display a panel
    MAX_TIME_PER_PANEL = 5.0  # Maximum seconds to display a panel
    MIN_NARRATION_SPEED = 0.85  # Slowest TTS speed
    MAX_NARRATION_SPEED = 1.3  # Fastest TTS speed
    
    def __init__(self, target_duration: float = 120.0):
        self.target_duration = target_duration
        self._adjustments: List[str] = []
    
    def set_target_duration(self, duration: float) -> None:
        """Set the target video duration."""
        self.target_duration = duration
        logger.info(f"Target duration set to {duration}s")
    
    def calculate_max_panels(self, available_duration: float = None) -> int:
        """
        Calculate maximum number of panels for target duration.
        
        Args:
            available_duration: Override duration (uses target if None)
            
        Returns:
            Maximum recommended panel count
        """
        duration = available_duration or self.target_duration
        
        # Reserve time for transitions
        transition_time = 0.3  # seconds per transition
        
        # Calculate with minimum panel display time
        max_panels = int(duration / (self.MIN_TIME_PER_PANEL + transition_time))
        
        return max(1, min(max_panels, 40))  # Allow up to 40 panels for ~2 min videos
    
    def calculate_max_words(self, available_duration: float = None) -> int:
        """
        Calculate maximum word count for narration.
        
        Args:
            available_duration: Override duration (uses target if None)
            
        Returns:
            Maximum word count
        """
        duration = available_duration or self.target_duration
        
        # Leave some buffer for natural pacing
        buffer_ratio = 0.9
        max_words = int(duration * self.WORDS_PER_SECOND * buffer_ratio)
        
        return max(10, max_words)
    
    def plan_duration(
        self,
        narration_text: str,
        panel_count: int,
        transition_duration: float = 0.3
    ) -> DurationPlan:
        """
        Create a plan to fit content into target duration.
        
        Args:
            narration_text: The narration script
            panel_count: Number of panels to display
            transition_duration: Duration of transitions
            
        Returns:
            DurationPlan with adjusted parameters
        """
        self._adjustments = []
        
        word_count = len(narration_text.split())
        narration_duration = word_count / self.WORDS_PER_SECOND
        
        # Calculate total transition time
        total_transition_time = transition_duration * max(0, panel_count - 1)
        
        # Calculate time available for panels
        available_for_panels = self.target_duration - total_transition_time
        
        # Initial estimates
        estimated_duration = narration_duration
        adjusted_word_count = word_count
        adjusted_panel_count = panel_count
        narration_speed = 1.0
        
        # Strategy 1: If narration is too long, reduce word count
        if narration_duration > self.target_duration:
            target_words = self.calculate_max_words()
            
            if word_count > target_words:
                adjusted_word_count = target_words
                self._adjustments.append(
                    f"Reduced narration from {word_count} to {target_words} words"
                )
                estimated_duration = target_words / self.WORDS_PER_SECOND
        
        # Strategy 2: If still too long, increase narration speed
        if estimated_duration > self.target_duration:
            required_speed = estimated_duration / self.target_duration
            narration_speed = min(required_speed, self.MAX_NARRATION_SPEED)
            estimated_duration = estimated_duration / narration_speed
            self._adjustments.append(f"Adjusted narration speed to {narration_speed:.2f}x")
        
        # Strategy 3: Adjust panel count if needed
        max_panels = self.calculate_max_panels()
        if adjusted_panel_count > max_panels:
            adjusted_panel_count = max_panels
            self._adjustments.append(f"Reduced panels from {panel_count} to {max_panels}")
        
        # Strategy 4: If narration is shorter than target, use natural pacing
        if estimated_duration < self.target_duration * 0.5:
            # Narration is very short, slow it down
            narration_speed = max(
                estimated_duration / (self.target_duration * 0.7),
                self.MIN_NARRATION_SPEED
            )
            estimated_duration = estimated_duration / narration_speed
            self._adjustments.append(f"Slowed narration to {narration_speed:.2f}x for better pacing")
        
        # Calculate time per panel
        time_per_panel = max(
            self.MIN_TIME_PER_PANEL,
            min(
                (estimated_duration - total_transition_time) / max(1, adjusted_panel_count),
                self.MAX_TIME_PER_PANEL
            )
        )
        
        # Final duration estimate
        final_duration = (
            (time_per_panel * adjusted_panel_count) +
            (transition_duration * max(0, adjusted_panel_count - 1))
        )
        
        return DurationPlan(
            target_duration=self.target_duration,
            estimated_duration=final_duration,
            panel_count=adjusted_panel_count,
            time_per_panel=time_per_panel,
            narration_word_count=adjusted_word_count,
            narration_speed=narration_speed,
            adjustments_made=self._adjustments.copy()
        )
    
    def adjust_script_length(
        self,
        script: str,
        max_words: int = None
    ) -> str:
        """
        Truncate script to fit within word limit.
        
        Args:
            script: Original narration script
            max_words: Maximum word count (calculates from target if None)
            
        Returns:
            Adjusted script
        """
        if max_words is None:
            max_words = self.calculate_max_words()
        
        words = script.split()
        
        if len(words) <= max_words:
            return script
        
        # Truncate and try to end at sentence boundary
        truncated = words[:max_words]
        result = " ".join(truncated)
        
        # Find last sentence ending
        for punct in ['. ', '! ', '? ']:
            last_idx = result.rfind(punct)
            if last_idx > len(result) * 0.7:  # Keep at least 70%
                result = result[:last_idx + 1]
                break
        
        logger.info(f"Truncated script from {len(words)} to {len(result.split())} words")
        
        return result.strip()
    
    def select_panels_for_duration(
        self,
        panel_scores: List[Tuple[int, float]],
        target_panel_count: int = None
    ) -> List[int]:
        """
        Select panels based on scores and duration constraints.
        
        Args:
            panel_scores: List of (panel_index, score) tuples
            target_panel_count: Target number of panels (calculates if None)
            
        Returns:
            List of selected panel indices
        """
        if target_panel_count is None:
            target_panel_count = self.calculate_max_panels()
        
        # Sort by score (descending)
        sorted_panels = sorted(panel_scores, key=lambda x: -x[1])
        
        # Select top panels
        selected = [idx for idx, _ in sorted_panels[:target_panel_count]]
        
        # Sort back to original order
        selected.sort()
        
        logger.info(f"Selected {len(selected)} panels for {self.target_duration}s video")
        
        return selected
