"""
Base Video Generator Interface

Defines the contract for video generation from panels and narration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
from enum import Enum
import numpy as np


class TransitionType(Enum):
    """Available transition types."""
    NONE = "none"
    FADE = "fade"
    CROSSFADE = "crossfade"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    WIPE = "wipe"
    FLASH_WHITE = "flash_white"
    DISSOLVE = "dissolve"


class ZoomEffect(Enum):
    """Available zoom effects."""
    NONE = "none"
    KEN_BURNS_IN = "ken_burns_in"  # Slow zoom in
    KEN_BURNS_OUT = "ken_burns_out"  # Slow zoom out
    FOCUS_CENTER = "focus_center"
    FOCUS_TOP = "focus_top"
    FOCUS_BOTTOM = "focus_bottom"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"


@dataclass
class PanelTiming:
    """Timing information for a panel in the video."""
    panel_index: int
    start_time: float
    end_time: float
    duration: float
    transition_in: TransitionType = TransitionType.FADE
    transition_out: TransitionType = TransitionType.NONE
    zoom_effect: ZoomEffect = ZoomEffect.KEN_BURNS_IN
    transition_duration: float = 0.5


@dataclass
class SubtitleSegment:
    """A subtitle segment for the video."""
    text: str
    start_time: float
    end_time: float
    style: Dict[str, any] = field(default_factory=dict)


@dataclass
class VideoConfig:
    """Configuration for video generation."""
    width: int = 1080
    height: int = 1920
    fps: int = 30
    max_duration: float = 60.0
    transition_duration: float = 0.5
    default_transition: TransitionType = TransitionType.CROSSFADE
    default_zoom: ZoomEffect = ZoomEffect.KEN_BURNS_IN
    include_subtitles: bool = True
    subtitle_style: Dict[str, any] = field(default_factory=lambda: {
        "font": "Arial Black",
        "fontsize": 72,
        "color": "yellow",
        "stroke_color": "black",
        "stroke_width": 2,
        "position": "bottom"
    })
    background_color: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class VideoResult:
    """Result of video generation."""
    output_path: Path
    duration: float
    width: int
    height: int
    fps: int
    panel_count: int
    has_audio: bool
    has_subtitles: bool
    file_size_bytes: int
    metadata: Dict[str, any] = field(default_factory=dict)


class BaseVideoGenerator(ABC):
    """
    Abstract base class for video generation.
    
    Implementations should create videos from panels and narration
    with transitions, effects, and subtitles.
    """
    
    @abstractmethod
    def generate(
        self,
        panels: List[np.ndarray],
        audio_path: Optional[Path],
        output_path: Path,
        config: Optional[VideoConfig] = None
    ) -> VideoResult:
        """
        Generate video from panels and audio.
        
        Args:
            panels: List of panel images (in display order)
            audio_path: Path to narration audio file
            output_path: Path to save output video
            config: Video configuration
            
        Returns:
            VideoResult with output information
        """
        pass
    
    @abstractmethod
    def calculate_timings(
        self,
        panel_count: int,
        audio_duration: float,
        config: VideoConfig
    ) -> List[PanelTiming]:
        """
        Calculate optimal timing for each panel.
        
        Args:
            panel_count: Number of panels
            audio_duration: Duration of narration audio
            config: Video configuration
            
        Returns:
            List of PanelTiming for each panel
        """
        pass
    
    @abstractmethod
    def apply_transition(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        transition_type: TransitionType,
        progress: float
    ) -> np.ndarray:
        """
        Apply transition between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            transition_type: Type of transition
            progress: Transition progress (0.0 to 1.0)
            
        Returns:
            Blended frame
        """
        pass
    
    @abstractmethod
    def apply_zoom_effect(
        self,
        frame: np.ndarray,
        zoom_effect: ZoomEffect,
        progress: float,
        config: VideoConfig
    ) -> np.ndarray:
        """
        Apply zoom/pan effect to frame.
        
        Args:
            frame: Source frame
            zoom_effect: Effect to apply
            progress: Effect progress (0.0 to 1.0)
            config: Video configuration
            
        Returns:
            Frame with effect applied
        """
        pass
    
    @abstractmethod
    def add_subtitles(
        self,
        video_path: Path,
        subtitles: List[SubtitleSegment],
        output_path: Path,
        config: VideoConfig
    ) -> Path:
        """
        Add subtitles to video.
        
        Args:
            video_path: Path to input video
            subtitles: List of subtitle segments
            output_path: Path to save output video
            config: Video configuration
            
        Returns:
            Path to video with subtitles
        """
        pass
    
    @abstractmethod
    def generate_subtitles_from_script(
        self,
        script: str,
        audio_duration: float,
        words_per_segment: int = 8
    ) -> List[SubtitleSegment]:
        """
        Generate subtitle segments from narration script.
        
        Args:
            script: Narration script
            audio_duration: Duration of audio
            words_per_segment: Words per subtitle segment
            
        Returns:
            List of SubtitleSegment
        """
        pass
    
    @abstractmethod
    def resize_for_format(
        self,
        image: np.ndarray,
        target_width: int,
        target_height: int,
        fit_mode: str = "contain"
    ) -> np.ndarray:
        """
        Resize image for target video format.
        
        Args:
            image: Source image
            target_width: Target width
            target_height: Target height
            fit_mode: 'contain', 'cover', or 'stretch'
            
        Returns:
            Resized image
        """
        pass
