"""
Base Anime Generator Interface

Defines the contract for AI-powered manga-to-anime style generation.
Converts manga panels into anime-styled images/videos using Stable Diffusion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum

import numpy as np


class AnimeStyle(Enum):
    """Available anime art styles."""
    MODERN_ANIME = "modern_anime"       # Clean modern anime look (Anything V5 style)
    CLASSIC_ANIME = "classic_anime"     # 90s anime aesthetic
    GHIBLI = "ghibli"                   # Studio Ghibli watercolor style
    SHONEN = "shonen"                   # Bold shonen action style
    CHIBI = "chibi"                     # Cute chibi/SD style (good for bloopers)
    VIBRANT = "vibrant"                 # Highly saturated, vivid colors


class AnimationMode(Enum):
    """Animation generation modes."""
    STATIC = "static"           # Single frame per panel (no motion)
    KEN_BURNS = "ken_burns"     # Pan/zoom animation on static frames
    INTERPOLATED = "interpolated"  # AI frame interpolation between panels
    ANIMATED = "animated"       # Full AnimateDiff motion generation
    PUPPET = "puppet"           # Industry-style puppet control (segment/rig/pose/lipsync)


@dataclass
class AnimeConfig:
    """Configuration for anime generation."""
    style: AnimeStyle = AnimeStyle.MODERN_ANIME
    animation_mode: AnimationMode = AnimationMode.KEN_BURNS
    strength: float = 0.75          # img2img denoising strength (0.6-0.85 recommended)
    guidance_scale: float = 7.5     # CFG scale — how closely to follow prompt
    num_inference_steps: int = 25   # Diffusion steps (more = better, slower)
    fps: int = 24                   # Output video FPS
    duration_per_panel: float = 2.5 # Seconds per panel in output video
    width: int = 768                # Output width (multiple of 8)
    height: int = 512               # Output height (multiple of 8)
    seed: int = -1                  # Random seed (-1 = random)
    use_controlnet: bool = True     # Use ControlNet for composition preservation
    negative_prompt: str = (
        "blurry, low quality, worst quality, jpeg artifacts, watermark, text, "
        "deformed, ugly, duplicate, morbid, extra fingers, mutated hands, "
        "poorly drawn hands, poorly drawn face, mutation, extra limbs, "
        "bad anatomy, bad proportions, disfigured, gross proportions"
    )
    positive_prompt_suffix: str = (
        "masterpiece, best quality, highly detailed, anime style, "
        "vibrant colors, sharp lines, professional, beautiful lighting"
    )
    model_id: str = ""              # HuggingFace model ID (auto-selected if empty)
    device: str = "auto"            # "auto", "cuda", "cpu"
    enable_attention_slicing: bool = True   # Reduce VRAM usage
    enable_vae_tiling: bool = True          # Reduce VRAM for large images


@dataclass
class AnimeFrame:
    """A single generated anime frame."""
    image: np.ndarray               # BGR image (OpenCV format)
    panel_index: int                # Source panel index
    style: str                      # Style used
    seed: int                       # Seed used for reproducibility
    prompt: str = ""                # Prompt used


@dataclass
class AnimeResult:
    """Result of anime generation."""
    frames: List[AnimeFrame]        # Generated anime frames
    video_path: Optional[Path]      # Path to output video (if generated)
    width: int = 0
    height: int = 0
    fps: int = 24
    duration: float = 0.0
    panel_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": str(self.video_path) if self.video_path else None,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration": self.duration,
            "panel_count": self.panel_count,
            "frame_count": len(self.frames),
            "metadata": self.metadata,
        }


class BaseAnimeGenerator(ABC):
    """Abstract base class for anime generation from manga panels."""

    @abstractmethod
    def generate(
        self,
        panels: List[np.ndarray],
        output_path: Path,
        config: Optional[AnimeConfig] = None,
        scene_descriptions: Optional[List[str]] = None,
    ) -> AnimeResult:
        """
        Generate anime-style content from manga panels.

        Args:
            panels: List of manga panel images (BGR, numpy).
            output_path: Path to save output video / frames.
            config: Anime generation configuration.
            scene_descriptions: Optional per-panel text descriptions for
                                better prompt guidance.

        Returns:
            AnimeResult with generated frames and optional video path.
        """
        pass

    @abstractmethod
    def stylize_panel(
        self,
        panel: np.ndarray,
        prompt: str = "",
        config: Optional[AnimeConfig] = None,
    ) -> AnimeFrame:
        """
        Stylize a single manga panel into anime art.

        Args:
            panel: Single manga panel image (BGR).
            prompt: Text prompt for style guidance.
            config: Anime generation config.

        Returns:
            AnimeFrame with the generated image.
        """
        pass
