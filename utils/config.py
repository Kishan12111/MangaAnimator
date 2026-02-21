"""
Configuration Management

Handles loading and validating configuration for the MangaVID pipeline.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Configuration for the MangaVID pipeline.
    
    All settings can be overridden from config.json.
    """

    # Video settings
    max_video_duration: float = 120.0
    video_width: int = 1080
    video_height: int = 1920
    video_fps: int = 30

    # Narration settings
    narration_style: str = "engaging"
    tts_model: str = "pyttsx3"
    narrator_speed: float = 1.0
    narrator_voice: str = "default"

    # Panel selection
    panel_selection_mode: str = "hybrid"
    max_panels: int = 35

    # LLM / Story Intelligence
    llm_model: str = "placeholder"

    # Colorization
    colorization_model: str = "placeholder"
    enable_colorization: bool = True

    # OCR
    ocr_language: str = "en"

    # Reading direction
    reading_direction: str = "rtl"

    # Effects
    enable_transitions: bool = True
    enable_zoom_effects: bool = True
    transition_duration: float = 0.5

    # Subtitles
    enable_subtitles: bool = True
    subtitle_font_size: int = 72
    subtitle_color: str = "yellow"

    # Output
    output_directory: str = "outputs"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # API keys
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # ElevenLabs
    elevenlabs_api_key: Optional[str] = None

    # Anime/manga title
    anime_title: str = ""

    # Character color mapping
    # Format: {"character_name": {"hair": [R,G,B], "skin": [R,G,B]}}
    character_colors: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)

    # Upload settings
    auto_upload: bool = False
    upload_platform: str = "local"

    # Anime generation settings
    enable_anime_gen: bool = False
    anime_style: str = "modern_anime"
    anime_animation_mode: str = "ken_burns"
    anime_strength: float = 0.65
    anime_steps: int = 25
    anime_controlnet: bool = True
    anime_fps: int = 24

    # Extra settings
    extra_settings: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""

        known_fields = {
            'max_video_duration', 'video_width', 'video_height', 'video_fps',
            'narration_style', 'tts_model', 'narrator_speed', 'narrator_voice',
            'panel_selection_mode', 'max_panels', 'llm_model',
            'colorization_model', 'enable_colorization', 'ocr_language',
            'reading_direction', 'enable_transitions', 'enable_zoom_effects',
            'transition_duration', 'enable_subtitles', 'subtitle_font_size',
            'subtitle_color', 'output_directory', 'log_level', 'log_file',
            'openai_api_key', 'gemini_api_key', 'elevenlabs_api_key',
            'anime_title', 'character_colors', 'auto_upload',
            'upload_platform', 'enable_anime_gen', 'anime_style',
            'anime_animation_mode',
            'anime_strength', 'anime_steps', 'anime_controlnet',
            'anime_fps', 'narrator_voice',
        }
        config_data = {}
        extra = {}

        for key, value in data.items():
            if key in known_fields:
                config_data[key] = value
            else:
                extra[key] = value

        config_data["extra_settings"] = extra

        return cls(**config_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        result = {
            "max_video_duration": self.max_video_duration,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "video_fps": self.video_fps,
            "narration_style": self.narration_style,
            "tts_model": self.tts_model,
            "narrator_speed": self.narrator_speed,
            "narrator_voice": self.narrator_voice,
            "panel_selection_mode": self.panel_selection_mode,
            "max_panels": self.max_panels,
            "llm_model": self.llm_model,
            "colorization_model": self.colorization_model,
            "enable_colorization": self.enable_colorization,
            "ocr_language": self.ocr_language,
            "reading_direction": self.reading_direction,
            "enable_transitions": self.enable_transitions,
            "enable_zoom_effects": self.enable_zoom_effects,
            "transition_duration": self.transition_duration,
            "enable_subtitles": self.enable_subtitles,
            "subtitle_font_size": self.subtitle_font_size,
            "subtitle_color": self.subtitle_color,
            "output_directory": self.output_directory,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "auto_upload": self.auto_upload,
            "upload_platform": self.upload_platform,
            "anime_title": self.anime_title,
            "enable_anime_gen": self.enable_anime_gen,
            "anime_style": self.anime_style,
            "anime_animation_mode": self.anime_animation_mode,
            "anime_strength": self.anime_strength,
            "anime_steps": self.anime_steps,
            "anime_controlnet": self.anime_controlnet,
            "anime_fps": self.anime_fps,
        }

        # Include any extra settings passed through from config
        # file that weren't recognized as known fields

        result.update(self.extra_settings)
        return result

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if self.max_video_duration < 5:
            warnings.append("max_video_duration is very short (< 5 seconds)")

        if self.max_video_duration > 180:
            warnings.append("max_video_duration may be too long for short-form platforms")

        if self.max_panels < 3:
            warnings.append("max_panels is very low, video may be too short")

        if self.video_fps < 24:
            warnings.append("video_fps is low, video may look choppy")

        if self.panel_selection_mode not in ('llm', 'heuristic', 'hybrid'):
            warnings.append(f"Unknown panel_selection_mode: {self.panel_selection_mode}")

        if self.reading_direction not in ('rtl', 'ltr'):
            warnings.append(f"Unknown reading_direction: {self.reading_direction}")

        return warnings


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from file or return defaults.
    
    Args:
        config_path: Path to config.json file
        
    Returns:
        Config instance
    """
    if config_path is None:
        config_path = Path("config.json")

    config_path = Path(config_path)

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            config = Config.from_dict(data)
            logger.info(f"Loaded configuration from {config_path}")

            # Validate and warn
            warnings = config.validate()
            for warning in warnings:
                logger.warning(f"Config warning: {warning}")

            return config

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    logger.info("Using default configuration")
    return Config()


def save_config(config: Config, config_path: Optional[Path] = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Config instance to save
        config_path: Path to save to (default: config.json)
    """
    if config_path is None:
        config_path = Path("config.json")

    config_path = Path(config_path)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

        logger.info(f"Saved configuration to {config_path}")

    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise
