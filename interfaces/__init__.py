"""
MangaVID Interfaces Package

This package contains all abstract base classes that define the contracts
for each module in the MangaVID pipeline. All implementations must inherit
from these interfaces to ensure modularity and swappability.
"""

from interfaces.base_input_handler import BaseInputHandler
from interfaces.base_panel_detector import BasePanelDetector
from interfaces.base_ocr_engine import BaseOCREngine
from interfaces.base_story_intelligence import BaseStoryIntelligence
from interfaces.base_panel_selector import BasePanelSelector
from interfaces.base_colorizer import BaseColorizer
from interfaces.base_narrator import BaseNarrator
from interfaces.base_video_generator import BaseVideoGenerator
from interfaces.base_uploader import BaseUploader

__all__ = ['BaseInputHandler', 'BasePanelDetector', 'BaseOCREngine', 'BaseStoryIntelligence', 'BasePanelSelector', 'BaseColorizer', 'BaseNarrator', 'BaseVideoGenerator', 'BaseUploader']
