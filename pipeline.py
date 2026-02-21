"""
MangaVID Pipeline

Main pipeline orchestrator that coordinates all modules to convert
manga files into narrated short-form videos.
"""

import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from interfaces.base_story_intelligence import StoryIntelligenceInput, StoryIntelligenceOutput
from interfaces.base_video_generator import VideoConfig
from interfaces.base_uploader import UploadConfig, Platform

from modules.input_handler import InputHandler
from modules.panel_detector import PanelDetector
from modules.ocr_engine import OCREngine
from modules.story_intelligence import StoryIntelligenceEngine
from modules.panel_selector import PanelSelector
from modules.colorizer import Colorizer
from modules.narrator import Narrator
from modules.video_generator import VideoGenerator
from modules.uploader import Uploader

from utils.config import Config, load_config
from utils.logger import setup_logging, PipelineLogger
from utils.duration_controller import DurationController

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    success: bool
    video_path: Optional[Path] = None
    duration: float = 0.0
    panel_count: int = 0
    narration_script: str = ""
    error_message: Optional[str] = None
    stage_timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "video_path": str(self.video_path) if self.video_path else None,
            "duration": self.duration,
            "panel_count": self.panel_count,
            "narration_script": self.narration_script,
            "error_message": self.error_message,
            "stage_timings": self.stage_timings,
            "metadata": self.metadata
        }


class MangaVideoPipeline:
    """
    Main pipeline for converting manga to video.
    
    Orchestrates all modules in the correct order:
    Input → Panel Detection → OCR → Story Understanding → 
    Panel Selection → Colorization → Narration → Video → Upload
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration object (loads from file if None)
        """
        self.config = config or load_config()
        
        # Set up logging
        setup_logging(
            level=self.config.log_level,
            log_file=Path(self.config.log_file) if self.config.log_file else None
        )
        
        self._logger = PipelineLogger("MangaVideoPipeline")
        
        # Initialize modules
        self._init_modules()
        
        # Initialize duration controller
        self.duration_controller = DurationController(self.config.max_video_duration)
        
        logger.info("MangaVID Pipeline initialized")
    
    def _init_modules(self) -> None:
        """Initialize all pipeline modules."""
        self._logger.info("Initializing modules...")
        
        self.input_handler = InputHandler()
        self.panel_detector = PanelDetector()
        
        # Skip EasyOCR when Gemini Vision is available — Gemini reads panel text
        # directly from images (faster, no RAM overhead from EasyOCR models)
        self._use_vision_ocr = bool(self.config.gemini_api_key)
        if self._use_vision_ocr:
            self.ocr_engine = None
            logger.info("Gemini Vision active — skipping EasyOCR (saves RAM & time)")
        else:
            self.ocr_engine = OCREngine()
            self.ocr_engine.set_language(self.config.ocr_language)
        
        self.story_intelligence = StoryIntelligenceEngine(
            model_name=self.config.llm_model,
            api_key=self.config.gemini_api_key or self.config.openai_api_key,
            anime_title=getattr(self.config, 'anime_title', ''),
        )
        self.panel_selector = PanelSelector()
        self.colorizer = Colorizer(
            model_name=self.config.colorization_model,
            api_key=self.config.gemini_api_key,
            character_colors=self.config.character_colors if hasattr(self.config, 'character_colors') else None,
            anime_title=getattr(self.config, 'anime_title', ''),
        )
        self.narrator = Narrator(
            model_name=self.config.tts_model,
            voice=self.config.narrator_voice if self.config.narrator_voice != 'default' else None,
            api_key=getattr(self.config, 'elevenlabs_api_key', None)
        )
        self.video_generator = VideoGenerator()
        self.uploader = Uploader()
        
        self._logger.info("All modules initialized")
    
    def process(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        title: Optional[str] = None
    ) -> PipelineResult:
        """
        Process a manga file/folder into a video.
        
        Args:
            input_path: Path to ZIP, PDF, or folder with manga pages
            output_path: Path for output video (auto-generated if None)
            title: Optional title for the video
            
        Returns:
            PipelineResult with video path and metadata
        """
        start_time = datetime.now()
        stage_timings = {}
        
        input_path = Path(input_path)
        
        # Generate output path if not provided
        if output_path is None:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"manga_video_{timestamp}.mp4"
        
        output_path = Path(output_path)
        
        if title is None:
            title = input_path.stem
        
        try:
            # Stage 1: Load Input
            self._logger.start_stage("Input Loading")
            input_result = self.input_handler.load(input_path)
            stage_timings["input_loading"] = self._get_elapsed(start_time)
            self._logger.end_stage("Input Loading")
            logger.info(f"Loaded {input_result.total_pages} pages")
            
            # Stage 2: Detect Panels
            self._logger.start_stage("Panel Detection")
            all_panels = []
            global_idx = 0
            for page in input_result.pages:
                detection_result = self.panel_detector.detect(page.image, page.index)
                for panel in detection_result.panels:
                    # Assign globally unique index
                    panel.index = global_idx
                    global_idx += 1
                    all_panels.append(panel)
            stage_timings["panel_detection"] = self._get_elapsed(start_time)
            self._logger.end_stage("Panel Detection")
            logger.info(f"Detected {len(all_panels)} panels")
            
            # Auto-split warning: if chapter has way too many panels for a single video,
            # flag it in metadata. The caller (app/batch) can use this to suggest batch mode.
            auto_split_suggested = False
            max_panels_for_video = self.duration_controller.calculate_max_panels()
            if len(all_panels) > max_panels_for_video * 1.5:
                auto_split_suggested = True
                logger.warning(
                    f"Large chapter detected: {len(all_panels)} panels (max recommended: {max_panels_for_video}). "
                    f"Consider using batch mode for better coverage."
                )
            
            # Stage 3: Visual-first Panel Selection
            # Select panels BEFORE OCR to avoid running OCR on all panels
            self._logger.start_stage("Panel Selection")
            
            # Auto-scale panel count to content size:
            # - Small chapters (≤15 panels): use most of them
            # - Medium chapters (16-40): scale to fit ~2 min video
            # - Large chapters (40+): cap at max_panels, AI picks best
            detected_count = len(all_panels)
            max_for_duration = self.duration_controller.calculate_max_panels()
            effective_max = min(
                max(detected_count, 6),  # At least use what we have if under limit
                self.config.max_panels,
                max_for_duration,
            )
            # For small chapters, keep almost everything
            if detected_count <= 15:
                effective_max = min(detected_count, max_for_duration)
            logger.info(f"Auto-scaled panel limit: {effective_max} (detected={detected_count}, config_max={self.config.max_panels}, duration_max={max_for_duration})")
            
            # Create minimal OCR results for scoring (text-independent)
            from interfaces.base_ocr_engine import OCRResult
            placeholder_ocr = [
                OCRResult(panel_index=p.index, text_boxes=[], full_text="", confidence=0.0)
                for p in all_panels
            ]
            
            selection_result = self.panel_selector.select(
                panels=all_panels,
                ocr_results=placeholder_ocr,
                max_panels=effective_max,
                strategy=self.config.panel_selection_mode
            )
            
            selected_panels = [
                panel for panel in all_panels
                if panel.index in set(selection_result.selected_indices)
            ]
            
            # Preserve panel importance scores for dynamic pacing in video generation
            # (higher score = more screen time for dramatic/important panels)
            score_lookup = {ps.panel_index: ps.total_score for ps in selection_result.panel_scores}
            selected_panel_scores = [score_lookup.get(p.index, 0.5) for p in selected_panels]
            
            stage_timings["panel_selection"] = self._get_elapsed(start_time)
            self._logger.end_stage("Panel Selection")
            logger.info(f"Selected {len(selected_panels)} panels from {len(all_panels)} total")
            
            # Stage 4: OCR (only on selected panels)
            self._logger.start_stage("OCR")
            if self._use_vision_ocr:
                # Gemini Vision handles text extraction — skip EasyOCR entirely
                # This saves 1-2GB RAM and eliminates the slowest pipeline stage
                from interfaces.base_ocr_engine import OCRResult as _OCRResult
                ocr_results = [
                    _OCRResult(panel_index=p.index, text_boxes=[], full_text="", confidence=0.0)
                    for p in selected_panels
                ]
                logger.info("OCR skipped — Gemini Vision will read panel text directly")
            else:
                ocr_results = []
                total_panels = len(selected_panels)
                for i, panel in enumerate(selected_panels):
                    self._logger.progress("OCR", i + 1, total_panels)
                    logger.info(f"OCR processing panel {i+1}/{total_panels}")
                    ocr_result = self.ocr_engine.extract_text(panel.image, panel.index)
                    ocr_results.append(ocr_result)
            stage_timings["ocr"] = self._get_elapsed(start_time)
            self._logger.end_stage("OCR")
            
            ocr_texts = [ocr.full_text for ocr in ocr_results]
            non_empty = sum(1 for t in ocr_texts if t.strip())
            logger.info(f"OCR text from {non_empty}/{len(selected_panels)} panels" + (" (vision mode)" if self._use_vision_ocr else ""))
            
            # Stage 5: Story Intelligence
            self._logger.start_stage("Story Analysis")
            panel_texts = ocr_texts
            panel_indices = [ocr.panel_index for ocr in ocr_results]
            
            story_input = StoryIntelligenceInput(
                panel_texts=panel_texts,
                panel_indices=panel_indices,
                max_duration_seconds=self.config.max_video_duration,
                narration_style=self.config.narration_style,
                metadata={'chapter_title': title},
            )
            
            # Pass panel images for Gemini Vision analysis
            panel_images = [panel.image for panel in selected_panels]
            story_output = self.story_intelligence.analyze(story_input, panel_images=panel_images)
            stage_timings["story_analysis"] = self._get_elapsed(start_time)
            self._logger.end_stage("Story Analysis")
            logger.info(f"Generated script with {len(story_output.summary_script.split())} words")
            
            # Generate AI intro image (non-blocking — falls back to best panel)
            # Priority: 1) SD txt2img (local, fast, no rate limits)
            #           2) Gemini image gen (cloud, may hit quota)
            #           3) Gemini-selected best panel
            #           4) Largest panel heuristic
            intro_image = None
            char_details = story_output.metadata.get('character_details', [])
            manga_title_str = getattr(self.config, 'anime_title', '') or title

            # ── Try SD txt2img first (local GPU) ──
            try:
                from modules.anime_generator import AnimeGenerator as _AG
                sd_gen = _AG()
                sd_prompt = self._build_sd_intro_prompt(
                    char_details, story_output.characters,
                    story_output.tone, manga_title_str,
                )
                intro_image = sd_gen.generate_intro_thumbnail(
                    prompt=sd_prompt,
                    width=512,
                    height=768,
                    steps=30,
                    guidance_scale=8.0,
                )
                if intro_image is not None:
                    logger.info("SD txt2img intro image ready")
            except Exception as e:
                logger.warning(f"SD txt2img intro skipped: {e}")

            # ── Fallback: Gemini image gen ──
            if intro_image is None:
                try:
                    intro_image = self.story_intelligence.generate_intro_image(
                        summary=story_output.summary_script,
                        tone=story_output.tone,
                        characters=story_output.characters,
                        manga_title=manga_title_str,
                        character_details=char_details,
                    )
                    if intro_image is not None:
                        logger.info("Gemini AI intro image ready")
                except Exception as e:
                    logger.warning(f"Gemini intro image skipped: {e}")
            
            # ── Fallback: Gemini-selected best panel ──
            if intro_image is None:
                intro_panel_idx = story_output.metadata.get('intro_panel_index')
                if intro_panel_idx is not None:
                    for p in selected_panels:
                        if p.index == intro_panel_idx:
                            intro_image = p.image.copy()
                            logger.info(f"Using Gemini-selected panel {intro_panel_idx} as intro background")
                            break

            # ── Last resort: largest panel ──
            if intro_image is None and selected_panels:
                best_panel = max(
                    selected_panels,
                    key=lambda p: p.image.shape[0] * p.image.shape[1],
                )
                intro_image = best_panel.image.copy()
                logger.info(f"Using largest panel (idx {best_panel.index}) as intro background (heuristic fallback)")
            
            # Stage 6: Duration Planning
            self._logger.start_stage("Duration Planning")
            duration_plan = self.duration_controller.plan_duration(
                narration_text=story_output.summary_script,
                panel_count=len(selected_panels),
                transition_duration=self.config.transition_duration
            )
            
            # Adjust script if needed
            adjusted_script = self.duration_controller.adjust_script_length(
                story_output.summary_script,
                duration_plan.narration_word_count
            )
            stage_timings["duration_planning"] = self._get_elapsed(start_time)
            self._logger.end_stage("Duration Planning")
            
            for adjustment in duration_plan.adjustments_made:
                logger.info(f"Duration adjustment: {adjustment}")
            
            # Apply panel reduction from duration planning
            if duration_plan.panel_count < len(selected_panels):
                import numpy as np
                indices = np.linspace(0, len(selected_panels) - 1, duration_plan.panel_count, dtype=int)
                selected_panels = [selected_panels[i] for i in indices]
                selected_panel_scores = [selected_panel_scores[i] for i in indices]
                logger.info(f"Trimmed to {len(selected_panels)} panels after duration planning")
            
            # ── Apply Gemini's panel selection ──
            # Gemini returned selected_panel_indices: the subset of panels the
            # narration actually describes.  Re-order selected_panels to match
            # so the video shows exactly what the narration talks about.
            gemini_indices = story_output.selected_panels  # absolute panel indices
            if gemini_indices and len(gemini_indices) >= 4:
                # Build lookup: panel.index → position in selected_panels
                idx_to_pos = {p.index: i for i, p in enumerate(selected_panels)}
                matched_positions = [idx_to_pos[gi] for gi in gemini_indices if gi in idx_to_pos]
                
                if len(matched_positions) >= 4:
                    selected_panels = [selected_panels[i] for i in matched_positions]
                    selected_panel_scores = [selected_panel_scores[i] for i in matched_positions]
                    logger.info(f"Reordered to {len(selected_panels)} panels matching Gemini's narration selection")
                else:
                    logger.info(f"Gemini panel indices didn't match enough panels ({len(matched_positions)}/{len(gemini_indices)}), keeping original order")
            
            # Stage 7: Colorization (optional)
            if self.config.enable_colorization:
                self._logger.start_stage("Colorization")
                colorized_panels = []
                for panel in selected_panels:
                    color_result = self.colorizer.colorize(panel.image, panel.index)
                    colorized_panels.append(color_result.colorized_image)
                stage_timings["colorization"] = self._get_elapsed(start_time)
                self._logger.end_stage("Colorization")
            else:
                colorized_panels = [panel.image for panel in selected_panels]
            
            # Stage 8: Narration
            self._logger.start_stage("Narration")
            audio_path = output_path.with_suffix('.wav')
            
            narration_result = self.narrator.generate_to_file(
                script=adjusted_script,
                output_path=audio_path
            )
            stage_timings["narration"] = self._get_elapsed(start_time)
            self._logger.end_stage("Narration")
            
            # Check for TTS fallback and log it
            narration_warnings = []
            narration_meta = narration_result.metadata or {}
            if narration_meta.get('tts_fallback'):
                reason = narration_meta.get('tts_fallback_reason', 'unknown error')
                original = narration_meta.get('original_engine', 'elevenlabs')
                actual = narration_meta.get('model', 'edge-tts')
                warning_msg = f"{original.title()} failed ({reason}). Fell back to {actual}."
                narration_warnings.append(warning_msg)
                logger.warning(warning_msg)
            
            logger.info(f"Generated narration: {narration_result.duration_seconds:.1f}s (engine: {narration_meta.get('model', 'unknown')})")
            
            # Stage 9: Video Generation
            self._logger.start_stage("Video Generation")
            video_config = VideoConfig(
                width=self.config.video_width,
                height=self.config.video_height,
                fps=self.config.video_fps,
                max_duration=self.config.max_video_duration,
                transition_duration=self.config.transition_duration,
                include_subtitles=self.config.enable_subtitles,
                subtitle_style={
                    'fontsize': self.config.subtitle_font_size,
                    'color': self.config.subtitle_color
                }
            )
            
            video_result = self.video_generator.generate(
                panels=colorized_panels,
                audio_path=audio_path if audio_path.exists() else None,
                output_path=output_path,
                config=video_config,
                panel_scores=selected_panel_scores,
                intro_hook=getattr(story_output, 'intro_hook', ''),
                manga_title=getattr(self.config, 'anime_title', '') or title,
                intro_image=intro_image,
            )
            stage_timings["video_generation"] = self._get_elapsed(start_time)
            self._logger.end_stage("Video Generation")
            
            # Stage 10: Add Subtitles (optional)
            if self.config.enable_subtitles:
                self._logger.start_stage("Subtitles")
                
                # Get intro offset so subtitles start after the intro card
                intro_offset = video_result.metadata.get('intro_duration', 0.0)
                
                # Prefer word-level timings from TTS engine (Edge-TTS provides these)
                # for perfectly synced subtitles, fall back to estimated timings
                tts_segments = narration_result.segments if hasattr(narration_result, 'segments') else []
                if tts_segments and isinstance(tts_segments, list) and len(tts_segments) > 0:
                    # Use TTS word-level timings for precise sync
                    subtitles = self.video_generator.generate_subtitles_from_word_timings(
                        word_timings=tts_segments,
                        audio_duration=video_result.duration,
                        intro_offset=intro_offset,
                    )
                    logger.info(f"Using TTS word-level timings for subtitle sync ({len(tts_segments)} words)")
                else:
                    subtitles = self.video_generator.generate_subtitles_from_script(
                        script=adjusted_script,
                        audio_duration=video_result.duration,
                        intro_offset=intro_offset,
                    )
                
                subtitled_path = output_path.with_stem(output_path.stem + "_subtitled")
                final_video_path = self.video_generator.add_subtitles(
                    video_path=output_path,
                    subtitles=subtitles,
                    output_path=subtitled_path,
                    config=video_config
                )
                
                # Replace original with subtitled version
                if final_video_path != output_path and final_video_path.exists():
                    output_path.unlink(missing_ok=True)
                    final_video_path.rename(output_path)
                
                stage_timings["subtitles"] = self._get_elapsed(start_time)
                self._logger.end_stage("Subtitles")
            
            # Stage 11: Anime Generation (optional) — BEFORE audio cleanup
            anime_path = None
            if getattr(self.config, 'enable_anime_gen', False):
                self._logger.start_stage("Anime Generation")
                try:
                    anime_path = self._generate_anime_clip(
                        colorized_panels, output_path, story_output,
                        audio_path=audio_path,
                        narration_duration=narration_result.duration_seconds,
                    )
                    stage_timings["anime_generation"] = self._get_elapsed(start_time)
                    self._logger.end_stage("Anime Generation")
                except Exception as e:
                    logger.warning(f"Anime generation failed (non-fatal): {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    self._logger.end_stage("Anime Generation", success=False)
            
            # Clean up temporary audio file (AFTER anime generation)
            audio_path.unlink(missing_ok=True)
            
            # Stage 12: Upload (optional)
            if self.config.auto_upload:
                self._logger.start_stage("Upload")
                upload_config = UploadConfig(
                    title=title,
                    description=f"Generated from manga: {input_path.name}",
                    tags=["manga", "anime", "short"],
                    metadata={'destination': str(output_path)}
                )
                
                platform = Platform(self.config.upload_platform)
                upload_result = self.uploader.upload(
                    video_path=output_path,
                    platform=platform,
                    config=upload_config
                )
                stage_timings["upload"] = self._get_elapsed(start_time)
                self._logger.end_stage("Upload", upload_result.success)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Pipeline completed in {total_time:.1f}s")
            logger.info(f"Output video: {output_path}")
            
            return PipelineResult(
                success=True,
                video_path=output_path,
                duration=video_result.duration,
                panel_count=len(selected_panels),
                narration_script=adjusted_script,
                stage_timings=stage_timings,
                metadata={
                    'input_path': str(input_path),
                    'total_pages': input_result.total_pages,
                    'total_panels_detected': len(all_panels),
                    'panels_selected': len(selected_panels),
                    'story_tone': story_output.tone,
                    'characters': story_output.characters,
                    'key_events': story_output.key_events,
                    'total_processing_time': total_time,
                    'tts_engine': narration_meta.get('model', 'unknown'),
                    'tts_voice': narration_meta.get('voice', 'unknown'),
                    'tts_fallback': narration_meta.get('tts_fallback', False),
                    'auto_split_suggested': auto_split_suggested,
                    'warnings': narration_warnings,
                    'anime_clip_path': str(anime_path) if anime_path else None,
                }
            )
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return PipelineResult(
                success=False,
                error_message=str(e),
                stage_timings=stage_timings
            )
    
    def _get_elapsed(self, start_time: datetime) -> float:
        """Get elapsed time since start."""
        return (datetime.now() - start_time).total_seconds()

    @staticmethod
    def _build_sd_intro_prompt(
        char_details: list,
        characters: list,
        tone: str,
        manga_title: str,
    ) -> str:
        """Build a Stable Diffusion txt2img prompt for the intro thumbnail.

        Prioritises female characters for eye-catching thumbnails.
        Uses SD1.5-optimised tag-style prompting (not natural language).
        """
        # Quality tags that work well with Anything V5
        quality = (
            "masterpiece, best quality, extremely detailed, "
            "beautiful detailed eyes, detailed face, "
            "anime style, vivid colors, dramatic lighting, "
            "cinematic composition, portrait, upper body"
        )

        focal_tags = ""

        # Priority 1: Female character with details
        female_chars = [c for c in char_details if c.get("gender", "").lower() == "female"]
        if female_chars:
            c = female_chars[0]
            features = c.get("features", "")
            name = c.get("name", "anime girl")
            focal_tags = (
                f"1girl, solo, {name}, beautiful anime girl, "
                f"gorgeous face, sparkling detailed eyes, glossy lips, "
                f"flowing hair, attractive, alluring pose, "
                f"form-fitting outfit, {features}, "
                f"looking at viewer, captivating expression"
            )
        # Priority 2: Male protagonist
        elif char_details:
            protag = [c for c in char_details if c.get("role", "").lower() == "protagonist"]
            c = protag[0] if protag else char_details[0]
            features = c.get("features", "")
            name = c.get("name", "anime character")
            gender = c.get("gender", "").lower()
            tag = "1boy" if gender == "male" else "1girl"
            focal_tags = (
                f"{tag}, solo, {name}, {features}, "
                f"powerful pose, dramatic expression, "
                f"cool, badass, intense eyes"
            )
        # Priority 3: Character name only
        elif characters:
            focal_tags = f"1girl, solo, {characters[0]}, beautiful anime girl, detailed eyes"
        # Priority 4: Generic attractive anime girl
        else:
            focal_tags = (
                "1girl, solo, beautiful anime girl, long hair, "
                "gorgeous face, detailed eyes, alluring, "
                "looking at viewer, captivating"
            )

        # Tone-specific atmosphere tags
        tone_tags = {
            "dramatic": "dramatic lighting, intense atmosphere, dark background",
            "hype": "dynamic angle, energy particles, glowing effects, epic",
            "emotional": "soft lighting, emotional, gentle atmosphere, bokeh",
            "intense": "battle aura, intense expression, action pose, particles",
            "somber": "melancholic, rain, muted warm tones, reflective",
            "comedic": "cheerful, bright colors, playful expression, sparkles",
            "suspenseful": "mysterious, shadows, tension, dark atmosphere",
            "triumphant": "golden light, victorious, shining, heroic pose",
        }.get(tone, "dramatic lighting, atmospheric")

        prompt = f"{quality}, {focal_tags}, {tone_tags}"
        return prompt

    def _generate_anime_clip(
        self,
        panels: list,
        main_video_path: Path,
        story_output,
        audio_path: Optional[Path] = None,
        narration_duration: float = 0.0,
    ) -> Optional[Path]:
        """Generate an anime-style clip from the chapter's panels.

        Uses Stable Diffusion 1.5 img2img with ControlNet for composition
        preservation.  Produces a separate video file alongside the main output.

        Returns:
            Path to the anime clip, or None on failure.
        """
        from modules.anime_generator import AnimeGenerator
        from interfaces.base_anime_generator import AnimeConfig, AnimeStyle, AnimationMode

        style_map = {
            "modern_anime": AnimeStyle.MODERN_ANIME,
            "classic_anime": AnimeStyle.CLASSIC_ANIME,
            "ghibli": AnimeStyle.GHIBLI,
            "shonen": AnimeStyle.SHONEN,
            "chibi": AnimeStyle.CHIBI,
            "vibrant": AnimeStyle.VIBRANT,
        }


        animation_mode_map = {
            "static": AnimationMode.STATIC,
            "ken_burns": AnimationMode.KEN_BURNS,
            "interpolated": AnimationMode.INTERPOLATED,
            "animated": AnimationMode.ANIMATED,
            "puppet": AnimationMode.PUPPET,
        }

        style = style_map.get(
            getattr(self.config, 'anime_style', 'modern_anime'),
            AnimeStyle.MODERN_ANIME,
        )

        anime_config = AnimeConfig(
            style=style,
            strength=getattr(self.config, 'anime_strength', 0.75),
            num_inference_steps=getattr(self.config, 'anime_steps', 25),
            use_controlnet=getattr(self.config, 'anime_controlnet', True),
            fps=getattr(self.config, 'anime_fps', 24),
            animation_mode=animation_mode_map.get(
                getattr(self.config, 'anime_animation_mode', 'ken_burns'),
                AnimationMode.KEN_BURNS,
            ),
            width=self.config.video_width,    # Match main video (1080)
            height=self.config.video_height,  # Match main video (1920)
        )

        # Use key events as per-panel scene descriptions for better prompts
        scene_descs = story_output.key_events if story_output.key_events else None

        # Select a subset of panels for the anime clip (max 10)
        import numpy as np
        max_anime_panels = min(len(panels), 10)
        if len(panels) > max_anime_panels:
            indices = np.linspace(0, len(panels) - 1, max_anime_panels, dtype=int)
            anime_panels = [panels[i] for i in indices]
        else:
            anime_panels = list(panels)

        anime_output = main_video_path.with_stem(main_video_path.stem + "_anime")
        anime_output = anime_output.with_suffix('.mp4')

        generator = AnimeGenerator()

        def _anime_progress(phase, current, total):
            """Report anime sub-progress so frontend moves beyond 90%."""
            if phase == "stylize":
                # Stylize = 90-96%
                pct = 90 + int((current / max(total, 1)) * 6)
            elif phase == "assemble":
                # Assemble = 97-98%
                pct = 97 + current
            else:
                pct = 99
            self._logger.info(f"Anime {phase}: {current}/{total}")
            # Try to update job progress directly if the logger has it
            try:
                self._logger.progress("Anime Generation", current, total)
            except Exception:
                pass

        result = generator.generate(
            panels=anime_panels,
            output_path=anime_output,
            config=anime_config,
            scene_descriptions=scene_descs,
            audio_path=audio_path if audio_path and audio_path.exists() else None,
            narration_duration=narration_duration,
            progress_callback=_anime_progress,
        )

        if result.video_path and result.video_path.exists():
            logger.info(f"Anime clip: {result.video_path} ({result.duration:.1f}s)")
            return result.video_path

        return None
    
    def set_config(self, config: Config) -> None:
        """Update configuration and reinitialize affected modules."""
        self.config = config
        self._init_modules()
        self.duration_controller.set_target_duration(config.max_video_duration)


def create_pipeline(config_path: Optional[Path] = None) -> MangaVideoPipeline:
    """
    Factory function to create a configured pipeline.
    
    Args:
        config_path: Path to config.json (uses default if None)
        
    Returns:
        Configured MangaVideoPipeline instance
    """
    config = load_config(config_path)
    return MangaVideoPipeline(config)
