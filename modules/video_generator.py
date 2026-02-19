"""
Video Generator Module

Creates cinematic videos from manga panels with high-quality transitions,
Ken Burns effects, colored animated captions, and vignette overlays.
"""

import logging
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from interfaces.base_video_generator import (
    BaseVideoGenerator,
    VideoConfig,
    VideoResult,
    PanelTiming,
    SubtitleSegment,
    TransitionType,
    ZoomEffect
)

logger = logging.getLogger(__name__)


def _ease_in_out_cubic(t: float) -> float:
    """Smooth cubic easing for transitions."""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2


def _ease_in_out_quad(t: float) -> float:
    """Smooth quadratic easing."""
    if t < 0.5:
        return 2 * t * t
    return 1 - pow(-2 * t + 2, 2) / 2


class VideoGenerator(BaseVideoGenerator):
    """
    High-quality video generation with cinematic effects.
    
    Features:
    - Smooth eased transitions (fade, crossfade, slide, zoom, wipe, dissolve)
    - Ken Burns pan/zoom with cubic easing
    - Colored animated captions with outline, shadow, rounded background
    - Cinematic vignette overlay
    - High quality H.264 encoding
    """
    
    def __init__(self):
        self._ffmpeg_available = self._check_ffmpeg()
        self._vignette_cache: dict = {}  # Cache vignette masks by (w, h)
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("FFmpeg not found. Some features may be limited.")
            return False
    
    def generate(
        self,
        panels: List[np.ndarray],
        audio_path: Optional[Path],
        output_path: Path,
        config: Optional[VideoConfig] = None,
        panel_scores: Optional[List[float]] = None,
        intro_hook: str = "",
        manga_title: str = "",
        intro_image: Optional[np.ndarray] = None,
    ) -> VideoResult:
        """Generate video from panels and audio."""
        config = config or VideoConfig()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating video with {len(panels)} panels")
        
        # Intro card duration (seconds) — only if we have a hook
        intro_duration = 2.5 if intro_hook else 0.0
        
        # Determine audio duration
        if audio_path and audio_path.exists():
            audio_duration = self._get_audio_duration(audio_path)
        else:
            audio_duration = config.max_duration
            audio_path = None
        
        # Calculate timings with dynamic pacing
        timings = self.calculate_timings(len(panels), audio_duration, config, panel_scores=panel_scores)
        
        # Offset all timings by intro duration so panels start after the intro card
        if intro_duration > 0:
            for timing in timings:
                timing.start_time += intro_duration
                timing.end_time += intro_duration
        
        # Generate video frames
        temp_video_path = output_path.with_suffix('.temp.mp4')
        
        self._generate_frames_to_video(
            panels, timings, config, temp_video_path,
            intro_hook=intro_hook, manga_title=manga_title,
            intro_duration=intro_duration,
            intro_image=intro_image,
        )
        
        # Add audio if available
        if audio_path and self._ffmpeg_available:
            self._add_audio_to_video(temp_video_path, audio_path, output_path, delay_ms=int(intro_duration * 1000))
            temp_video_path.unlink(missing_ok=True)
        else:
            temp_video_path.rename(output_path)
        
        # Get file size
        file_size = output_path.stat().st_size if output_path.exists() else 0
        
        # Calculate actual duration (include intro)
        actual_duration = (timings[-1].end_time if timings else 0)
        
        logger.info(f"Video generated: {output_path}")
        
        return VideoResult(
            output_path=output_path,
            duration=actual_duration,
            width=config.width,
            height=config.height,
            fps=config.fps,
            panel_count=len(panels),
            has_audio=audio_path is not None,
            has_subtitles=False,
            file_size_bytes=file_size,
            metadata={'intro_duration': intro_duration},
        )
    
    def calculate_timings(
        self,
        panel_count: int,
        audio_duration: float,
        config: VideoConfig,
        panel_scores: Optional[List[float]] = None,
    ) -> List[PanelTiming]:
        """Calculate timing with dynamic pacing based on panel importance.
        
        High-score panels (dramatic, character-heavy) get more screen time.
        Low-score panels (transitions, establishing shots) get rapid cuts.
        This creates the fast-slow-fast rhythm that keeps viewers engaged.
        """
        if panel_count == 0:
            return []
        
        # Ensure minimum duration
        target_duration = max(audio_duration, config.max_duration * 0.6, 20.0)
        
        # Calculate available time for panels
        transition_dur = config.transition_duration
        total_transition_time = transition_dur * (panel_count - 1)
        available_time = target_duration - total_transition_time
        
        # ── Dynamic pacing: vary time per panel based on importance ──
        avg_time = available_time / panel_count if panel_count > 0 else 3.0
        min_panel_dur = 2.0   # Rapid cuts for action/transition panels
        # Scale max based on time budget — generous budgets allow more lingering
        max_panel_dur = max(5.5, min(avg_time * 1.6, 8.0))
        
        if panel_scores and len(panel_scores) == panel_count:
            # Amplify score differences with power curve for more dramatic pacing
            scores = [max(s, 0.05) ** 1.8 for s in panel_scores]
            total_score = sum(scores)
            weights = [s / total_score for s in scores]
            
            # Distribute available time proportionally to importance
            raw_durations = [w * available_time for w in weights]
            
            # Iterative clamp-and-redistribute to exactly fill available_time
            durations = list(raw_durations)
            for _ in range(5):  # converges in 2-3 iterations
                clamped = []
                frozen_time = 0.0
                free_indices = []
                free_weight_sum = 0.0
                
                for j, d in enumerate(durations):
                    if d <= min_panel_dur:
                        clamped.append(min_panel_dur)
                        frozen_time += min_panel_dur
                    elif d >= max_panel_dur:
                        clamped.append(max_panel_dur)
                        frozen_time += max_panel_dur
                    else:
                        clamped.append(d)
                        free_indices.append(j)
                        free_weight_sum += weights[j]
                
                if not free_indices:
                    durations = clamped
                    break
                
                # Redistribute remaining time to unclamped panels
                remaining = available_time - frozen_time
                for j in free_indices:
                    clamped[j] = (weights[j] / free_weight_sum) * remaining
                
                durations = clamped
            
            # Final pass: scale uniformly to fill available_time exactly
            # This preserves relative differences while matching audio length
            dur_total = sum(durations)
            if dur_total > 0 and abs(dur_total - available_time) > 0.5:
                scale = available_time / dur_total
                durations = [max(min_panel_dur, d * scale) for d in durations]
        else:
            # Even distribution when no scores available
            even_dur = max(min(available_time / panel_count, max_panel_dur), min_panel_dur)
            durations = [even_dur] * panel_count
        
        # ── Pacing rhythm: first and last panels get extra time ──
        # (Hook panel and cliffhanger panel are most important for retention)
        if panel_count >= 4:
            durations[0] = max(durations[0], 3.5)   # Opening hook needs to land
            durations[-1] = max(durations[-1], 3.5)  # Ending beat needs weight
        
        # Richer transition cycle — vary to keep visual interest
        transitions = [
            TransitionType.CROSSFADE,
            TransitionType.SLIDE_LEFT,
            TransitionType.FLASH_WHITE,
            TransitionType.ZOOM_IN,
            TransitionType.DISSOLVE,
            TransitionType.FADE,
            TransitionType.SLIDE_RIGHT,
            TransitionType.ZOOM_OUT,
        ]
        
        # Richer zoom cycle
        zooms = [
            ZoomEffect.KEN_BURNS_IN,
            ZoomEffect.PAN_LEFT,
            ZoomEffect.KEN_BURNS_OUT,
            ZoomEffect.PAN_RIGHT,
        ]
        
        timings = []
        current_time = 0.0
        
        for i in range(panel_count):
            dur = durations[i]
            timing = PanelTiming(
                panel_index=i,
                start_time=current_time,
                end_time=current_time + dur,
                duration=dur,
                transition_in=transitions[i % len(transitions)],
                transition_out=TransitionType.NONE,
                zoom_effect=zooms[i % len(zooms)],
                transition_duration=transition_dur
            )
            timings.append(timing)
            current_time += dur
        
        return timings
    
    # ───────────────── Transitions ─────────────────
    
    def apply_transition(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        transition_type: TransitionType,
        progress: float
    ) -> np.ndarray:
        """Apply transition between two frames with easing."""
        t = np.clip(progress, 0.0, 1.0)
        t = _ease_in_out_cubic(t)  # smooth easing
        
        if transition_type == TransitionType.NONE:
            return frame2 if t >= 0.5 else frame1
        
        elif transition_type == TransitionType.FADE:
            return self._fade_transition(frame1, frame2, t)
        
        elif transition_type == TransitionType.CROSSFADE:
            return self._crossfade_transition(frame1, frame2, t)
        
        elif transition_type == TransitionType.SLIDE_LEFT:
            return self._slide_transition(frame1, frame2, t, direction='left')
        
        elif transition_type == TransitionType.SLIDE_RIGHT:
            return self._slide_transition(frame1, frame2, t, direction='right')
        
        elif transition_type == TransitionType.ZOOM_IN:
            return self._zoom_transition(frame1, frame2, t, zoom_in=True)
        
        elif transition_type == TransitionType.ZOOM_OUT:
            return self._zoom_transition(frame1, frame2, t, zoom_in=False)

        elif transition_type == TransitionType.FLASH_WHITE:
            return self._flash_white_transition(frame1, frame2, t)

        elif transition_type == TransitionType.DISSOLVE:
            return self._dissolve_transition(frame1, frame2, t)

        else:
            return self._crossfade_transition(frame1, frame2, t)
    
    def _fade_transition(self, f1: np.ndarray, f2: np.ndarray, t: float) -> np.ndarray:
        """Fade through black with smooth easing (integer-only, no float conversion)."""
        if t < 0.5:
            alpha = 1.0 - t * 2
            return cv2.multiply(f1, (alpha, alpha, alpha, 0), dtype=cv2.CV_8U)
        else:
            alpha = (t - 0.5) * 2
            return cv2.multiply(f2, (alpha, alpha, alpha, 0), dtype=cv2.CV_8U)
    
    def _crossfade_transition(self, f1: np.ndarray, f2: np.ndarray, t: float) -> np.ndarray:
        """Smooth crossfade blend."""
        return cv2.addWeighted(f1, 1.0 - t, f2, t, 0)
    
    def _slide_transition(self, f1: np.ndarray, f2: np.ndarray, t: float, direction: str) -> np.ndarray:
        """Slide transition with seam blur for smoothness."""
        h, w = f1.shape[:2]
        result = np.zeros_like(f1)
        offset = int(w * t)
        
        if direction == 'left':
            if w - offset > 0:
                result[:, :w - offset] = f1[:, offset:]
            if offset > 0:
                result[:, w - offset:] = f2[:, :offset]
        else:
            if w - offset > 0:
                result[:, offset:] = f1[:, :w - offset]
            if offset > 0:
                result[:, :offset] = f2[:, w - offset:]
        
        # Add subtle blur at the seam for smoothness
        seam_x = w - offset if direction == 'left' else offset
        blur_width = max(4, int(w * 0.02))
        x1 = max(0, seam_x - blur_width)
        x2 = min(w, seam_x + blur_width)
        if x2 > x1:
            result[:, x1:x2] = cv2.GaussianBlur(result[:, x1:x2], (0, 0), sigmaX=3)
        
        return result
    
    def _zoom_transition(self, f1: np.ndarray, f2: np.ndarray, t: float, zoom_in: bool) -> np.ndarray:
        """Zoom transition between frames."""
        h, w = f1.shape[:2]
        
        if t < 0.5:
            scale = 1.0 + (t * 0.4 if zoom_in else -t * 0.2)
            scale = max(scale, 0.5)
            frame = self._apply_zoom(f1, scale, w, h)
        else:
            scale = 1.0 + ((1.0 - t) * 0.4 if zoom_in else -(1.0 - t) * 0.2)
            scale = max(scale, 0.5)
            frame = self._apply_zoom(f2, scale, w, h)
        
        return frame

    def _flash_white_transition(self, f1: np.ndarray, f2: np.ndarray, t: float) -> np.ndarray:
        """Flash-to-white impact transition – commonly used in anime/manga edits.
        Goes: current → bright white flash → next panel."""
        white = np.full_like(f1, 255)
        if t < 0.35:
            # Brighten outgoing frame toward white
            alpha = t / 0.35
            return cv2.addWeighted(f1, 1.0 - alpha, white, alpha, 0)
        elif t < 0.55:
            # Hold white for impact
            return white
        else:
            # Fade from white into incoming frame
            alpha = (t - 0.55) / 0.45
            return cv2.addWeighted(white, 1.0 - alpha, f2, alpha, 0)

    def _dissolve_transition(self, f1: np.ndarray, f2: np.ndarray, t: float) -> np.ndarray:
        """Pixel-dissolve transition using noise threshold."""
        h, w = f1.shape[:2]
        # Generate a fixed noise pattern (seeded for consistency between frames
        # at same resolution, but we use a simple approach here)
        if not hasattr(self, '_dissolve_noise') or self._dissolve_noise.shape[:2] != (h, w):
            rng = np.random.RandomState(42)
            self._dissolve_noise = rng.rand(h, w).astype(np.float32)
        threshold = t
        mask = (self._dissolve_noise < threshold).astype(np.uint8)
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        return np.where(mask_3ch, f2, f1)
    
    # ───────────────── Zoom / Pan Effects ─────────────────
    
    def apply_zoom_effect(
        self,
        frame: np.ndarray,
        zoom_effect: ZoomEffect,
        progress: float,
        config: VideoConfig
    ) -> np.ndarray:
        """Apply smooth Ken Burns zoom/pan effect."""
        h, w = config.height, config.width
        p = _ease_in_out_quad(progress)  # smooth easing on zoom
        
        if zoom_effect == ZoomEffect.NONE:
            return frame
        
        elif zoom_effect == ZoomEffect.KEN_BURNS_IN:
            scale = 1.0 + p * 0.15
            return self._apply_zoom(frame, scale, w, h)
        
        elif zoom_effect == ZoomEffect.KEN_BURNS_OUT:
            scale = 1.15 - p * 0.15
            return self._apply_zoom(frame, scale, w, h)
        
        elif zoom_effect == ZoomEffect.PAN_LEFT:
            return self._apply_pan(frame, p, direction='left', config=config)
        
        elif zoom_effect == ZoomEffect.PAN_RIGHT:
            return self._apply_pan(frame, p, direction='right', config=config)
        
        else:
            return frame
    
    def _apply_zoom(self, frame: np.ndarray, scale: float, target_w: int, target_h: int) -> np.ndarray:
        """Apply centered zoom."""
        h, w = frame.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        x_start = max(0, (new_w - target_w) // 2)
        y_start = max(0, (new_h - target_h) // 2)
        
        cropped = scaled[y_start:y_start + target_h, x_start:x_start + target_w]
        
        if cropped.shape[0] < target_h or cropped.shape[1] < target_w:
            result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            rh, rw = min(cropped.shape[0], target_h), min(cropped.shape[1], target_w)
            result[:rh, :rw] = cropped[:rh, :rw]
            return result
        
        return cropped
    
    def _apply_pan(self, frame: np.ndarray, progress: float, direction: str, config: VideoConfig) -> np.ndarray:
        """Apply smooth pan effect using pre-padded frame (crop only, no resize)."""
        target_w, target_h = config.width, config.height
        h, w = frame.shape[:2]
        
        # Max offset based on available padding
        max_offset = max(0, (w - target_w) // 2)
        
        if direction == 'left':
            offset = int(max_offset * progress)
        else:
            offset = int(max_offset * (1 - progress))
        
        # Center vertically in padded frame
        y_start = max(0, (h - target_h) // 2)
        
        cropped = frame[y_start:y_start + target_h, offset:offset + target_w]
        
        if cropped.shape[0] < target_h or cropped.shape[1] < target_w:
            result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            rh = min(cropped.shape[0], target_h)
            rw = min(cropped.shape[1], target_w)
            result[:rh, :rw] = cropped[:rh, :rw]
            return result
        
        return cropped
    
    # ───────────────── Frame Overlays ─────────────────
    
    def _get_vignette_mask(self, w: int, h: int) -> np.ndarray:
        """Get or create cached vignette mask for given dimensions.
        Returns a uint8 mask (0-255) representing the vignette intensity."""
        key = (w, h)
        if key not in self._vignette_cache:
            Y, X = np.ogrid[:h, :w]
            cx, cy = w / 2, h / 2
            radius = math.sqrt(cx ** 2 + cy ** 2)
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            # Scale factor: 1.0 in center to 0.65 at extreme edges
            mask_float = 1.0 - np.clip((dist / radius - 0.4) / 0.6, 0, 1) * 0.35
            # Convert to uint8 scale (0-255) for fast integer blending
            mask_u8 = (mask_float * 255).astype(np.uint8)
            # Broadcast to 3 channels
            self._vignette_cache[key] = np.stack([mask_u8, mask_u8, mask_u8], axis=-1)
        return self._vignette_cache[key]

    def _apply_vignette_fast(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply vignette using fast integer-only path (no float conversion)."""
        # Use 16-bit integer multiply: (frame * mask + 127) / 255
        # This avoids the expensive float32 conversion entirely
        return cv2.multiply(frame, mask, scale=1.0 / 255.0, dtype=cv2.CV_8U)

    def _apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """Apply cinematic vignette overlay using cached mask."""
        h, w = frame.shape[:2]
        mask = self._get_vignette_mask(w, h)
        return cv2.multiply(frame, mask, scale=1.0 / 255.0, dtype=cv2.CV_8U)
    
    # ───────────────── Resize ─────────────────
    
    def resize_for_format(
        self,
        image: np.ndarray,
        target_width: int,
        target_height: int,
        fit_mode: str = "contain"
    ) -> np.ndarray:
        """Resize image for target video format."""
        h, w = image.shape[:2]
        
        if fit_mode == "stretch":
            return cv2.resize(image, (target_width, target_height))
        
        source_aspect = w / h
        target_aspect = target_width / target_height
        
        if fit_mode == "cover":
            if source_aspect > target_aspect:
                new_h = target_height
                new_w = int(target_height * source_aspect)
            else:
                new_w = target_width
                new_h = int(target_width / source_aspect)
            
            # Use INTER_AREA for downsample (fast + good quality), LINEAR for upsample
            interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LINEAR
            resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
            
            y_off = (new_h - target_height) // 2
            x_off = (new_w - target_width) // 2
            
            return resized[y_off:y_off + target_height, x_off:x_off + target_width]
        
        else:  # contain
            if source_aspect > target_aspect:
                new_w = target_width
                new_h = int(target_width / source_aspect)
            else:
                new_h = target_height
                new_w = int(target_height * source_aspect)
            
            interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LINEAR
            resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
            
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            y_off = (target_height - new_h) // 2
            x_off = (target_width - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            
            return canvas
    
    # ───────────────── Subtitles ─────────────────

    def generate_subtitles_from_script(
        self,
        script: str,
        audio_duration: float,
        words_per_segment: int = 3,
        intro_offset: float = 0.0,
    ) -> List[SubtitleSegment]:
        """Generate viral-style subtitle segments – 2-3 words at a time."""
        words = script.split()
        if not words:
            return []

        segments = []
        total_words = len(words)
        time_per_word = audio_duration / total_words

        # Adaptive grouping: 2-3 words per segment for punchy readability
        i = 0
        while i < total_words:
            # Use 3 words normally, 2 words for short punchy words or end
            remaining = total_words - i
            if remaining <= 2:
                chunk = remaining
            elif remaining == 4:
                chunk = 2  # split 4 into 2+2 not 3+1
            else:
                chunk = 3
            seg_words = words[i:i + chunk]
            start_time = i * time_per_word + intro_offset
            end_time = min((i + chunk) * time_per_word, audio_duration) + intro_offset
            segments.append(SubtitleSegment(
                text=" ".join(seg_words),
                start_time=start_time,
                end_time=end_time
            ))
            i += chunk

        return segments

    def generate_subtitles_from_word_timings(
        self,
        word_timings: List[Dict],
        audio_duration: float,
        words_per_segment: int = 3,
        intro_offset: float = 0.0,
    ) -> List[SubtitleSegment]:
        """Generate subtitle segments from TTS word-level timings for perfect sync.

        Args:
            word_timings: List of dicts with 'word', 'start', 'end' keys
                          (as returned by Edge-TTS WordBoundary events).
            audio_duration: Total audio duration in seconds.
            words_per_segment: Target words per segment (2-3 for viral style).
            intro_offset: Time offset to add (for intro card).

        Returns:
            List of SubtitleSegment with precise timing.
        """
        if not word_timings:
            return []

        segments: List[SubtitleSegment] = []
        total = len(word_timings)
        i = 0

        while i < total:
            remaining = total - i
            if remaining <= 2:
                chunk = remaining
            elif remaining == 4:
                chunk = 2
            else:
                chunk = words_per_segment

            group = word_timings[i:i + chunk]
            text = " ".join(w.get("word", w.get("text", "")) for w in group)
            start_time = group[0].get("start", 0.0) + intro_offset
            end_time = group[-1].get("end", start_time + 0.5) + intro_offset

            # Clamp to audio duration (plus intro offset)
            end_time = min(end_time, audio_duration + intro_offset)

            segments.append(SubtitleSegment(
                text=text,
                start_time=start_time,
                end_time=end_time,
            ))
            i += chunk

        return segments
    
    def add_subtitles(
        self,
        video_path: Path,
        subtitles: List[SubtitleSegment],
        output_path: Path,
        config: VideoConfig
    ) -> Path:
        """Add colored subtitles using FFmpeg ASS styling with fallback to SRT."""
        if not self._ffmpeg_available:
            logger.warning("FFmpeg not available, skipping subtitles")
            return video_path
        
        # Create ASS file with rich styling
        ass_path = output_path.with_suffix('.ass')
        self._create_ass_file(subtitles, ass_path, config)
        
        # Use FFmpeg to burn styled subtitles
        # Escape backslashes and colons for FFmpeg filter on Windows
        ass_path_str = str(ass_path).replace('\\', '/').replace(':', r'\:')
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-vf', f"ass='{ass_path_str}'",
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '20',
            '-c:a', 'copy',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            ass_path.unlink(missing_ok=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.warning(f"ASS subtitle burn failed, trying SRT fallback")
            ass_path.unlink(missing_ok=True)
            return self._add_subtitles_srt(video_path, subtitles, output_path, config)
    
    def _add_subtitles_srt(self, video_path, subtitles, output_path, config):
        """Fallback SRT subtitle method with centered bold styling."""
        srt_path = output_path.with_suffix('.srt')
        self._create_srt_file(subtitles, srt_path)

        font_size = max(48, int(58 * config.height / 1920))

        # SRT path escaping for FFmpeg on Windows
        srt_path_str = str(srt_path).replace('\\', '/').replace(':', r'\:')

        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-vf', f"subtitles='{srt_path_str}':force_style='FontName=Arial Black,"
                   f"FontSize={font_size},PrimaryColour=&H0000FFFF,"
                   f"OutlineColour=&H00000000,Outline=4,Shadow=3,"
                   f"BackColour=&HCC000000,BorderStyle=1,"
                   f"Alignment=5,Bold=1,MarginV=0'",
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '20',
            '-c:a', 'copy',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            srt_path.unlink(missing_ok=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"SRT subtitle burn also failed: {e.stderr[:300] if e.stderr else 'unknown error'}")
            srt_path.unlink(missing_ok=True)
            import shutil
            shutil.copy2(str(video_path), str(output_path))
            return output_path
    
    def _create_ass_file(self, subtitles: List[SubtitleSegment], path: Path, config: VideoConfig) -> None:
        """Create viral YouTube/TikTok-style ASS subtitles.

        Style features:
        - Centered on screen (middle, not bottom)
        - Large bold Impact/Arial Black font
        - 2-3 words per segment for punchy readability
        - Alternating yellow ↔ white color per segment
        - Thick outline + drop shadow for readability over any panel
        - Pop-in scale animation + fade
        """
        style = config.subtitle_style
        base_font_size = style.get('fontsize', 58)
        # Scale font for resolution (designed for 1080×1920)
        font_size = max(48, int(base_font_size * config.height / 1920))

        # Two alternating highlight colors for visual energy
        COLOR_YELLOW = '&H0000FFFF'   # bright yellow  (primary hits)
        COLOR_WHITE  = '&H00FFFFFF'   # clean white    (alternating)
        OUTLINE      = '&H00000000'   # black outline
        SHADOW       = '&HCC000000'   # semi-transparent black shadow

        with open(path, 'w', encoding='utf-8') as f:
            f.write("[Script Info]\n")
            f.write("Title: MangaVID Subtitles\n")
            f.write(f"PlayResX: {config.width}\n")
            f.write(f"PlayResY: {config.height}\n")
            f.write("ScriptType: v4.00+\n")
            f.write("WrapStyle: 0\n\n")

            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                    "Alignment, MarginL, MarginR, MarginV, Encoding\n")
            # Style A – yellow (Alignment=5 = center-middle of screen)
            f.write(f"Style: Pop_Yellow,Arial Black,{font_size},{COLOR_YELLOW},&H000000FF,"
                    f"{OUTLINE},{SHADOW},1,0,0,0,100,100,2,0,1,4,3,5,60,60,0,1\n")
            # Style B – white
            f.write(f"Style: Pop_White,Arial Black,{font_size},{COLOR_WHITE},&H000000FF,"
                    f"{OUTLINE},{SHADOW},1,0,0,0,100,100,2,0,1,4,3,5,60,60,0,1\n\n")

            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

            for idx, sub in enumerate(subtitles):
                start = self._format_ass_time(sub.start_time)
                end = self._format_ass_time(sub.end_time)
                style_name = 'Pop_Yellow' if idx % 2 == 0 else 'Pop_White'
                # Animation: fast pop-in scale (80→100%) + fade-in 100ms / fade-out 80ms
                clean_text = self._sanitize_subtitle_text(sub.text)
                anim = (r"{\fad(100,80)"
                        r"\t(0,80,\fscx110\fscy110)"
                        r"\t(80,180,\fscx100\fscy100)}")
                f.write(f"Dialogue: 0,{start},{end},{style_name},,0,0,0,,{anim}{clean_text}\n")
    
    def _format_ass_time(self, seconds: float) -> str:
        """Format time for ASS file."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"
    
    def _create_srt_file(self, subtitles: List[SubtitleSegment], path: Path) -> None:
        """Create SRT subtitle file with uppercase text for viral style."""
        with open(path, 'w', encoding='utf-8') as f:
            for i, sub in enumerate(subtitles, 1):
                start = self._format_srt_time(sub.start_time)
                end = self._format_srt_time(sub.end_time)
                clean_text = self._sanitize_subtitle_text(sub.text)
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{clean_text}\n\n")
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT file."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _sanitize_subtitle_text(text: str) -> str:
        """Clean subtitle text: uppercase, escape ASS special chars, strip junk.

        Fixes:
        - Escapes curly braces { } so they aren't parsed as ASS override tags
        - Strips non-printable / zero-width Unicode characters that Edge-TTS
          occasionally emits (ZWSP, ZWNJ, BOM, soft-hyphens, etc.)
        - Replaces problematic punctuation variants with ASCII equivalents
        - Converts to uppercase for viral subtitle style
        """
        import unicodedata

        if not text:
            return ""

        # Strip zero-width and invisible Unicode characters
        # These cause "weird letters" when rendered in subtitles
        cleaned = []
        for ch in text:
            cat = unicodedata.category(ch)
            # Skip: Cf (format chars like ZWSP/ZWNJ/BOM), Cc (control chars),
            # Mn (combining marks that appear as floating accents without base)
            if cat in ('Cf', 'Cc'):
                continue
            # Skip specific known troublemakers
            if ord(ch) in (
                0x200B,  # Zero-width space
                0x200C,  # Zero-width non-joiner
                0x200D,  # Zero-width joiner
                0xFEFF,  # BOM / zero-width no-break space
                0x00AD,  # Soft hyphen
                0x2060,  # Word joiner
                0x2028,  # Line separator
                0x2029,  # Paragraph separator
            ):
                continue
            cleaned.append(ch)

        result = "".join(cleaned)

        # Normalize Unicode to NFC (compose combining characters)
        result = unicodedata.normalize('NFC', result)

        # Replace fancy quotes/dashes with ASCII equivalents
        replacements = {
            '\u2018': "'", '\u2019': "'",  # Smart single quotes
            '\u201C': '"', '\u201D': '"',  # Smart double quotes
            '\u2014': '-', '\u2013': '-',  # Em/en dash → hyphen
            '\u2026': '...',               # Ellipsis character
            '\u00A0': ' ',                 # Non-breaking space
        }
        for old, new in replacements.items():
            result = result.replace(old, new)

        # Escape ASS override tag delimiters so they render as literal text
        # In ASS format, { } are used for style overrides — literal braces must be escaped
        result = result.replace('\\', '\\\\')  # Escape backslashes first
        result = result.replace('{', '\\{')
        result = result.replace('}', '\\}')

        # Uppercase for viral subtitle style
        result = result.upper()

        # Collapse multiple spaces
        result = ' '.join(result.split())

        return result.strip()
    
    # ───────────────── Frame Generation ─────────────────
    
    def _generate_frames_to_video(
        self,
        panels: List[np.ndarray],
        timings: List[PanelTiming],
        config: VideoConfig,
        output_path: Path,
        intro_hook: str = "",
        manga_title: str = "",
        intro_duration: float = 0.0,
        intro_image: Optional[np.ndarray] = None,
    ) -> None:
        """Generate video frames and pipe directly to FFmpeg for fast H.264 encoding.
        
        Uses subprocess pipe instead of cv2.VideoWriter to avoid the slow
        mp4v codec + re-encode roundtrip.  Frames are written as raw BGR
        bytes directly into FFmpeg's stdin.
        """
        # Prepare panels - resize to slightly larger than target for zoom headroom
        zoom_headroom = 1.20  # 20% extra for Ken Burns
        padded_w = int(config.width * zoom_headroom)
        padded_h = int(config.height * zoom_headroom)
        prepared_panels = [
            self.resize_for_format(panel, padded_w, padded_h, "cover")
            for panel in panels
        ]
        
        # Pre-warm the vignette cache for the target resolution
        vignette_mask = self._get_vignette_mask(config.width, config.height)
        
        # Pre-compute the previous panel's final frame for each panel (for transitions)
        prev_final_frames = [None] * len(timings)
        for i, timing in enumerate(timings):
            if i > 0:
                prev_panel = prepared_panels[timings[i - 1].panel_index]
                prev_final_frames[i] = self.apply_zoom_effect(
                    prev_panel, timings[i - 1].zoom_effect, 1.0, config
                )
        
        total_frames = int(timings[-1].end_time * config.fps) if timings else 0
        log_interval = max(total_frames // 10, 1)
        
        # ── Pipe raw frames to FFmpeg for direct H.264 encoding ──
        # This avoids the slow cv2.VideoWriter → mp4v → re-encode path
        if self._ffmpeg_available:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{config.width}x{config.height}',
                '-pix_fmt', 'bgr24',
                '-r', str(config.fps),
                '-i', '-',         # read from stdin
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '20',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                str(output_path),
            ]
            pipe = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            write_frame = lambda f: pipe.stdin.write(f.tobytes())
            use_pipe = True
        else:
            # Fallback to slower cv2.VideoWriter if no FFmpeg
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path), fourcc, config.fps,
                (config.width, config.height),
            )
            write_frame = lambda f: writer.write(f)
            use_pipe = False
        
        frame_idx = 0
        
        # ── Intro card: atmospheric blurred panel + title + hook text ──
        if intro_duration > 0 and intro_hook and len(panels) > 0:
            intro_frames_count = int(intro_duration * config.fps)
            intro_card = self._render_intro_card(
                panels[0], config, intro_hook, manga_title,
                intro_image=intro_image,
            )
            for f in range(intro_frames_count):
                progress = f / max(intro_frames_count, 1)
                # Fade in from black over first 0.6s, hold, then crossfade to first panel
                fade_in_frames = int(0.6 * config.fps)
                fade_out_frames = int(0.4 * config.fps)
                fade_out_start = intro_frames_count - fade_out_frames
                
                if f < fade_in_frames:
                    # Fade in from black
                    alpha = f / max(fade_in_frames, 1)
                    frame = cv2.multiply(intro_card, (alpha, alpha, alpha, 0), dtype=cv2.CV_8U)
                elif f >= fade_out_start:
                    # Crossfade into first panel
                    t = (f - fade_out_start) / max(fade_out_frames, 1)
                    first_panel_frame = self.apply_zoom_effect(
                        prepared_panels[0], timings[0].zoom_effect, 0.0, config
                    )
                    first_panel_frame = cv2.multiply(
                        first_panel_frame, vignette_mask, scale=1.0 / 255.0, dtype=cv2.CV_8U
                    )
                    frame = cv2.addWeighted(intro_card, 1.0 - t, first_panel_frame, t, 0)
                else:
                    # Subtle slow zoom during hold
                    hold_progress = (f - fade_in_frames) / max(fade_out_start - fade_in_frames, 1)
                    scale = 1.0 + hold_progress * 0.03
                    frame = self._apply_zoom(intro_card, scale, config.width, config.height)
                
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                write_frame(frame)
                frame_idx += 1
        
        for timing in timings:
            panel_idx = timing.panel_index
            panel = prepared_panels[panel_idx]
            
            start_frame = int(timing.start_time * config.fps)
            end_frame = int(timing.end_time * config.fps)
            transition_frames = int(timing.transition_duration * config.fps)
            panel_duration_frames = end_frame - start_frame
            
            for f in range(start_frame, end_frame):
                frame_in_panel = f - start_frame
                progress = frame_in_panel / max(panel_duration_frames, 1)
                
                # Apply zoom/pan effect with easing
                frame = self.apply_zoom_effect(panel, timing.zoom_effect, progress, config)
                
                # Apply transition at start (if not first panel)
                if panel_idx > 0 and frame_in_panel < transition_frames:
                    prev_frame = prev_final_frames[panel_idx]
                    if prev_frame is not None:
                        trans_progress = frame_in_panel / max(transition_frames, 1)
                        frame = self.apply_transition(prev_frame, frame, timing.transition_in, trans_progress)
                
                # Apply cinematic vignette (integer-only, no float conversion)
                frame = cv2.multiply(frame, vignette_mask, scale=1.0 / 255.0, dtype=cv2.CV_8U)
                
                # Ensure contiguous memory for pipe write
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                
                write_frame(frame)
                frame_idx += 1
                
                if frame_idx % log_interval == 0:
                    logger.debug(f"Frame {frame_idx}/{total_frames} ({frame_idx * 100 // total_frames}%)")
        
        if use_pipe:
            pipe.stdin.close()
            pipe.wait()
        else:
            writer.release()
        
        logger.info(f"Generated {frame_idx} frames")
    
    # ───────────────── Intro Card ─────────────────

    def _render_intro_card(
        self,
        first_panel: np.ndarray,
        config: VideoConfig,
        intro_hook: str,
        manga_title: str = "",
        intro_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render a cinematic intro card using AI-generated image or blurred panel.

        When an AI-generated intro image is available, it fills the frame with a
        vivid, coloured scene that sets the story context visually.  A subtle
        gradient overlay and the manga title are added on top.

        Falls back to the classic blurred-panel-with-text style when no AI image
        is provided.
        """
        w, h = config.width, config.height

        if intro_image is not None:
            # ── AI-generated image path ──
            bg = self.resize_for_format(intro_image, w, h, "cover")

            # Light bottom-gradient so the title text is readable
            gradient = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                # Subtle gradient: transparent at top, darker at bottom 30 %
                frac = y / h
                if frac > 0.65:
                    darkness = int(((frac - 0.65) / 0.35) * 160)
                    gradient[y, :] = darkness
                # Also slight darkening at the very top for title
                elif frac < 0.18:
                    darkness = int(((0.18 - frac) / 0.18) * 90)
                    gradient[y, :] = darkness
            bg = cv2.subtract(bg, gradient)

            # Apply subtle vignette
            vignette = self._get_vignette_mask(w, h)
            bg = cv2.multiply(bg, vignette, scale=1.0 / 255.0, dtype=cv2.CV_8U)

            # ── Minimal title overlay ──
            if manga_title:
                title_text = manga_title.split('\n')[0].strip()[:40]
                if title_text:
                    title_font = cv2.FONT_HERSHEY_DUPLEX
                    title_scale = min(w / 450, 1.8)
                    title_thickness = max(int(title_scale * 2), 2)
                    (tw, th), _ = cv2.getTextSize(title_text, title_font, title_scale, title_thickness)
                    tx = (w - tw) // 2
                    ty = int(h * 0.10)
                    # Shadow
                    cv2.putText(bg, title_text, (tx + 3, ty + 3), title_font, title_scale,
                                (0, 0, 0), title_thickness + 2, cv2.LINE_AA)
                    cv2.putText(bg, title_text, (tx, ty), title_font, title_scale,
                                (255, 255, 255), title_thickness, cv2.LINE_AA)

            return bg

        # Resize first panel to cover the frame
        bg = self.resize_for_format(first_panel, w, h, "cover")

        # Heavy Gaussian blur for atmospheric background
        bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=25, sigmaY=25)

        # Dark semi-transparent overlay (gradient: darker at center for text contrast)
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        # Vertical gradient: darker in center-upper area where text will be
        for y in range(h):
            # Peaks at 40% from top (where text sits)
            center_dist = abs(y / h - 0.42)
            darkness = int(max(0.55 - center_dist * 0.4, 0.3) * 255)
            overlay[y, :] = darkness
        # Blend: darken the blurred background
        bg = cv2.subtract(bg, overlay)

        # Apply vignette for cinematic feel
        vignette = self._get_vignette_mask(w, h)
        bg = cv2.multiply(bg, vignette, scale=1.0 / 255.0, dtype=cv2.CV_8U)

        # ── Draw text ──
        # We use cv2.putText for reliability (no PIL dependency)
        # Title (larger, bold)
        if manga_title:
            # Clean title: take first meaningful part
            title_text = manga_title.split('\n')[0].strip()[:40]
            if title_text:
                title_font = cv2.FONT_HERSHEY_DUPLEX
                title_scale = min(w / 400, 2.0)
                title_thickness = max(int(title_scale * 2), 2)

                # Get text size for centering
                (tw, th), baseline = cv2.getTextSize(title_text, title_font, title_scale, title_thickness)
                tx = (w - tw) // 2
                ty = int(h * 0.35)

                # Shadow
                cv2.putText(bg, title_text, (tx + 3, ty + 3), title_font, title_scale,
                            (0, 0, 0), title_thickness + 2, cv2.LINE_AA)
                # Main text (white)
                cv2.putText(bg, title_text, (tx, ty), title_font, title_scale,
                            (255, 255, 255), title_thickness, cv2.LINE_AA)

        # Hook text (smaller, atmospheric line below title)
        if intro_hook:
            hook_font = cv2.FONT_HERSHEY_SIMPLEX
            hook_scale = min(w / 600, 1.2)
            hook_thickness = max(int(hook_scale * 1.5), 1)

            # Word-wrap the hook text to fit screen width with margin
            max_text_width = int(w * 0.8)
            hook_lines = self._wrap_text_cv2(intro_hook, hook_font, hook_scale, hook_thickness, max_text_width)

            # Starting Y position: below title or center if no title
            start_y = int(h * 0.45) if manga_title else int(h * 0.40)
            line_gap = int(40 * hook_scale)

            for i, line in enumerate(hook_lines):
                (lw, lh), _ = cv2.getTextSize(line, hook_font, hook_scale, hook_thickness)
                lx = (w - lw) // 2
                ly = start_y + i * line_gap

                # Shadow
                cv2.putText(bg, line, (lx + 2, ly + 2), hook_font, hook_scale,
                            (0, 0, 0), hook_thickness + 2, cv2.LINE_AA)
                # Main text (warm accent color — golden yellow)
                cv2.putText(bg, line, (lx, ly), hook_font, hook_scale,
                            (0, 215, 255), hook_thickness, cv2.LINE_AA)

        # Subtle bottom line / bar for visual polish
        bar_y = int(h * 0.58)
        bar_w = int(w * 0.3)
        bar_x = (w - bar_w) // 2
        cv2.line(bg, (bar_x, bar_y), (bar_x + bar_w, bar_y), (0, 180, 255), 2, cv2.LINE_AA)

        return bg

    def _wrap_text_cv2(
        self, text: str, font: int, scale: float, thickness: int, max_width: int
    ) -> List[str]:
        """Word-wrap text to fit within max_width pixels using cv2 text metrics."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            (tw, _), _ = cv2.getTextSize(test_line, font, scale, thickness)
            if tw <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [text]

    # ───────────────── Audio ─────────────────
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file."""
        try:
            import wave
            with wave.open(str(audio_path), 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                return frames / rate
        except Exception:
            pass
        
        if self._ffmpeg_available:
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
                    capture_output=True, text=True
                )
                return float(result.stdout.strip())
            except Exception:
                pass
        
        return 30.0
    
    def _add_audio_to_video(self, video_path: Path, audio_path: Path, output_path: Path, delay_ms: int = 0) -> None:
        """Add audio track to video using FFmpeg — copy video stream, only encode audio.
        
        Args:
            delay_ms: Delay audio start by this many milliseconds (for intro card offset).
        """
        # Build audio filter for delay if needed
        audio_filter = []
        if delay_ms > 0:
            audio_filter = ['-af', f'adelay={delay_ms}|{delay_ms}']
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-c:v', 'copy',         # No re-encode — video already H.264 from pipe
            *audio_filter,
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-movflags', '+faststart',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.debug("Added audio to video with high quality encoding")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add audio: {e}")
            video_path.rename(output_path)
