"""
Anime Generator Module — v2

AI-powered manga-to-anime style transfer using Stable Diffusion 1.5.
Converts manga panels into vibrant anime-style images and assembles
them into a narrated video with cinematic camera motion and audio.

Key improvements over v1:
  - HuggingFace cache redirected to E: drive (C: has <1 GB)
  - Portrait 1080×1920 output matching the main video format
  - Per-panel duration matched to narration audio length
  - Cinematic camera: varied pan, zoom, and parallax per panel
  - Audio muxed in via FFmpeg after encoding
  - Much stronger style transfer (strength 0.75) for visible results
  - Clear logging when SD model fails → fallback runs

Designed for RTX 3050 6 GB (~4 GB VRAM with attention slicing + VAE tiling).
"""

import gc
import logging
import math
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from interfaces.base_anime_generator import (
    AnimeConfig,
    AnimeFrame,
    AnimeResult,
    AnimeStyle,
    AnimationMode,
    BaseAnimeGenerator,
)

logger = logging.getLogger(__name__)

# ── Redirect HuggingFace cache to E: (C: has <1 GB free) ──
_HF_CACHE = Path("E:/MangaVID/.cache/huggingface")
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_CACHE / "hub"))

# ── Model IDs (all free on HuggingFace) ──
STYLE_MODELS: Dict[AnimeStyle, str] = {
    AnimeStyle.MODERN_ANIME: "stablediffusionapi/anything-v5",
    AnimeStyle.CLASSIC_ANIME: "stablediffusionapi/anything-v5",
    AnimeStyle.GHIBLI: "nitrosocke/Ghibli-Diffusion",
    AnimeStyle.SHONEN: "stablediffusionapi/anything-v5",
    AnimeStyle.CHIBI: "stablediffusionapi/anything-v5",
    AnimeStyle.VIBRANT: "stablediffusionapi/anything-v5",
}

STYLE_PROMPTS: Dict[AnimeStyle, str] = {
    AnimeStyle.MODERN_ANIME: "anime style, modern anime, clean lines, vivid colors, colorful",
    AnimeStyle.CLASSIC_ANIME: "90s anime style, retro anime, cel-shaded, nostalgic, colorful",
    AnimeStyle.GHIBLI: "ghibli style, watercolor, soft lighting, whimsical, painterly, colorful",
    AnimeStyle.SHONEN: "shonen anime, dynamic action, bold colors, intense, dramatic lighting, colorful",
    AnimeStyle.CHIBI: "chibi style, cute, super deformed, adorable, colorful, kawaii",
    AnimeStyle.VIBRANT: "anime style, extremely vibrant colors, HDR, neon accents, saturated, colorful",
}

# ControlNet model for edge preservation
CONTROLNET_CANNY_ID = "lllyasviel/sd-controlnet-canny"

# ── Camera motion presets for cinematic variety ──
CAMERA_MOTIONS = [
    {"name": "zoom_in",      "zoom": (1.00, 1.15), "pan_x": (0.0, 0.0),  "pan_y": (0.0, 0.0)},
    {"name": "zoom_out",     "zoom": (1.15, 1.00), "pan_x": (0.0, 0.0),  "pan_y": (0.0, 0.0)},
    {"name": "pan_left",     "zoom": (1.12, 1.12), "pan_x": (0.3, -0.3), "pan_y": (0.0, 0.0)},
    {"name": "pan_right",    "zoom": (1.12, 1.12), "pan_x": (-0.3, 0.3), "pan_y": (0.0, 0.0)},
    {"name": "pan_up",       "zoom": (1.12, 1.12), "pan_x": (0.0, 0.0),  "pan_y": (0.3, -0.3)},
    {"name": "pan_down",     "zoom": (1.12, 1.12), "pan_x": (0.0, 0.0),  "pan_y": (-0.3, 0.3)},
    {"name": "zoom_pan_r",   "zoom": (1.00, 1.18), "pan_x": (-0.2, 0.2), "pan_y": (0.0, 0.0)},
    {"name": "zoom_pan_l",   "zoom": (1.00, 1.18), "pan_x": (0.2, -0.2), "pan_y": (0.0, 0.0)},
    {"name": "drift_down",   "zoom": (1.05, 1.10), "pan_x": (0.0, 0.0),  "pan_y": (-0.15, 0.15)},
    {"name": "drift_up",     "zoom": (1.05, 1.10), "pan_x": (0.0, 0.0),  "pan_y": (0.15, -0.15)},
]


def _select_device(preference: str = "auto") -> str:
    """Pick the best available device.  Strongly prefers CUDA."""
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"CUDA GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        return "cuda"

    if preference == "cpu":
        logger.info("CPU mode requested")
        return "cpu"

    # CUDA not available — log detailed diagnostics
    logger.error(
        "CUDA NOT available!  torch.cuda.is_available() == False.\n"
        f"  torch version : {torch.__version__}\n"
        f"  torch file    : {torch.__file__}\n"
        "  SD 1.5 on CPU will use 16 GB+ RAM and take hours.\n"
        "  Install the CUDA build: pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu124"
    )
    return "cpu"


class AnimeGenerator(BaseAnimeGenerator):
    """
    Generates anime-style narrated video from manga panels.

    Pipeline:
    1. Style-transfer each panel via SD 1.5 img2img (+ ControlNet)
    2. Fit panels into 1080×1920 portrait canvas (letterbox/pillarbox)
    3. Apply cinematic camera motion (varied zoom / pan per panel)
    4. Crossfade between panels
    5. Mux narration audio via FFmpeg
    6. Output a single .mp4 matching the narration duration

    Runs on RTX 3050 6 GB with attention slicing + VAE tiling.
    """

    def __init__(self):
        self._pipe = None
        self._controlnet_pipe = None
        self._loaded_model_id: str = ""
        self._device: str = "cpu"
        self._ffmpeg_available = self._check_ffmpeg()
        self._sd_loaded = False  # Track whether real SD loaded

    # ────────────────── Public API ──────────────────

    def generate(
        self,
        panels: List[np.ndarray],
        output_path: Path,
        config: Optional[AnimeConfig] = None,
        scene_descriptions: Optional[List[str]] = None,
        audio_path: Optional[Path] = None,
        narration_duration: float = 0.0,
        progress_callback=None,
    ) -> AnimeResult:
        """Generate anime-style narrated video from manga panels.

        Args:
            panels: BGR numpy images.
            output_path: Destination .mp4 path.
            config: Anime config (style, strength, etc.).
            scene_descriptions: Per-panel text for SD prompts.
            audio_path: Path to narration WAV to mux in.
            narration_duration: Total narration length in seconds.
                If >0, panel display times are spread to match.
        """
        config = config or AnimeConfig()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Anime generation: {len(panels)} panels, style={config.style.value}, "
            f"device_pref={config.device}, narration={narration_duration:.1f}s"
        )

        self._device = _select_device(config.device)
        logger.info(f"Using device: {self._device}")

        # Refuse to run SD on CPU — it uses 16 GB+ RAM and takes hours
        skip_sd = False
        if self._device == "cpu":
            logger.warning(
                "Skipping Stable Diffusion (no CUDA GPU). "
                "Using colour-enhanced fallback instead."
            )
            skip_sd = True

        # ── Phase 1: Stylize each panel ──
        self._sd_loaded = False

        # Pre-load pipeline ONCE (not per-panel)
        if not skip_sd:
            self._ensure_pipeline(config)

        frames: List[AnimeFrame] = []
        for i, panel in enumerate(panels):
            desc = (
                scene_descriptions[i]
                if scene_descriptions and i < len(scene_descriptions)
                else ""
            )
            prompt = self._build_prompt(config, desc)
            logger.info(f"Stylizing panel {i+1}/{len(panels)}: '{prompt[:60]}…'")

            frame = self.stylize_panel(
                panel, prompt=prompt, config=config, skip_sd=skip_sd
            )
            frame.panel_index = i
            frames.append(frame)

            if progress_callback:
                progress_callback("stylize", i + 1, len(panels))

        if self._sd_loaded:
            logger.info("✓ All panels stylized with Stable Diffusion")
        else:
            logger.warning(
                "⚠ Stable Diffusion could not load — using colour-enhanced fallback. "
                "The first run needs to download ~5 GB of models to "
                f"{_HF_CACHE}. Check disk space and network."
            )

        # ── Phase 2: Compute per-panel durations ──
        total_duration = narration_duration if narration_duration > 0 else (
            len(frames) * config.duration_per_panel
        )
        # Evenly distribute duration across panels
        per_panel = total_duration / max(len(frames), 1)
        # Clamp between 2s and 15s per panel
        per_panel = max(2.0, min(per_panel, 15.0))
        panel_durations = [per_panel] * len(frames)
        logger.info(
            f"Panel timing: {per_panel:.1f}s × {len(frames)} = "
            f"{per_panel * len(frames):.1f}s total"
        )

        # ── Phase 3: Assemble into video ──
        if progress_callback:
            progress_callback("assemble", 0, 1)
        silent_path = output_path.with_stem(output_path.stem + "_silent")
        video_path = self._assemble_video(
            frames, silent_path, config, panel_durations
        )
        if progress_callback:
            progress_callback("assemble", 1, 1)

        # ── Phase 4: Mux audio ──
        if video_path and audio_path and audio_path.exists():
            final = self._mux_audio(video_path, audio_path, output_path)
            # Clean up silent intermediate
            if final and final != video_path:
                video_path.unlink(missing_ok=True)
            video_path = final
        elif video_path and video_path != output_path:
            # No audio available — just rename
            if output_path.exists():
                output_path.unlink()
            video_path.rename(output_path)
            video_path = output_path

        # Free VRAM
        self._unload_pipeline()

        actual_dur = sum(panel_durations)
        return AnimeResult(
            frames=frames,
            video_path=video_path,
            width=config.width,
            height=config.height,
            fps=config.fps,
            duration=actual_dur,
            panel_count=len(panels),
            metadata={
                "style": config.style.value,
                "animation_mode": config.animation_mode.value,
                "device": self._device,
                "strength": config.strength,
                "steps": config.num_inference_steps,
                "sd_loaded": self._sd_loaded,
                "has_audio": audio_path is not None and audio_path.exists(),
            },
        )

    def stylize_panel(
        self,
        panel: np.ndarray,
        prompt: str = "",
        config: Optional[AnimeConfig] = None,
        skip_sd: bool = False,
    ) -> AnimeFrame:
        """Stylize a single manga panel into anime art using img2img."""
        config = config or AnimeConfig()

        if not prompt:
            prompt = self._build_prompt(config)

        seed = config.seed if config.seed >= 0 else random.randint(0, 2**32 - 1)

        # SD generation dimensions (landscape for SD, we'll letterbox later)
        gen_w, gen_h = 768, 512

        if skip_sd:
            result_np = self._enhance_fallback(panel, gen_w, gen_h)
            return AnimeFrame(
                image=result_np, panel_index=0,
                style=config.style.value, seed=seed, prompt=prompt,
            )

        # Prepare input image
        pil_image = self._numpy_to_pil(panel, gen_w, gen_h)

        # Generate
        import torch
        generator = torch.Generator(device=self._device).manual_seed(seed)

        try:
            if self._controlnet_pipe and config.use_controlnet:
                control_image = self._extract_canny(panel, gen_w, gen_h)
                result = self._controlnet_pipe(
                    prompt=prompt,
                    negative_prompt=config.negative_prompt,
                    image=pil_image,
                    control_image=control_image,
                    strength=config.strength,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps,
                    generator=generator,
                ).images[0]
                self._sd_loaded = True
            elif self._pipe:
                result = self._pipe(
                    prompt=prompt,
                    negative_prompt=config.negative_prompt,
                    image=pil_image,
                    strength=config.strength,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps,
                    generator=generator,
                ).images[0]
                self._sd_loaded = True
            else:
                raise RuntimeError("No Stable Diffusion pipeline loaded")
        except Exception as e:
            logger.error(f"SD inference failed: {e}")
            result_np = self._enhance_fallback(panel, gen_w, gen_h)
            return AnimeFrame(
                image=result_np, panel_index=0,
                style=config.style.value, seed=seed, prompt=prompt,
            )

        # Convert PIL → numpy BGR
        result_np = np.array(result)
        if result_np.ndim == 3 and result_np.shape[2] == 3:
            result_np = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        return AnimeFrame(
            image=result_np, panel_index=0,
            style=config.style.value, seed=seed, prompt=prompt,
        )

    # ────────────────── Pipeline Management ──────────────────

    def _ensure_pipeline(self, config: AnimeConfig) -> None:
        """Load or swap the Stable Diffusion pipeline as needed."""
        model_id = config.model_id or STYLE_MODELS.get(
            config.style, STYLE_MODELS[AnimeStyle.MODERN_ANIME]
        )

        if self._pipe is not None and self._loaded_model_id == model_id:
            return

        self._unload_pipeline()

        import torch

        logger.info(f"Loading Stable Diffusion model: {model_id}")
        logger.info(f"HuggingFace cache: {os.environ.get('HF_HOME', 'default')}")
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        try:
            if config.use_controlnet:
                self._load_controlnet_pipeline(model_id, dtype, config)
            else:
                self._load_basic_pipeline(model_id, dtype, config)
        except Exception as e:
            logger.warning(f"ControlNet load failed ({e}), trying basic img2img…")
            self._controlnet_pipe = None
            try:
                self._load_basic_pipeline(model_id, dtype, config)
            except Exception as e2:
                logger.error(
                    f"★ Pipeline load FAILED entirely: {e2}\n"
                    f"  Models are downloaded to {_HF_CACHE} (~5 GB needed).\n"
                    f"  Falling back to colour-enhanced original panels."
                )
                self._pipe = None

        self._loaded_model_id = model_id

    def _load_basic_pipeline(self, model_id: str, dtype, config: AnimeConfig) -> None:
        from diffusers import StableDiffusionImg2ImgPipeline

        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=str(_HF_CACHE / "hub"),
        )
        self._pipe.to(self._device)

        if config.enable_attention_slicing:
            self._pipe.enable_attention_slicing()
        if config.enable_vae_tiling and hasattr(self._pipe, "enable_vae_tiling"):
            self._pipe.enable_vae_tiling()

        logger.info(f"✓ Loaded img2img pipeline on {self._device} ({dtype})")

    def _load_controlnet_pipeline(self, model_id: str, dtype, config: AnimeConfig) -> None:
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetImg2ImgPipeline,
        )

        logger.info(f"Loading ControlNet: {CONTROLNET_CANNY_ID}")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_CANNY_ID,
            torch_dtype=dtype,
            cache_dir=str(_HF_CACHE / "hub"),
        )

        self._controlnet_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=str(_HF_CACHE / "hub"),
        )
        self._controlnet_pipe.to(self._device)

        if config.enable_attention_slicing:
            self._controlnet_pipe.enable_attention_slicing()
        if config.enable_vae_tiling and hasattr(self._controlnet_pipe, "enable_vae_tiling"):
            self._controlnet_pipe.enable_vae_tiling()

        self._pipe = self._controlnet_pipe
        logger.info(f"✓ Loaded ControlNet img2img pipeline on {self._device}")

    def _unload_pipeline(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if self._controlnet_pipe is not None:
            del self._controlnet_pipe
            self._controlnet_pipe = None
        self._loaded_model_id = ""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ────────────────── Text-to-Image (Intro Thumbnail) ──────────────────

    def generate_intro_thumbnail(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 768,
        steps: int = 30,
        guidance_scale: float = 8.0,
        seed: int = -1,
    ) -> Optional[np.ndarray]:
        """Generate an anime image from text using SD 1.5 txt2img.

        Used for intro thumbnail / title-card background.
        Loads a txt2img pipeline (shares the same model weights as img2img),
        generates the image, then unloads to free VRAM.

        Args:
            prompt: Text prompt describing the image.
            negative_prompt: Things to avoid.
            width: Image width (multiple of 8, default 512 for portrait).
            height: Image height (multiple of 8, default 768 for portrait).
            steps: Diffusion steps (more = better quality, slower).
            guidance_scale: CFG scale — how closely to follow prompt.
            seed: Random seed (-1 = random).

        Returns:
            BGR numpy array or None on failure.
        """
        import torch

        device = _select_device("auto")
        if device == "cpu":
            logger.warning("No CUDA GPU — skipping SD txt2img for intro thumbnail")
            return None

        model_id = "stablediffusionapi/anything-v5"
        dtype = torch.float16

        if not negative_prompt:
            negative_prompt = (
                "blurry, low quality, worst quality, jpeg artifacts, watermark, "
                "text, speech bubble, dialogue, deformed, ugly, duplicate, "
                "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, "
                "mutation, extra limbs, bad anatomy, bad proportions, disfigured, "
                "nsfw, nude, naked, explicit, grotesque, gore"
            )

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        txt2img_pipe = None
        try:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

            logger.info(f"Loading SD txt2img pipeline: {model_id}")
            txt2img_pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                cache_dir=str(_HF_CACHE / "hub"),
            )
            txt2img_pipe.to(device)
            txt2img_pipe.enable_attention_slicing()
            if hasattr(txt2img_pipe, "enable_vae_tiling"):
                txt2img_pipe.enable_vae_tiling()

            # Use DPM++ 2M Karras scheduler for better quality
            txt2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                txt2img_pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )

            generator = torch.Generator(device=device).manual_seed(seed)

            logger.info(f"Generating intro thumbnail: {width}x{height}, {steps} steps, seed={seed}")
            result = txt2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

            # Convert PIL → numpy BGR
            result_np = np.array(result)
            if result_np.ndim == 3 and result_np.shape[2] == 3:
                result_np = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

            logger.info(f"✓ Intro thumbnail generated: {result_np.shape[1]}x{result_np.shape[0]}")
            return result_np

        except Exception as e:
            logger.error(f"SD txt2img failed: {e}")
            return None

        finally:
            # Always free VRAM
            if txt2img_pipe is not None:
                del txt2img_pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ────────────────── Image Processing ──────────────────

    @staticmethod
    def _numpy_to_pil(image: np.ndarray, target_w: int, target_h: int):
        """Convert numpy BGR → PIL RGB and resize."""
        import PIL.Image

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb)

        w = (target_w // 8) * 8
        h = (target_h // 8) * 8
        pil_img = pil_img.resize((w, h), PIL.Image.LANCZOS)
        return pil_img

    @staticmethod
    def _extract_canny(image: np.ndarray, target_w: int, target_h: int):
        """Extract Canny edge map for ControlNet guidance."""
        import PIL.Image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        w = (target_w // 8) * 8
        h = (target_h // 8) * 8
        gray = cv2.resize(gray, (w, h))
        edges = cv2.Canny(gray, 50, 150)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return PIL.Image.fromarray(edges_3ch)

    @staticmethod
    def _enhance_fallback(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Fallback: heavy anime-style colour enhancement when SD is unavailable."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        tw = (target_w // 8) * 8
        th = (target_h // 8) * 8
        image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        # Convert greyscale-ish manga to a warm-toned colour version
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Add colour to desaturated areas (manga lines)
        low_sat = hsv[:, :, 1] < 30
        hsv[low_sat, 0] = random.choice([15, 25, 110, 130, 170])  # tint hue
        hsv[low_sat, 1] = 60  # add saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Bilateral filter for anime-like smoothing
        image = cv2.bilateralFilter(image, 9, 75, 75)

        return image

    @staticmethod
    def _fit_to_canvas(
        image: np.ndarray, canvas_w: int, canvas_h: int
    ) -> np.ndarray:
        """Fit a panel into 1080×1920 portrait canvas with blurred background fill."""
        h, w = image.shape[:2]
        aspect = w / h
        target_aspect = canvas_w / canvas_h

        # Create blurred background from the panel itself
        bg = cv2.resize(image, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)
        bg = cv2.GaussianBlur(bg, (51, 51), 30)
        bg = (bg.astype(np.float32) * 0.4).astype(np.uint8)  # darken

        # Scale panel to fit within canvas (preserving aspect)
        if aspect > target_aspect:
            # Wider than canvas → fit width, letterbox top/bottom
            new_w = canvas_w
            new_h = int(canvas_w / aspect)
        else:
            # Taller → fit height, pillarbox left/right
            new_h = canvas_h
            new_w = int(canvas_h * aspect)

        # Cap to canvas bounds
        new_w = min(new_w, canvas_w)
        new_h = min(new_h, canvas_h)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Center on canvas
        x_off = (canvas_w - new_w) // 2
        y_off = (canvas_h - new_h) // 2
        bg[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        return bg

    # ────────────────── Prompt Building ──────────────────

    @staticmethod
    def _build_prompt(config: AnimeConfig, scene_description: str = "") -> str:
        style_prompt = STYLE_PROMPTS.get(config.style, STYLE_PROMPTS[AnimeStyle.MODERN_ANIME])
        parts = []
        if scene_description:
            parts.append(scene_description)
        parts.append(style_prompt)
        if config.positive_prompt_suffix:
            parts.append(config.positive_prompt_suffix)
        return ", ".join(parts)

    # ────────────────── Cinematic Camera ──────────────────

    @staticmethod
    def _camera_frame(
        image: np.ndarray,
        progress: float,
        motion: dict,
    ) -> np.ndarray:
        """Apply cinematic camera motion (zoom + pan) at a given progress [0..1].

        `motion` has keys: zoom (start, end), pan_x (start, end), pan_y (start, end).
        Pan values are fractions of the available slack after zoom-crop.
        """
        h, w = image.shape[:2]

        # Interpolate zoom
        z_start, z_end = motion["zoom"]
        scale = z_start + (z_end - z_start) * progress

        # Crop dimensions at current zoom
        crop_w = int(w / scale)
        crop_h = int(h / scale)

        # Available slack for panning
        slack_x = w - crop_w
        slack_y = h - crop_h

        # Interpolate pan offset (as fraction of slack)
        px_start, px_end = motion["pan_x"]
        py_start, py_end = motion["pan_y"]
        pan_x = px_start + (px_end - px_start) * progress  # -1..1
        pan_y = py_start + (py_end - py_start) * progress

        # Convert to pixel offset (center ± pan fraction of half-slack)
        cx = slack_x // 2 + int(pan_x * slack_x * 0.5)
        cy = slack_y // 2 + int(pan_y * slack_y * 0.5)

        # Clamp
        cx = max(0, min(cx, w - crop_w))
        cy = max(0, min(cy, h - crop_h))

        cropped = image[cy : cy + crop_h, cx : cx + crop_w]

        # Resize back to original dimensions
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return result

    # ────────────────── Video Assembly ──────────────────

    def _assemble_video(
        self,
        frames: List[AnimeFrame],
        output_path: Path,
        config: AnimeConfig,
        panel_durations: List[float],
    ) -> Optional[Path]:
        """Assemble anime frames into portrait video with cinematic movement.

        Streams frames directly to FFmpeg instead of buffering in memory,
        so RAM usage stays constant regardless of video length.
        """
        if not frames:
            return None
        if not self._ffmpeg_available:
            logger.warning("FFmpeg not found — saving frames only")
            self._save_frames(frames, output_path.parent)
            return None

        fps = config.fps
        canvas_w, canvas_h = config.width, config.height
        transition_dur = 0.6
        transition_frames = int(transition_dur * fps)

        # Assign camera motion per panel
        motions = self._pick_motions(len(frames))

        # Pre-compute how many raw frames each panel will have
        panel_n_frames = []
        for idx in range(len(frames)):
            dur = panel_durations[idx] if idx < len(panel_durations) else config.duration_per_panel
            panel_n_frames.append(max(int(dur * fps), 1))

        # Estimate total frame count for logging
        total_est = sum(panel_n_frames) - transition_frames * max(len(frames) - 1, 0)
        total_est = max(total_est, 1)

        # Start FFmpeg pipe BEFORE generating frames
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{canvas_w}x{canvas_h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

        logger.info(
            f"Encoding anime video: ~{total_est} frames "
            f"(~{total_est/fps:.1f}s) → {output_path}"
        )
        pipe = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def _write(f: np.ndarray) -> None:
            if not f.flags["C_CONTIGUOUS"]:
                f = np.ascontiguousarray(f)
            pipe.stdin.write(f.tobytes())

        # Keep only the tail of the previous panel for crossfading
        prev_tail: List[np.ndarray] = []
        written = 0

        for idx, frame in enumerate(frames):
            n = panel_n_frames[idx]
            canvas = self._fit_to_canvas(frame.image, canvas_w, canvas_h)
            motion = motions[idx]
            logger.info(f"Assembling panel {idx+1}/{len(frames)} ({n} frames, motion={motion['name']})")

            # How many frames to trim off the end for crossfade with next panel
            trim_tail = transition_frames if idx < len(frames) - 1 else 0
            # How many frames to use for crossfade with prev panel
            xfade_head = min(transition_frames, len(prev_tail), n) if idx > 0 else 0

            if idx > 0 and xfade_head > 0:
                # Crossfade: blend prev_tail with current head frames
                for t in range(xfade_head):
                    alpha = t / max(xfade_head - 1, 1)
                    progress = t / max(n - 1, 1)
                    curr_f = self._camera_frame(canvas, progress, motion)
                    blended = cv2.addWeighted(prev_tail[t], 1.0 - alpha, curr_f, alpha, 0)
                    _write(blended)
                    written += 1

            # Body frames: after crossfade head, before crossfade tail
            body_start = xfade_head
            body_end = n - trim_tail

            for f_idx in range(body_start, body_end):
                progress = f_idx / max(n - 1, 1)
                cam = self._camera_frame(canvas, progress, motion)
                _write(cam)
                written += 1

            # Store tail frames for next crossfade (only keep in memory briefly)
            if trim_tail > 0:
                prev_tail = []
                for f_idx in range(max(n - trim_tail, 0), n):
                    progress = f_idx / max(n - 1, 1)
                    cam = self._camera_frame(canvas, progress, motion)
                    prev_tail.append(cam)
            else:
                prev_tail = []

        pipe.stdin.close()
        rc = pipe.wait()
        if rc != 0:
            logger.error(f"FFmpeg encoding failed with exit code {rc}")
            return None

        logger.info(f"✓ Anime video encoded: {written} frames → {output_path}")
        return output_path

    @staticmethod
    def _pick_motions(count: int) -> List[dict]:
        """Select varied camera motions (no two consecutive same type)."""
        available = list(CAMERA_MOTIONS)
        result = []
        last_name = ""
        for _ in range(count):
            candidates = [m for m in available if m["name"] != last_name]
            if not candidates:
                candidates = available
            chosen = random.choice(candidates)
            result.append(chosen)
            last_name = chosen["name"]
        return result

    @staticmethod
    def _merge_with_crossfades(
        panel_arrays: List[List[np.ndarray]],
        transition_frames: int,
    ) -> List[np.ndarray]:
        """Merge per-panel frame lists with crossfade transitions between them."""
        if not panel_arrays:
            return []
        if len(panel_arrays) == 1:
            return panel_arrays[0]

        result: List[np.ndarray] = []

        for idx, panel_frames in enumerate(panel_arrays):
            if idx == 0:
                # First panel: everything except last transition_frames
                cut = max(len(panel_frames) - transition_frames, 1)
                result.extend(panel_frames[:cut])
            else:
                prev_frames = panel_arrays[idx - 1]
                curr_frames = panel_frames

                # Crossfade region
                t_frames = min(transition_frames, len(prev_frames), len(curr_frames))
                for t in range(t_frames):
                    alpha = t / max(t_frames - 1, 1)
                    prev_f = prev_frames[len(prev_frames) - t_frames + t]
                    curr_f = curr_frames[t]
                    blended = cv2.addWeighted(prev_f, 1.0 - alpha, curr_f, alpha, 0)
                    result.append(blended)

                # Rest of current panel (after transition head, before transition tail)
                if idx < len(panel_arrays) - 1:
                    cut = max(len(curr_frames) - transition_frames, t_frames)
                    result.extend(curr_frames[t_frames:cut])
                else:
                    result.extend(curr_frames[t_frames:])

        return result

    # ────────────────── Audio Muxing ──────────────────

    @staticmethod
    def _mux_audio(
        video_path: Path, audio_path: Path, output_path: Path
    ) -> Optional[Path]:
        """Mux narration audio into the anime video."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✓ Audio muxed into anime clip: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio mux failed: {e.stderr.decode(errors='replace')[-300:]}")
            # Return the silent video as fallback
            if output_path != video_path:
                video_path.rename(output_path)
            return output_path

    @staticmethod
    def _save_frames(frames: List[AnimeFrame], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imwrite(str(output_dir / f"anime_frame_{i:04d}.png"), frame.image)
        logger.info(f"Saved {len(frames)} anime frames to {output_dir}")

    @staticmethod
    def _check_ffmpeg() -> bool:
        try:
            return subprocess.run(
                ["ffmpeg", "-version"], capture_output=True
            ).returncode == 0
        except FileNotFoundError:
            return False
