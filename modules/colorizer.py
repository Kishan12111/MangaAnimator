"""
Colorizer Module

Colorizes black and white manga panels using the Zhang et al. "Colorful Image
Colorization" ECCV 2016 model (PyTorch checkpoint from HuggingFace).

Character palette overlay for anime-accurate character colours.
No external API calls.  No rate limiting.  ~1-2 s per panel on CPU.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from skimage import color as skcolor

from interfaces.base_colorizer import BaseColorizer, ColorizationResult

logger = logging.getLogger(__name__)

# HuggingFace coordinates for the model checkpoint
_HF_REPO = "ckpt/colorization"
_HF_FILE = "colorization_release_v2-9b330a0b.pth"


# ━━━━━━━━━━━━━━━━━━━━ Zhang et al. Network Architecture ━━━━━━━━━━━━━━━━━
class _ECCVGenerator(nn.Module):
    """Zhang et al. ECCV-2016 colorization network (8-block VGG-style)."""

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()

        model1 = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64),
        ]
        model2 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]
        model3 = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]
        model4 = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]
        model5 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]
        model6 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]
        model7 = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]
        model8 = [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),
        ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

    def forward(self, input_l: torch.Tensor) -> torch.Tensor:
        conv1 = self.model1(self.normalize_l(input_l))
        conv2 = self.model2(conv1)
        conv3 = self.model3(conv2)
        conv4 = self.model4(conv3)
        conv5 = self.model5(conv4)
        conv6 = self.model6(conv5)
        conv7 = self.model7(conv6)
        conv8 = self.model8(conv7)
        out_reg = self.model_out(self.softmax(conv8))
        return self.unnormalize_ab(self.upsample4(out_reg))

    @staticmethod
    def normalize_l(in_l: torch.Tensor) -> torch.Tensor:
        return (in_l - 50.0) / 100.0

    @staticmethod
    def unnormalize_ab(in_ab: torch.Tensor) -> torch.Tensor:
        return in_ab * 110.0


# ━━━━━━━━━━━━━━━━━━━━ Colour-space helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_img_to_l(img_rgb: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Convert an RGB uint8 image -> (L tensor [1,1,H,W], original_L float64 [H,W])."""
    img_lab = skcolor.rgb2lab(img_rgb)            # float64 [H,W,3]
    img_l = img_lab[:, :, 0]                      # [H,W], range [0,100]
    tens_l = torch.from_numpy(img_l).float()[None, None, :, :]  # [1,1,H,W]
    return tens_l, img_l


def _ab_to_rgb(img_l: np.ndarray, out_ab: torch.Tensor) -> np.ndarray:
    """Combine original L with predicted ab -> RGB uint8."""
    H, W = img_l.shape
    ab = out_ab[0].detach().cpu().numpy().transpose(1, 2, 0)  # [h,w,2]
    # Resize predicted ab to original resolution
    ab = cv2.resize(ab, (W, H))
    img_lab = np.concatenate([img_l[:, :, np.newaxis], ab], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_rgb = np.clip(skcolor.lab2rgb(img_lab), 0, 1)
    return (img_rgb * 255).astype(np.uint8)


# ━━━━━━━━━━━━━━━━━━━━ Main Colorizer class ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Colorizer(BaseColorizer):
    """
    Manga colorization using Zhang et al. ECCV-2016 model (PyTorch).

    Pipeline per panel:
      1. Neural colorize → produces natural-looking colours
      2. Character palette overlay → anime-accurate character tints
      3. Post-process (CLAHE + saturation boost)
    """

    AVAILABLE_MODELS = ["placeholder", "ddcolor"]

    def __init__(
        self,
        model_name: str = "placeholder",
        api_key: Optional[str] = None,
        character_colors: Optional[Dict[str, Dict[str, List[int]]]] = None,
        anime_title: str = "",
    ):
        self._model_name = model_name
        self._model_params: Dict[str, Any] = {}
        self._net: Optional[_ECCVGenerator] = None
        self._api_key = api_key
        self._character_colors = character_colors or {}
        self._anime_title = anime_title
        self._initialize_model()

    # ─────────────────────────────────────────────── Init ───────────────────
    def _initialize_model(self) -> None:
        if self._model_name == "ddcolor":
            try:
                self._init_neural_colorizer()
            except Exception as e:
                logger.warning(f"Neural colorizer init failed: {e}. Heuristic fallback active.")
                self._net = None
        elif self._model_name != "placeholder":
            logger.warning(f"Unknown model '{self._model_name}'. Using placeholder.")
            self._model_name = "placeholder"
        else:
            logger.info("Using placeholder colorizer (enhanced grayscale)")

    def _init_neural_colorizer(self) -> None:
        """Download (once, cached) and load the ECCV-2016 PyTorch model."""
        logger.info("Loading Zhang et al. ECCV-2016 colorization model …")
        ckpt_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)

        net = _ECCVGenerator()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()

        self._net = net
        self._model = "neural_eccv16"
        logger.info("Neural colorization model loaded (PyTorch ECCV-2016)")

    # ─────────────────────────────────────────── Public API ─────────────────
    def set_model(self, model_name: str, **model_params) -> None:
        self._model_name = model_name
        self._model_params = model_params
        self._initialize_model()

    def get_available_models(self) -> List[str]:
        return self.AVAILABLE_MODELS.copy()

    def set_character_colors(self, colors: Dict[str, Dict[str, List[int]]]) -> None:
        self._character_colors = colors
        logger.info(f"Set character colors for: {list(colors.keys())}")

    def is_already_colored(self, image: np.ndarray) -> bool:
        if len(image.shape) != 3 or image.shape[2] != 3:
            return False
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return float(np.mean(hsv[:, :, 1])) > 20

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def postprocess(self, image: np.ndarray) -> np.ndarray:
        """CLAHE + saturation boost for punchy anime look."""
        if image.dtype != np.uint8:
            image = np.clip(
                image * 255 if image.max() <= 1.0 else image, 0, 255
            ).astype(np.uint8)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        # Saturation boost (a, b channels)
        a = lab[:, :, 1].astype(np.float32)
        b = lab[:, :, 2].astype(np.float32)
        lab[:, :, 1] = np.clip((a - 128) * 1.3 + 128, 0, 255).astype(np.uint8)
        lab[:, :, 2] = np.clip((b - 128) * 1.3 + 128, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ─────────────────────────────────────────── Colorize ───────────────────
    def colorize(self, panel_image: np.ndarray, panel_index: int) -> ColorizationResult:
        logger.debug(f"Colorizing panel {panel_index}")
        preprocessed = self.preprocess(panel_image)

        if self.is_already_colored(preprocessed):
            return ColorizationResult(
                panel_index=panel_index,
                original_image=panel_image,
                colorized_image=preprocessed,
                confidence=1.0,
                model_used="passthrough",
                metadata={"already_colored": True},
            )

        # 1. Neural colorization (or heuristic fallback)
        if self._net is not None:
            colorized = self._neural_colorize(preprocessed)
            model_used = "neural_eccv16"
            confidence = 0.80
        else:
            colorized = self._heuristic_colorize(preprocessed)
            model_used = "heuristic"
            confidence = 0.45

        # 2. Character palette overlay
        if self._character_colors:
            colorized = self._apply_character_palette(colorized, preprocessed)
            model_used += "+palette"
            confidence = min(confidence + 0.1, 0.95)

        # 3. Post-process
        colorized = self.postprocess(colorized)

        return ColorizationResult(
            panel_index=panel_index,
            original_image=panel_image,
            colorized_image=colorized,
            confidence=confidence,
            model_used=model_used,
        )

    def colorize_batch(
        self, panels: List[tuple[np.ndarray, int]]
    ) -> List[ColorizationResult]:
        return [self.colorize(img, idx) for img, idx in panels]

    # ────────────────────────── Neural colorization (Zhang ECCV-16) ────────
    @torch.no_grad()
    def _neural_colorize(self, image: np.ndarray) -> np.ndarray:
        """Run the ECCV-2016 network.  Input/output are RGB uint8."""
        tens_l, orig_l = _load_img_to_l(image)
        out_ab = self._net(tens_l)
        return _ab_to_rgb(orig_l, out_ab)

    # ────────────────────────── Heuristic fallback ─────────────────────────
    def _heuristic_colorize(self, image: np.ndarray) -> np.ndarray:
        """Strong heuristic colouring when neural model unavailable."""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        h, w = gray.shape[:2]
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        edges = cv2.Canny(gray, 30, 100)
        is_line = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
        smooth = cv2.GaussianBlur(gray, (7, 7), 0).astype(np.float32)

        a_ch = np.full((h, w), 128.0, dtype=np.float32)
        b_ch = np.full((h, w), 128.0, dtype=np.float32)

        # Skin – warm peach
        skin = cv2.morphologyEx(
            ((smooth > 150) & (smooth < 235) & ~is_line).astype(np.uint8),
            cv2.MORPH_OPEN, np.ones((5, 5)),
        )
        sf = cv2.GaussianBlur(skin.astype(np.float32), (15, 15), 0)
        a_ch += sf * 25
        b_ch += sf * 30

        # Hair – cool
        hair = cv2.morphologyEx(
            ((gray < 55) & ~is_line).astype(np.uint8),
            cv2.MORPH_OPEN, np.ones((5, 5)),
        )
        hf = cv2.GaussianBlur(hair.astype(np.float32), (11, 11), 0)
        a_ch -= hf * 15
        b_ch -= hf * 20

        # Background white – warm cream
        bg = cv2.morphologyEx(
            ((smooth > 230) & ~is_line).astype(np.uint8),
            cv2.MORPH_OPEN, np.ones((7, 7)),
        )
        bf = cv2.GaussianBlur(bg.astype(np.float32), (21, 21), 0)
        a_ch += bf * 5
        b_ch += bf * 12

        a_ch[is_line] = 128
        b_ch[is_line] = 128
        a_ch = cv2.GaussianBlur(a_ch, (15, 15), 0)
        b_ch = cv2.GaussianBlur(b_ch, (15, 15), 0)

        lab[:, :, 1] = np.clip(a_ch, 0, 255)
        lab[:, :, 2] = np.clip(b_ch, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # ────────────────────────── Character palette overlay ───────────────────
    def _apply_character_palette(
        self, colorized: np.ndarray, original: np.ndarray
    ) -> np.ndarray:
        """Overlay anime-accurate character colours using intensity masks."""
        if not self._character_colors:
            return colorized

        gray = (
            cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            if len(original.shape) == 3
            else original
        )
        edges = cv2.Canny(gray, 30, 100)
        is_line = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
        output = colorized.copy()

        for palette in self._character_colors.values():
            if "skin" not in palette:
                continue
            sc = np.array(palette["skin"], dtype=np.float32)
            mask = cv2.morphologyEx(
                ((gray > 160) & (gray < 240) & ~is_line).astype(np.uint8),
                cv2.MORPH_OPEN, np.ones((7, 7)),
            )
            mf = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0)[:, :, np.newaxis]
            gn = (gray.astype(np.float32) / 255.0)[:, :, np.newaxis]
            output = np.clip(
                output.astype(np.float32) * (1 - mf * 0.5) + (gn * sc) * mf * 0.5,
                0, 255,
            ).astype(np.uint8)
            break

        for palette in self._character_colors.values():
            if "hair" not in palette:
                continue
            hc = np.array(palette["hair"], dtype=np.float32)
            mask = cv2.morphologyEx(
                ((gray < 60) & ~is_line).astype(np.uint8),
                cv2.MORPH_OPEN, np.ones((5, 5)),
            )
            mf = cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0)[:, :, np.newaxis]
            gn = (gray.astype(np.float32) / 255.0)[:, :, np.newaxis]
            output = np.clip(
                output.astype(np.float32) * (1 - mf * 0.65) + (gn * hc) * mf * 0.65,
                0, 255,
            ).astype(np.uint8)
            break

        for palette in self._character_colors.values():
            if "outfit" not in palette:
                continue
            oc = np.array(palette["outfit"], dtype=np.float32)
            mask = cv2.morphologyEx(
                ((gray > 60) & (gray < 150) & ~is_line).astype(np.uint8),
                cv2.MORPH_OPEN, np.ones((7, 7)),
            )
            mf = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0)[:, :, np.newaxis]
            gn = (gray.astype(np.float32) / 255.0)[:, :, np.newaxis]
            output = np.clip(
                output.astype(np.float32) * (1 - mf * 0.45) + (gn * oc) * mf * 0.45,
                0, 255,
            ).astype(np.uint8)
            break

        return output

    # ────────────────────────── Misc ───────────────────────────────────────
    def apply_color_hints(self, image: np.ndarray, hints: dict) -> np.ndarray:
        """Apply manual colour hints (region -> colour mapping)."""
        result = image.copy()
        for _, hint in hints.items():
            region = hint.get("region")
            color_val = hint.get("color")
            if region and color_val:
                x, y, w, h = region
                roi = result[y : y + h, x : x + w].astype(np.float32) / 255.0
                ca = np.array(color_val, dtype=np.float32) / 255.0
                roi = roi * 0.7 + ca * 0.3
                result[y : y + h, x : x + w] = (roi * 255).astype(np.uint8)
        return result
