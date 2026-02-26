"""Asset loading for PinPad environments.

Loads pad (food) and agent (animal) images from the assets directory.
Pad images are used for the colored regions; agent images replace the black dot.
"""

from pathlib import Path

import numpy as np
from PIL import Image

# Assets directory relative to this file
ASSETS_DIR = Path(__file__).parent / "assets"

# Pad images (foods) - mapped to layout digits 1-8, cycling if fewer than 8
PAD_IMAGE_FILES = [
    "pear.png",
    "pizza.png",
    "taco.png",
    "apple.png",
    "hamburger.png",
    "lemon.png",
]

# Agent images (animals)
AGENT_IMAGE_FILES = [
    "dog.png",
    "cat.png",
    "frog.png",
]

# Pad char '1' -> index 0, '2' -> 1, ..., '8' -> 7 (cycles through PAD_IMAGE_FILES)
PAD_CHAR_TO_IMAGE_INDEX = {str(i): (i - 1) % len(PAD_IMAGE_FILES) for i in range(1, 9)}

# Pad char to food name (for prompts). Matches PAD_IMAGE_FILES order.
PAD_CHAR_TO_FOOD_NAME = {
    "1": "apple",
    "2": "hamburger",
    "3": "lemon",
    "4": "pear",
    "5": "pizza",
    "6": "taco",
    "7": "apple",
    "8": "hamburger",
}

# Agent (animal) size multiplier - makes agent larger than one cell
AGENT_SIZE_FACTOR = 3.0


def _load_image(path: Path, size: tuple[int, int]) -> Image.Image:
    """Load and resize an image to the given (width, height)."""
    img = Image.open(path).convert("RGBA")
    return img.resize(size, Image.Resampling.LANCZOS)


def _pil_to_rgba_array(pil_img: Image.Image) -> np.ndarray:
    """Return RGBA numpy array (H, W, 4) for alpha compositing."""
    return np.asarray(pil_img.convert("RGBA"))


def _composite_rgba_onto_rgb(
    background: np.ndarray,
    rgba: np.ndarray,
    px: int,
    py: int,
) -> None:
    """Alpha-composite RGBA overlay onto RGB background in-place. Transparent areas show background."""
    h, w = rgba.shape[:2]
    bg_slice = background[py : py + h, px : px + w]
    # Handle clipping: rgba may extend beyond background
    clip_h = min(h, background.shape[0] - py)
    clip_w = min(w, background.shape[1] - px)
    if clip_h <= 0 or clip_w <= 0:
        return
    rgba_clip = rgba[:clip_h, :clip_w]
    bg_clip = bg_slice[:clip_h, :clip_w]
    alpha = rgba_clip[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba_clip[:, :, :3].astype(np.float32)
    blended = (alpha * rgb + (1 - alpha) * bg_clip.astype(np.float32)).astype(np.uint8)
    bg_slice[:clip_h, :clip_w] = blended


def load_pad_image(pad_char: str, width: int, height: int) -> np.ndarray:
    """Load and resize the pad image. Returns RGBA for alpha compositing."""
    idx = PAD_CHAR_TO_IMAGE_INDEX.get(pad_char, 0)
    path = ASSETS_DIR / PAD_IMAGE_FILES[idx]
    img = _load_image(path, (width, height))
    return _pil_to_rgba_array(img)


def load_agent_image(agent_index: int, size: int) -> np.ndarray:
    """Load and resize the agent image. Returns RGBA for alpha compositing."""
    idx = agent_index % len(AGENT_IMAGE_FILES)
    path = ASSETS_DIR / AGENT_IMAGE_FILES[idx]
    img = _load_image(path, (size, size))
    return _pil_to_rgba_array(img)


def get_num_agent_images() -> int:
    """Return the number of available agent images."""
    return len(AGENT_IMAGE_FILES)


def get_num_pad_images() -> int:
    """Return the number of available pad images."""
    return len(PAD_IMAGE_FILES)
