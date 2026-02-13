"""
Synthetic handwriting-style data generator for Nepali word classification.

Renders each vocabulary word as an image with various augmentations to
simulate handwriting variation:
  - Multiple fonts (Nirmala, NotoSans Devanagari, bold variants)
  - Random font sizes (simulating different handwriting sizes)
  - Random stroke thickness variation
  - Slight rotation / shear
  - Gaussian noise
  - Random padding / offset
  - Elastic distortion (simulates pen wobble)

Each generated sample is a 64×192 grayscale image with black text on white bg.
"""

import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from typing import List, Tuple, Optional

# ── Image spec ──────────────────────────────────────────────────────────
IMG_HEIGHT = 64
IMG_WIDTH = 192

# ── Fonts ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FONT_PATHS: List[str] = []
_candidates = [
    os.path.join(_PROJECT_ROOT, "backend", "static", "fonts", "NotoSansDevanagari-Regular.ttf"),
    r"C:\Windows\Fonts\Nirmala.ttf",
    r"C:\Windows\Fonts\NirmalaB.ttf",
    r"C:\Windows\Fonts\NirmalaS.ttf",
]
for p in _candidates:
    if os.path.isfile(p):
        FONT_PATHS.append(p)

if not FONT_PATHS:
    raise RuntimeError("No Devanagari font found. Install Nirmala or NotoSans Devanagari.")


# ── Augmentation helpers ────────────────────────────────────────────────

def _random_font(size_range: Tuple[int, int] = (28, 44)) -> ImageFont.FreeTypeFont:
    """Pick a random font at a random size."""
    path = random.choice(FONT_PATHS)
    size = random.randint(*size_range)
    return ImageFont.truetype(path, size)


def _add_gaussian_noise(img: np.ndarray, std: float = 8.0) -> np.ndarray:
    """Add Gaussian noise to a uint8 grayscale image."""
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def _elastic_distort(img: np.ndarray, alpha: float = 4.0, sigma: float = 3.0) -> np.ndarray:
    """Light elastic distortion to mimic pen wobble."""
    from PIL import Image as _PILImage, ImageFilter as _IF
    h, w = img.shape
    # Random displacement fields
    dx = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
    dy = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
    # Smooth with Gaussian
    dx_img = _PILImage.fromarray((dx * 128 + 128).astype(np.uint8))
    dy_img = _PILImage.fromarray((dy * 128 + 128).astype(np.uint8))
    dx_img = dx_img.filter(_IF.GaussianBlur(radius=sigma))
    dy_img = dy_img.filter(_IF.GaussianBlur(radius=sigma))
    dx = (np.array(dx_img, dtype=np.float32) - 128) / 128 * alpha
    dy = (np.array(dy_img, dtype=np.float32) - 128) / 128 * alpha

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    map_x = np.clip(x_coords + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y_coords + dy, 0, h - 1).astype(np.float32)

    # Nearest-neighbor remap
    out = img[map_y.astype(int), map_x.astype(int)]
    return out


def _random_erode_dilate(img: np.ndarray) -> np.ndarray:
    """Randomly thicken or thin strokes using simple morphological ops."""
    from PIL import Image as _PILImage, ImageFilter as _IF
    pil = _PILImage.fromarray(img)
    choice = random.random()
    if choice < 0.3:
        # Dilate (thicken strokes)
        pil = pil.filter(_IF.MinFilter(3))
    elif choice < 0.5:
        # Erode (thin strokes)
        pil = pil.filter(_IF.MaxFilter(3))
    return np.array(pil)


def _add_stroke_variation(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont,
                          x: int, y: int):
    """Draw text with slight offset copies to simulate variable pen pressure."""
    # Main stroke
    draw.text((x, y), text, fill=0, font=font)
    # Sometimes add a slightly offset copy (bolder effect)
    if random.random() < 0.4:
        offset = random.choice([(1, 0), (0, 1), (1, 1)])
        draw.text((x + offset[0], y + offset[1]), text, fill=0, font=font)


# ── Main rendering function ────────────────────────────────────────────

def render_word(word: str,
                size: Tuple[int, int] = (IMG_WIDTH, IMG_HEIGHT),
                augment: bool = True) -> Image.Image:
    """
    Render a single Nepali word as a grayscale image.

    Args:
        word: The Nepali word to render.
        size: (width, height) of the output image.
        augment: Whether to apply random augmentations.

    Returns:
        PIL Image (mode='L', white background, black text).
    """
    w, h = size

    # Pick font
    font_size_range = (26, 42) if augment else (32, 36)
    font = _random_font(font_size_range)

    # Measure text bounding box
    dummy = Image.new('L', (1, 1), 255)
    dummy_draw = ImageDraw.Draw(dummy)
    bbox = dummy_draw.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # If text is wider than canvas, shrink font
    if tw > w - 10:
        ratio = (w - 10) / tw
        new_size = max(16, int(font.size * ratio))
        font = ImageFont.truetype(font.path, new_size)
        bbox = dummy_draw.textbbox((0, 0), word, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Create larger canvas and center text (we'll crop/resize later)
    canvas_w = max(tw + 40, w)
    canvas_h = max(th + 30, h)
    img = Image.new('L', (canvas_w, canvas_h), 255)
    draw = ImageDraw.Draw(img)

    # Center the text with random offset if augmenting
    cx = (canvas_w - tw) // 2
    cy = (canvas_h - th) // 2
    if augment:
        cx += random.randint(-8, 8)
        cy += random.randint(-4, 4)

    _add_stroke_variation(draw, word, font, cx - bbox[0], cy - bbox[1])

    # ── Augmentations ───────────────────────────────────────────────
    if augment:
        # Random rotation (-8 to +8 degrees)
        angle = random.uniform(-8, 8)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255, expand=False)

        # Random affine shear
        if random.random() < 0.4:
            shear = random.uniform(-0.15, 0.15)
            img = img.transform(
                img.size, Image.AFFINE,
                (1, shear, -shear * canvas_h / 2, 0, 1, 0),
                resample=Image.BILINEAR, fillcolor=255
            )

    # Convert to numpy for pixel-level augmentation
    arr = np.array(img)

    if augment:
        # Morphological variation
        arr = _random_erode_dilate(arr)
        # Elastic distortion
        if random.random() < 0.5:
            arr = _elastic_distort(arr, alpha=random.uniform(2, 6), sigma=random.uniform(2, 4))
        # Gaussian noise
        if random.random() < 0.6:
            arr = _add_gaussian_noise(arr, std=random.uniform(3, 12))

    # ── Crop to content bounding box ────────────────────────────────
    dark_mask = arr < 200
    rows = np.any(dark_mask, axis=1)
    cols = np.any(dark_mask, axis=0)
    if np.any(rows) and np.any(cols):
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        margin = 4
        rmin = max(0, rmin - margin)
        rmax = min(arr.shape[0] - 1, rmax + margin)
        cmin = max(0, cmin - margin)
        cmax = min(arr.shape[1] - 1, cmax + margin)
        arr = arr[rmin:rmax + 1, cmin:cmax + 1]

    # ── Resize to target keeping aspect ratio, pad with white ───────
    crop_img = Image.fromarray(arr)
    crop_w, crop_h = crop_img.size

    # Scale to fit within (w, h) maintaining aspect ratio
    scale = min(w / crop_w, h / crop_h) * random.uniform(0.85, 0.98) if augment else min(w / crop_w, h / crop_h) * 0.9
    new_w = max(1, int(crop_w * scale))
    new_h = max(1, int(crop_h * scale))
    crop_img = crop_img.resize((new_w, new_h), Image.LANCZOS)

    # Paste onto white target canvas with random offset
    out = Image.new('L', (w, h), 255)
    ox = (w - new_w) // 2
    oy = (h - new_h) // 2
    if augment:
        ox += random.randint(-4, 4)
        oy += random.randint(-2, 2)
    ox = max(0, min(ox, w - new_w))
    oy = max(0, min(oy, h - new_h))
    out.paste(crop_img, (ox, oy))

    # Final slight blur to smooth anti-aliasing artifacts
    if augment and random.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(radius=0.5))

    return out


# ── Dataset generation ──────────────────────────────────────────────────

def generate_dataset(vocab: List[str],
                     samples_per_word: int = 50,
                     output_dir: Optional[str] = None,
                     img_size: Tuple[int, int] = (IMG_WIDTH, IMG_HEIGHT),
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a full synthetic dataset of word images.

    Args:
        vocab: List of words (index = class label).
        samples_per_word: Number of augmented samples per word.
        output_dir: If set, also save images to disk for inspection.
        img_size: (width, height).

    Returns:
        (images, labels) — numpy arrays ready for DataLoader.
        images: float32, shape (N, 1, H, W), normalized [0, 1].
        labels: int64, shape (N,).
    """
    all_images = []
    all_labels = []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for idx, word in enumerate(vocab):
        if word == "<unknown>":
            continue  # Skip the unknown token in training

        for s in range(samples_per_word):
            img = render_word(word, size=img_size, augment=True)
            arr = np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]
            all_images.append(arr)
            all_labels.append(idx)

            if output_dir and s < 3:  # Save first 3 samples for visual check
                word_dir = os.path.join(output_dir, f"{idx:03d}_{word}")
                os.makedirs(word_dir, exist_ok=True)
                img.save(os.path.join(word_dir, f"sample_{s}.png"))

    images = np.array(all_images, dtype=np.float32)[:, np.newaxis, :, :]  # (N, 1, H, W)
    labels = np.array(all_labels, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(labels))
    images = images[perm]
    labels = labels[perm]

    return images, labels


if __name__ == "__main__":
    from vocab import VOCAB
    print(f"Generating dataset for {len(VOCAB)} words …")
    imgs, lbls = generate_dataset(VOCAB, samples_per_word=3,
                                  output_dir=os.path.join(_PROJECT_ROOT, "handwriting_recognition", "cnn_model", "sample_output"))
    print(f"Generated {len(lbls)} samples.  images shape: {imgs.shape}")
