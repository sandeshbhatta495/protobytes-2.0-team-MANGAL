"""
Inference module for the Nepali word CNN classifier.

Used by the backend to recognize handwritten Nepali words from canvas images.
"""

import os
import json
import numpy as np
from typing import Tuple, List, Optional
from PIL import Image
import torch
import torch.nn.functional as F

# Import model architecture
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model import NepaliWordCNN

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "static", "handwriting_model")
CKPT_PATH = os.path.join(MODEL_DIR, "nepali_word_cnn.pt")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

# ── Image spec (must match training) ────────────────────────────────────
IMG_HEIGHT = 64
IMG_WIDTH = 192


class NepaliWordRecognizer:
    """
    Inference wrapper for the trained CNN word classifier.

    Usage:
        recognizer = NepaliWordRecognizer()
        if recognizer.load():
            word, confidence, alternatives = recognizer.recognize(pil_image)
    """

    def __init__(self):
        self.model: Optional[NepaliWordCNN] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab: List[str] = []
        self.idx_to_word: dict = {}
        self.loaded = False

    def load(self) -> bool:
        """
        Load the trained model and vocab from disk.

        Returns True if successful, False otherwise.
        """
        if not os.path.isfile(CKPT_PATH):
            print(f"[NepaliWordRecognizer] Model not found: {CKPT_PATH}")
            return False
        if not os.path.isfile(META_PATH):
            print(f"[NepaliWordRecognizer] Meta not found: {META_PATH}")
            return False

        try:
            # Load metadata
            with open(META_PATH, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.vocab = meta.get("vocab", [])
            self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
            num_classes = meta.get("num_classes", len(self.vocab))

            # Load model
            ckpt = torch.load(CKPT_PATH, map_location=self.device)
            self.model = NepaliWordCNN(num_classes=num_classes)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            self.loaded = True
            print(f"[NepaliWordRecognizer] Loaded model with {num_classes} classes (val_acc={ckpt.get('val_acc', 0):.3f})")
            return True

        except Exception as e:
            print(f"[NepaliWordRecognizer] Load error: {e}")
            return False

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a PIL image for the model.

        Steps:
          1. Convert to grayscale
          2. Composite transparent background onto white
          3. Invert if needed (model expects black-on-white)
          4. Crop to content bounding box
          5. Resize to IMG_WIDTH × IMG_HEIGHT keeping aspect ratio
          6. Normalize to [0, 1]

        Args:
            image: PIL Image (any mode)

        Returns:
            Tensor of shape (1, 1, IMG_HEIGHT, IMG_WIDTH)
        """
        # Handle RGBA / transparent background
        if image.mode in ('RGBA', 'LA', 'PA'):
            # Composite onto white
            white = Image.new('RGBA', image.size, (255, 255, 255, 255))
            white.paste(image, mask=image.split()[-1])
            image = white.convert('L')
        else:
            image = image.convert('L')

        arr = np.array(image, dtype=np.float32)

        # Check if inverted (white text on dark background) and fix
        mean_intensity = arr.mean()
        if mean_intensity < 128:
            arr = 255 - arr

        # Crop to content bounding box
        dark = arr < 240
        rows = np.any(dark, axis=1)
        cols = np.any(dark, axis=0)
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            margin = 4
            rmin = max(0, rmin - margin)
            rmax = min(arr.shape[0] - 1, rmax + margin)
            cmin = max(0, cmin - margin)
            cmax = min(arr.shape[1] - 1, cmax + margin)
            arr = arr[rmin:rmax + 1, cmin:cmax + 1]

        # Resize keeping aspect ratio
        h, w = arr.shape
        scale = min(IMG_WIDTH / w, IMG_HEIGHT / h) * 0.9
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        crop_img = Image.fromarray(arr.astype(np.uint8))
        crop_img = crop_img.resize((new_w, new_h), Image.LANCZOS)

        # Paste centered onto white canvas
        out = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 255)
        ox = (IMG_WIDTH - new_w) // 2
        oy = (IMG_HEIGHT - new_h) // 2
        out.paste(crop_img, (ox, oy))

        # Normalize to [0, 1]
        arr = np.array(out, dtype=np.float32) / 255.0

        # Add batch and channel dims
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return tensor.to(self.device)

    def recognize(self, image: Image.Image, top_k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Recognize a handwritten Nepali word from an image.

        Args:
            image: PIL Image of the handwritten word.
            top_k: Number of alternative predictions to return.

        Returns:
            (best_word, confidence, alternatives)
            - best_word: The predicted word (str)
            - confidence: Probability of the best word (float)
            - alternatives: List of (word, probability) for top-k predictions
        """
        if not self.loaded or self.model is None:
            return "<unknown>", 0.0, []

        tensor = self.preprocess(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_idx = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)

        topk_probs = topk_probs[0].cpu().numpy()
        topk_idx = topk_idx[0].cpu().numpy()

        alternatives = []
        for p, i in zip(topk_probs, topk_idx):
            word = self.idx_to_word.get(int(i), "<unknown>")
            alternatives.append((word, float(p)))

        best_word = alternatives[0][0] if alternatives else "<unknown>"
        confidence = alternatives[0][1] if alternatives else 0.0

        return best_word, confidence, alternatives


# ── Singleton instance ──────────────────────────────────────────────────
_recognizer_instance: Optional[NepaliWordRecognizer] = None


def get_recognizer() -> NepaliWordRecognizer:
    """Get the singleton recognizer instance (lazy loaded)."""
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = NepaliWordRecognizer()
        _recognizer_instance.load()
    return _recognizer_instance


# ── Convenience function for backend ────────────────────────────────────
def recognize_handwriting_image(pil_image: Image.Image, top_k: int = 5) -> dict:
    """
    Recognize handwritten Nepali text from a PIL image.

    Returns a dict compatible with the backend's /recognize-handwriting response:
        {"text": "...", "confidence": 0.95, "success": True, "alternatives": [...]}
    """
    rec = get_recognizer()
    if not rec.loaded:
        return {"text": "", "confidence": 0.0, "success": False, "error": "Model not loaded"}

    try:
        word, conf, alts = rec.recognize(pil_image, top_k=top_k)
        return {
            "text": word if word != "<unknown>" else "",
            "confidence": conf,
            "success": conf > 0.3 and word != "<unknown>",
            "alternatives": [{"word": w, "prob": p} for w, p in alts],
        }
    except Exception as e:
        return {"text": "", "confidence": 0.0, "success": False, "error": str(e)}


if __name__ == "__main__":
    # Quick test with a rendered sample
    from data_generator import render_word
    from vocab import VOCAB

    recognizer = NepaliWordRecognizer()
    if recognizer.load():
        # Test with a few words
        test_words = ["राम", "शर्मा", "काठमाडौं", "विवाह", "पुरुष"]
        for w in test_words:
            img = render_word(w, augment=True)
            pred, conf, alts = recognizer.recognize(img)
            status = "✓" if pred == w else "✗"
            print(f"  {status} '{w}' → '{pred}' (conf={conf:.3f})  top3: {alts[:3]}")
    else:
        print("Model not loaded — run train.py first!")
