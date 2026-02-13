"""
Lightweight CNN classifier for Nepali word recognition.

Architecture:
  Input:  1 × 64 × 192   (grayscale, height × width)
  Conv → BN → ReLU → Pool  ×3
  Global Average Pool
  FC → Dropout → FC (num_classes)
  CrossEntropy loss

Total params: ~200K–500K  →  < 2 MB saved model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NepaliWordCNN(nn.Module):
    """
    3-layer CNN for word-level classification.

    Image flows:
        1×64×192
      → Conv 32 (3×3) → BN → ReLU → MaxPool(2) → 32×32×96
      → Conv 64 (3×3) → BN → ReLU → MaxPool(2) → 64×16×48
      → Conv 128(3×3) → BN → ReLU → MaxPool(2) → 128×8×24
      → AdaptiveAvgPool(1) → 128
      → FC 128 → ReLU → Dropout → FC num_classes
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # → 128×1×1

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 64, 192)  float [0, 1]
        Returns:
            logits: (B, num_classes)
        """
        x = self.features(x)
        x = self.global_pool(x)          # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)        # (B, 128)
        x = self.classifier(x)           # (B, num_classes)
        return x

    def predict_topk(self, x: torch.Tensor, k: int = 5):
        """
        Return top-k predictions with probabilities.

        Returns:
            probs: (B, k) float
            indices: (B, k) int
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, k, dim=-1)
        return topk_probs, topk_idx


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from vocab import VOCAB_SIZE
    model = NepaliWordCNN(num_classes=VOCAB_SIZE)
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Test forward pass
    dummy = torch.randn(2, 1, 64, 192)
    out = model(dummy)
    print(f"Input: {dummy.shape}  →  Output: {out.shape}")

    probs, idx = model.predict_topk(dummy, k=3)
    print(f"Top-3 probs: {probs}")
    print(f"Top-3 idx:   {idx}")
