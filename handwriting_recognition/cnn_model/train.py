"""
Training script for the Nepali word CNN classifier.

Usage:
    cd handwriting_recognition/cnn_model
    python train.py                       # default: 50 samples/word, 30 epochs
    python train.py --samples 100 --epochs 50
    python train.py --save-samples        # also save sample images for inspection
"""

import os
import sys
import argparse
import time
import json
import numpy as np

# Disable PyTorch compile to avoid sympy import issues in some PyTorch versions
os.environ['PYTORCH_DISABLE_MPS_FALLBACK'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import torch
# Disable dynamo before any optimizer import
torch._dynamo.config.disable = True if hasattr(torch, '_dynamo') else None
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# Ensure cnn_model package is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from vocab import VOCAB, VOCAB_SIZE, WORD_TO_IDX, IDX_TO_WORD, UNKNOWN_TOKEN
from model import NepaliWordCNN, count_parameters
from data_generator import generate_dataset, IMG_HEIGHT, IMG_WIDTH


# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "static", "handwriting_model")
CKPT_PATH = os.path.join(MODEL_DIR, "nepali_word_cnn.pt")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Generate synthetic data ─────────────────────────────────────
    sample_dir = os.path.join(_THIS_DIR, "sample_output") if args.save_samples else None
    t0 = time.time()
    print(f"Generating synthetic data: {args.samples} samples/word × {VOCAB_SIZE - 1} words …")
    images, labels = generate_dataset(
        VOCAB,
        samples_per_word=args.samples,
        output_dir=sample_dir,
        img_size=(IMG_WIDTH, IMG_HEIGHT),
    )
    print(f"  Generated {len(labels)} total samples in {time.time() - t0:.1f}s")
    print(f"  Image tensor shape: {images.shape}")

    # ── Train / val split ───────────────────────────────────────────
    X = torch.from_numpy(images)   # (N, 1, H, W)
    Y = torch.from_numpy(labels)   # (N,)

    dataset = TensorDataset(X, Y)
    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    print(f"  Train: {train_size}  |  Val: {val_size}")

    # ── Create model ────────────────────────────────────────────────
    num_classes = VOCAB_SIZE  # includes <unknown>
    model = NepaliWordCNN(num_classes=num_classes, dropout=0.3).to(device)
    print(f"  Model params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ── Training loop ───────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # ── Validation ──────────────────────────────────────────────
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={lr_now:.6f}")

        # ── Early stopping / checkpoint ─────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'vocab_size': VOCAB_SIZE,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
                'val_acc': val_acc,
                'epoch': epoch,
            }, CKPT_PATH)
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"  Model saved to: {CKPT_PATH}")

    # ── Save vocab + meta info ──────────────────────────────────────
    meta = {
        "model_type": "cnn_word_classifier",
        "num_classes": num_classes,
        "vocab_size": VOCAB_SIZE,
        "img_height": IMG_HEIGHT,
        "img_width": IMG_WIDTH,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "vocab": VOCAB,
    }
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Meta saved to: {META_PATH}")

    # Also save updated vocab.json  
    vocab_json_path = os.path.join(MODEL_DIR, "vocab.json")
    with open(vocab_json_path, 'w', encoding='utf-8') as f:
        json.dump({"vocab": VOCAB, "word_to_idx": WORD_TO_IDX}, f, ensure_ascii=False, indent=2)
    print(f"  Vocab saved to: {vocab_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Nepali word CNN classifier")
    parser.add_argument("--samples", type=int, default=50,
                        help="Augmented samples per word (default: 50)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate (default: 0.01)")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (default: 8)")
    parser.add_argument("--save-samples", action="store_true",
                        help="Also save sample images to disk for visual inspection")
    args = parser.parse_args()
    train(args)
