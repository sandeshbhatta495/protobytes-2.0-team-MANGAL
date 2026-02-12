# Dataset Preparation Guide

This guide explains how to prepare training data for the handwriting recognition model.

## Data Format

Training data should be in JSON format with the following structure:

```json
[
    {
        "strokes": [
            [[x1, y1, pen1], [x2, y2, pen2], ...],
            [[x3, y3, pen3], [x4, y4, pen4], ...],
            ...
        ],
        "label": "text label in Nepali/English"
    },
    ...
]
```

### Stroke Format

Each stroke is a list of points where:
- `x`: X coordinate (pixel position)
- `y`: Y coordinate (pixel position)
- `pen`: Pen state (1 = pen down/drawing, 0 = pen up/stroke end)

### Example

```json
{
    "strokes": [
        [[10, 50, 1], [15, 55, 1], [20, 60, 1], [25, 55, 0]],
        [[30, 50, 1], [35, 55, 1], [40, 50, 0]]
    ],
    "label": "क"
}
```

## Data Collection Methods

### Method 1: Canvas Collection Tool

Use the included HTML tool to collect handwriting samples:

```html
<!-- See tools/data_collector.html -->
```

### Method 2: Convert Existing Datasets

If you have handwriting data in other formats (images, IAM format, etc.), convert to stroke format:

```python
from model.dataset import convert_image_to_strokes

# For traced images
strokes = convert_image_to_strokes("sample.png")
```

### Method 3: Use Public Datasets

Several public datasets can be converted:

1. **IAM Handwriting Database**: English handwriting
2. **RIMES**: French handwriting (similar stroke patterns)
3. **Custom Nepali datasets**: Contact local research institutions

## Dataset Structure

```
data/
├── train.json          # Training data (80%)
├── val.json            # Validation data (10%)
├── test.json           # Test data (10%)
└── synthetic/          # Generated synthetic data
    ├── nepali_chars/
    └── english_chars/
```

## Synthetic Data Generation

For initial training, generate synthetic data:

```bash
python tools/generate_synthetic.py --output data/synthetic/train.json --samples 10000
```

This creates simple stroke patterns for each character.

## Data Augmentation

The training pipeline automatically applies:
- Random rotation (±10°)
- Random scaling (90-110%)
- Random translation
- Gaussian noise

## Quality Requirements

For best results:
- Minimum 100 samples per character
- Diverse handwriting styles
- Include common ligatures for Nepali
- Balance between characters

## Character Coverage

Ensure your dataset covers:

| Category | Characters | Recommended Samples |
|----------|-----------|---------------------|
| Nepali vowels | अ आ इ ई उ ऊ ए ऐ ओ औ | 500+ each |
| Nepali consonants | क-ह (33) | 500+ each |
| Nepali digits | ०-९ | 300+ each |
| English letters | a-z, A-Z | 200+ each |
| English digits | 0-9 | 200+ each |
| Common words | नाम, ठेगाना, etc. | 100+ each |

## Validation

Check your dataset before training:

```bash
python tools/validate_dataset.py data/train.json

# Output:
# Total samples: 15000
# Unique characters: 118
# Character distribution: [histogram]
# Average stroke length: 45.2
# Warnings: 0
```

## Tips

1. **Normalize canvas size**: Use consistent canvas dimensions (400x200 recommended)
2. **Clean strokes**: Remove very short strokes (< 3 points)
3. **Label accuracy**: Double-check labels for accuracy
4. **Diverse writers**: Collect from multiple people for robustness
