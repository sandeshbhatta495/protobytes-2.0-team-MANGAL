# Offline Handwriting Recognition for Sarkari Sarathi

A fully offline, browser-based handwriting recognition system supporting Nepali (Devanagari), English letters, digits, and spaces.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (TensorFlow.js)                  │
├─────────────────────────────────────────────────────────────┤
│  Canvas Input → Stroke Capture → Normalization → BiLSTM+CTC │
│                                                             │
│  ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌────────────┐ │
│  │ Canvas  │ → │ Strokes  │ → │ Δx, Δy  │ → │ Recognition│ │
│  │  Input  │   │(x,y,pen) │   │Sequence │   │   Result   │ │
│  └─────────┘   └──────────┘   └─────────┘   └────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Fully Offline**: No server/API calls required
- **Lightweight**: < 10MB model size
- **Fast**: < 100ms inference latency
- **Multi-script**: Nepali Devanagari + English + Digits + Space

## Directory Structure

```
handwriting_recognition/
├── README.md                    # This file
├── model/
│   ├── architecture.py          # BiLSTM + CTC model definition
│   ├── train.py                 # Training pipeline
│   ├── dataset.py               # Data loading and preprocessing
│   ├── convert_to_tfjs.py       # TensorFlow.js conversion
│   └── config.py                # Model configuration
├── tfjs_model/                  # Converted TF.js model (generated)
│   ├── model.json
│   └── group1-shard*.bin
├── browser/
│   ├── handwriting.js           # Main inference module
│   ├── stroke_processor.js      # Stroke normalization
│   └── ctc_decoder.js           # CTC decoding (greedy + beam)
└── data/
    └── README.md                # Dataset preparation guide
```

## Quick Start

### 1. Training (Python)

```bash
cd handwriting_recognition
pip install -r requirements.txt
python model/train.py --epochs 50 --batch_size 32
```

### 2. Convert to TensorFlow.js

```bash
python model/convert_to_tfjs.py --model_path saved_models/best_model.h5
```

### 3. Browser Integration

Copy `tfjs_model/` and `browser/` to your web server, then:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
<script src="browser/handwriting.js"></script>
```

## Character Set

| Category | Characters | Count |
|----------|-----------|-------|
| Nepali Vowels | अ आ इ ई उ ऊ ए ऐ ओ औ अं अः | 12 |
| Nepali Consonants | क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह | 33 |
| Nepali Digits | ० १ २ ३ ४ ५ ६ ७ ८ ९ | 10 |
| English Letters | a-z, A-Z | 52 |
| English Digits | 0-9 | 10 |
| Special | Space, CTC Blank | 2 |
| **Total** | | **119** |

## Performance Targets

- Model Size: < 10 MB
- Inference Time: < 100ms (desktop), < 200ms (mobile)
- Accuracy: > 90% character-level, > 85% word-level

## License

MIT License - Part of Sarkari Sarathi Project
