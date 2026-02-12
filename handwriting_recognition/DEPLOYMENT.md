# Offline Handwriting Recognition - Deployment Guide

Complete guide for deploying the offline handwriting recognition system for Sarkari Sarathi.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Training the Model](#training-the-model)
3. [Converting to TensorFlow.js](#converting-to-tensorflowjs)
4. [Deploying to Browser](#deploying-to-browser)
5. [Optimization Guide](#optimization-guide)
6. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐ │
│  │  Stroke  │ → │ Normalize │ → │ BiLSTM + │ → │   Saved      │ │
│  │   Data   │    │  & Pad   │    │   CTC    │    │   Model      │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────┘ │
│                                                        ↓           │
├────────────────────────────────────────────────────────────────────┤
│                         CONVERSION                                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐   │
│  │ Keras Model  │ → │ TensorFlow.js│ → │  model.json +      │   │
│  │   (.h5)      │    │  Converter   │    │  weight files      │   │
│  └──────────────┘    └──────────────┘    └────────────────────┘   │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│                      BROWSER INFERENCE                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌────────────────┐   │
│  │  Canvas  │→│   Stroke   │→│ TF.js    │→│ CTC Decoder    │   │
│  │  Input   │  │ Processor  │  │ Inference│  │ (Greedy/Beam)  │   │
│  └──────────┘  └────────────┘  └──────────┘  └────────────────┘   │
│       ↑                                              ↓             │
│    User                                      Recognized Text       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Model Architecture

**BiLSTM + CTC (Connectionist Temporal Classification)**

```
Input: (batch, seq_len=256, features=3)  # Δx, Δy, pen_state
  ↓
Dense(64) + LayerNorm + Dropout(0.3)
  ↓
Bidirectional LSTM (128 units) × 2 layers
  ↓
Dense(128) + Dropout(0.3)
  ↓
Dense(vocab_size, softmax)
  ↓
Output: (batch, seq_len, vocab_size=119)
```

**Estimated Model Size:**
- Float32: ~2-3 MB
- Quantized (uint8): ~0.5-1 MB

---

## Training the Model

### Prerequisites

```bash
cd handwriting_recognition
pip install -r requirements.txt
```

### Step 1: Prepare Data

See `data/README.md` for detailed data preparation instructions.

```bash
# Generate synthetic data for initial testing
python model/train.py --use_synthetic --num_synthetic 10000 --epochs 10
```

### Step 2: Train

```bash
# Train with real data
python model/train.py \
    --data_dir data/ \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001

# Resume from checkpoint
python model/train.py \
    --data_dir data/ \
    --epochs 100 \
    --resume saved_models/checkpoint_epoch_050.h5
```

### Step 3: Evaluate

```bash
python model/train.py --evaluate saved_models/best_model.h5
```

### Training Tips

1. **Start with synthetic data** to validate the pipeline
2. **Gradually add real data** for improved accuracy
3. **Monitor validation loss** - stop if overfitting
4. **Use learning rate scheduling** for better convergence

---

## Converting to TensorFlow.js

### Basic Conversion

```bash
python model/convert_to_tfjs.py \
    --model_path saved_models/best_model.h5 \
    --output_dir tfjs_model \
    --quantize
```

### Conversion Options

| Option | Description |
|--------|-------------|
| `--quantize` | Enable weight quantization (recommended) |
| `--quantization_dtype` | `uint8` (smallest), `uint16`, `float16` |
| `--no-quantize` | Keep float32 weights |

### Output Files

```
tfjs_model/
├── model.json          # Model architecture + weight manifest
├── group1-shard1of1.bin  # Weight data
├── vocab.json          # Character vocabulary
└── config.json         # Model configuration
```

---

## Deploying to Browser

### Step 1: Copy Files

Copy the model files to your web server:

```bash
# Create directory
mkdir -p backend/static/handwriting_model

# Copy model files
cp -r tfjs_model/* backend/static/handwriting_model/

# Copy browser modules
cp -r browser/* backend/static/handwriting/
```

### Step 2: Update HTML

The `frontend/index.html` already includes:

```html
<!-- TensorFlow.js -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>

<!-- Handwriting modules -->
<script src="/static/handwriting/stroke_processor.js"></script>
<script src="/static/handwriting/ctc_decoder.js"></script>
<script src="/static/handwriting/handwriting.js"></script>
<script src="offline_handwriting.js"></script>
```

### Step 3: Configure Paths

Edit `frontend/offline_handwriting.js`:

```javascript
const CONFIG = {
    modelPath: '/static/handwriting_model',  // Path to TF.js model
    useBeamSearch: true,
    beamWidth: 10,
    enableFallback: true,  // Fall back to server if model unavailable
};
```

### Step 4: Test

1. Start the backend server
2. Open browser console
3. Look for: `[OfflineHandwriting] Model loaded successfully!`
4. Try handwriting recognition

---

## Optimization Guide

### Target Performance

| Metric | Target | How to Achieve |
|--------|--------|----------------|
| Model Size | < 10 MB | Quantization, fewer layers |
| Inference | < 100ms | Smaller model, WebGL backend |
| Memory | < 50 MB | Dispose tensors, batch=1 |

### Key Optimizations

#### 1. Model Size Reduction

```python
# In model/config.py
MODEL_CONFIG = {
    "hidden_units": 128,  # Reduce from 256
    "num_layers": 2,      # Reduce from 3
}
```

#### 2. Weight Quantization

```bash
# uint8 quantization (4x smaller)
python model/convert_to_tfjs.py --quantize --quantization_dtype uint8
```

#### 3. TensorFlow.js Backend

```javascript
// Force WebGL backend for GPU acceleration
await tf.setBackend('webgl');

// Or use WASM for consistent performance
await tf.setBackend('wasm');
```

#### 4. Input Optimization

```javascript
// Limit sequence length
const MAX_POINTS = 200;  // Reduce from 256

// Sample strokes if too long
if (points.length > MAX_POINTS) {
    points = samplePoints(points, MAX_POINTS);
}
```

#### 5. Memory Management

```javascript
// Always dispose tensors
const input = tf.tensor3d(...);
const output = model.predict(input);
const result = await output.data();
input.dispose();
output.dispose();
```

### Performance Testing

```javascript
// Enable profiling
const recognizer = new HandwritingRecognizer({
    enableProfiling: true
});

// Check metrics
console.log(recognizer.getMetrics());
// Output: { loadTime: 1234, inferenceTime: 45, ... }
```

---

## Troubleshooting

### Model Loading Fails

**Error:** `Failed to load vocab.json: 404`

**Solution:** Check file paths and CORS settings:
```python
# backend/main.py - ensure static files are served
app.mount("/static", StaticFiles(directory="static"), name="static")
```

### Slow Inference

**Issue:** Inference > 100ms

**Solutions:**
1. Use WebGL backend
2. Reduce model size
3. Check for memory leaks

```javascript
// Debug: Check TF.js backend
console.log(tf.getBackend());  // Should be 'webgl'

// Check memory
console.log(tf.memory());
```

### Poor Accuracy

**Issue:** Low recognition accuracy

**Solutions:**
1. Collect more training data
2. Add data augmentation
3. Increase model capacity (more hidden units)
4. Use beam search instead of greedy decoding

### Memory Leaks

**Issue:** Memory usage grows over time

**Solution:** Ensure proper tensor disposal:
```javascript
// Wrap in tf.tidy for automatic cleanup
const result = tf.tidy(() => {
    const input = tf.tensor3d(...);
    return model.predict(input);
});
```

---

## API Reference

### HandwritingRecognizer

```javascript
const recognizer = new HandwritingRecognizer({
    useBeamSearch: true,
    beamWidth: 10,
    enableProfiling: false
});

await recognizer.init('/path/to/model');
const result = await recognizer.recognize(strokes);

// Result object:
{
    text: "recognized text",
    confidence: 0.95,
    timing: {
        preprocess: 5,
        inference: 45,
        decode: 10,
        total: 60
    }
}
```

### StrokeCapture

```javascript
const capture = new StrokeCapture(canvasElement);

// Get strokes after user draws
const strokes = capture.getStrokes();

// Clear
capture.clear();
```

### CTCDecoder

```javascript
const decoder = new CTCDecoder(vocab, {
    beamWidth: 10,
    useBeamSearch: true
});

const result = decoder.decode(probabilities, seqLength);
```

---

## File Structure Summary

```
handwriting_recognition/
├── README.md                    # Overview
├── DEPLOYMENT.md               # This file
├── requirements.txt            # Python dependencies
├── model/
│   ├── __init__.py
│   ├── config.py               # Vocabulary & model config
│   ├── architecture.py         # BiLSTM + CTC model
│   ├── dataset.py              # Data loading
│   ├── train.py                # Training pipeline
│   └── convert_to_tfjs.py      # TF.js conversion
├── browser/
│   ├── stroke_processor.js     # Stroke preprocessing
│   ├── ctc_decoder.js          # CTC decoding
│   └── handwriting.js          # Main recognizer
├── data/
│   └── README.md               # Data preparation guide
└── tfjs_model/                 # Generated (after conversion)
    ├── model.json
    ├── vocab.json
    └── config.json
```

---

## Quick Start Checklist

- [ ] Install Python requirements
- [ ] Prepare training data (or use synthetic)
- [ ] Train model: `python model/train.py`
- [ ] Convert to TF.js: `python model/convert_to_tfjs.py`
- [ ] Copy files to `backend/static/`
- [ ] Verify HTML includes TensorFlow.js
- [ ] Test in browser
- [ ] Check console for successful model load
- [ ] Test handwriting recognition

For questions or issues, check the troubleshooting section or open an issue.
