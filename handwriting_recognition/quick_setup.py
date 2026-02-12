"""
Quick setup script to generate a test model for development.
This creates a small untrained model for testing the browser pipeline.

Usage:
    python quick_setup.py
"""

import os
import sys
import json

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("Handwriting Recognition - Quick Setup")
    print("=" * 60)
    
    # Check dependencies
    print("\n[1/4] Checking dependencies...")
    try:
        import tensorflow as tf
        print(f"  TensorFlow: {tf.__version__}")
    except ImportError:
        print("  ERROR: TensorFlow not found. Install with: pip install tensorflow")
        sys.exit(1)
    
    try:
        import tensorflowjs as tfjs
        print(f"  TensorFlow.js converter: Available")
    except ImportError:
        print("  WARNING: tensorflowjs not found. Install with: pip install tensorflowjs")
        print("  Skipping TF.js conversion for now.")
        tfjs = None
    
    # Create model
    print("\n[2/4] Creating test model...")
    from model.architecture import build_model_from_config, estimate_model_size
    from model.config import VOCAB, MODEL_CONFIG
    
    base_model, _, _ = build_model_from_config()
    
    size_info = estimate_model_size(base_model)
    print(f"  Model created")
    print(f"  Parameters: {size_info['total_params']:,}")
    print(f"  Size (float32): {size_info['size_float32_mb']:.2f} MB")
    
    # Save model
    print("\n[3/4] Saving model...")
    os.makedirs("saved_models", exist_ok=True)
    
    # Save Keras model
    keras_path = "saved_models/test_model.h5"
    base_model.save(keras_path)
    print(f"  Saved: {keras_path}")
    
    # Save SavedModel format
    saved_model_path = "saved_models/test_saved_model"
    base_model.save(saved_model_path, save_format="tf")
    print(f"  Saved: {saved_model_path}")
    
    # Convert to TF.js
    if tfjs:
        print("\n[4/4] Converting to TensorFlow.js...")
        os.makedirs("tfjs_model", exist_ok=True)
        
        tfjs.converters.save_keras_model(base_model, "tfjs_model")
        
        # Save vocabulary
        vocab_data = {
            "vocab": VOCAB,
            "vocab_size": len(VOCAB),
            "idx_to_char": {str(i): c for i, c in enumerate(VOCAB)},
            "blank_index": 0,
        }
        with open("tfjs_model/vocab.json", 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        # Save config
        config_data = {
            "max_seq_length": MODEL_CONFIG["max_seq_length"],
            "max_label_length": MODEL_CONFIG["max_label_length"],
            "input_dim": MODEL_CONFIG["input_dim"],
            "vocab_size": len(VOCAB),
        }
        with open("tfjs_model/config.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        # List output files
        print("  Output files:")
        total_size = 0
        for filename in sorted(os.listdir("tfjs_model")):
            filepath = os.path.join("tfjs_model", filename)
            size = os.path.getsize(filepath)
            total_size += size
            print(f"    {filename}: {size/1024:.1f} KB")
        print(f"  Total: {total_size/1024/1024:.2f} MB")
    else:
        print("\n[4/4] Skipping TF.js conversion (tensorflowjs not installed)")
    
    # Instructions
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("""
Next steps:

1. Copy model to backend static folder:
   mkdir -p ../backend/static/handwriting_model
   cp -r tfjs_model/* ../backend/static/handwriting_model/
   
   mkdir -p ../backend/static/handwriting
   cp -r browser/* ../backend/static/handwriting/

2. Train with real data for accuracy:
   python model/train.py --data_dir data/ --epochs 50

3. Re-convert trained model:
   python model/convert_to_tfjs.py --model_path saved_models/best_model.h5

Note: The test model is UNTRAINED and will produce random results.
      You need to train it with real data for actual handwriting recognition.
""")


if __name__ == "__main__":
    main()
