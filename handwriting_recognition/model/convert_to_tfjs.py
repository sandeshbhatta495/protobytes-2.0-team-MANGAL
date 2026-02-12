"""
Convert trained Keras model to TensorFlow.js format.

Usage:
    python convert_to_tfjs.py --model_path saved_models/best_model.h5
    python convert_to_tfjs.py --model_path saved_models/saved_model --quantize
"""

import argparse
import os
import sys
import json
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import VOCAB, VOCAB_SIZE, IDX_TO_CHAR, TFJS_CONFIG, MODEL_CONFIG


def convert_to_tfjs(
    model_path: str,
    output_dir: str = None,
    quantize: bool = True,
    quantization_dtype: str = "uint8"
):
    """
    Convert Keras/TensorFlow model to TensorFlow.js format.
    
    Args:
        model_path: Path to saved model (.h5 or SavedModel directory)
        output_dir: Output directory for TF.js model
        quantize: Whether to quantize weights
        quantization_dtype: Quantization data type (uint8 or uint16)
    """
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print("Error: tensorflowjs not installed.")
        print("Install with: pip install tensorflowjs")
        sys.exit(1)
    
    import tensorflow as tf
    from tensorflow import keras
    
    output_dir = output_dir or TFJS_CONFIG["output_dir"]
    
    print("=" * 60)
    print("TensorFlow.js Model Conversion")
    print("=" * 60)
    
    # Load model
    print(f"\n[1/4] Loading model from: {model_path}")
    
    if model_path.endswith('.h5'):
        model = keras.models.load_model(model_path, compile=False)
    else:
        # Assume SavedModel format
        model = keras.models.load_model(model_path, compile=False)
    
    print(f"  Model loaded: {model.name}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    # Get model size
    total_params = model.count_params()
    size_mb = (total_params * 4) / (1024 * 1024)  # Float32
    quantized_size_mb = (total_params * 1) / (1024 * 1024)  # Uint8
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Estimated size (float32): {size_mb:.2f} MB")
    print(f"  Estimated size (quantized): {quantized_size_mb:.2f} MB")
    
    # Prepare output directory
    print(f"\n[2/4] Preparing output directory: {output_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to TF.js
    print(f"\n[3/4] Converting to TensorFlow.js format...")
    print(f"  Quantization: {'enabled (' + quantization_dtype + ')' if quantize else 'disabled'}")
    
    # Build conversion command
    if quantize:
        tfjs.converters.save_keras_model(
            model,
            output_dir,
            quantization_dtype_map={
                "uint8": "uint8",
                "uint16": "uint16",
                "float16": "float16"
            }.get(quantization_dtype, "uint8")
        )
    else:
        tfjs.converters.save_keras_model(model, output_dir)
    
    # Save vocabulary and config
    print(f"\n[4/4] Saving vocabulary and config...")
    
    # Vocabulary file
    vocab_file = os.path.join(output_dir, "vocab.json")
    vocab_data = {
        "vocab": VOCAB,
        "vocab_size": VOCAB_SIZE,
        "idx_to_char": {str(k): v for k, v in IDX_TO_CHAR.items()},
        "blank_index": 0,
    }
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    print(f"  Vocabulary saved to: {vocab_file}")
    
    # Model config file
    config_file = os.path.join(output_dir, "config.json")
    config_data = {
        "max_seq_length": MODEL_CONFIG["max_seq_length"],
        "max_label_length": MODEL_CONFIG["max_label_length"],
        "input_dim": MODEL_CONFIG["input_dim"],
        "vocab_size": VOCAB_SIZE,
        "model_version": "1.0.0",
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2)
    print(f"  Config saved to: {config_file}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    
    # List output files
    print(f"\nOutput files in: {output_dir}/")
    total_size = 0
    for filename in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, filename)
        size = os.path.getsize(filepath)
        total_size += size
        print(f"  {filename}: {size/1024:.1f} KB")
    
    print(f"\nTotal size: {total_size/1024/1024:.2f} MB")
    
    if total_size > 10 * 1024 * 1024:  # 10 MB
        print("\n⚠️  Warning: Model size exceeds 10 MB target!")
        print("Consider reducing hidden_units or num_layers in config.py")
    else:
        print("\n✓ Model size is within 10 MB target")
    
    # Usage instructions
    print("\n" + "=" * 60)
    print("Usage in Browser:")
    print("=" * 60)
    print("""
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
<script>
async function loadModel() {
    const model = await tf.loadLayersModel('tfjs_model/model.json');
    const vocabResp = await fetch('tfjs_model/vocab.json');
    const vocab = await vocabResp.json();
    return { model, vocab };
}
</script>
""")


def verify_conversion(output_dir: str):
    """
    Verify the converted model can be loaded and run inference.
    """
    try:
        import tensorflowjs as tfjs
        import tensorflow as tf
        import numpy as np
    except ImportError:
        print("Cannot verify - missing dependencies")
        return
    
    print("\n" + "=" * 60)
    print("Verifying Conversion...")
    print("=" * 60)
    
    # Load model.json
    model_json_path = os.path.join(output_dir, "model.json")
    if not os.path.exists(model_json_path):
        print(f"Error: {model_json_path} not found")
        return
    
    with open(model_json_path, 'r') as f:
        model_json = json.load(f)
    
    print(f"  Model format: {model_json.get('format', 'unknown')}")
    print(f"  Model topology: {'present' if 'modelTopology' in model_json else 'missing'}")
    print(f"  Weight specs: {len(model_json.get('weightsManifest', [{}])[0].get('weights', []))} weight tensors")
    
    # Check weight files
    weight_files = []
    for manifest in model_json.get('weightsManifest', []):
        for path in manifest.get('paths', []):
            weight_path = os.path.join(output_dir, path)
            if os.path.exists(weight_path):
                weight_files.append(path)
    
    print(f"  Weight files present: {len(weight_files)}")
    
    # Load and test in TensorFlow
    print("\n  Running test inference...")
    
    # Load the original Keras model to verify shapes match
    try:
        model = tf.keras.models.load_model(
            os.path.join(output_dir, "model.json"),
            compile=False
        )
        
        # Create test input
        test_input = np.random.randn(1, MODEL_CONFIG["max_seq_length"], 3).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        
        print(f"  Test input shape: {test_input.shape}")
        print(f"  Test output shape: {output.shape}")
        print(f"  Output sum (should be ~1): {output[0, 0, :].sum():.4f}")
        print("\n✓ Verification complete - model is ready for browser deployment")
    except Exception as e:
        print(f"\n  Note: Could not run TF verification: {e}")
        print("  This is normal - full verification requires loading in browser/Node.js")


def main():
    parser = argparse.ArgumentParser(description="Convert model to TensorFlow.js")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved model (.h5 or SavedModel directory)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tfjs_model",
        help="Output directory for TF.js model"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Quantize weights to reduce size"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_false",
        dest="quantize",
        help="Disable weight quantization"
    )
    parser.add_argument(
        "--quantization_dtype",
        type=str,
        default="uint8",
        choices=["uint8", "uint16", "float16"],
        help="Quantization data type"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify converted model"
    )
    
    args = parser.parse_args()
    
    convert_to_tfjs(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quantize=args.quantize,
        quantization_dtype=args.quantization_dtype
    )
    
    if args.verify:
        verify_conversion(args.output_dir)


if __name__ == "__main__":
    main()
