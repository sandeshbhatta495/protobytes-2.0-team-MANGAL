"""
BiLSTM + CTC Model Architecture for Handwriting Recognition.

This module defines a lightweight neural network optimized for:
- Browser deployment (< 10MB)
- Low latency inference (< 100ms)
- Multi-script support (Nepali Devanagari + English)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from config import MODEL_CONFIG, VOCAB_SIZE


def create_bilstm_ctc_model(
    input_dim: int = 3,
    hidden_units: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    vocab_size: int = None,
    max_seq_length: int = 256,
) -> keras.Model:
    """
    Create a BiLSTM + CTC model for handwriting recognition.
    
    Architecture:
    1. Input: (batch, time_steps, 3) - normalized (Δx, Δy, pen_state)
    2. Dense projection layer
    3. Stacked Bidirectional LSTM layers with dropout
    4. Dense output layer with softmax
    5. CTC loss computed externally
    
    Args:
        input_dim: Input feature dimension (default: 3 for Δx, Δy, pen_state)
        hidden_units: LSTM hidden units per direction
        num_layers: Number of BiLSTM layers
        dropout: Dropout rate
        vocab_size: Output vocabulary size (including CTC blank)
        max_seq_length: Maximum input sequence length
    
    Returns:
        Keras Model for training and inference
    """
    if vocab_size is None:
        vocab_size = VOCAB_SIZE
    
    # Input layer: stroke sequence (Δx, Δy, pen_state)
    inputs = layers.Input(
        shape=(max_seq_length, input_dim),
        name="stroke_input"
    )
    
    # Input projection - expand features before LSTM
    x = layers.Dense(64, activation="relu", name="input_projection")(inputs)
    x = layers.LayerNormalization(name="input_norm")(x)
    x = layers.Dropout(dropout)(x)
    
    # Stacked Bidirectional LSTM layers
    for i in range(num_layers):
        return_sequences = True  # Always return sequences for CTC
        
        # Use CuDNN-compatible LSTM for speed
        lstm = layers.LSTM(
            hidden_units,
            return_sequences=return_sequences,
            dropout=dropout if i < num_layers - 1 else 0,
            recurrent_dropout=0,  # Must be 0 for CuDNN
            name=f"lstm_{i}"
        )
        
        x = layers.Bidirectional(
            lstm,
            merge_mode="concat",
            name=f"bilstm_{i}"
        )(x)
        
        # Add layer normalization between LSTM layers
        if i < num_layers - 1:
            x = layers.LayerNormalization(name=f"norm_{i}")(x)
    
    # Output projection
    x = layers.Dense(hidden_units, activation="relu", name="output_projection")(x)
    x = layers.Dropout(dropout)(x)
    
    # Final output layer (vocabulary size including blank)
    outputs = layers.Dense(
        vocab_size,
        activation="softmax",
        name="output"
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="handwriting_bilstm_ctc")
    
    return model


class CTCLayer(layers.Layer):
    """
    Custom CTC loss layer for training.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost
    
    def call(self, y_true, y_pred, input_length, label_length):
        """
        Compute CTC loss.
        
        Args:
            y_true: Ground truth labels (batch, max_label_length)
            y_pred: Model predictions (batch, time_steps, vocab_size)
            input_length: Actual sequence lengths (batch, 1)
            label_length: Actual label lengths (batch, 1)
        
        Returns:
            CTC loss value
        """
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


def create_training_model(
    base_model: keras.Model,
    max_label_length: int = 64
) -> keras.Model:
    """
    Create a training model with CTC loss layer.
    
    Args:
        base_model: Base BiLSTM model
        max_label_length: Maximum label sequence length
    
    Returns:
        Training model with CTC loss
    """
    # Get input shape from base model
    max_seq_length = base_model.input_shape[1]
    input_dim = base_model.input_shape[2]
    
    # Inputs
    stroke_input = layers.Input(
        shape=(max_seq_length, input_dim),
        name="stroke_input"
    )
    labels = layers.Input(
        shape=(max_label_length,),
        dtype="int32",
        name="labels"
    )
    input_length = layers.Input(
        shape=(1,),
        dtype="int32",
        name="input_length"
    )
    label_length = layers.Input(
        shape=(1,),
        dtype="int32",
        name="label_length"
    )
    
    # Get base model output
    y_pred = base_model(stroke_input)
    
    # CTC loss layer
    ctc_layer = CTCLayer(name="ctc_loss")
    output = ctc_layer(labels, y_pred, input_length, label_length)
    
    # Create training model
    training_model = keras.Model(
        inputs=[stroke_input, labels, input_length, label_length],
        outputs=output,
        name="training_model"
    )
    
    return training_model


def create_inference_model(base_model: keras.Model) -> keras.Model:
    """
    Create an inference model (without CTC loss layer).
    This is the model that will be converted to TensorFlow.js.
    
    Args:
        base_model: Base BiLSTM model
    
    Returns:
        Inference model
    """
    # For inference, we just use the base model directly
    # The output is (batch, time_steps, vocab_size) softmax probabilities
    return base_model


def estimate_model_size(model: keras.Model) -> dict:
    """
    Estimate model size in MB.
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary with size estimates
    """
    # Count parameters
    trainable = sum(
        tf.reduce_prod(var.shape).numpy()
        for var in model.trainable_variables
    )
    non_trainable = sum(
        tf.reduce_prod(var.shape).numpy()
        for var in model.non_trainable_variables
    )
    total = trainable + non_trainable
    
    # Estimate size (4 bytes per float32, 1 byte per uint8 if quantized)
    size_float32_mb = (total * 4) / (1024 * 1024)
    size_uint8_mb = (total * 1) / (1024 * 1024)
    
    return {
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "total_params": total,
        "size_float32_mb": round(size_float32_mb, 2),
        "size_uint8_quantized_mb": round(size_uint8_mb, 2),
    }


def build_model_from_config():
    """
    Build model using configuration from config.py.
    
    Returns:
        Tuple of (base_model, training_model, inference_model)
    """
    config = MODEL_CONFIG
    
    base_model = create_bilstm_ctc_model(
        input_dim=config["input_dim"],
        hidden_units=config["hidden_units"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        vocab_size=config["vocab_size"],
        max_seq_length=config["max_seq_length"],
    )
    
    training_model = create_training_model(
        base_model,
        max_label_length=config["max_label_length"]
    )
    
    inference_model = create_inference_model(base_model)
    
    return base_model, training_model, inference_model


if __name__ == "__main__":
    # Test model creation
    print("Creating models...")
    base_model, training_model, inference_model = build_model_from_config()
    
    print("\n=== Base Model Summary ===")
    base_model.summary()
    
    print("\n=== Model Size Estimate ===")
    size_info = estimate_model_size(base_model)
    for key, value in size_info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    import numpy as np
    batch_size = 2
    test_input = np.random.randn(batch_size, 256, 3).astype(np.float32)
    output = inference_model.predict(test_input, verbose=0)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sum per timestep (should be ~1.0): {output[0, 0, :].sum():.4f}")
