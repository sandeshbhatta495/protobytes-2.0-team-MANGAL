"""
Training pipeline for handwriting recognition model.

Usage:
    python train.py --epochs 50 --batch_size 32
    python train.py --use_synthetic --num_synthetic 10000
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TRAINING_CONFIG, MODEL_CONFIG, decode_indices, IDX_TO_CHAR
from architecture import build_model_from_config, estimate_model_size
from dataset import prepare_datasets, HandwritingDataset


def setup_callbacks(
    checkpoint_dir: str,
    log_dir: str,
    patience: int = 10
) -> list:
    """
    Setup training callbacks.
    
    Args:
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for TensorBoard logs
        patience: Early stopping patience
    
    Returns:
        List of Keras callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        # Model checkpoint - save best model
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Save periodic checkpoints
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"checkpoint_epoch_{{epoch:03d}}.h5"),
            save_freq="epoch",
            save_weights_only=True,
            verbose=0
        ),
        
        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=TRAINING_CONFIG["lr_decay_factor"],
            patience=TRAINING_CONFIG["lr_decay_patience"],
            min_lr=1e-6,
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_dir, timestamp),
            histogram_freq=1,
            update_freq="epoch"
        ),
        
        # CSV logger
        keras.callbacks.CSVLogger(
            filename=os.path.join(log_dir, f"training_log_{timestamp}.csv"),
            separator=",",
            append=False
        )
    ]
    
    return callbacks


class CTCAccuracyCallback(keras.callbacks.Callback):
    """
    Custom callback to compute and log CTC decoding accuracy.
    """
    
    def __init__(self, validation_dataset, base_model, num_samples: int = 100):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.base_model = base_model
        self.num_samples = num_samples
    
    def on_epoch_end(self, epoch, logs=None):
        """Compute accuracy at end of epoch."""
        if logs is None:
            logs = {}
        
        correct = 0
        total = 0
        
        for batch in self.validation_dataset.take(self.num_samples // 32 + 1):
            inputs, _ = batch
            stroke_input = inputs["stroke_input"]
            labels = inputs["labels"]
            input_lengths = inputs["input_length"]
            label_lengths = inputs["label_length"]
            
            # Get predictions from base model
            predictions = self.base_model.predict(stroke_input, verbose=0)
            
            # Greedy decode
            for i in range(len(predictions)):
                pred_indices = greedy_decode(
                    predictions[i],
                    input_lengths[i, 0].numpy()
                )
                
                # Get ground truth
                true_len = label_lengths[i, 0].numpy()
                true_indices = labels[i, :true_len].numpy().tolist()
                
                # Compare
                if pred_indices == true_indices:
                    correct += 1
                total += 1
                
                if total >= self.num_samples:
                    break
            
            if total >= self.num_samples:
                break
        
        accuracy = correct / max(total, 1)
        logs["val_accuracy"] = accuracy
        print(f"\nValidation Accuracy (greedy): {accuracy:.4f} ({correct}/{total})")


def greedy_decode(predictions: np.ndarray, seq_length: int) -> list:
    """
    Greedy CTC decoding.
    
    Args:
        predictions: Model output probabilities (seq_len, vocab_size)
        seq_length: Actual sequence length
    
    Returns:
        List of decoded character indices (without blanks/duplicates)
    """
    # Get most likely class at each timestep
    best_path = np.argmax(predictions[:seq_length], axis=1)
    
    # Remove duplicates and blanks (blank = index 0)
    decoded = []
    prev_idx = -1
    
    for idx in best_path:
        if idx != prev_idx and idx != 0:  # 0 is blank
            decoded.append(idx)
        prev_idx = idx
    
    return decoded


def train_model(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    data_dir: str = None,
    use_synthetic: bool = False,
    num_synthetic: int = 5000,
    checkpoint_dir: str = None,
    log_dir: str = None,
    resume_from: str = None
):
    """
    Main training function.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        data_dir: Directory containing training data
        use_synthetic: Use synthetic data for training
        num_synthetic: Number of synthetic samples
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for logs
        resume_from: Path to checkpoint to resume from
    """
    # Set directories
    checkpoint_dir = checkpoint_dir or TRAINING_CONFIG["checkpoint_dir"]
    log_dir = log_dir or TRAINING_CONFIG["log_dir"]
    
    print("=" * 60)
    print("Handwriting Recognition Model Training")
    print("=" * 60)
    
    # Build models
    print("\n[1/5] Building models...")
    base_model, training_model, inference_model = build_model_from_config()
    
    # Print model info
    size_info = estimate_model_size(base_model)
    print(f"  Total parameters: {size_info['total_params']:,}")
    print(f"  Estimated size (float32): {size_info['size_float32_mb']:.2f} MB")
    print(f"  Estimated size (uint8): {size_info['size_uint8_quantized_mb']:.2f} MB")
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"\n  Loading weights from: {resume_from}")
        training_model.load_weights(resume_from)
    
    # Compile training model
    print("\n[2/5] Compiling model...")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    training_model.compile(optimizer=optimizer)
    
    # Prepare datasets
    print("\n[3/5] Preparing datasets...")
    train_dataset, val_dataset = prepare_datasets(
        data_dir=data_dir,
        validation_split=TRAINING_CONFIG["validation_split"],
        batch_size=batch_size,
        use_synthetic=use_synthetic,
        num_synthetic=num_synthetic
    )
    
    # Setup callbacks
    print("\n[4/5] Setting up callbacks...")
    callbacks = setup_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        patience=TRAINING_CONFIG["early_stopping_patience"]
    )
    
    # Add accuracy callback
    accuracy_callback = CTCAccuracyCallback(
        validation_dataset=val_dataset,
        base_model=base_model,
        num_samples=100
    )
    callbacks.append(accuracy_callback)
    
    # Train
    print("\n[5/5] Training...")
    print("-" * 60)
    
    history = training_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final models
    print("\n" + "=" * 60)
    print("Training complete! Saving models...")
    
    # Save training model
    training_model.save(os.path.join(checkpoint_dir, "final_training_model.h5"))
    
    # Save base model (for inference)
    base_model.save(os.path.join(checkpoint_dir, "final_inference_model.h5"))
    
    # Save in SavedModel format for TF.js conversion
    base_model.save(os.path.join(checkpoint_dir, "saved_model"), save_format="tf")
    
    print(f"  Training model: {checkpoint_dir}/final_training_model.h5")
    print(f"  Inference model: {checkpoint_dir}/final_inference_model.h5")
    print(f"  SavedModel: {checkpoint_dir}/saved_model/")
    
    # Print final metrics
    print("\n" + "=" * 60)
    print("Final Metrics:")
    print(f"  Final train loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final val loss: {history.history['val_loss'][-1]:.4f}")
    if 'val_accuracy' in history.history:
        print(f"  Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return history, base_model


def evaluate_model(model_path: str, data_dir: str = None, use_synthetic: bool = True):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        data_dir: Directory containing test data
        use_synthetic: Use synthetic data for evaluation
    """
    print("Loading model...")
    model = keras.models.load_model(model_path, compile=False)
    
    print("Preparing test data...")
    test_dataset = HandwritingDataset(augment=False)
    if use_synthetic:
        test_dataset.load_synthetic_samples(500)
    
    print("Evaluating...")
    correct_char = 0
    total_char = 0
    correct_seq = 0
    total_seq = 0
    
    for idx in range(len(test_dataset)):
        features, labels, input_len, label_len = test_dataset[idx]
        
        # Predict
        pred = model.predict(features[np.newaxis, ...], verbose=0)[0]
        pred_indices = greedy_decode(pred, input_len)
        
        # Ground truth
        true_indices = labels[:label_len].tolist()
        
        # Character-level accuracy
        for p, t in zip(pred_indices, true_indices):
            if p == t:
                correct_char += 1
            total_char += 1
        
        # Sequence accuracy
        if pred_indices == true_indices:
            correct_seq += 1
        total_seq += 1
    
    print(f"\nResults:")
    print(f"  Character accuracy: {correct_char/max(total_char,1):.4f}")
    print(f"  Sequence accuracy: {correct_seq/total_seq:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train handwriting recognition model")
    
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--num_synthetic", type=int, default=5000, help="Number of synthetic samples")
    parser.add_argument("--checkpoint_dir", type=str, default="saved_models", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--evaluate", type=str, default=None, help="Evaluate model at path")
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_model(
            model_path=args.evaluate,
            data_dir=args.data_dir,
            use_synthetic=args.use_synthetic
        )
    else:
        train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            data_dir=args.data_dir,
            use_synthetic=args.use_synthetic,
            num_synthetic=args.num_synthetic,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            resume_from=args.resume
        )


if __name__ == "__main__":
    main()
