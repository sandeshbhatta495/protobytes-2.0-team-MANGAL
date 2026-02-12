"""
Dataset loading and preprocessing for handwriting recognition.

Handles:
- Loading stroke data from various formats
- Stroke normalization and resampling
- Data augmentation
- Creating TensorFlow datasets
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
import tensorflow as tf
from config import (
    PREPROCESSING_CONFIG,
    TRAINING_CONFIG,
    MODEL_CONFIG,
    encode_text,
    CHAR_TO_IDX,
)


class StrokeProcessor:
    """
    Process raw strokes into normalized sequences for model input.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or PREPROCESSING_CONFIG
        self.canvas_width = self.config["canvas_width"]
        self.canvas_height = self.config["canvas_height"]
    
    def normalize_coordinates(self, strokes: List[List[Tuple[float, float, int]]]) -> np.ndarray:
        """
        Normalize stroke coordinates to [-1, 1] range.
        
        Args:
            strokes: List of strokes, each stroke is list of (x, y, pen_state)
        
        Returns:
            Normalized coordinates as numpy array
        """
        # Flatten all points
        all_points = []
        for stroke in strokes:
            for point in stroke:
                all_points.append(point)
        
        if not all_points:
            return np.zeros((1, 3), dtype=np.float32)
        
        points = np.array(all_points, dtype=np.float32)
        
        # Get bounding box
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Add small epsilon to avoid division by zero
        x_range = max(x_max - x_min, 1.0)
        y_range = max(y_max - y_min, 1.0)
        
        # Normalize to [-1, 1]
        points[:, 0] = 2.0 * (points[:, 0] - x_min) / x_range - 1.0
        points[:, 1] = 2.0 * (points[:, 1] - y_min) / y_range - 1.0
        
        return points
    
    def to_delta_coordinates(self, points: np.ndarray) -> np.ndarray:
        """
        Convert absolute coordinates to delta (relative) coordinates.
        
        Args:
            points: Array of shape (N, 3) with (x, y, pen_state)
        
        Returns:
            Delta coordinates array (Δx, Δy, pen_state)
        """
        if len(points) < 2:
            return np.zeros((1, 3), dtype=np.float32)
        
        deltas = np.zeros_like(points)
        
        # First point delta is 0
        deltas[0, :2] = 0
        deltas[0, 2] = points[0, 2]
        
        # Compute deltas for remaining points
        deltas[1:, :2] = points[1:, :2] - points[:-1, :2]
        deltas[1:, 2] = points[1:, 2]
        
        return deltas
    
    def resample_strokes(self, points: np.ndarray, min_distance: float = 3.0) -> np.ndarray:
        """
        Resample strokes to have uniform point spacing.
        
        Args:
            points: Array of shape (N, 3)
            min_distance: Minimum distance between points
        
        Returns:
            Resampled points array
        """
        if len(points) < 2:
            return points
        
        resampled = [points[0]]
        accumulated_dist = 0.0
        
        for i in range(1, len(points)):
            dx = points[i, 0] - points[i-1, 0]
            dy = points[i, 1] - points[i-1, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            accumulated_dist += dist
            
            # Check if pen was lifted (stroke boundary)
            if points[i, 2] == 0 and points[i-1, 2] == 1:
                resampled.append(points[i])
                accumulated_dist = 0.0
            elif accumulated_dist >= min_distance:
                resampled.append(points[i])
                accumulated_dist = 0.0
        
        # Always include the last point
        if len(resampled) > 0 and not np.array_equal(resampled[-1], points[-1]):
            resampled.append(points[-1])
        
        return np.array(resampled, dtype=np.float32)
    
    def process_strokes(self, strokes: List[List[Tuple[float, float, int]]]) -> np.ndarray:
        """
        Full preprocessing pipeline for strokes.
        
        Args:
            strokes: Raw strokes from canvas
        
        Returns:
            Processed feature sequence (N, 3) with (Δx, Δy, pen_state)
        """
        # Normalize coordinates
        points = self.normalize_coordinates(strokes)
        
        # Resample for uniform spacing
        if self.config.get("resample_points", True):
            points = self.resample_strokes(
                points,
                self.config.get("resample_distance", 3.0)
            )
        
        # Convert to delta coordinates
        if self.config.get("use_delta", True):
            points = self.to_delta_coordinates(points)
        
        return points.astype(np.float32)


class DataAugmenter:
    """
    Data augmentation for handwriting stroke data.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or PREPROCESSING_CONFIG.get("augmentation", {})
    
    def augment(self, points: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to stroke data.
        
        Args:
            points: Input points (N, 3)
        
        Returns:
            Augmented points
        """
        points = points.copy()
        
        # Random rotation
        rotation_range = self.config.get("rotation_range", 10)
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            angle_rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            x, y = points[:, 0], points[:, 1]
            points[:, 0] = x * cos_a - y * sin_a
            points[:, 1] = x * sin_a + y * cos_a
        
        # Random scaling
        scale_range = self.config.get("scale_range", (0.9, 1.1))
        if scale_range:
            scale = np.random.uniform(scale_range[0], scale_range[1])
            points[:, :2] *= scale
        
        # Random translation (for absolute coordinates)
        translation_range = self.config.get("translation_range", 0.1)
        if translation_range > 0:
            tx = np.random.uniform(-translation_range, translation_range)
            ty = np.random.uniform(-translation_range, translation_range)
            points[:, 0] += tx
            points[:, 1] += ty
        
        # Add Gaussian noise
        noise_std = self.config.get("noise_std", 0.02)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, points[:, :2].shape)
            points[:, :2] += noise.astype(np.float32)
        
        return points


class HandwritingDataset:
    """
    Dataset class for loading and preparing handwriting data.
    """
    
    def __init__(
        self,
        data_dir: str = None,
        max_seq_length: int = None,
        max_label_length: int = None,
        augment: bool = True
    ):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length or MODEL_CONFIG["max_seq_length"]
        self.max_label_length = max_label_length or MODEL_CONFIG["max_label_length"]
        self.augment = augment
        
        self.processor = StrokeProcessor()
        self.augmenter = DataAugmenter() if augment else None
        
        self.samples = []  # List of (strokes, label) tuples
    
    def load_from_json(self, json_path: str):
        """
        Load dataset from JSON file.
        
        Expected format:
        [
            {
                "strokes": [[[x1, y1, pen1], [x2, y2, pen2], ...], ...],
                "label": "text label"
            },
            ...
        ]
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            strokes = item["strokes"]
            label = item["label"]
            self.samples.append((strokes, label))
        
        print(f"Loaded {len(self.samples)} samples from {json_path}")
    
    def load_synthetic_samples(self, num_samples: int = 1000):
        """
        Generate synthetic training samples for testing.
        This is useful for validating the pipeline before real data is available.
        """
        from config import CHAR_SET
        
        print(f"Generating {num_samples} synthetic samples...")
        
        for _ in range(num_samples):
            # Random label (1-5 characters)
            label_len = np.random.randint(1, 6)
            label = "".join(np.random.choice(CHAR_SET[:20], label_len))
            
            # Generate synthetic strokes
            strokes = self._generate_synthetic_strokes(label)
            self.samples.append((strokes, label))
        
        print(f"Generated {len(self.samples)} synthetic samples")
    
    def _generate_synthetic_strokes(self, label: str) -> List[List[Tuple[float, float, int]]]:
        """Generate simple synthetic strokes for a label."""
        strokes = []
        x_offset = 0
        
        for char in label:
            # Simple stroke pattern per character
            stroke = []
            num_points = np.random.randint(10, 30)
            
            for i in range(num_points):
                t = i / max(num_points - 1, 1)
                # Simple curve pattern
                x = x_offset + t * 30 + np.random.normal(0, 2)
                y = 50 + 30 * np.sin(t * np.pi) + np.random.normal(0, 2)
                pen = 1 if i < num_points - 1 else 0
                stroke.append((x, y, pen))
            
            strokes.append(stroke)
            x_offset += 40
        
        return strokes
    
    def _pad_sequence(self, seq: np.ndarray, max_len: int, pad_value: float = 0.0) -> np.ndarray:
        """Pad or truncate sequence to fixed length."""
        if len(seq) >= max_len:
            return seq[:max_len]
        
        padding = np.full((max_len - len(seq), seq.shape[1]), pad_value, dtype=np.float32)
        return np.concatenate([seq, padding], axis=0)
    
    def _pad_label(self, label_indices: List[int], max_len: int, pad_value: int = 0) -> np.ndarray:
        """Pad label indices to fixed length."""
        padded = np.full(max_len, pad_value, dtype=np.int32)
        length = min(len(label_indices), max_len)
        padded[:length] = label_indices[:length]
        return padded
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Get a single processed sample.
        
        Returns:
            (stroke_features, label_indices, input_length, label_length)
        """
        strokes, label = self.samples[idx]
        
        # Process strokes
        features = self.processor.process_strokes(strokes)
        
        # Apply augmentation during training
        if self.augment and self.augmenter:
            features = self.augmenter.augment(features)
        
        # Record actual lengths before padding
        input_length = min(len(features), self.max_seq_length)
        label_length = min(len(label), self.max_label_length)
        
        # Pad sequences
        features = self._pad_sequence(features, self.max_seq_length)
        
        # Encode label
        label_indices = encode_text(label)
        label_indices = self._pad_label(label_indices, self.max_label_length)
        
        return features, label_indices, input_length, label_length
    
    def create_tf_dataset(self, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset for training/evaluation.
        
        Returns:
            tf.data.Dataset yielding batches
        """
        def generator():
            indices = list(range(len(self.samples)))
            if shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                features, labels, input_len, label_len = self[idx]
                yield (
                    {
                        "stroke_input": features,
                        "labels": labels,
                        "input_length": np.array([input_len], dtype=np.int32),
                        "label_length": np.array([label_len], dtype=np.int32),
                    },
                    np.zeros((1,), dtype=np.float32)  # Dummy target (loss computed in layer)
                )
        
        output_signature = (
            {
                "stroke_input": tf.TensorSpec(shape=(self.max_seq_length, 3), dtype=tf.float32),
                "labels": tf.TensorSpec(shape=(self.max_label_length,), dtype=tf.int32),
                "input_length": tf.TensorSpec(shape=(1,), dtype=tf.int32),
                "label_length": tf.TensorSpec(shape=(1,), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def prepare_datasets(
    data_dir: str = None,
    validation_split: float = 0.15,
    batch_size: int = 32,
    use_synthetic: bool = False,
    num_synthetic: int = 5000
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepare training and validation datasets.
    
    Args:
        data_dir: Directory containing data files
        validation_split: Fraction of data for validation
        batch_size: Batch size
        use_synthetic: Whether to use synthetic data
        num_synthetic: Number of synthetic samples to generate
    
    Returns:
        (train_dataset, val_dataset)
    """
    # Create dataset instances
    train_dataset = HandwritingDataset(data_dir=data_dir, augment=True)
    val_dataset = HandwritingDataset(data_dir=data_dir, augment=False)
    
    # Load or generate data
    if use_synthetic:
        train_dataset.load_synthetic_samples(num_synthetic)
    elif data_dir:
        train_json = os.path.join(data_dir, "train.json")
        if os.path.exists(train_json):
            train_dataset.load_from_json(train_json)
    
    # Split data
    all_samples = train_dataset.samples
    np.random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * (1 - validation_split))
    train_dataset.samples = all_samples[:split_idx]
    val_dataset.samples = all_samples[split_idx:]
    
    print(f"Training samples: {len(train_dataset.samples)}")
    print(f"Validation samples: {len(val_dataset.samples)}")
    
    # Create TensorFlow datasets
    train_tf = train_dataset.create_tf_dataset(batch_size, shuffle=True)
    val_tf = val_dataset.create_tf_dataset(batch_size, shuffle=False)
    
    return train_tf, val_tf


if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset module...")
    
    # Create dataset with synthetic data
    dataset = HandwritingDataset()
    dataset.load_synthetic_samples(100)
    
    # Test single sample
    features, labels, input_len, label_len = dataset[0]
    print(f"\nSingle sample:")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Input length: {input_len}")
    print(f"  Label length: {label_len}")
    
    # Test TensorFlow dataset
    tf_dataset = dataset.create_tf_dataset(batch_size=4)
    for batch in tf_dataset.take(1):
        inputs, targets = batch
        print(f"\nBatch shapes:")
        print(f"  stroke_input: {inputs['stroke_input'].shape}")
        print(f"  labels: {inputs['labels'].shape}")
        print(f"  input_length: {inputs['input_length'].shape}")
        print(f"  label_length: {inputs['label_length'].shape}")
