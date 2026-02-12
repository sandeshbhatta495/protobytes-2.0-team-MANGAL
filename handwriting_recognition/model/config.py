"""
Configuration file for handwriting recognition model.
"""

# Character set definition
NEPALI_VOWELS = list("अआइईउऊएऐओऔअंअः")
NEPALI_CONSONANTS = list("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
NEPALI_DIGITS = list("०१२३४५६७८९")
ENGLISH_LOWER = list("abcdefghijklmnopqrstuvwxyz")
ENGLISH_UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ENGLISH_DIGITS = list("0123456789")
SPECIAL_CHARS = [" "]  # Space

# Full character set (CTC blank token will be added at training)
CHAR_SET = (
    NEPALI_VOWELS + 
    NEPALI_CONSONANTS + 
    NEPALI_DIGITS + 
    ENGLISH_LOWER + 
    ENGLISH_UPPER + 
    ENGLISH_DIGITS + 
    SPECIAL_CHARS
)

# CTC blank token (always index 0)
CTC_BLANK = "<blank>"
VOCAB = [CTC_BLANK] + CHAR_SET
VOCAB_SIZE = len(VOCAB)

# Character to index mapping
CHAR_TO_IDX = {char: idx for idx, char in enumerate(VOCAB)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(VOCAB)}

# Model architecture parameters
MODEL_CONFIG = {
    "input_dim": 3,           # (Δx, Δy, pen_state)
    "hidden_units": 128,      # LSTM hidden units (balanced for size/accuracy)
    "num_layers": 2,          # Number of BiLSTM layers
    "dropout": 0.3,           # Dropout rate
    "vocab_size": VOCAB_SIZE, # Number of output classes
    "max_seq_length": 256,    # Maximum stroke sequence length
    "max_label_length": 64,   # Maximum text label length
}

# Training parameters
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "lr_decay_factor": 0.5,
    "lr_decay_patience": 5,
    "early_stopping_patience": 10,
    "validation_split": 0.15,
    "checkpoint_dir": "saved_models",
    "log_dir": "logs",
}

# Data preprocessing parameters
PREPROCESSING_CONFIG = {
    "canvas_width": 400,
    "canvas_height": 200,
    "stroke_normalize": True,   # Normalize coordinates to [-1, 1]
    "use_delta": True,          # Use delta (relative) coordinates
    "resample_points": True,    # Resample strokes to uniform spacing
    "resample_distance": 3.0,   # Minimum distance between points
    "augmentation": {
        "rotation_range": 10,   # Degrees
        "scale_range": (0.9, 1.1),
        "translation_range": 0.1,
        "noise_std": 0.02,
    }
}

# TensorFlow.js conversion settings
TFJS_CONFIG = {
    "output_dir": "tfjs_model",
    "quantize": True,           # Quantize weights to reduce size
    "quantization_dtype": "uint8",
}

# Inference settings
INFERENCE_CONFIG = {
    "beam_width": 10,           # Beam search width
    "use_beam_search": True,    # Use beam search (vs greedy)
    "blank_threshold": 0.5,     # Threshold for blank detection
}

def get_char_index(char):
    """Get index for a character, return blank index if not found."""
    return CHAR_TO_IDX.get(char, 0)

def get_char_from_index(idx):
    """Get character from index."""
    return IDX_TO_CHAR.get(idx, "")

def encode_text(text):
    """Encode text string to list of indices."""
    return [get_char_index(c) for c in text]

def decode_indices(indices):
    """Decode list of indices to text string."""
    return "".join(get_char_from_index(idx) for idx in indices)

if __name__ == "__main__":
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Character set: {''.join(CHAR_SET)}")
    print(f"Model config: {MODEL_CONFIG}")
