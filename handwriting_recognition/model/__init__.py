"""
Handwriting Recognition Model Package

This package provides a BiLSTM + CTC model for offline handwriting recognition.
"""

from .config import VOCAB, VOCAB_SIZE, encode_text, decode_indices, MODEL_CONFIG
from .architecture import create_bilstm_ctc_model, build_model_from_config

__all__ = [
    'VOCAB',
    'VOCAB_SIZE',
    'encode_text',
    'decode_indices',
    'MODEL_CONFIG',
    'create_bilstm_ctc_model',
    'build_model_from_config',
]
