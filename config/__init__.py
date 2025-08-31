"""
Configuration module for synthesis pipeline.
"""

from enum import Enum
from .synthesis_config import (
    DatasetConfig,
    ModelConfig,
    GenerationConfig,
    SynthesisConfig
)

class DatasetType(Enum):
    """Supported dataset types."""
    VCTK = "vctk"
    LJSPEECH = "ljspeech"

class ModelType(Enum):
    """Supported model types."""
    XTTS_V2 = "xtts_v2"

class GenerationMethod(Enum):
    """Generation method types."""
    METHOD1 = "method1"  # 100 1:1 pairs
    METHOD2 = "method2"  # 10 refs × 10 syn each

__all__ = [
    "DatasetConfig",
    "ModelConfig",
    "GenerationConfig",
    "SynthesisConfig",
    "DatasetType",
    "ModelType",
    "GenerationMethod"
]