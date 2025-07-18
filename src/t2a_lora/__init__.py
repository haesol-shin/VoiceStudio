"""T2A-LoRA: Text-to-Audio LoRA Generation via Hypernetworks for Real-time Voice Adaptation."""

__version__ = "0.1.0"
__author__ = "LatentForge"
__description__ = "Text-to-Audio LoRA Generation via Hypernetworks for Real-time Voice Adaptation"

from .models import (
    HyperNetwork,
    MultimodalEncoder,
    T2ALoRAGenerator,
)
from .training import (
    T2ALoRATrainer,
    TrainingConfig,
)
from .generation import (
    T2ALoRAGenerator as Generator,
    GenerationConfig,
)

__all__ = [
    "HyperNetwork",
    "MultimodalEncoder", 
    "T2ALoRAGenerator",
    "T2ALoRATrainer",
    "TrainingConfig",
    "Generator",
    "GenerationConfig",
]
