"""
Base synthesizer for TTS models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List


class BaseSynthesizer(ABC):
    """Base class for TTS synthesizers."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load the TTS model."""
        pass

    @abstractmethod
    def synthesize(
        self,
        text: List[str],
        output_path: List[Path],
        reference_audio: List[Optional[Path]],
        style_prompt: List[Optional[str]],
        speaker_id: List[Optional[str]]
    ) -> bool:
        """Synthesize speech from text using reference audio or style prompt.

        Args:
            text: List of texts to synthesize
            output_path: List of paths to save synthesized audio
            reference_audio: List of paths to reference audio for voice cloning
            style_prompt: List of optional style prompts for synthesis
            speaker_id: List of optional speaker identifiers

        Returns:
            True if successful, False otherwise
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        if not self.is_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False
