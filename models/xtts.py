"""
XTTS v2 synthesizer implementation.
"""

from pathlib import Path
from typing import Optional, List
import torch
from tqdm import tqdm

from .base import BaseSynthesizer


class XTTSSynthesizer(BaseSynthesizer):
    """XTTS v2 synthesizer using Coqui TTS."""

    def __init__(self, config):
        super().__init__(config)

    def load_model(self) -> None:
        """Load XTTS v2 model."""
        try:
            from TTS.api import TTS

            # Load XTTS v2 model
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            self.model = TTS(model_name).to(self.config.device)
            self.is_loaded = True

            print(f"Loaded XTTS v2 model on {self.config.device}")

        except ImportError:
            raise RuntimeError("Coqui TTS not installed. Install with: pip install TTS")
        except Exception as e:
            raise RuntimeError(f"Failed to load XTTS model: {e}")

    def synthesize(
        self,
        text: List[str],
        output_path: List[Path],
        reference_audio: List[Path],
        style_prompt: List[Optional[str]],
        speaker_id: List[Optional[str]]
    ) -> bool:
        """Synthesize speech using XTTS voice cloning.

        Args:
            text: List of texts to synthesize
            output_path: List of paths to save synthesized audio
            reference_audio: List of paths to reference audio for voice cloning
            style_prompt: List of optional style prompts (unused in XTTS)
            speaker_id: List of optional speaker identifiers (unused in XTTS)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_loaded:
            self.load_model()

        is_batch = isinstance(text, list)
        if not is_batch:
            text = [text]
            output_path = [output_path]
            reference_audio = [reference_audio]
            style_prompt = [style_prompt] if style_prompt else [None]
            speaker_id = [speaker_id] if speaker_id else [None]

        if style_prompt is None:
            style_prompt = [None] * len(text)

        try:
            for i in tqdm(range(len(text)), desc="Synthesizing..."):
                current_output_path = output_path[i]
                current_output_path.parent.mkdir(parents=True, exist_ok=True)

                self.model.tts_to_file(
                    text=text[i],
                    speaker_wav=str(reference_audio[i]),
                    language=self.config.language,
                    file_path=str(current_output_path),
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p
                )

                if not current_output_path.exists() or current_output_path.stat().st_size == 0:
                    print(f"Warning: Output file {current_output_path} was not created or is empty")

            return True

        except Exception as e:
            print(f"Failed to synthesize audio with XTTS: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up XTTS model resources."""
        super().cleanup()

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
