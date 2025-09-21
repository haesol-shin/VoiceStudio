"""
Dia-TTS synthesizer implementation.
"""

from pathlib import Path
from typing import Optional
import torch
import torchaudio
from .base import BaseSynthesizer

class DiaSynthesizer(BaseSynthesizer):
    """Dia-TTS synthesizer using nari-labs/Dia-1.6B-0626."""

    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.sampling_rate = 44100  # Dia-TTS default sampling rate

    def load_model(self) -> None:
        """Load Dia-TTS model and processor."""
        try:
            from transformers import AutoProcessor, DiaForConditionalGeneration
        except ImportError:
            raise RuntimeError("transformers not installed. Install with: pip install transformers")

        try:
            model_name = "nari-labs/Dia-1.6B-0626"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = DiaForConditionalGeneration.from_pretrained(model_name).to(self.config.device)
            self.is_loaded = True
            print(f"Loaded Dia-TTS model on {self.config.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load Dia-TTS model: {e}")

    def synthesize(
            self,
            text: str,
            output_path: Path,
            reference_audio: Optional[Path] = None,
            style_prompt: Optional[str] = None,
            speaker_id: Optional[str] = None
    ) -> bool:
        """Synthesize speech using Dia-TTS.

        Args:
            text: Text to synthesize.
            output_path: Path to save synthesized audio.
            reference_audio: Path to reference audio for voice cloning.
            style_prompt: Optional style prompt (unused).
            speaker_id: Optional speaker identifier (unused).

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_loaded:
            self.load_model()

        if reference_audio is None:
            print("Error: Dia-TTS requires a reference audio for voice cloning.")
            return False

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Load and process reference audio
            audio_waveform, sr = torchaudio.load(reference_audio)

            if audio_waveform.shape[1] == 0:
                print(f"Warning: Reference audio is empty, skipping synthesis. Path: {reference_audio}")
                return False


            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
                audio_waveform = resampler(audio_waveform)

            input_text = [f"[S1] {text} [S2] {text}"]

            inputs = self.processor(
                text=input_text,
                audio=audio_waveform.squeeze().numpy(),
                sampling_rate=self.sampling_rate,
                padding=True,
                return_tensors="pt"
            ).to(self.config.device)

            prompt_len = self.processor.get_audio_prompt_len(inputs["decoder_attention_mask"])

            # Generate audio
            outputs = self.model.generate(**inputs, max_new_tokens=prompt_len + 128)

            # Decode and save audio
            decoded_outputs = self.processor.batch_decode(outputs, audio_prompt_len=prompt_len)
            self.processor.save_audio(decoded_outputs, str(output_path))

            if not output_path.exists() or output_path.stat().st_size == 0:
                print(f"Warning: Output file {output_path} was not created or is empty")
                return False
            return True

        except Exception as e:
            print(f"Failed to synthesize audio with Dia-TTS: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up Dia model resources."""
        super().cleanup()
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()