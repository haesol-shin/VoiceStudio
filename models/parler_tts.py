"""
Parler-TTS synthesizer implementation.
"""

from pathlib import Path
from typing import Optional, List
import torch
import soundfile as sf
from tqdm import tqdm

from .base import BaseSynthesizer


# Default description if none provided
DEFAULT_DESCRIPTION = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

class ParlerTTSSynthesizer(BaseSynthesizer):
    """Parler-TTS synthesizer."""

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = None
        self.sampling_rate = None
        self.batch_size = 32

    def load_model(self) -> None:
        """Load Parler-TTS model and tokenizer."""
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except ImportError:
            raise RuntimeError("Parler-TTS not installed. Install with: pip install parler-tts")

        try:
            model_name = "parler-tts/parler-tts-mini-v1"
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self.config.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sampling_rate = self.model.config.sampling_rate
            self.is_loaded = True
            print(f"Loaded Parler-TTS model on {self.config.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load Parler-TTS model: {e}")

    def synthesize(
        self,
        text: List[str],
        output_path: List[Path],
        reference_audio: List[Optional[Path]],
        style_prompt: List[str],
        speaker_id: List[Optional[str]]
    ) -> bool:
        """Synthesize speech using Parler-TTS.

        Args:
            text: List of texts to synthesize (prompt)
            output_path: List of paths to save synthesized audio
            reference_audio: List of paths to reference audio (unused in Parler-TTS)
            style_prompt: List of optional style prompts (used as description)
            speaker_id: List of optional speaker identifiers (unused in Parler-TTS)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_loaded:
            self.load_model()

        try:
            is_batch = isinstance(text, list)
            if not is_batch:
                text = [text]
                output_path = [output_path]
                style_prompt = [style_prompt]

            if style_prompt is None:
                style_prompt = [DEFAULT_DESCRIPTION] * len(text)

            if len(style_prompt) != len(text):
                style_prompt = [style_prompt[0]] * len(text)

            for i in tqdm(range(0, len(text), self.batch_size), desc=f"Generating in batches of {self.batch_size}"):
                batch_texts = text[i:i+self.batch_size]
                batch_output_paths = output_path[i:i+self.batch_size]
                batch_style_prompts = style_prompt[i:i+self.batch_size]

                # Ensure output directories exist
                for path in batch_output_paths:
                    path.parent.mkdir(parents=True, exist_ok=True)

                # Tokenize batch
                inputs = self.tokenizer(batch_style_prompts, return_tensors="pt", padding=True).to(self.config.device)
                prompt_input_ids = self.tokenizer(batch_texts, return_tensors="pt", padding=True).to(self.config.device)

                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42)
                    torch.cuda.manual_seed_all(42)

                # Generate for the current batch
                generation = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    prompt_input_ids=prompt_input_ids.input_ids,
                    prompt_attention_mask=prompt_input_ids.attention_mask,
                    do_sample=False,
                    num_beams=1,
                    return_dict_in_generate=True,
                )

                # Save audio files for the current batch
                for j in range(len(batch_texts)):
                    audio_arr = generation.sequences[j, :generation.audios_length[j]]
                    sf.write(str(batch_output_paths[j]), audio_arr.cpu().numpy().squeeze(), self.sampling_rate)

                    if not batch_output_paths[j].exists() or batch_output_paths[j].stat().st_size == 0:
                        print(f"Warning: Output file {batch_output_paths[j]} was not created or is empty")

            return True

        except Exception as e:
            print(f"Failed to synthesize audio: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up model resources."""
        super().cleanup()
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()