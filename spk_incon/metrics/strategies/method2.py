"""
Method 2: 10 references × 10 synthesis each for speaker consistency.
"""
import traceback
from tqdm.auto import tqdm

from .base import BaseGenerationStrategy


class Method2Strategy(BaseGenerationStrategy):
    """Generate 10 reference audios with 10 synthesis each."""

    COMPARISON_TEXTS = [
        # Short (10-15w)
        "The earth is not devoid of resemblance to a jail.",
        "The chaos in which his ardour extinguished itself was a cold indifferent knowledge of himself.",
        # Short-medium (16-20w)
        "Wylder was laughing rather redly, with the upper part of his face very surly, I thought.",
        "I cannot allow the examination to be held if one of the papers has been tampered with.",
        "Supposing that it was his sister coming back from one of her farms, he kept on with his work.",
        "The behaviourist, who attempts to make psychology a record of behaviour, has to trust his memory in making the record.",
        # Medium (21-26w)
        "I opened a line of credit sufficient to cover the babirusa and, Conseil at my heels, I jumped into a carriage.",
        "The little knot of Indians drew back in a body, and suffered, as they thought, the conjurer and his inspired assistant to proceed.",
        "And there you dwelt as became the children of the gods, excelling all men in virtue, and many famous actions are recorded of you.",
        "So the world was happy, and the face of the black belt green and luxuriant with thickening flecks of the coming foam of the cotton.",
        "It is evident, therefore, that the present trend of the development is in the direction of heightening the utility of conspicuous consumption as compared with leisure.",
        # Long (35w+)
        "Louis turned hastily towards the side, and in fact, on his right, brilliant in his character of Autumn, De Guiche awaited until the king should look at him, in order that he might address him.",
        "Had the telegraph been invented in the days of ancient Rome, would the romans have accepted it, or have stoned Wheatstone? So thinking, I resolved that I was before my age, and that I must pay the allotted penalty.",
        "By reason of these affections the soul is at first without intelligence, but as time goes on the stream of nutriment abates, and the courses of the soul regain their proper motion, and apprehend the same and the other rightly, and become rational.",
    ]

    def generate_all(self, dataset_name: str, model_name: str) -> bool:
        print(f"Starting Method 2 generation for {dataset_name} -> {model_name}")

        ref_dir, syn_dir = self.create_output_paths(dataset_name, model_name, "method2")

        num_refs = self.config.generation.method2_ref_samples
        syn_per_ref = self.config.generation.method2_syn_per_ref
        sample_indices = self.select_unique_speakers(num_refs)

        total_success = 0

        for ref_idx, sample_idx in enumerate(
            tqdm(sample_indices, desc="Processing references")
        ):
            try:
                transcript, audio_path, style_prompt, speaker_id = (
                    self.dataset.get_sample(sample_idx)
                )

                ref_filename = f"ref_{ref_idx:03d}.wav"
                ref_output_path = ref_dir / ref_filename

                if not self.copy_reference_audio(audio_path, ref_output_path):
                    print(f"Failed to copy reference audio {ref_idx}")
                    continue

                set_dir = syn_dir / f"set_{ref_idx:03d}"
                set_dir.mkdir(exist_ok=True)

                set_success = 0
                set_metadata = {}

                for syn_idx in tqdm(
                    range(syn_per_ref), desc=f"Set {ref_idx}", leave=False
                ):
                    syn_filename = f"syn_{ref_idx:03d}_{syn_idx:02d}.wav"
                    syn_output_path = set_dir / syn_filename

                    text_to_synthesize = (
                        transcript if syn_idx == 0
                        else self.COMPARISON_TEXTS[syn_idx - 1]
                    )

                    set_metadata[syn_filename] = {
                        "target_text": text_to_synthesize,
                        "speaker_id": speaker_id,
                        "reference_audio": str(audio_path)
                    }

                    if self.synthesizer.synthesize(
                        text=text_to_synthesize,
                        output_path=syn_output_path,
                        reference_audio=audio_path,
                        style_prompt=style_prompt,
                        speaker_id=speaker_id,
                    ):
                        set_success += 1
                    else:
                        print(f"Failed synthesis: set {ref_idx}, syn {syn_idx}")

                self.save_metadata(set_dir, set_metadata)
                total_success += set_success
                print(f"Set {ref_idx}: {set_success}/{syn_per_ref} synthesis generated")

            except Exception as e:
                print(f"Error processing reference {ref_idx}")
                traceback.print_exc()
                continue

        expected_total = len(sample_indices) * syn_per_ref
        print(f"Method 2 completed: {total_success}/{expected_total} synthesis generated")
        return total_success > 0

    def generate_batch_all(self, dataset_name: str, model_name: str, verbose: bool = False) -> bool:
        if verbose: print(f"Starting Method 2 batch generation for {dataset_name} -> {model_name}")

        ref_dir, syn_dir = self.create_output_paths(dataset_name, model_name, "method2")

        num_refs = self.config.generation.method2_ref_samples
        syn_per_ref = self.config.generation.method2_syn_per_ref
        sample_indices = self.select_unique_speakers(num_refs)

        total_success = 0

        for ref_idx, sample_idx in enumerate(
            tqdm(sample_indices, desc="Processing references", leave=verbose)
        ):
            try:
                transcript, audio_path, style_prompt, speaker_id = (
                    self.dataset.get_sample(sample_idx)
                )

                ref_filename = f"ref_{ref_idx:03d}.wav"
                ref_output_path = ref_dir / ref_filename

                if not self.copy_reference_audio(audio_path, ref_output_path):
                    print(f"Failed to copy reference audio {ref_idx}")
                    continue

                set_dir = syn_dir / f"set_{ref_idx:03d}"
                set_dir.mkdir(exist_ok=True)

                set_metadata = {}
                batch_text, batch_prompt = [], []
                batch_ref, batch_spk_id = [], []
                batch_output_path = []

                for syn_idx in tqdm(
                    range(syn_per_ref), desc=f"Set {ref_idx}", leave=False
                ):
                    syn_filename = f"syn_{ref_idx:03d}_{syn_idx:02d}.wav"
                    syn_output_path = set_dir / syn_filename

                    text_to_synthesize = (
                        transcript if syn_idx == 0
                        else self.COMPARISON_TEXTS[syn_idx - 1]
                    )

                    set_metadata[syn_filename] = {
                        "target_text": text_to_synthesize,
                        "speaker_id": speaker_id,
                        "reference_audio": str(audio_path)
                    }

                    batch_text.append(text_to_synthesize)
                    batch_prompt.append(style_prompt)
                    batch_ref.append(audio_path)
                    batch_spk_id.append(speaker_id)
                    batch_output_path.append(syn_output_path)

                success = self.synthesizer.synthesize(
                    text=batch_text,
                    output_path=batch_output_path,
                    reference_audio=batch_ref,
                    style_prompt=batch_prompt,
                    speaker_id=batch_spk_id,
                )
                set_success = len(batch_text) if success else 0
                if not success:
                    print(f"Failed synthesis: set {ref_idx}")
                total_success += set_success

                self.save_metadata(set_dir, set_metadata)
                if verbose: print(f"Set {ref_idx}: {set_success}/{syn_per_ref} synthesis generated")

            except Exception as e:
                print(f"Error processing reference {ref_idx}")
                traceback.print_exc()
                continue

        expected_total = len(sample_indices) * syn_per_ref
        if verbose: print(f"Method 2 batch completed: {total_success}/{expected_total} synthesis generated")
        return total_success > 0

    def generate_batch_group_all(self, dataset_name: str, model_name: str, verbose: bool = False) -> bool:
        if verbose: print(f"Starting Method 2 generation for {dataset_name} -> {model_name}")

        ref_dir, syn_dir = self.create_output_paths(dataset_name, model_name, "method2")

        num_refs = self.config.generation.method2_ref_samples
        syn_per_ref = self.config.generation.method2_syn_per_ref
        sample_indices = self.select_unique_speakers(num_refs)

        batch_text, batch_prompt = [], []
        batch_ref, batch_spk_id = [], []
        batch_output_path = []

        for ref_idx, sample_idx in enumerate(
            tqdm(sample_indices, desc="Processing references", leave=verbose)
        ):
            transcript, audio_path, style_prompt, speaker_id = (
                self.dataset.get_sample(sample_idx)
            )

            ref_filename = f"ref_{ref_idx:03d}.wav"
            ref_output_path = ref_dir / ref_filename

            if not self.copy_reference_audio(audio_path, ref_output_path):
                print(f"Failed to copy reference audio {ref_idx}")
                continue

            set_dir = syn_dir / f"set_{ref_idx:03d}"
            set_dir.mkdir(exist_ok=True)

            set_metadata = {}

            for syn_idx in range(syn_per_ref):
                syn_filename = f"syn_{ref_idx:03d}_{syn_idx:02d}.wav"
                syn_output_path = set_dir / syn_filename

                text_to_synthesize = (
                    transcript if syn_idx == 0
                    else self.COMPARISON_TEXTS[syn_idx - 1]
                )

                set_metadata[syn_filename] = {
                    "target_text": text_to_synthesize,
                    "speaker_id": speaker_id,
                    "reference_audio": str(audio_path)
                }

                batch_text.append(text_to_synthesize)
                batch_prompt.append(style_prompt)
                batch_ref.append(audio_path)
                batch_spk_id.append(speaker_id)
                batch_output_path.append(syn_output_path)

            self.save_metadata(set_dir, set_metadata)

        self.synthesizer.synthesize(
            text=batch_text,
            output_path=batch_output_path,
            reference_audio=batch_ref,
            style_prompt=batch_prompt,
            speaker_id=batch_spk_id,
        )
        total_success = len(batch_text)

        expected_total = len(sample_indices) * syn_per_ref
        if verbose: print(f"Method 2 completed: {total_success}/{expected_total} synthesis generated")
        return total_success > 0
