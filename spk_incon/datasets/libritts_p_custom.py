"""LibriTTS-P Curated Dataset.

Curated version of LibriTTS-P with outlier filtering and dynamic prompt generation.
"""

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torchaudio
from datasets import Dataset as HFDataset
from torch import Tensor
from torch.utils.data import Dataset

from .libritts_p3 import (
    download_libritts_audio,
    download_libritts_p_metadata,
    get_libritts_p_metadata_paths,
    load_libritts_p_prompts,
)


class LIBRITTS_P_Custom(Dataset):
    """Custom LibriTTS-P dataset with outlier filtering and dynamic prompt generation.

    This dataset merges curated quality labels (Z-scores) with original LibriTTS-P metadata.
    It provides a unified interface for loading audio and generating mixed prompts
    deterministically based on the current training epoch.

    When ``use_synthetic=True``, the dataset filters to only samples whose speaker ID
    has a corresponding entry in the synthetic speaker description CSV. In this mode,
    ``combined_prompt`` in ``__getitem__`` is replaced by a deterministically selected
    ``speaker_description`` from the synthetic CSV, bypassing ``_mix_prompts()``.

    Args:
        root (str or Path): Root directory of the dataset.
        url (str, optional): Dataset split. Allowed values: ``"dev-clean"``,
            ``"dev-other"``, ``"test-clean"``, ``"test-other"``, ``"train-clean-100"``,
            ``"train-clean-360"``, ``"train-other-500"``.
            (default: ``"train-clean-100"``)
        annotator (str, optional): Speaker prompt annotator. Must be one of
            ``["df1", "df2", "df3"]``. (default: ``"df1"``)
        max_z_score (float, optional): Maximum modified Z-score for outlier filtering.
            Samples with distance Z-scores above this threshold are excluded.
            (default: ``3.5``)
        min_group_size (int, optional): Minimum number of samples required per speaker group.
            Groups with fewer samples than this threshold are excluded.
            (default: ``10``)
        use_synthetic (bool, optional): If True, loads synthetic speaker descriptions and
            uses them as ``combined_prompt`` instead of ``_mix_prompts()``. Samples whose
            speaker ID has no synthetic entry are excluded.
            (default: ``False``)
        transform (Callable, optional): A function/transform that takes in a sample
            dictionary and returns a transformed version. (default: ``None``)
        download (bool, optional): Whether to download the dataset if not found at root.
            (default: ``False``)
        force_reload (bool, optional): If True, re-creates the cache even if it exists.
            (default: ``False``)
    """
    def __init__(
        self,
        root: Union[str, Path],
        url: str = "train-clean-100",
        annotator: str = "df1",
        max_z_score: float = 3.5,
        min_group_size: int = 10,
        use_synthetic: bool = False,
        transform: Optional[Callable] = None,
        download: bool = False,
        force_reload: bool = False,
    ) -> None:
        self.root = Path(root)
        self.url = url
        self.annotator = annotator
        self.max_z_score = max_z_score
        self.min_group_size = min_group_size
        self.use_synthetic = use_synthetic
        self.transform = transform
        self.download = download
        self.epoch = 0

        self.cache_dir = self.root / ".cache" / f"libritts_p_{url}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.data_source = self._load_or_create_dataset(force_reload)
        self.style_map, self.speaker_map = load_libritts_p_prompts(self.root, self.annotator)

        # Load synthetic speaker descriptions if requested
        self.synthetic_map: Dict[int, Dict[str, str]] = {}
        if self.use_synthetic:
            self._load_synthetic_descriptions()

        self._filter_outliers()
        self._filter_min_group_size()

        # Filter to speakers with synthetic descriptions when use_synthetic=True
        if self.use_synthetic:
            self._filter_synthetic_speakers()

    # ------------------------------------------------------------------
    # Synthetic description loading
    # ------------------------------------------------------------------

    def _load_synthetic_descriptions(self) -> None:
        """Loads synthetic speaker descriptions from CSV into ``self.synthetic_map``.

        The CSV is expected at ``root / synthetic_speaker_descriptions_{annotator}.csv``
        with columns: ``speaker_id``, ``prompt_key``, ``speaker_description``.

        ``self.synthetic_map`` maps ``speaker_id (int)`` ->
        ``prompt_key (str)`` -> ``speaker_description (str)``.
        """
        synthetic_path = self.root / f"synthetic_speaker_descriptions_{self.annotator}.csv"
        if not synthetic_path.exists():
            raise FileNotFoundError(
                f"Synthetic speaker description file not found: {synthetic_path}"
            )

        print(f"[INFO] Loading synthetic speaker descriptions from {synthetic_path}...")
        df = pd.read_csv(synthetic_path)
        df["speaker_id"] = df["speaker_id"].astype(int)

        self.synthetic_map = {
            int(spk_id): group.set_index("prompt_key")["speaker_description"].to_dict()
            for spk_id, group in df.groupby("speaker_id")
        }

        print(f"[INFO] Loaded synthetic descriptions for {len(self.synthetic_map)} speakers.")

    def _filter_synthetic_speakers(self) -> None:
        """Excludes samples whose speaker ID has no synthetic description entry."""
        before = len(self.data)
        print("[INFO] Filtering samples without synthetic speaker descriptions...")
        self.data = self.data.filter(
            lambda x: int(x["speaker_id"]) in self.synthetic_map
        )
        print(f"[INFO] Filtered: {before} -> {len(self.data)} samples.")

    # ------------------------------------------------------------------
    # Dataset loading / caching
    # ------------------------------------------------------------------

    def _load_or_create_dataset(self, force_reload: bool) -> HFDataset:
        """Loads the dataset from disk or triggers creation if missing.

        Args:
            force_reload (bool): If True, forces re-creation of the cache.

        Returns:
            HFDataset: The loaded or newly created Hugging Face dataset.
        """
        cache_path = self.cache_dir / "dataset"

        if not force_reload and cache_path.exists():
            try:
                print(f"[INFO] Loading cached dataset from {cache_path}...")
                return HFDataset.load_from_disk(str(cache_path))
            except Exception as e:
                print(f"[INFO] Failed to load cache: {e}. Recreating...")

        return self._create_and_cache_dataset(cache_path)

    @staticmethod
    def add_file_metadata(batch: Dict[str, Any], root: Path, url: str) -> Dict[str, List[Any]]:
        """Maps utterance IDs to absolute file system paths and loads normalized text.

        Args:
            batch (Dict): A batch of samples from the Arrow dataset.
            root (Path): Root directory for audio and text resolution.
            url (str): Dataset split URL.

        Returns:
            Dict: A dictionary containing the lists of resolved audio paths and normalized texts.
        """
        audio_paths = []
        normalized_texts = []

        for fileid in batch['utterance_id']:
            parts = fileid.split('_')
            spk_id, ch_id = parts[0], parts[1]

            audio_p = root / "LibriTTS" / url / spk_id / ch_id / f"{fileid}.wav"
            audio_paths.append(str(audio_p))

            norm_p = root / "LibriTTS" / url / spk_id / ch_id / f"{fileid}.normalized.txt"
            try:
                with open(norm_p, 'r', encoding='utf-8') as f:
                    normalized_texts.append(f.read().strip())
            except Exception:
                normalized_texts.append("")

        return {
            "audio_path": audio_paths,
            "normalized_text": normalized_texts
        }

    def _create_and_cache_dataset(self, cache_path: Path) -> HFDataset:
        """Processes raw CSVs and merges them into a unified Arrow dataset.

        Args:
            cache_path (Path): Destination for the saved Arrow dataset.

        Returns:
            HFDataset: The processed Hugging Face dataset.
        """
        print("[INFO] Creating dataset cache... This may take a while.")

        if self.download:
            download_libritts_audio(self.root, self.url)
            download_libritts_p_metadata(self.root, self.annotator)

        url_norm = self.url.replace("-", "_")
        curated_filename = f"libritts_p_curated_{url_norm}.csv"
        curated_path = self.root / curated_filename

        paths = get_libritts_p_metadata_paths(self.root)
        orig_meta_df = pd.read_csv(paths["metadata"])

        if curated_path.exists():
            print("[INFO] Merging curated and original metadata...")
            curated_df = pd.read_csv(curated_path)
            df = pd.merge(
                curated_df,
                orig_meta_df,
                left_on="utterance_id",
                right_on="item_name",
                how="inner",
            )
        else:
            print(f"[INFO] Curated file not found for '{self.url}', using full metadata (no curation).")
            # Filter to current split by matching utterance_id prefix pattern
            # LibriTTS utterance_id format: {spk_id}_{chapter_id}_{...}
            # Audio files exist at root/LibriTTS/{url}/{spk_id}/{chapter_id}/
            split_audio_root = self.root / "LibriTTS" / self.url
            if split_audio_root.exists():
                valid_speakers = {p.name for p in split_audio_root.iterdir() if p.is_dir()}
                orig_meta_df["_spk_id_str"] = orig_meta_df["item_name"].str.split("_").str[0]
                orig_meta_df = orig_meta_df[orig_meta_df["_spk_id_str"].isin(valid_speakers)]
                orig_meta_df = orig_meta_df.drop(columns=["_spk_id_str"])
            df = orig_meta_df.copy()
            df["utterance_id"] = df["item_name"]
            df["group_id"] = ""
            df["group_size"] = 0
            df["distance_z_score"] = 0.0
            df["intra_group_distance"] = 0.0

        df = df.rename(columns={
            'spk_id': 'speaker_id',
            'content_prompt': 'original_text'
        })

        dataset = HFDataset.from_pandas(df)

        print("[INFO] Mapping audio paths and loading text metadata...")
        dataset = dataset.map(
            self.add_file_metadata,
            batched=True,
            fn_kwargs={"root": self.root, "url": self.url}
        )

        cols_to_remove = [
            c for c in ["item_name", "chapter_id", "__index_level_0__"]
            if c in dataset.column_names
        ]
        dataset = dataset.remove_columns(cols_to_remove)

        print(f"[INFO] Saving dataset cache to {cache_path}...")
        dataset.save_to_disk(str(cache_path))
        return dataset

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_outliers(self) -> None:
        """Excludes samples with modified Z-scores exceeding the defined threshold.
        
        Skipped if curated metadata is unavailable (group_size == 0).
        """
        if self.max_z_score is None:
            self.data = self.data_source
            return

        # Skip if no curation data available (fallback mode)
        if len(self.data_source) == 0 or self.data_source[0].get('group_size', 0) == 0:
            print("[INFO] Skipping outlier filter (no curated metadata).")
            self.data = self.data_source
            return

        print(f"[INFO] Filtering outliers (max_z_score={self.max_z_score})...")
        self.data = self.data_source.filter(
            lambda x: x['distance_z_score'] <= self.max_z_score if x['distance_z_score'] is not None else True
        )
        print(f"[INFO] Filtered: {len(self.data_source)} -> {len(self.data)} samples.")

    def _filter_min_group_size(self) -> None:
        """Excludes samples whose speaker groups have fewer than min_group_size samples.
        
        Skipped if curated metadata is unavailable (group_size == 0).
        """
        if self.min_group_size is None:
            return

        # Skip if no curation data available (fallback mode)
        if len(self.data) == 0 or self.data[0].get('group_size', 0) == 0:
            print("[INFO] Skipping min group size filter (no curated metadata).")
            return

        before = len(self.data)
        print(f"[INFO] Filtering groups with fewer than {self.min_group_size} samples...")
        self.data = self.data.filter(
            lambda x: x['group_size'] >= self.min_group_size
        )
        print(f"[INFO] Filtered: {before} -> {len(self.data)} samples.")

    # ------------------------------------------------------------------
    # Epoch / prompt helpers
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Sets the current epoch to ensure deterministic variety in prompt mixing.

        Args:
            epoch (int): The current training epoch number.
        """
        self.epoch = epoch

    def _load_audio(self, path: str) -> Tuple[Tensor, int]:
        """Loads an audio file from the filesystem."""
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate

    def _mix_prompts(self, style_prompts: List[str], speaker_prompts: List[str], idx: int) -> str:
        """Combines style and speaker prompts into a single descriptive string.

        Args:
            style_prompts (List[str]): List of available style prompt variants.
            speaker_prompts (List[str]): List of available speaker trait variants.
            idx (int): Global index of the sample for deterministic selection.

        Returns:
            str: The final combined prompt.
        """
        seed = int(self.epoch * 1e6 + idx)
        rng = random.Random(seed)

        style = rng.choice(style_prompts)

        spk_prompts_copy = list(speaker_prompts).copy()
        rng.shuffle(spk_prompts_copy)
        spk_str = ", ".join(spk_prompts_copy)

        style = style.rstrip(". ")
        return f"{style}. The speaker's identity can be described as {spk_str}."

    def _select_synthetic_prompt(self, speaker_id: int, prompt_key: str) -> str:
        """Returns the synthetic speaker description for a given speaker and prompt key.

        Args:
            speaker_id (int): Speaker ID to look up in ``self.synthetic_map``.
            prompt_key (str): Style prompt key (e.g. ``M_p-low_s-slow_e-low``).

        Returns:
            str: The synthetic speaker description string.

        Raises:
            KeyError: If speaker_id or prompt_key is not found in synthetic_map.
        """
        return self.synthetic_map[speaker_id][prompt_key]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns the n-th sample from the dataset.

        When ``use_synthetic=False`` (default), ``combined_prompt`` is generated via
        ``_mix_prompts()``. When ``use_synthetic=True``, ``combined_prompt`` is a
        deterministically selected synthetic speaker description.

        Returns:
            Dict containing:
                - waveform (Tensor)
                - sample_rate (int)
                - original_text (str)
                - normalized_text (str)
                - speaker_id (int)
                - utterance_id (str)
                - group_id (str)
                - group_size (int)
                - distance_z_score (float)
                - intra_group_distance (float)
                - combined_prompt (str)
                - style_prompts (List[str])
                - speaker_prompts (List[str])
        """
        item = self.data[idx]

        waveform, sr = self._load_audio(item['audio_path'])

        style_variants, _ = self.style_map.get(item['utterance_id'], (["Unknown Style"], None))

        speaker_id = int(item['speaker_id'])
        speaker_variants = self.speaker_map.get(speaker_id, ["Unknown Speaker"]) if self.speaker_map else ["No Annotator"]

        if self.use_synthetic:
            prompt_key = item.get('style_prompt_key', '')
            combined_prompt = self._select_synthetic_prompt(speaker_id, prompt_key)
        else:
            combined_prompt = self._mix_prompts(style_variants, speaker_variants, idx)

        return {
            'waveform': waveform,
            'sample_rate': sr,
            'original_text': item.get('original_text', ''),
            'normalized_text': item.get('normalized_text', ''),
            'style_prompts': style_variants,
            'speaker_prompts': speaker_variants,
            'combined_prompt': combined_prompt,
            'utterance_id': item['utterance_id'],
            'speaker_id': speaker_id,
            'group_id': item.get('group_id', ''),
            'group_size': item.get('group_size', 0),
            'distance_z_score': item.get('distance_z_score', 0.0),
            'intra_group_distance': item.get('intra_group_distance', 0.0)
        }

    def get_sample(self, index: int) -> Tuple[str, Path, str, str]:
        """Returns a sample in the format expected by synthesis strategies.

        Args:
            index (int): Sample index.

        Returns:
            Tuple of:
                - transcript (str): Normalized text.
                - audio_path (Path): Path to the audio file.
                - style_prompt (str): Combined prompt (mixed or synthetic).
                - speaker_id (str): Speaker ID as string.
        """
        item = self[index]
        return (
            item['normalized_text'],
            Path(self.data[index]['audio_path']),
            item['combined_prompt'],
            str(item['speaker_id']),
        )
