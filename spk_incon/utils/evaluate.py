"""
Evaluation pipeline for synthesized audio quality assessment.
"""

import json
import random
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..metrics import MetricType, ModelConfig, create_calculator
from ..metrics.presets import DatasetType, GenerationMethod, ModelType


class EvaluationPipeline:
    """Pipeline for evaluating synthesized audio quality."""

    def __init__(self, base_dir: Path = Path("results"), html: bool = False, verbose: bool = True):
        self.base_dir = Path(base_dir)
        self.ref_dir = self.base_dir / "ref"
        self.syn_dir = self.base_dir / "syn"
        self.verbose = verbose
        self.html = html

    @staticmethod
    def _load_metadata(set_dir: Path) -> dict:
        """Load metadata.json from a set directory.
        
        Args:
            set_dir: Path to the set directory
            
        Returns:
            Dictionary containing metadata, or empty dict if file doesn't exist
        """
        metadata_path = set_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def get_audio_pairs_with_metadata(
        self,
        dataset_type: DatasetType,
        model_type: ModelType,
        method: GenerationMethod
    ) -> list[dict]:
        """Get reference-synthesis audio pairs with metadata for proper grouping.

        Args:
            dataset_type: Dataset type
            model_type: Model type
            method: Generation method

        Returns:
            List of dictionaries containing pair info and metadata
        """
        ref_base = self.ref_dir / dataset_type.value / method.value
        syn_base = self.syn_dir / dataset_type.value / model_type.value / method.value

        pairs = []

        if method == GenerationMethod.METHOD1:
            # Method1: Direct 1:1 pairs
            ref_files = sorted(ref_base.glob("ref_*.wav"))

            for ref_file in ref_files:
                # Extract index from ref_001.wav -> syn_001.wav
                index = ref_file.stem.split('_')[1]
                syn_file = syn_base / f"syn_{index}.wav"

                if syn_file.exists():
                    pairs.append({
                        "ref_path": ref_file,
                        "syn_path": syn_file,
                        "ref_id": index,
                        "target_text": None  # Not needed for Method1
                    })
                else:
                    print(f"Warning: Missing synthesis file {syn_file}")

        elif method == GenerationMethod.METHOD2:
            set_dirs = sorted(d for d in syn_base.iterdir() if d.is_dir() and d.name.startswith('set_'))

            for set_dir in set_dirs:
                ref_index = set_dir.stem.split('_')[1]
                syn_files = sorted(set_dir.glob("syn_*.wav"))

                if len(syn_files) < 2:
                    print(f"Warning: Skipping {set_dir} for METHOD2, found {len(syn_files)} files.")
                    continue

                # Load metadata
                metadata = self._load_metadata(set_dir)

                # For reference-based metrics: (syn_0, syn_1), (syn_0, syn_2), ...
                consistency_ref_file = syn_files[0]
                for syn_file in syn_files[1:]:
                    pairs.append({
                        "ref_path": consistency_ref_file,
                        "syn_path": syn_file,
                        "ref_id": ref_index,
                        "target_text": metadata.get(syn_file.name, {}).get("target_text")
                    })
                
                # Special dummy pair to ensure syn_files[0] is seen by no-reference metrics
                pairs.append({
                    "ref_path": consistency_ref_file,
                    "syn_path": consistency_ref_file,
                    "ref_id": ref_index,
                    "target_text": metadata.get(consistency_ref_file.name, {}).get("target_text")
                })

        elif method == GenerationMethod.METHOD3:
            set_dirs = sorted(d for d in syn_base.iterdir() if d.is_dir() and d.name.startswith('set_'))

            for set_dir in set_dirs:
                ref_index = set_dir.stem.split('_')[1]
                syn_files = sorted(set_dir.glob("syn_*.wav"))

                if len(syn_files) < 3:
                    print(f"Warning: Skipping {set_dir} for METHOD3, found {len(syn_files)} files.")
                    continue

                # Load metadata
                metadata = self._load_metadata(set_dir)

                consistency_ref_file = syn_files[0]
                for syn_file in syn_files[1:]:
                    pairs.append({
                        "ref_path": consistency_ref_file,
                        "syn_path": syn_file,
                        "ref_id": ref_index,
                        "target_text": metadata.get(syn_file.name, {}).get("target_text")
                    })
                
                # Special dummy pair to ensure syn_files[0] is seen by no-reference metrics
                pairs.append({
                    "ref_path": consistency_ref_file,
                    "syn_path": consistency_ref_file,
                    "ref_id": ref_index,
                    "target_text": metadata.get(consistency_ref_file.name, {}).get("target_text")
                })

        return pairs

    @staticmethod
    def evaluate_pairs_with_grouping(
        pairs: list[dict],
        metric_types: list[MetricType],
        batch_size: int = 16,
        verbose: bool = True
    ) -> dict[MetricType, dict[str, list[float]]]:
        """Evaluate pairs and group results by reference ID.

        Args:
            pairs: List of dictionaries containing pair info and metadata
            metric_types: List of metrics to calculate
            batch_size: Batch size for metric calculation

        Returns:
            Dictionary mapping metric_type -> reference_id -> scores
        """
        results = {}

        for metric_type in metric_types:
            if verbose: print(f"\nCalculating {metric_type.value}...")

            config = ModelConfig(
                name=metric_type.value,
                batch_size=batch_size,
                device="cuda:0"
            )

            try:
                with create_calculator(metric_type, config) as calculator:
                    is_no_reference = metric_type in [MetricType.UTMOS]
                    is_wer = metric_type == MetricType.WER
                    
                    if is_no_reference:
                        # For no-reference metrics, we only care about unique synthesis files
                        unique_syn_paths = sorted(set([p["syn_path"] for p in pairs]))
                        # Create dummy pairs for calculate_batch_optimized
                        calc_pairs = [(p, p) for p in unique_syn_paths]
                        valid_pairs = calculator.validate_audio_files(calc_pairs)
                        
                        scores_output = calculator.calculate_batch_optimized(valid_pairs)
                        path_to_score = {valid_pairs[i][1]: scores_output[i] for i in range(len(valid_pairs))}
                        
                        # Build mapping of syn_path to ref_id (each file belongs to one ref_id)
                        syn_to_ref = {}
                        for pair_info in pairs:
                            syn_path = pair_info["syn_path"]
                            ref_id = pair_info["ref_id"]
                            if syn_path not in syn_to_ref:
                                syn_to_ref[syn_path] = ref_id
                        
                        # Group scores by ref_id (each unique file added once)
                        grouped_scores = {}
                        for syn_path, score in path_to_score.items():
                            if score is not None and not np.isnan(score):
                                ref_id = syn_to_ref.get(syn_path)
                                if ref_id is not None:
                                    if ref_id not in grouped_scores:
                                        grouped_scores[ref_id] = []
                                    grouped_scores[ref_id].append(score)
                            
                    else:
                        audio_pairs = [(p["ref_path"], p["syn_path"]) for p in pairs]
                        # Filter out our dummy self-pairs for reference-based metrics
                        audio_pairs = [(r, s) for r, s in audio_pairs if r != s]
                        
                        valid_pairs = calculator.validate_audio_files(audio_pairs)
                        
                        # For WER, we need to pass target_text as kwargs
                        if is_wer:
                            # Build target_text mapping for WER
                            pair_to_target = {}
                            for pair_info in pairs:
                                if pair_info["ref_path"] != pair_info["syn_path"]:
                                    key = (pair_info["ref_path"], pair_info["syn_path"])
                                    pair_to_target[key] = pair_info.get("target_text")
                            
                            # Calculate WER with target texts
                            scores = []
                            for ref_path, syn_path in valid_pairs:
                                target_text = pair_to_target.get((ref_path, syn_path))
                                # WER calculator will use target_text if provided, otherwise transcribe reference
                                score = calculator(synthesis=syn_path, reference=ref_path, target_text=target_text)
                                scores.append(score)
                        else:
                            scores = calculator.calculate_batch_optimized(valid_pairs)

                        grouped_scores = {}
                        pair_to_score = {valid_pairs[i]: scores[i] for i in range(len(valid_pairs))}

                        for pair_info in pairs:
                            ref_path = pair_info["ref_path"]
                            syn_path = pair_info["syn_path"]
                            if ref_path == syn_path:
                                continue  # Skip dummy pairs
                            
                            score = pair_to_score.get((ref_path, syn_path))
                            if score is not None and not np.isnan(score):
                                # Scaling for WER and FFE: clamp to [0, 1]
                                if metric_type in [MetricType.WER, MetricType.FFE]:
                                    score = min(1.0, max(0.0, float(score)))
                                
                                ref_id = pair_info["ref_id"]
                                if ref_id not in grouped_scores:
                                    grouped_scores[ref_id] = []
                                grouped_scores[ref_id].append(score)

                    results[metric_type] = grouped_scores
                    total_scores = sum(len(scores) for scores in grouped_scores.values())
                    if verbose: print(f"Grouped scores: {total_scores} scores in {len(grouped_scores)} groups")

            except Exception as e:
                print(f"Error calculating {metric_type.value}: {e}")
                results[metric_type] = {}

        return results

    @staticmethod
    def calculate_method1_statistics(
        grouped_results: dict[MetricType, dict[str, list[float]]]
    ) -> dict[str, float]:
        """Calculate statistics for Method1 results (simple averages)."""
        stats = {}

        for metric_type, ref_groups in grouped_results.items():
            if not ref_groups:
                continue

            metric_name = metric_type.value

            # Flatten all scores from all groups
            all_scores = []
            for scores in ref_groups.values():
                all_scores.extend(scores)

            if not all_scores:
                continue

            stats[f"{metric_name}_mean"] = np.mean(all_scores)
            stats[f"{metric_name}_std"] = np.std(all_scores)
            stats[f"{metric_name}_median"] = np.median(all_scores)

        return stats

    @staticmethod
    def calculate_method2_statistics(
        grouped_results: dict[MetricType, dict[str, list[float]]]
    ) -> dict[str, float]:
        """Calculate statistics for Method2 results with proper grouping."""
        stats = {}

        for metric_type, ref_groups in grouped_results.items():
            if not ref_groups:
                continue

            metric_name = metric_type.value
            all_scores = []
            group_stds = []
            group_cvs = []

            # Calculate statistics for each reference group
            for ref_id, scores in ref_groups.items():
                if not scores:
                    continue

                all_scores.extend(scores)

                if len(scores) > 1:
                    group_std = np.std(scores, ddof=1)
                    group_stds.append(group_std)

                    mean_score = np.mean(scores)
                    if mean_score > 0:
                        cv = group_std / mean_score
                        group_cvs.append(cv)

                elif len(scores) == 1:
                    group_stds.append(0.0)
                    group_cvs.append(0.0)

            if not all_scores:
                continue

            # Core statistics
            stats[f"{metric_name}_mean"] = np.mean(all_scores)
            stats[f"{metric_name}_std"] = np.std(all_scores)
            stats[f"{metric_name}_median"] = np.median(all_scores)

            # Speaker consistency metrics (core purpose of Method2)
            if group_stds:
                stats[f"{metric_name}_avg_std"] = np.mean(group_stds)

            if group_cvs:
                stats[f"{metric_name}_avg_cv"] = np.mean(group_cvs)

        return stats

    def evaluate_dataset_model(
        self,
        dataset_type: DatasetType,
        model_type: ModelType,
        metric_types: list[MetricType] = None,
        methods: list[GenerationMethod] = None
    ) -> dict[GenerationMethod, dict[str, float]]:
        """Evaluate a specific dataset-model combination."""
        if metric_types is None:
            metric_types = [MetricType.UTMOS, MetricType.WER, MetricType.SIM, MetricType.FFE, MetricType.MCD]

        if methods is None:
            methods = [GenerationMethod.METHOD1, GenerationMethod.METHOD2, GenerationMethod.METHOD3]

        results = {}

        for method in methods:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating: {dataset_type.value} -> {model_type.value} -> {method.value}")
                print(f"{'='*60}")

            pairs = self.get_audio_pairs_with_metadata(dataset_type, model_type, method)
            if self.verbose: print(f"Found {len(pairs)} primary audio pairs/files")

            if not pairs:
                print(f"No audio samples found for {method.value}")
                continue

            grouped_results = self.evaluate_pairs_with_grouping(pairs, metric_types, verbose=self.verbose)

            if method == GenerationMethod.METHOD1:
                stats = self.calculate_method1_statistics(grouped_results)
            else:
                stats = self.calculate_method2_statistics(grouped_results)

            results[method] = stats
            self.print_markdown_table(method, stats, self.html)

        return results

    @staticmethod
    def print_markdown_table(method: GenerationMethod, stats: dict[str, float], html: bool = False) -> None:
        """Print statistics in a Markdown table format with specific ordering and proper alignment."""
        # Metric order: UTMOS, WER, COS (SIM), FFE, MCD
        # Stat order: Mean, Std, Median, Avg Std, Avg CV

        metrics = [
            ("UTMOS", "utmos"),
            ("WER", "wer"),
            ("COS", "sim"),
            ("FFE", "ffe"),
            ("MCD", "mcd")
        ]

        stat_keys = [
            ("Mean", "mean"),
            ("Std", "std"),
            ("Median", "median"),
            ("Avg Std", "avg_std"),
            ("Avg CV", "avg_cv")
        ]

        # Build data rows
        data_rows = []
        for label, metric_prefix in metrics:
            row_items = [label]
            has_data = False
            for _, stat_suffix in stat_keys:
                key = f"{metric_prefix}_{stat_suffix}"
                val = stats.get(key)
                if val is not None:
                    row_items.append(f"{val:.4f}")
                    has_data = True
                else:
                    row_items.append("-")

            if has_data:
                data_rows.append(row_items)

        if not data_rows:
            print(f"\n### Evaluation Results: {method.value}")
            print("No data available")
            return

        header_row = ["Metric"] + [s[0] for s in stat_keys]
        if html:
            from IPython.display import display, HTML

            # Build HTML table
            html_str = f"<h3>Evaluation Results: {method.value}</h3>"
            html_str += "<table style='border-collapse: collapse; font-family: monospace;'>"

            # Header
            html_str += "<thead><tr>"
            for i, h in enumerate(header_row):
                align = "left" if i == 0 else "center"
                html_str += f"<th style='border: 1px solid #ccc; padding: 6px 12px; text-align: {align}; background: #f0f0f0;'>{h}</th>"
            html_str += "</tr></thead>"

            # Data rows
            html_str += "<tbody>"
            for row in data_rows:
                html_str += "<tr>"
                for i, item in enumerate(row):
                    align = "left" if i == 0 else "center"
                    html_str += f"<td style='border: 1px solid #ccc; padding: 6px 12px; text-align: {align};'>{item}</td>"
                html_str += "</tr>"
            html_str += "</tbody></table>"

            display(HTML(html_str))
        else:
            # Calculate column widths
            col_widths = [len(h) for h in header_row]

            for row in data_rows:
                for i, item in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(item))

            # Add padding
            col_widths = [w + 2 for w in col_widths]

            print(f"\n### Evaluation Results: {method.value}")

            # Print header
            header_str = "|"
            for i, item in enumerate(header_row):
                if i == 0:
                    header_str += " " + item.ljust(col_widths[i] - 1) + "|"
                else:
                    header_str += item.center(col_widths[i]) + "|"
            print(header_str)

            # Print separator
            sep_str = "|"
            sep_str += " " + ("-" * (col_widths[0] - 2)) + " |"
            for w in col_widths[1:]:
                sep_str += ":" + ("-" * (w - 2)) + ":|"
            print(sep_str)

            # Print data rows
            for row in data_rows:
                row_str = "|"
                for i, item in enumerate(row):
                    if i == 0:
                        row_str += " " + item.ljust(col_widths[i] - 1) + "|"
                    else:
                        row_str += item.center(col_widths[i]) + "|"
                print(row_str)

    @staticmethod
    def save_results_to_csv(
        results: dict[GenerationMethod, dict[str, float]],
        dataset_type: DatasetType,
        model_type: ModelType,
        output_dir: Path = Path("results")
    ) -> None:
        """Save evaluation results to CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for method, stats in results.items():
            if not stats:
                continue

            df = pd.DataFrame([stats])
            df.insert(0, 'dataset', dataset_type.value)
            df.insert(1, 'model', model_type.value)
            df.insert(2, 'method', method.value)

            filename = f"{dataset_type.value}_{model_type.value}_{method.value}_results.csv"
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)

            print(f"Saved results to {filepath}")


class Evaluator:
    """Flexible evaluator using metadata files"""

    _DEFAULT_METRICS = [
        MetricType.UTMOS, MetricType.WER, MetricType.SIM,
    ]

    def __init__(
        self,
        base_dir: Union[str, Path] = "results",
        dataset_name: str = "libritts",
        device: str = "cuda",
        seed: int = 42,
        batch_size: int = 16,
        syn_base_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.dataset_name = dataset_name
        self.device = device
        self.seed = seed
        self.batch_size = batch_size
        self._syn_base = Path(syn_base_dir) if syn_base_dir else None
        self._ckpt_dir = self.base_dir / ".checkpoints"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fix_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _syn_dir(self, model_name: str, exp: str) -> Path:
        if self._syn_base is not None:
            return self._syn_base / model_name / exp
        return self.base_dir / "syn" / self.dataset_name / model_name / exp

    def _ckpt_path(self, model_name: str, exp: str) -> Path:
        return self._ckpt_dir / f"{model_name}_{exp}.json"

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    def load_pairs(self, model_name: str, exp: str) -> List[dict]:
        """Load evaluation pairs from ``metadata.json``."""
        meta_path = self._syn_dir(model_name, exp) / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {meta_path}\n"
                f"Run synthesis first, or check that the model/exp path is correct."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_pairs = data.get("pairs", data)
        if not raw_pairs:
            raise RuntimeError(f"Metadata at {meta_path} contains no pairs.")

        pairs = []
        for entry in raw_pairs:
            if "ref_path" not in entry or "syn_path" not in entry or "index" not in entry:
                raise KeyError(
                    f"Metadata entry missing required fields (ref_path / syn_path / index): {entry}"
                )
            pairs.append({
                "ref_path":    Path(entry["ref_path"]),
                "syn_path":    Path(entry["syn_path"]),
                "ref_id":      str(entry.get("ref_id") or entry["index"]),
                "target_text": entry.get("target_text"),
                "speaker_id":  entry.get("speaker_id"),
            })

        print(f"[Evaluator] Loaded {len(pairs)} pairs from {meta_path}")
        return pairs

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self, model_name: str, exp: str) -> dict:
        p = self._ckpt_path(model_name, exp)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_checkpoint(self, model_name: str, exp: str, grouped: dict) -> None:
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        serialisable = {
            mt.value: {rid: scores for rid, scores in ref_groups.items()}
            for mt, ref_groups in grouped.items()
        }
        with open(self._ckpt_path(model_name, exp), "w", encoding="utf-8") as f:
            json.dump(serialisable, f)

    def clear_checkpoint(self, model_name: str, exp: str) -> None:
        p = self._ckpt_path(model_name, exp)
        if p.exists():
            p.unlink()
            print(f"[Evaluator] Checkpoint cleared: {p}")
        else:
            print(f"[Evaluator] No checkpoint to clear for {model_name}/{exp}")

    # ------------------------------------------------------------------
    # Core metric computation
    # ------------------------------------------------------------------

    def _run_metrics(
        self,
        pairs: List[dict],
        metrics: List[MetricType],
        model_name: str,
        exp: str,
        resume: bool = False,
    ) -> dict:
        self._fix_seed()

        grouped: dict = {}
        if resume:
            raw_ckpt = self._load_checkpoint(model_name, exp)
            if raw_ckpt:
                for mt in metrics:
                    if mt.value in raw_ckpt:
                        grouped[mt] = raw_ckpt[mt.value]
                        print(f"[Evaluator] Resumed {mt.value} from checkpoint "
                              f"({sum(len(v) for v in grouped[mt].values())} scores).")

        for metric_type in metrics:
            if metric_type in grouped:
                print(f"[Evaluator] Skipping {metric_type.value} (already in checkpoint).")
                continue

            print(f"\n[Evaluator] Calculating {metric_type.value} ...")
            config = ModelConfig(
                name=metric_type.value,
                batch_size=self.batch_size,
                device=self.device,
                additional_params={"seed": self.seed},
            )

            with create_calculator(metric_type, config) as calc:
                is_no_ref = metric_type in {MetricType.UTMOS}
                is_wer    = metric_type == MetricType.WER

                if is_no_ref:
                    unique_syn = sorted({p["syn_path"] for p in pairs})
                    calc_pairs = [(p, p) for p in unique_syn]
                    valid = calc.validate_audio_files(calc_pairs)
                    raw_scores = calc.calculate_batch_optimized(valid)
                    path_to_score = {valid[i][1]: raw_scores[i] for i in range(len(valid))}

                    syn_to_ref: dict = {}
                    for pi in pairs:
                        syn_to_ref.setdefault(pi["syn_path"], pi["ref_id"])

                    ref_scores: dict = {}
                    for syn_path, score in path_to_score.items():
                        if score is not None and not np.isnan(float(score)):
                            rid = syn_to_ref.get(syn_path)
                            if rid is not None:
                                ref_scores.setdefault(rid, []).append(float(score))
                    grouped[metric_type] = ref_scores

                elif is_wer:
                    audio_pairs = [(p["ref_path"], p["syn_path"]) for p in pairs]
                    valid = calc.validate_audio_files(audio_pairs)
                    pair_to_target = {
                        (pi["ref_path"], pi["syn_path"]): pi.get("target_text")
                        for pi in pairs
                    }

                    ref_scores = {}
                    for ref_path, syn_path in valid:
                        tgt = pair_to_target.get((ref_path, syn_path))
                        score = calc(synthesis=syn_path, reference=ref_path, target_text=tgt)
                        if score is not None and not np.isnan(float(score)):
                            score = float(min(1.0, max(0.0, score)))
                            rid = next(
                                (pi["ref_id"] for pi in pairs
                                 if pi["ref_path"] == ref_path and pi["syn_path"] == syn_path),
                                None,
                            )
                            if rid is not None:
                                ref_scores.setdefault(rid, []).append(score)
                    grouped[metric_type] = ref_scores

                else:
                    audio_pairs = [(p["ref_path"], p["syn_path"]) for p in pairs]
                    valid = calc.validate_audio_files(audio_pairs)
                    raw_scores = calc.calculate_batch_optimized(valid)

                    ref_scores = {}
                    pair_to_score = {valid[i]: raw_scores[i] for i in range(len(valid))}
                    for pi in pairs:
                        key = (pi["ref_path"], pi["syn_path"])
                        score = pair_to_score.get(key)
                        if score is not None and not np.isnan(float(score)):
                            ref_scores.setdefault(pi["ref_id"], []).append(float(score))
                    grouped[metric_type] = ref_scores

            n = sum(len(v) for v in grouped[metric_type].values())
            print(f"  → {n} scores collected")

            self._save_checkpoint(model_name, exp, grouped)

        return grouped

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_stats(grouped: dict) -> dict:
        stats: dict = {}
        for metric_type, ref_groups in grouped.items():
            all_scores = [s for v in ref_groups.values() for s in v]
            if not all_scores:
                continue
            k = metric_type.value
            stats[f"{k}_mean"]   = float(np.mean(all_scores))
            stats[f"{k}_std"]    = float(np.std(all_scores))
            stats[f"{k}_median"] = float(np.median(all_scores))
        return stats

    # ------------------------------------------------------------------
    # SIM pairwise helper (exp2 only)
    # ------------------------------------------------------------------

    def _evaluate_sim_intra(self, pairs: list[dict]) -> dict:
        """Pairwise intra-group speaker consistency for exp2.

        For each ref_id group:
            1. Extract ECAPA embeddings for all syn audio
            2. Compute cosine similarity for all N*(N-1)/2 pairs
            3. Report mean pairwise sim (↑ better)

        Groups with fewer than 2 valid embeddings are excluded.
        """
        groups: dict[str, list[Path]] = {}
        for p in pairs:
            groups.setdefault(p["ref_id"], []).append(p["syn_path"])

        config = ModelConfig(
            name="sim",
            batch_size=self.batch_size,
            device=self.device,
            additional_params={"seed": self.seed},
        )

        group_means: list[float] = []

        with create_calculator(MetricType.SIM, config) as calc:
            all_syn_paths = [p for paths in groups.values() for p in paths]
            embeddings = calc.extract_embeddings(all_syn_paths)

        for ref_id, syn_paths in tqdm(groups.items(), desc="Computing pairwise SIM", leave=False):
            group_embeds = [
                embeddings[p] for p in syn_paths if p in embeddings
            ]
            if len(group_embeds) < 2:
                continue

            # All pairwise cosine similarities (embeddings already L2-normalized)
            sims = [
                torch.dot(group_embeds[i], group_embeds[j]).item()
                for i, j in combinations(range(len(group_embeds)), 2)
            ]
            group_means.append(float(np.mean(sims)))

        stats: dict = {}
        if group_means:
            stats["sim_intra_mean"] = float(np.mean(group_means))  # ↑ better
            stats["sim_num_groups"] = len(group_means)

        return stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_exp1(
        self,
        model_name: str,
        metrics: Optional[List[MetricType]] = None,
        resume: bool = False,
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = list(self._DEFAULT_METRICS)

        pairs = self.load_pairs(model_name, "exp1")
        grouped = self._run_metrics(pairs, metrics, model_name, "exp1", resume=resume)
        stats = self._flatten_stats(grouped)

        df = pd.DataFrame([stats])
        df.insert(0, "model", model_name)
        df.insert(1, "exp", "exp1")
        return df

    def evaluate_exp2(
        self,
        model_name: str,
        metrics: Optional[List[MetricType]] = None,
        resume: bool = False,
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = list(self._DEFAULT_METRICS)

        pairs = self.load_pairs(model_name, "exp2")

        scalar_metrics = [m for m in metrics if m != MetricType.SIM]
        grouped = self._run_metrics(
            pairs, scalar_metrics, model_name, "exp2", resume=resume
        )

        stats: dict = {}

        # ── Scalar metrics (UTMOS, WER) ───────────────────────────────
        for metric_type, ref_groups in grouped.items():
            all_scores = [s for v in ref_groups.values() for s in v]
            if not all_scores:
                continue
            k = metric_type.value
            stats[f"{k}_mean"]   = float(np.mean(all_scores))
            stats[f"{k}_std"]    = float(np.std(all_scores))
            stats[f"{k}_median"] = float(np.median(all_scores))

            group_stds = []
            for scores in ref_groups.values():
                if len(scores) > 1:
                    group_stds.append(float(np.std(scores, ddof=1)))

            if group_stds:
                stats[f"{k}_avg_std"] = float(np.mean(group_stds))

        # ── SIM: pairwise consistency ─────────────────────────────────
        if MetricType.SIM in metrics:
            sim_stats = self._evaluate_sim_intra(pairs)
            stats.update(sim_stats)

        df = pd.DataFrame([stats])
        df.insert(0, "model", model_name)
        df.insert(1, "exp", "exp2")
        return df

    def compare_models(
        self,
        model_names: List[str],
        exp: str = "exp1",
        metrics: Optional[List[MetricType]] = None,
        resume: bool = False,
    ) -> pd.DataFrame:
        frames = []
        for model_name in model_names:
            print(f"\n{'='*60}")
            print(f"  Model: {model_name}  |  Exp: {exp}")
            print(f"{'='*60}")
            try:
                if exp == "exp1":
                    df = self.evaluate_exp1(model_name, metrics, resume=resume)
                elif exp == "exp2":
                    df = self.evaluate_exp2(model_name, metrics, resume=resume)
                else:
                    raise ValueError(f"Unknown exp: {exp!r}. Use 'exp1' or 'exp2'.")
                frames.append(df)
            except FileNotFoundError as e:
                print(f"[SKIP] {model_name}: {e}")

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def save_results(
        self,
        df: pd.DataFrame,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to save.")

        if output_path is None:
            exp = df["exp"].iloc[0] if "exp" in df.columns else "results"
            output_path = self.base_dir / f"eval_{self.dataset_name}_{exp}.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[Evaluator] Saved → {output_path}")
        return output_path


def main():
    """Main evaluation function."""
    evaluator = EvaluationPipeline()

    methods_to_run = [GenerationMethod.METHOD1, GenerationMethod.METHOD2, GenerationMethod.METHOD3]

    results = evaluator.evaluate_dataset_model(
        dataset_type=DatasetType.LIBRITTS,
        model_type=ModelType.PARLER_TTS_MINI_V1,
        methods=methods_to_run
    )

    evaluator.save_results_to_csv(
        results,
        DatasetType.LIBRITTS,
        ModelType.PARLER_TTS_MINI_V1
    )


if __name__ == "__main__":
    main()
