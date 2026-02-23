from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from voicestudio.models.parler_tts import ParlerTTSForConditionalGeneration
from voicestudio._qwen3_tts.inference.qwen3_tts_model import Qwen3TTSModel
from voicestudio._qwen3_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

from spk_incon.datasets import LIBRITTS_P_Custom
from spk_incon.metrics import MetricType
from spk_incon.utils.evaluate import Evaluator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_MODELS = {
    "parler-mini": "parler-tts/parler-tts-mini-v1",
    "qwen-vd": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

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

ALL_METRICS = ["utmos", "wer", "sim"]

METRIC_MAP = {
    "utmos": MetricType.UTMOS,
    "wer": MetricType.WER,
    "sim": MetricType.SIM,
}


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model type detection
# ---------------------------------------------------------------------------


def detect_model_type(model_name: str) -> str:
    """Return model type string. Only parler-mini and qwen-vd are supported."""
    if model_name == "parler-mini":
        return "parler"
    if model_name == "qwen-vd":
        return "qwen-vd"
    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Supported: {list(BASELINE_MODELS.keys())}"
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_samples(
    data_root: Path,
    url: str = "test-clean",
    annotator: str = "df1",
    num_samples: int = -1,
    use_synthetic: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load samples for Exp1 (N × 1-to-1 pairs) using LIBRITTS_P_Custom.

    Filtering is disabled (max_z_score=inf, min_group_size=1) so the full
    split is available for random sampling.
    """
    print(f"[Dataset] Loading {url} (annotator: {annotator})")

    ds = LIBRITTS_P_Custom(
        root=data_root,
        url=url,
        annotator=annotator,
        max_z_score=float("inf"),
        min_group_size=1,
        use_synthetic=use_synthetic,
        download=True,
    )

    total = len(ds)
    rng = random.Random(seed)
    n = total if (num_samples == -1 or num_samples >= total) else num_samples
    indices = rng.sample(range(total), min(n, total))

    samples, skipped = [], 0
    for idx in indices:
        item = ds[idx]
        audio_path = Path(ds.data[idx]["audio_path"])
        if not audio_path.exists():
            skipped += 1
            continue

        samples.append(
            {
                "audio_path": audio_path,
                "normalized_text": item["normalized_text"],
                "utterance_id": item["utterance_id"],
                "speaker_id": str(item["speaker_id"]),
                "combined_prompt": item["combined_prompt"],
            }
        )

    if skipped:
        print(f"[Dataset] Skipped {skipped} missing files")
    print(f"[Dataset] Loaded {len(samples)} samples")
    return samples


def load_groups(
    data_root: Path,
    url: str = "test-clean",
    annotator: str = "df1",
    num_groups: int = 50,
    use_synthetic: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load unique (group_id, speaker) groups for Exp2 using LIBRITTS_P_Custom.

    Deduplication uses (group_id, speaker_id) from the dataset directly,
    which works correctly for both normal and synthetic prompt modes.

    Filtering is disabled so the full split is available for group selection.
    """
    print(f"[Dataset] Loading groups from {url} (annotator: {annotator})")

    ds = LIBRITTS_P_Custom(
        root=data_root,
        url=url,
        annotator=annotator,
        max_z_score=float("inf"),
        min_group_size=1,
        use_synthetic=use_synthetic,
        download=True,
    )

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    seen: set = set()
    groups: List[Dict[str, Any]] = []

    for idx in indices:
        item = ds[idx]
        audio_path = Path(ds.data[idx]["audio_path"])
        if not audio_path.exists():
            continue

        spk_id = str(item["speaker_id"])
        key = (item["group_id"], spk_id)
        if key in seen:
            continue
        seen.add(key)

        groups.append(
            {
                "group_id": f"g{len(groups):03d}",
                "ref_path": audio_path,
                "prompt": item["combined_prompt"],
                "speaker_id": spk_id,
            }
        )

        if len(groups) >= num_groups:
            break

    print(f"[Dataset] Loaded {len(groups)} groups")
    return groups


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_name: str, device_map: str) -> tuple:
    mtype = detect_model_type(model_name)
    model_id = BASELINE_MODELS[model_name]

    if mtype == "parler":
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        sr = model.config.audio_encoder.sampling_rate

    else:  # qwen-vd
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        tokenizer = model.processor.tokenizer
        sr = getattr(model.model.config, "sampling_rate", None)
        if sr is None:
            sr = getattr(model.model.config.speaker_encoder_config, "sample_rate", 24000)

    if hasattr(model, "eval"):
        model.eval()

    inner = getattr(model, "model", model)
    if hasattr(inner, "speech_tokenizer") and inner.speech_tokenizer is None:
        st_dir = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
        inner.load_speech_tokenizer(
            Qwen3TTSTokenizer.from_pretrained(st_dir, device_map=device_map)
        )

    print(f"[Model] {model_id} loaded (SR: {sr})")
    return model, tokenizer, mtype, int(sr)


# ---------------------------------------------------------------------------
# Synthesis function factory
# ---------------------------------------------------------------------------


def make_synthesize_fn(
    model,
    tokenizer,
    mtype: str,
    sr: int,
    device: torch.device,
    seed: int = 42,
) -> Callable:
    if tokenizer and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def synthesize_single(sample: dict, syn_path: Path) -> Optional[dict]:
        fix_seed(seed)  # intentional: same seed per sample for reproducibility
        text = sample["normalized_text"]
        style = sample["combined_prompt"]

        with torch.inference_mode():
            if mtype == "parler":
                d_in = tokenizer(style, return_tensors="pt").to(device)
                p_in = tokenizer(text, return_tensors="pt").to(device)
                gen = model.generate(
                    input_ids=d_in.input_ids,
                    prompt_input_ids=p_in.input_ids,
                )
                wav = gen.cpu().numpy().squeeze()

            elif mtype == "qwen-vd":
                wavs, _ = model.generate_voice_design(
                    text=text, instruct=style, language="English"
                )
                wav = wavs[0]

            else:
                raise ValueError(f"Unsupported model type: {mtype}")

        syn_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(syn_path), wav, sr)

        if not syn_path.exists() or syn_path.stat().st_size == 0:
            return None

        return {
            "ref_path": str(sample["audio_path"].resolve()),
            "syn_path": str(syn_path.resolve()),
            "target_text": text,
            "utterance_id": sample["utterance_id"],
            "speaker_id": sample["speaker_id"],
        }

    return synthesize_single


# ---------------------------------------------------------------------------
# Synthesis — Exp1
# ---------------------------------------------------------------------------


def _write_metadata(meta_path: Path, model_name: str, exp: str, pairs: list) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": model_name, "exp": exp, "pairs": pairs},
            f,
            indent=2,
            ensure_ascii=False,
        )


def synthesize_exp1(
    samples: List[Dict[str, Any]],
    model_name: str,
    synthesize_fn: Callable,
    output_dir: Path,
    dataset_name: str = "",
    resume: bool = False,
) -> None:
    """Exp1: N × 1-to-1 pairs."""
    syn_dir = output_dir / "syn" / dataset_name / model_name / "exp1"
    meta_path = syn_dir / "metadata.json"
    syn_dir.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, dict] = {}
    if resume and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for entry in json.load(f).get("pairs", []):
                existing[str(entry["index"])] = entry
        print(f"[Synth1] Resume: {len(existing)} already done")

    pairs, pending = [], []
    for i, sample in enumerate(samples):
        idx_str = f"{i:03d}"
        if resume and idx_str in existing:
            pairs.append(existing[idx_str])
        else:
            pending.append((idx_str, sample))

    if not pending:
        _write_metadata(meta_path, model_name, "exp1", pairs)
        print(f"[Synth1] All done -> {meta_path}")
        return

    pbar = tqdm(total=len(samples), initial=len(pairs), desc=f"[Synth1] {model_name}")
    for idx_str, sample in pending:
        syn_path = syn_dir / f"syn_{idx_str}.wav"
        pbar.set_postfix(uid=sample["utterance_id"])
        try:
            res = synthesize_fn(sample, syn_path)
            if res:
                res["index"] = idx_str
                pairs.append(res)
            else:
                print(f"  [WARN] {idx_str}: synthesis failed")
        except Exception as exc:
            print(f"  [ERROR] {idx_str}: {exc}")
        pbar.update(1)
        _write_metadata(meta_path, model_name, "exp1", pairs)

    _write_metadata(meta_path, model_name, "exp1", pairs)
    print(f"[Synth1] Done — {len(pairs)}/{len(samples)} pairs -> {meta_path}")


# ---------------------------------------------------------------------------
# Synthesis — Exp2
# ---------------------------------------------------------------------------


def synthesize_exp2(
    groups: List[Dict[str, Any]],
    model_name: str,
    synthesize_fn: Callable,
    output_dir: Path,
    dataset_name: str = "",
    resume: bool = False,
) -> None:
    """Exp2: num_groups × len(COMPARISON_TEXTS) synthesis runs."""
    syn_dir = output_dir / "syn" / dataset_name / model_name / "exp2"
    meta_path = syn_dir / "metadata.json"
    syn_dir.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, dict] = {}
    if resume and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for entry in json.load(f).get("pairs", []):
                existing[str(entry["index"])] = entry
        print(f"[Synth2] Resume: {len(existing)} already done")

    total = len(groups) * len(COMPARISON_TEXTS)
    pairs, pending = [], []
    for g_idx, group in enumerate(groups):
        for t_idx, text in enumerate(COMPARISON_TEXTS):
            idx_str = f"{g_idx:03d}_{t_idx:02d}"
            if resume and idx_str in existing:
                pairs.append(existing[idx_str])
            else:
                pending.append((idx_str, group, text))

    if not pending:
        _write_metadata(meta_path, model_name, "exp2", pairs)
        print(f"[Synth2] All done -> {meta_path}")
        return

    pbar = tqdm(total=total, initial=len(pairs), desc=f"[Synth2] {model_name}")
    for idx_str, group, text in pending:
        sample = {
            "audio_path": group["ref_path"],
            "normalized_text": text,
            "utterance_id": idx_str,
            "speaker_id": group["speaker_id"],
            "combined_prompt": group["prompt"],
        }
        syn_path = syn_dir / f"syn_{idx_str}.wav"
        try:
            res = synthesize_fn(sample, syn_path)
            if res:
                res["index"] = idx_str
                res["ref_id"] = group["group_id"]
                pairs.append(res)
            else:
                print(f"  [WARN] {idx_str}: synthesis failed")
        except Exception as exc:
            print(f"  [ERROR] {idx_str}: {exc}")
        pbar.update(1)
        _write_metadata(meta_path, model_name, "exp2", pairs)

    _write_metadata(meta_path, model_name, "exp2", pairs)
    print(f"[Synth2] Done — {len(pairs)}/{total} pairs -> {meta_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_exp1(
    model_name: str,
    output_dir: Path,
    metrics: List[str],
    dataset_name: str = "",
    device: str = "cuda",
    batch_size: int = 16,
    seed: int = 42,
    resume: bool = False,
) -> dict:
    metric_types = [METRIC_MAP[m] for m in metrics]
    evaluator = Evaluator(
        base_dir=output_dir,
        dataset_name="",
        device=device,
        seed=seed,
        batch_size=batch_size,
        syn_base_dir=output_dir / "syn" / dataset_name,
    )
    df = evaluator.evaluate_exp1(model_name, metric_types, resume=resume)
    row = df.iloc[0].to_dict() if not df.empty else {}
    row.pop("model", None)
    row.pop("exp", None)
    return row


def evaluate_exp2(
    model_name: str,
    output_dir: Path,
    dataset_name: str = "",
    device: str = "cuda",
    batch_size: int = 16,
    seed: int = 42,
    resume: bool = False,
) -> dict:
    evaluator = Evaluator(
        base_dir=output_dir,
        dataset_name="",
        device=device,
        seed=seed,
        batch_size=batch_size,
        syn_base_dir=output_dir / "syn" / dataset_name,
    )
    df = evaluator.evaluate_exp2(
        model_name,
        [MetricType.UTMOS, MetricType.WER, MetricType.SIM],
        resume=resume,
    )
    row = df.iloc[0].to_dict() if not df.empty else {}
    row.pop("model", None)
    row.pop("exp", None)
    return row


# ---------------------------------------------------------------------------
# Single-model run
# ---------------------------------------------------------------------------


def run_one(
    model_name: str,
    args,
    samples: Optional[list],
    groups: Optional[list],
    dataset_name: str,
) -> Dict[str, dict]:
    output_dir = Path(args.output_dir)

    if args.device == "cuda" and torch.cuda.is_available():
        device, device_map = torch.device("cuda:0"), "cuda:0"
    else:
        device, device_map = torch.device("cpu"), "cpu"

    run_exp1 = args.exp in ("1", "all")
    run_exp2 = args.exp in ("2", "all")

    # ── Synthesis ──────────────────────────────────────────────────────
    need_synth = (run_exp1 or run_exp2) and not args.eval_only
    if need_synth:
        model, tokenizer, mtype, sr = load_model(model_name, device_map)
        synth_fn = make_synthesize_fn(model, tokenizer, mtype, sr, device, args.seed)

        if run_exp1:
            synthesize_exp1(
                samples=samples,
                model_name=model_name,
                synthesize_fn=synth_fn,
                output_dir=output_dir,
                dataset_name=dataset_name,
                resume=args.resume,
            )

        if run_exp2:
            synthesize_exp2(
                groups=groups,
                model_name=model_name,
                synthesize_fn=synth_fn,
                output_dir=output_dir,
                dataset_name=dataset_name,
                resume=args.resume,
            )

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.synth_only:
        return {}

    # ── Evaluation ─────────────────────────────────────────────────────
    results = {}

    if run_exp1:
        print(f"\n[Eval1] {model_name} ...")
        results["exp1"] = evaluate_exp1(
            model_name=model_name,
            output_dir=output_dir,
            metrics=args.metrics,
            dataset_name=dataset_name,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            resume=args.resume,
        )

    if run_exp2:
        print(f"\n[Eval2] {model_name} ...")
        results["exp2"] = evaluate_exp2(
            model_name=model_name,
            output_dir=output_dir,
            dataset_name=dataset_name,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            resume=args.resume,
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp1 + Exp2: synthesis and evaluation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        choices=list(BASELINE_MODELS.keys()),
        help=f"Model name. Supported: {list(BASELINE_MODELS.keys())}",
    )
    group.add_argument(
        "--all-baselines",
        action="store_true",
        help="Run all baseline models sequentially.",
    )
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./results")
    p.add_argument(
        "--exp",
        choices=["1", "2", "all"],
        default="1",
        help="Which experiment to run (default: 1).",
    )
    p.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic speaker descriptions as prompts.",
    )
    p.add_argument(
        "--metrics", nargs="+", default=ALL_METRICS, choices=list(METRIC_MAP.keys())
    )
    p.add_argument(
        "--num-samples", type=int, default=-1, help="Exp1 samples. -1 = all."
    )
    p.add_argument(
        "--num-groups", type=int, default=50, help="Exp2 groups (default: 50)."
    )
    p.add_argument("--url", default="test-clean")
    p.add_argument("--annotator", default="df1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--synth-only", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    fix_seed(args.seed)

    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)
    dataset_name = re.sub(r"[^\w.\-]", "_", args.url)

    output_dir.mkdir(parents=True, exist_ok=True)

    run_exp1 = args.exp in ("1", "all")
    run_exp2 = args.exp in ("2", "all")

    # ── Model list ─────────────────────────────────────────────────────
    models_to_run: List[str] = (
        list(BASELINE_MODELS.keys()) if args.all_baselines else [args.model]
    )

    # ── Load data ──────────────────────────────────────────────────────
    samples, groups = None, None

    if run_exp1 and not args.eval_only:
        samples = load_samples(
            data_root=data_root,
            url=args.url,
            annotator=args.annotator,
            num_samples=args.num_samples,
            use_synthetic=args.use_synthetic,
            seed=args.seed,
        )

    if run_exp2 and not args.eval_only:
        groups = load_groups(
            data_root=data_root,
            url=args.url,
            annotator=args.annotator,
            num_groups=args.num_groups,
            use_synthetic=args.use_synthetic,
            seed=args.seed,
        )

    # ── Output CSV paths ───────────────────────────────────────────────
    out_csv1 = output_dir / f"exp1_{dataset_name}.csv"
    out_csv2 = output_dir / f"exp2_{dataset_name}.csv"

    # ── Run models ─────────────────────────────────────────────────────
    for model_name in models_to_run:
        model_id = BASELINE_MODELS[model_name]
        print(f"\n{'=' * 70}")
        print(f"  model : {model_id}  ({model_name})")
        print(f"  url   : {args.url}  |  exp: {args.exp}")
        print(f"{'=' * 70}")

        try:
            results = run_one(
                model_name=model_name,
                args=args,
                samples=samples,
                groups=groups,
                dataset_name=dataset_name,
            )
        except FileNotFoundError as e:
            print(f"[SKIP] {model_name}: {e}")
            continue

        if args.synth_only or not results:
            continue

        # ── Save Exp1 ──────────────────────────────────────────────────
        if "exp1" in results and results["exp1"]:
            stats = results["exp1"]
            stats["model"] = model_name
            stats["dataset"] = args.url
            cols = ["model", "dataset"] + [
                c for c in stats if c not in ("model", "dataset")
            ]
            df = pd.DataFrame([stats])[cols]
            print("\n[Exp1 Results]")
            print(df.to_string(index=False))

            if out_csv1.exists():
                prev = pd.read_csv(out_csv1)
                mask = (prev["model"] != model_name) | (
                    prev.get("dataset", "") != args.url
                )
                df = pd.concat([prev[mask], df], ignore_index=True)
            df.to_csv(out_csv1, index=False)
            print(f"[run_exp] Saved -> {out_csv1}")

        # ── Save Exp2 ──────────────────────────────────────────────────
        if "exp2" in results and results["exp2"]:
            stats = results["exp2"]
            stats["model"] = model_name
            stats["dataset"] = args.url
            cols = ["model", "dataset"] + [
                c for c in stats if c not in ("model", "dataset")
            ]
            df = pd.DataFrame([stats])[cols]
            print("\n[Exp2 Results]")
            print(df.to_string(index=False))

            if out_csv2.exists():
                prev = pd.read_csv(out_csv2)
                mask = (prev["model"] != model_name) | (
                    prev.get("dataset", "") != args.url
                )
                df = pd.concat([prev[mask], df], ignore_index=True)
            df.to_csv(out_csv2, index=False)
            print(f"[run_exp] Saved -> {out_csv2}")

    # ── Print final tables ─────────────────────────────────────────────
    for label, path in [("Exp1", out_csv1), ("Exp2", out_csv2)]:
        if path.exists():
            final_df = pd.read_csv(path)
            print(f"\n{'=' * 70}")
            print(f"[{label} Final Results] Tab-separated:")
            print("=" * 70)
            print(final_df.to_csv(sep="\t", index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
