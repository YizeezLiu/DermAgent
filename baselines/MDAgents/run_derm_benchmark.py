#!/usr/bin/env python3
# =============================================================================
# Adapted from MDAgents (https://github.com/mitmedialab/MDAgents)
# Original work: Kim et al., "MDAgents: An Adaptive Collaboration of LLMs
#   for Medical Decision-Making", NeurIPS 2024 (Oral).
# Upstream license: no license file. Used with attribution; see
#   baselines/MDAgents/ATTRIBUTION.md for details.
# Modifications by the DermAgent authors for dermatology benchmarks.
# =============================================================================

"""
MDAgents benchmark runner for DermAgent tasks.

Adapts the MDAgents adaptive multi-agent collaboration framework
(basic / intermediate / advanced routing) to run on dermatology
benchmark datasets.

Usage:
    python run_derm_benchmark.py --dataset HAM10000 --difficulty basic --num_samples 500
    python run_derm_benchmark.py --dataset SNU_500 --difficulty basic --num_samples 500
"""

import os
import sys
import json
import csv
import re
import random
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root is DermAgent_submission/ (two levels up from baselines/MDAgents/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Allow importing DermAgent's benchmark.metrics
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    Agent, determine_difficulty,
    process_basic_query, process_intermediate_query, process_advanced_query,
)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASETS = {
    # ---- Task 1: Diagnosis (zero-shot classification) ----
    "HAM10000": {
        "task": 1,
        "csv": "datasets/ham10000/HAM10000_benchmark_500.csv",
        "image_dir": "datasets/ham10000/images",
        "filename_col": "image_id",
        "label_col": "dx",
        "preserve_subdir": False,
        "ext": ".jpg",
        "label_map": {
            "nv": "melanocytic nevi", "mel": "melanoma",
            "bcc": "basal cell carcinoma", "akiec": "actinic keratosis",
            "bkl": "benign keratosis", "df": "dermatofibroma",
            "vasc": "vascular lesion",
        },
        "classes": [
            "melanocytic nevi", "melanoma", "basal cell carcinoma",
            "actinic keratosis", "benign keratosis", "dermatofibroma",
            "vascular lesion",
        ],
    },
    "SNU_500": {
        "task": 1,
        "csv": "datasets/SNU/MAKE_SNU_500.csv",
        "image_dir": "datasets/SNU/images",
        "filename_col": "filename",
        "label_col": "diag",
        "preserve_subdir": False,
        "classes": None,  # open-set, collected from CSV at runtime
    },
    # ---- Task 2: Concept annotation (multi-label) ----
    "derm7pt": {
        "task": 2,
        "csv": "datasets/derm7pt/meta_task2_sample_500.csv",
        "image_dir": "datasets/derm7pt/final_images",
        "filename_col": "ImageID",
        "concepts": [
            "pigment_network", "blue_whitish_veil", "vascular_structures",
            "pigmentation", "streaks", "dots_and_globules",
            "regression_structures",
        ],
    },
    "derm7pt_100": {
        "task": 2,
        "csv": "datasets/derm7pt/meta_task2_sample_100.csv",
        "image_dir": "datasets/derm7pt/final_images",
        "filename_col": "ImageID",
        "concepts": [
            "pigment_network", "blue_whitish_veil", "vascular_structures",
            "pigmentation", "streaks", "dots_and_globules",
            "regression_structures",
        ],
    },
    "skincon": {
        "task": 2,
        "csv": "datasets/skincon/meta_task2_sample_500.csv",
        "image_dir": "datasets/skincon/final_images",
        "filename_col": "ImageID",
        "concepts": [
            "Vesicle", "Papule", "Macule", "Plaque", "Pustule", "Bulla",
            "Patch", "Nodule", "Ulcer", "Crust", "Erosion", "Excoriation",
            "Atrophy", "Exudate", "Fissure", "Induration", "Xerosis",
            "Telangiectasia", "Scale", "Scar", "Friable", "Pedunculated",
            "Exophytic/Fungating", "Warty/Papillomatous", "Dome-shaped",
            "Umbilicated", "Brown(Hyperpigmentation)",
            "White(Hypopigmentation)", "Purple", "Yellow", "Black",
            "Erythema",
        ],
    },
    "skincon_100": {
        "task": 2,
        "csv": "datasets/skincon/meta_task2_sample_100.csv",
        "image_dir": "datasets/skincon/final_images",
        "filename_col": "ImageID",
        "concepts": [
            "Vesicle", "Papule", "Macule", "Plaque", "Pustule", "Bulla",
            "Patch", "Nodule", "Ulcer", "Crust", "Erosion", "Excoriation",
            "Atrophy", "Exudate", "Fissure", "Induration", "Xerosis",
            "Telangiectasia", "Scale", "Scar", "Friable", "Pedunculated",
            "Exophytic/Fungating", "Warty/Papillomatous", "Dome-shaped",
            "Umbilicated", "Brown(Hyperpigmentation)",
            "White(Hypopigmentation)", "Purple", "Yellow", "Black",
            "Erythema",
        ],
    },
    # ---- Task 3: Image captioning ----
    "skin_cap": {
        "task": 3,
        "csv": "datasets/skin_cap/skin_cap_meta_100.csv",
        "image_dir": "datasets/skin_cap/images",
        "filename_col": "filename",
        "caption_col": "caption_zh_polish_en",
        "preserve_subdir": False,
    },
}

# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

def resolve_image_path(row: dict, cfg: dict) -> str:
    """Return the absolute image path for a data row."""
    raw = row[cfg["filename_col"]]
    img_dir = PROJECT_ROOT / cfg["image_dir"]
    if cfg.get("ext"):
        return str(img_dir / f"{raw}{cfg['ext']}")
    if cfg.get("preserve_subdir"):
        return str(img_dir / raw)
    return str(img_dir / Path(raw).name)


# ---------------------------------------------------------------------------
# Ground-truth extraction helpers
# ---------------------------------------------------------------------------

def get_gt_label(row: dict, cfg: dict) -> str:
    """Return ground truth label string for Task 1."""
    raw = row[cfg["label_col"]]
    label_map = cfg.get("label_map")
    if label_map:
        return label_map.get(raw, raw)
    return raw


def get_gt_concepts(row: dict, cfg: dict) -> list:
    """Return list of present concept names for Task 2."""
    return [c for c in cfg["concepts"] if str(row.get(c, "0")).strip() == "1"]


def get_gt_caption(row: dict, cfg: dict) -> str:
    """Return ground truth caption for Task 3."""
    return row[cfg["caption_col"]]


def get_gt_vqa(row: dict) -> str:
    """Return ground truth answer for Task 4."""
    return row["gt_answer"]


def get_options(row: dict, cfg: dict) -> list:
    """Return non-empty option strings for Task 4."""
    opts = []
    for col in cfg["option_cols"]:
        val = row.get(col, "")
        if val and str(val).strip():
            opts.append(str(val).strip())
    return opts


# ---------------------------------------------------------------------------
# Question formatters
# ---------------------------------------------------------------------------

def format_question_task1(cfg: dict) -> str:
    """Build the question text for a Task 1 (diagnosis) sample."""
    classes = cfg.get("classes")
    if classes is None:
        # Open-set (SNU): no option list
        return (
            "Examine this clinical/dermoscopic skin image carefully. "
            "What is the most likely dermatological diagnosis? "
            "Answer with only the disease name."
        )
    # Closed-set: list options
    option_str = ", ".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(classes))
    return (
        "Examine this clinical/dermoscopic skin image carefully. "
        "Which of the following diagnoses is most likely?\n"
        f"Options: {option_str}\n"
        "Answer with the disease name only."
    )


def format_question_task2(cfg: dict) -> str:
    """Build the question text for a Task 2 (concept annotation) sample."""
    concept_str = ", ".join(cfg["concepts"])
    return (
        "Examine this dermoscopic skin image carefully. "
        "Which of the following dermoscopic concepts are present in the image?\n"
        f"Concepts: {concept_str}\n"
        "List ONLY the concepts that are clearly present, separated by commas. "
        "If none are present, answer 'none'."
    )


def format_question_task3() -> str:
    """Build the question text for a Task 3 (captioning) sample."""
    return (
        "Provide a detailed medical description of this skin lesion image, "
        "including the morphology, possible diagnosis, and clinical features."
    )


def format_question_task4(row: dict, cfg: dict) -> str:
    """Build the question text for a Task 4 (VQA) sample."""
    question = row["question"]
    options = get_options(row, cfg)
    option_str = " ".join(
        f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options)
    )
    return f"{question}\nOptions: {option_str}\nAnswer with the option letter or text."


# ---------------------------------------------------------------------------
# Answer parsers
# ---------------------------------------------------------------------------

def parse_answer_task1(response: str, cfg: dict) -> str:
    """Extract diagnosis from MDAgents response for Task 1."""
    text = _extract_answer_text(response)
    classes = cfg.get("classes")
    if classes is None:
        return text  # open-set

    text_lower = text.lower().strip()
    # Exact match
    for c in classes:
        if c.lower() == text_lower:
            return c
    # Partial / contains match
    for c in classes:
        if c.lower() in text_lower or text_lower in c.lower():
            return c
    # Try to find any class mentioned anywhere in the response
    resp_lower = response.lower() if isinstance(response, str) else str(response).lower()
    for c in classes:
        if c.lower() in resp_lower:
            return c
    return text


def _normalize_concept(s: str) -> str:
    """Normalize concept string for robust matching.

    Strips whitespace around parentheses/slashes so that
    'Brown (Hyperpigmentation)' matches 'Brown(Hyperpigmentation)'.
    """
    s = re.sub(r"\s*\(\s*", "(", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    s = re.sub(r"\s*/\s*", "/", s)
    return s.strip().lower()


def parse_answer_task2(response: str, cfg: dict) -> list:
    """Extract concept list from MDAgents response for Task 2."""
    text = _extract_answer_text(response)
    if text.lower().strip() in ("none", "n/a", ""):
        return []

    raw = re.split(r"[,;\n]", text)
    raw = [r.strip() for r in raw if r.strip()]

    # Build mapping: normalized_concept -> original_concept
    valid_map = {_normalize_concept(c): c for c in cfg["concepts"]}
    matched = []
    for r in raw:
        r_norm = _normalize_concept(r)
        # Exact match after normalization
        if r_norm in valid_map and valid_map[r_norm] not in matched:
            matched.append(valid_map[r_norm])
        else:
            # Fuzzy substring match (also normalized)
            for vl, vo in valid_map.items():
                if (r_norm in vl or vl in r_norm) and vo not in matched:
                    matched.append(vo)
                    break
    return matched


def parse_answer_task3(response) -> str:
    """Extract caption from MDAgents response for Task 3.

    Unlike MCQ tasks, captioning needs the FULL response text, not just the
    last line.  We only unwrap the MDAgents dict layers and strip markdown
    headings, keeping the complete description intact.
    """
    # Unwrap dict layers (same as _extract_answer_text, but keep full text)
    if isinstance(response, dict):
        if "majority" in response:
            response = response["majority"]
        if isinstance(response, dict):
            response = next(iter(response.values()), "")
    if not isinstance(response, str):
        response = str(response)

    text = response.strip()
    if not text:
        return ""

    # Strip markdown headings (### Morphology:  ->  Morphology:)
    text = re.sub(r"^#{1,4}\s*", "", text, flags=re.MULTILINE)
    # Strip markdown bold markers
    text = text.replace("**", "")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_answer_task4(response: str, row: dict, cfg: dict) -> str:
    """Extract VQA answer from MDAgents response for Task 4."""
    text = _extract_answer_text(response)
    options = get_options(row, cfg)

    if not text:
        text = response.strip() if isinstance(response, str) else str(response)

    text_lower = text.lower().strip()

    # Single letter answer (A, B, C, ...)
    if len(text_lower) == 1 and text_lower in "abcdefgh":
        idx = ord(text_lower) - ord("a")
        if idx < len(options):
            return options[idx]

    # Exact match
    for opt in options:
        if opt.lower() == text_lower:
            return opt
    # Partial match
    for opt in options:
        if opt.lower() in text_lower or text_lower in opt.lower():
            return opt
    return text


def _extract_answer_text(response) -> str:
    """
    Pull out the core answer from MDAgents' response.
    MDAgents wraps the final answer in a dict keyed by temperature (basic)
    or in a dict keyed by 'majority' -> {temp: text} (intermediate/advanced).
    """
    # Unwrap dict layers that MDAgents returns
    if isinstance(response, dict):
        # e.g. {'majority': {0.0: 'Answer: ...'}}
        if "majority" in response:
            response = response["majority"]
        if isinstance(response, dict):
            # e.g. {0.0: 'Answer: ...'}
            response = next(iter(response.values()), "")

    if not isinstance(response, str):
        response = str(response)

    # Try to find "Answer: ..." pattern
    m = re.search(r"Answer:\s*(?:\([A-Za-z]\)\s*)?(.+?)(?:\n|$)", response, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("`\"'")

    # Fallback: last non-empty line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list:
    """Read a CSV file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------------
# Metrics (imported from DermAgent)
# ---------------------------------------------------------------------------

try:
    # Import metrics directly to avoid benchmark/__init__.py which pulls in torch
    import importlib.util
    _metrics_path = str(PROJECT_ROOT / "benchmark" / "metrics.py")
    _spec = importlib.util.spec_from_file_location("benchmark_metrics", _metrics_path)
    _metrics_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_metrics_mod)

    compute_classification_metrics = _metrics_mod.compute_classification_metrics
    compute_multilabel_metrics = _metrics_mod.compute_multilabel_metrics
    compute_captioning_metrics = _metrics_mod.compute_captioning_metrics
    compute_vqa_metrics = _metrics_mod.compute_vqa_metrics
    METRICS_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Could not import benchmark.metrics: {e}. "
          "Metrics will be computed with basic accuracy only.")
    METRICS_AVAILABLE = False


def compute_basic_accuracy(y_true: list, y_pred: list) -> dict:
    """Fallback accuracy when sklearn / nltk are not installed."""
    correct = sum(1 for a, b in zip(y_true, y_pred) if str(a).lower() == str(b).lower())
    return {"accuracy": correct / len(y_true) if y_true else 0.0}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmark(args):
    dataset_name = args.dataset
    if dataset_name not in DATASETS:
        print(f"[ERROR] Unknown dataset '{dataset_name}'. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    cfg = DATASETS[dataset_name]
    task = cfg["task"]

    # Load CSV
    csv_path = PROJECT_ROOT / cfg["csv"]
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)
    data = load_csv(str(csv_path))
    print(f"[INFO] Loaded {len(data)} samples from {csv_path}")

    # For SNU open-set: collect unique classes from CSV
    if task == 1 and cfg.get("classes") is None:
        cfg["classes"] = sorted(set(row[cfg["label_col"]] for row in data))
        print(f"[INFO] SNU open-set: collected {len(cfg['classes'])} unique classes")

    # Limit samples
    if args.num_samples and args.num_samples < len(data):
        data = data[:args.num_samples]
    print(f"[INFO] Running on {len(data)} samples | task={task} | "
          f"difficulty={args.difficulty} | model={args.model}")

    # Build a mock args object compatible with MDAgents' process_*_query functions
    mdagents_args = argparse.Namespace(dataset='derm', model=args.model)

    results = []
    y_true_list = []
    y_pred_list = []

    for idx, row in enumerate(tqdm(data, desc=f"[{dataset_name}]")):
        img_path = resolve_image_path(row, cfg)

        # Validate image exists
        if not os.path.isfile(img_path):
            print(f"  [WARN] Image not found: {img_path}, skipping")
            continue

        # ----- Build question -----
        if task == 1:
            question = format_question_task1(cfg)
            gt = get_gt_label(row, cfg)
        elif task == 2:
            question = format_question_task2(cfg)
            gt = get_gt_concepts(row, cfg)
        elif task == 3:
            question = format_question_task3()
            gt = get_gt_caption(row, cfg)
        elif task == 4:
            question = format_question_task4(row, cfg)
            gt = get_gt_vqa(row)
        else:
            raise ValueError(f"Unknown task: {task}")

        # ----- Determine difficulty & run MDAgents pipeline -----
        difficulty = determine_difficulty(question, args.difficulty)
        if difficulty is None:
            difficulty = "basic"

        task_hint = "captioning" if task == 3 else None
        try:
            if difficulty == "basic":
                response = process_basic_query(
                    question, [], args.model, mdagents_args,
                    img_path=img_path, task_hint=task_hint)
            elif difficulty == "intermediate":
                response = process_intermediate_query(
                    question, [], args.model, mdagents_args,
                    img_path=img_path, task_hint=task_hint)
            elif difficulty == "advanced":
                response = process_advanced_query(
                    question, args.model, mdagents_args,
                    img_path=img_path, task_hint=task_hint)
            else:
                response = process_basic_query(
                    question, [], args.model, mdagents_args,
                    img_path=img_path, task_hint=task_hint)
        except Exception as e:
            print(f"  [ERROR] Sample {idx}: {e}")
            response = ""

        # ----- Parse answer -----
        if task == 1:
            pred = parse_answer_task1(response, cfg)
            y_true_list.append(gt)
            y_pred_list.append(pred)
        elif task == 2:
            pred = parse_answer_task2(response, cfg)
            y_true_list.append(gt)
            y_pred_list.append(pred)
        elif task == 3:
            pred = parse_answer_task3(response)
            y_true_list.append(gt)
            y_pred_list.append(pred)
        elif task == 4:
            pred = parse_answer_task4(response, row, cfg)
            y_true_list.append(gt)
            y_pred_list.append(pred)

        results.append({
            "index": idx,
            "image": img_path,
            "ground_truth": gt,
            "prediction": pred,
            "difficulty": difficulty,
            "raw_response": str(response)[:2000],  # truncate for storage
        })

        # Progress print every 10 samples
        if (idx + 1) % 10 == 0:
            if task == 1:
                correct = sum(1 for r in results if str(r["ground_truth"]).lower() == str(r["prediction"]).lower())
                print(f"  [{idx+1}/{len(data)}] running accuracy: {correct}/{len(results)} "
                      f"= {correct/len(results):.3f}")

    # ----- Compute metrics -----
    print(f"\n{'='*60}")
    print(f"EVALUATION: {dataset_name} (Task {task})")
    print(f"{'='*60}")

    metrics = {}
    if task == 1:
        metrics = _evaluate_task1(y_true_list, y_pred_list, cfg)
    elif task == 2:
        metrics = _evaluate_task2(y_true_list, y_pred_list, cfg)
    elif task == 3:
        metrics = _evaluate_task3(y_true_list, y_pred_list)
    elif task == 4:
        metrics = _evaluate_task4(y_true_list, y_pred_list)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # ----- Save results -----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{dataset_name}_{args.difficulty}_{timestamp}.json"

    output = {
        "config": {
            "dataset": dataset_name,
            "task": task,
            "model": args.model,
            "difficulty": args.difficulty,
            "num_samples": len(results),
            "timestamp": timestamp,
        },
        "metrics": metrics,
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[INFO] Results saved to {out_path}")

    return metrics


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def _evaluate_task1(y_true: list, y_pred: list, cfg: dict) -> dict:
    """Evaluate Task 1: zero-shot classification."""
    import numpy as np

    classes = cfg["classes"]
    class_to_idx = {c.lower(): i for i, c in enumerate(classes)}

    y_true_idx = [class_to_idx.get(str(y).lower(), -1) for y in y_true]
    y_pred_idx = [class_to_idx.get(str(y).lower(), -1) for y in y_pred]

    # Basic accuracy (always available)
    correct = sum(1 for a, b in zip(y_true_idx, y_pred_idx) if a == b and a != -1)
    total = len(y_true_idx)
    metrics = {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct}

    # Unmatched predictions
    unmatched = sum(1 for b in y_pred_idx if b == -1)
    if unmatched:
        metrics["unmatched_predictions"] = unmatched

    if METRICS_AVAILABLE:
        try:
            y_t = np.array(y_true_idx)
            y_p = np.array(y_pred_idx)
            # Filter out unmatched
            mask = (y_t >= 0) & (y_p >= 0)
            if mask.sum() > 0:
                full_metrics = compute_classification_metrics(
                    y_t[mask], y_p[mask], classes
                )
                metrics.update(full_metrics)
        except Exception as e:
            print(f"  [WARN] Full metrics failed: {e}")

    return metrics


def _evaluate_task2(y_true: list, y_pred: list, cfg: dict) -> dict:
    """Evaluate Task 2: concept annotation (multi-label)."""
    import numpy as np

    concepts = cfg["concepts"]
    concept_set = {c.lower(): i for i, c in enumerate(concepts)}

    def to_multihot(concept_list):
        vec = [0] * len(concepts)
        for c in concept_list:
            idx = concept_set.get(c.lower())
            if idx is not None:
                vec[idx] = 1
        return vec

    y_true_mh = np.array([to_multihot(gt) for gt in y_true])
    y_pred_mh = np.array([to_multihot(pred) for pred in y_pred])

    metrics = {}
    if METRICS_AVAILABLE:
        try:
            metrics = compute_multilabel_metrics(y_true_mh, y_pred_mh, concepts)
        except Exception as e:
            print(f"  [WARN] Full metrics failed: {e}")

    if not metrics:
        # Fallback
        exact = np.all(y_true_mh == y_pred_mh, axis=1).mean()
        metrics = {"subset_accuracy": float(exact)}

    return metrics


def _evaluate_task3(y_true: list, y_pred: list) -> dict:
    """Evaluate Task 3: image captioning."""
    if METRICS_AVAILABLE:
        try:
            return compute_captioning_metrics(y_true, y_pred)
        except Exception as e:
            print(f"  [WARN] Captioning metrics failed: {e}")

    # Fallback: simple ROUGE-like overlap
    scores = []
    for ref, hyp in zip(y_true, y_pred):
        ref_words = set(ref.lower().split())
        hyp_words = set(hyp.lower().split())
        if ref_words:
            scores.append(len(ref_words & hyp_words) / len(ref_words))
    return {"word_overlap": sum(scores) / len(scores) if scores else 0.0}


def _evaluate_task4(y_true: list, y_pred: list) -> dict:
    """Evaluate Task 4: visual QA."""
    if METRICS_AVAILABLE:
        try:
            return compute_vqa_metrics(y_true, y_pred)
        except Exception as e:
            print(f"  [WARN] VQA metrics failed: {e}")

    return compute_basic_accuracy(y_true, y_pred)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MDAgents benchmark runner for DermAgent tasks"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASETS.keys()),
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--difficulty", type=str, default="adaptive",
        choices=["basic", "intermediate", "advanced", "adaptive"],
        help="MDAgents difficulty routing mode (default: adaptive)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        choices=["gpt-3.5", "gpt-4", "gpt-4o", "gpt-4o-mini"],
        help="LLM model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Max number of samples to evaluate (default: all)",
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
