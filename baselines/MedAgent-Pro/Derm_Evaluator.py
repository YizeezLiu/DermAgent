# =============================================================================
# Adapted from MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro)
# Original work: Wang et al., "MedAgent-Pro: Towards Evidence-Based
#   Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow",
#   ICLR 2026.
# Upstream license: no license file. Used with attribution; see
#   baselines/MedAgent-Pro/ATTRIBUTION.md for details.
# Modifications by the DermAgent authors for dermatology benchmarks.
# =============================================================================

"""
Evaluation module for dermatology benchmarks (Task 1-3).

Reads final_diagnosis.json from each sample's record directory and computes
task-specific metrics against the ground-truth CSV.

Usage:
    python Derm_Evaluator.py --task 1 \\
        --record-dir Dermatology/task1_diagnosis/record/<timestamp> \\
        --csv-path ../../datasets/ham10000/HAM10000_benchmark_500.csv

    python Derm_Evaluator.py --task 2 \\
        --record-dir Dermatology/task2_concept/record/<timestamp> \\
        --csv-path ../../datasets/derm7pt/meta_task2_sample_100.csv

    python Derm_Evaluator.py --task 3 \\
        --record-dir Dermatology/task3_captioning/record/<timestamp> \\
        --csv-path ../../datasets/skin_cap/skin_cap_meta_100.csv
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Candidate class lists
# ============================================================================
HAM10000_CLASSES = [
    "melanocytic nevi", "melanoma", "basal cell carcinoma",
    "actinic keratosis", "benign keratosis", "dermatofibroma",
    "vascular lesion",
]

DERM7PT_CONCEPTS = [
    "pigment_network",
    "blue_whitish_veil",
    "vascular_structures",
    "pigmentation",
    "streaks",
    "dots_and_globules",
    "regression_structures",
]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MedAgent-Pro dermatology results")
    p.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--record-dir", type=str, required=True)
    p.add_argument("--csv-path", type=str, required=True)
    p.add_argument("--image-col", type=str, default=None)
    p.add_argument("--output", type=str, default=None, help="Path to write metrics JSON")
    return p.parse_args()


# ============================================================================
# Task 1: Multi-class classification
# ============================================================================

def evaluate_task1(record_dir: str, csv_path: str, image_col: str = "filename"):
    df = pd.read_csv(csv_path)
    y_true, y_pred = [], []
    matched, missed = 0, 0

    for _, row in df.iterrows():
        img_file = str(row[image_col])
        sample_id = Path(img_file).stem
        fd_path = os.path.join(record_dir, sample_id, "final_diagnosis.json")

        if not os.path.exists(fd_path):
            missed += 1
            continue

        with open(fd_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        pred = str(result.get("overall", {}).get("diagnosis", "")).strip().lower()
        gt = str(row.get("diag", "")).strip().lower()

        y_true.append(gt)
        y_pred.append(pred)
        matched += 1

    if not y_true:
        print("[ERROR] No matched samples found.")
        return {}

    # Overall accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)

    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1

    per_class_acc = {}
    for cls in sorted(class_total.keys()):
        per_class_acc[cls] = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0

    # Macro-averaged metrics
    macro_acc = np.mean(list(per_class_acc.values())) if per_class_acc else 0.0

    # F1 scores
    try:
        from sklearn.metrics import f1_score, classification_report
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_true, y_pred, zero_division=0)
    except ImportError:
        macro_f1 = weighted_f1 = 0.0
        report = "sklearn not available"

    metrics = {
        "task": 1,
        "total_samples": len(df),
        "matched": matched,
        "missed": missed,
        "accuracy": round(accuracy, 4),
        "macro_accuracy": round(macro_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_accuracy": {k: round(v, 4) for k, v in per_class_acc.items()},
    }

    print("\n" + "=" * 60)
    print("Task 1: Zero-shot Classification Results")
    print("=" * 60)
    print(f"Samples: {matched}/{len(df)} (missed: {missed})")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro Accuracy: {macro_acc:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Weighted F1:    {weighted_f1:.4f}")
    print("\nPer-class accuracy:")
    for cls, acc in per_class_acc.items():
        print(f"  {cls:30s}: {acc:.4f}  ({class_correct[cls]}/{class_total[cls]})")
    if isinstance(report, str) and report != "sklearn not available":
        print(f"\n{report}")

    return metrics


# ============================================================================
# Task 2: Multi-label concept annotation
# ============================================================================

def evaluate_task2(record_dir: str, csv_path: str, image_col: str = "ImageID"):
    df = pd.read_csv(csv_path)
    concept_cols = [c for c in DERM7PT_CONCEPTS if c in df.columns]
    if not concept_cols:
        print("[ERROR] No concept columns found in CSV")
        return {}

    y_true_all = []
    y_pred_all = []
    matched, missed = 0, 0

    for _, row in df.iterrows():
        img_file = str(row[image_col])
        sample_id = Path(img_file).stem
        fd_path = os.path.join(record_dir, sample_id, "final_diagnosis.json")

        if not os.path.exists(fd_path):
            missed += 1
            continue

        with open(fd_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        concepts_result = result.get("overall", {}).get("concepts", {})

        gt_vec = [int(row.get(c, 0)) for c in concept_cols]

        pred_vec = []
        for c in concept_cols:
            found = False
            c_display = c.replace("_", " ")
            for key, val in concepts_result.items():
                if c_display in key.lower() or c.lower() in key.lower():
                    found = val.get("present", False) if isinstance(val, dict) else False
                    break
            pred_vec.append(1 if found else 0)

        y_true_all.append(gt_vec)
        y_pred_all.append(pred_vec)
        matched += 1

    if not y_true_all:
        print("[ERROR] No matched samples found.")
        return {}

    y_true_np = np.array(y_true_all)
    y_pred_np = np.array(y_pred_all)

    # Per-concept metrics
    per_concept = {}
    for i, c in enumerate(concept_cols):
        tp = np.sum((y_true_np[:, i] == 1) & (y_pred_np[:, i] == 1))
        fp = np.sum((y_true_np[:, i] == 0) & (y_pred_np[:, i] == 1))
        fn = np.sum((y_true_np[:, i] == 1) & (y_pred_np[:, i] == 0))
        tn = np.sum((y_true_np[:, i] == 0) & (y_pred_np[:, i] == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        per_concept[c] = {"accuracy": round(acc, 4), "precision": round(prec, 4),
                          "recall": round(rec, 4), "f1": round(f1, 4)}

    # Aggregate metrics
    try:
        from sklearn.metrics import f1_score as skf1, hamming_loss, jaccard_score
        f1_micro = skf1(y_true_np, y_pred_np, average="micro", zero_division=0)
        f1_macro = skf1(y_true_np, y_pred_np, average="macro", zero_division=0)
        h_loss = hamming_loss(y_true_np, y_pred_np)
        jaccard = jaccard_score(y_true_np, y_pred_np, average="samples", zero_division=0)
    except ImportError:
        f1_micro = f1_macro = h_loss = jaccard = 0.0

    metrics = {
        "task": 2,
        "total_samples": len(df),
        "matched": matched,
        "missed": missed,
        "f1_micro": round(f1_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "hamming_loss": round(h_loss, 4),
        "jaccard": round(jaccard, 4),
        "per_concept": per_concept,
    }

    print("\n" + "=" * 60)
    print("Task 2: Concept Annotation Results")
    print("=" * 60)
    print(f"Samples: {matched}/{len(df)} (missed: {missed})")
    print(f"F1-micro:      {f1_micro:.4f}")
    print(f"F1-macro:      {f1_macro:.4f}")
    print(f"Hamming Loss:  {h_loss:.4f}")
    print(f"Jaccard:       {jaccard:.4f}")
    print("\nPer-concept:")
    for c, m in per_concept.items():
        print(f"  {c:30s}: F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    return metrics


# ============================================================================
# Task 3: Image captioning
# ============================================================================

def evaluate_task3(record_dir: str, csv_path: str, image_col: str = "filename",
                   caption_col: str = "caption_zh_polish_en"):
    df = pd.read_csv(csv_path)
    refs, hyps = [], []
    matched, missed = 0, 0

    for _, row in df.iterrows():
        img_file = str(row[image_col])
        sample_id = Path(img_file).stem
        fd_path = os.path.join(record_dir, sample_id, "final_diagnosis.json")

        if not os.path.exists(fd_path):
            missed += 1
            continue

        with open(fd_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        pred_caption = str(result.get("overall", {}).get("caption", "")).strip()
        gt_caption = str(row.get(caption_col, "")).strip()

        if not pred_caption:
            missed += 1
            continue

        refs.append(gt_caption)
        hyps.append(pred_caption)
        matched += 1

    if not refs:
        print("[ERROR] No matched samples found.")
        return {}

    # BLEU scores
    bleu1, bleu4 = _compute_bleu(refs, hyps)

    # ROUGE-L
    rouge_l = _compute_rouge_l(refs, hyps)

    metrics = {
        "task": 3,
        "total_samples": len(df),
        "matched": matched,
        "missed": missed,
        "bleu_1": round(bleu1, 4),
        "bleu_4": round(bleu4, 4),
        "rouge_l": round(rouge_l, 4),
    }

    print("\n" + "=" * 60)
    print("Task 3: Image Captioning Results")
    print("=" * 60)
    print(f"Samples: {matched}/{len(df)} (missed: {missed})")
    print(f"BLEU-1:  {bleu1:.4f}")
    print(f"BLEU-4:  {bleu4:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")

    return metrics


def _compute_bleu(refs: list, hyps: list) -> tuple:
    """Compute corpus-level BLEU-1 and BLEU-4."""
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smoother = SmoothingFunction().method1
        refs_tok = [[r.split()] for r in refs]
        hyps_tok = [h.split() for h in hyps]
        b1 = corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0),
                         smoothing_function=smoother)
        b4 = corpus_bleu(refs_tok, hyps_tok, weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=smoother)
        return b1, b4
    except ImportError:
        print("[warn] nltk not found — BLEU scores unavailable")
        return 0.0, 0.0


def _compute_rouge_l(refs: list, hyps: list) -> float:
    """Compute average ROUGE-L F1."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
        return float(np.mean(scores)) if scores else 0.0
    except ImportError:
        print("[warn] rouge_score not found — ROUGE-L unavailable")
        return 0.0


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    if args.task == 1:
        image_col = args.image_col or "filename"
        metrics = evaluate_task1(args.record_dir, args.csv_path, image_col)
    elif args.task == 2:
        image_col = args.image_col or "ImageID"
        metrics = evaluate_task2(args.record_dir, args.csv_path, image_col)
    elif args.task == 3:
        image_col = args.image_col or "filename"
        metrics = evaluate_task3(args.record_dir, args.csv_path, image_col)
    else:
        print(f"Unknown task: {args.task}")
        return

    # Save metrics
    out_path = args.output or os.path.join(args.record_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()
