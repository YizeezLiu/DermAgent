#!/usr/bin/env python3
"""
Unified Benchmark Runner

Usage (run from the benchmark/ directory):
    cd benchmark
    python run.py --model dermlip-panderm --dataset PAD
    python run.py --model gpt4o --dataset PAD --max-samples 10
    python run.py --model all --dataset PAD  # Run all models
"""

import os
import sys
import json
import re
import inspect
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
import torch

from datasets import DATASETS, load_dataset
from models import MODELS, get_model
TASK_MODEL_WHITELIST = {
    "diagnosis": [
        "dermlip-vit",
        "dermlip-panderm",
        "llavamed",
        "huatuogpt",
        "qwen3vl",
        "dermogpt",
        "dermogpt-structured",
        "skinvl",
        "skinvl-zs",
        "hulumed",
        "gpt4o",
        "gpt52",
    ],
    "concepts": [
        "make-concepts",
        "llavamed",
        "huatuogpt",
        "qwen3vl",
        "dermogpt",
        "dermogpt-structured",
        "skinvl",
        "skinvl-zs",
        "hulumed",
        "gpt4o",
        "gpt52",
    ],
    "captioning": [
        "llavamed",
        "huatuogpt",
        "qwen3vl",
        "dermogpt",
        "dermogpt-structured",
        "skinvl",
        "skinvl-zs",
        "hulumed",
        "gpt4o",
        "gpt52",
    ],
}
from metrics import (
    compute_classification_metrics,
    print_classification_report,
    compute_multilabel_metrics,
    print_multilabel_report,
    compute_captioning_metrics,
    compute_vqa_metrics,
)


def _extract_timestamp_from_name(path: Path) -> str:
    """Extract trailing timestamp token from benchmark filename stem."""
    match = re.search(r"(\d{8}_\d{6})$", path.stem)
    return match.group(1) if match else ""


def _find_latest_predictions_csv(
    output_dir: str,
    model_name: str,
    dataset_name: str,
    split: Optional[str] = None,
) -> Optional[Path]:
    """Find latest predictions CSV for model+dataset+split under output_dir."""
    split_suffix = f"_{split}" if split else "_all"
    pattern = f"predictions_{model_name}_{dataset_name}{split_suffix}_*.csv"
    candidates = sorted(Path(output_dir).glob(pattern))
    if not candidates:
        return None
    # Prefer lexical timestamp in filename, fallback to mtime if absent.
    candidates.sort(key=lambda p: (_extract_timestamp_from_name(p), p.stat().st_mtime))
    return candidates[-1]


def _pick_retry_key_columns(df: pd.DataFrame, pred_df: pd.DataFrame) -> List[str]:
    """Pick the most reliable shared key columns for retry row matching."""
    priority = [
        ["sample_id", "qa_index"],
        ["question_id"],
        ["filename", "question"],
        ["filename", "question_type", "question"],
        ["filename"],
    ]
    for cols in priority:
        if all(c in df.columns for c in cols) and all(c in pred_df.columns for c in cols):
            return cols
    return []


def _compute_retry_indices(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    filtered_pred_df: pd.DataFrame,
) -> List[int]:
    """Map filtered prediction rows back to dataset rows."""
    if len(pred_df) == len(df):
        return filtered_pred_df.index.tolist()

    key_cols = _pick_retry_key_columns(df, pred_df)
    if not key_cols:
        raise ValueError(
            "Cannot map filtered rows back to dataset rows: length mismatch and no shared key columns."
        )

    df_keys = df[key_cols].astype(str).agg("||".join, axis=1)
    filtered_keys = set(filtered_pred_df[key_cols].astype(str).agg("||".join, axis=1).tolist())
    matched = df.index[df_keys.isin(filtered_keys)].tolist()
    return matched


def check_gpu():
    """Check GPU availability."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        used = torch.cuda.memory_allocated(0) / 1e9
        print(f"GPU: {name} ({used:.1f}/{total:.1f} GB)")
        return True
    print("CUDA not available")
    return False


def _normalize_token(text: str) -> str:
    """Normalize text tokens for concept matching."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def parse_response_to_multilabel(response: str, class_names: List[str]) -> np.ndarray:
    """Convert a free-form VQA response into a multi-hot vector."""
    vector = np.zeros(len(class_names), dtype=int)
    if not response:
        return vector
    
    lower = response.lower()
    norm_names = [_normalize_token(name) for name in class_names]
    
    if "none" in lower:
        # Only early exit if no concept substrings are detected
        if not any(norm in _normalize_token(lower) for norm in norm_names if norm):
            return vector
    
    tokens = re.split(r"[\\n;,/]+", lower)
    for token in tokens:
        norm_token = _normalize_token(token)
        if not norm_token:
            continue
        for idx, norm_name in enumerate(norm_names):
            if not norm_name:
                continue
            if norm_name in norm_token or norm_token in norm_name:
                vector[idx] = 1
    
    if vector.sum() == 0:
        text_norm = _normalize_token(lower)
        for idx, norm_name in enumerate(norm_names):
            if norm_name and norm_name in text_norm:
                vector[idx] = 1
    
    return vector


def probabilities_to_multilabel(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold probability matrix and ensure at least one positive per sample."""
    if not isinstance(probs, np.ndarray) or probs.ndim != 2:
        raise ValueError("Probability array must be 2D for multi-label conversion")
    
    preds = (probs >= threshold).astype(int)
    row_sums = preds.sum(axis=1)
    if np.any(row_sums == 0):
        max_indices = probs.argmax(axis=1)
        preds[np.arange(len(probs)), max_indices] = 1
    return preds


def run_benchmark(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    multilabel_threshold: float = 0.5,
    retry_filtered_latest: bool = False,
    retry_filter_text: str = "[FILTERED]",
    **kwargs
) -> dict:
    """
    Run benchmark for a single model on a dataset.
    
    Args:
        model_name: Model name from MODELS registry
        dataset_name: Dataset name from DATASETS registry
        output_dir: Directory to save results
        split: Optional data split (train/val/test)
        max_samples: Optional limit on samples
        device: Device to use
        **kwargs: Additional model arguments
        
    Returns:
        metrics: Dictionary of computed metrics
    """
    # Get dataset config
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    dataset_config = DATASETS[dataset_name]
    
    if dataset_config.task == "concepts" and model_name.startswith("dermlip"):
        raise ValueError(
            "DermLIP models are disabled for Task2 concept annotation. "
            "Please select the MAKE model or a VQA model instead."
        )
    
    retry_meta = None
    if retry_filtered_latest:
        latest_pred_csv = _find_latest_predictions_csv(
            output_dir=output_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            split=split,
        )
        if latest_pred_csv is None:
            print(
                f"[Retry] No previous predictions found for {model_name}/{dataset_name}. "
                "Skipping retry."
            )
            return {
                "retry_filtered_latest": True,
                "retry_filter_text": retry_filter_text,
                "retry_filtered_count": 0,
                "skipped": True,
                "reason": "no_previous_predictions",
            }

        pred_df_for_retry = pd.read_csv(latest_pred_csv)
        response_col = None
        for col in ["model_answer", "raw_response", "generated_caption"]:
            if col in pred_df_for_retry.columns:
                response_col = col
                break
        if response_col is None:
            print(
                f"[Retry] No response column found in {latest_pred_csv.name}. "
                "Expected one of: model_answer/raw_response/generated_caption. Skipping."
            )
            return {
                "retry_filtered_latest": True,
                "retry_filter_text": retry_filter_text,
                "retry_filtered_count": 0,
                "skipped": True,
                "reason": "missing_response_column",
            }

        retry_mask = pred_df_for_retry[response_col].astype(str).str.contains(
            retry_filter_text, case=False, na=False, regex=False
        )
        filtered_pred_df = pred_df_for_retry[retry_mask].copy()
        if filtered_pred_df.empty:
            print(
                f"[Retry] Latest predictions ({latest_pred_csv.name}) has no rows matching "
                f"'{retry_filter_text}'. Skipping retry."
            )
            return {
                "retry_filtered_latest": True,
                "retry_filter_text": retry_filter_text,
                "retry_filtered_count": 0,
                "skipped": True,
                "reason": "no_filtered_rows",
            }

        retry_meta = {
            "latest_pred_csv": str(latest_pred_csv),
            "response_col": response_col,
            "pred_df": pred_df_for_retry,
            "filtered_pred_df": filtered_pred_df,
        }
        print(
            f"[Retry] Latest predictions: {latest_pred_csv.name} | "
            f"matched rows: {len(filtered_pred_df)}"
        )

    # Load dataset
    df, image_paths, y_true = load_dataset(
        dataset_config, split=split, max_samples=max_samples
    )
    y_true = np.array(y_true)

    if retry_meta is not None:
        retry_indices = _compute_retry_indices(
            df=df,
            pred_df=retry_meta["pred_df"],
            filtered_pred_df=retry_meta["filtered_pred_df"],
        )
        if not retry_indices:
            print("[Retry] No matched dataset rows after index/key mapping. Skipping retry.")
            return {
                "retry_filtered_latest": True,
                "retry_filter_text": retry_filter_text,
                "retry_filtered_count": 0,
                "skipped": True,
                "reason": "no_matched_rows",
            }
        df = df.iloc[retry_indices].reset_index(drop=True)
        image_paths = [image_paths[i] for i in retry_indices]
        y_true = np.array([y_true[i] for i in retry_indices])
        print(f"[Retry] Running on {len(df)} filtered samples only")
    
    # Convert string labels to indices if needed (for classification tasks)
    if dataset_config.label_mode not in {"multilabel", "captioning", "vqa"}:
        if y_true.dtype.kind in {'U', 'O'}:  # Unicode or object (string) array
            class_to_idx = {name: i for i, name in enumerate(dataset_config.class_names)}
            y_true = np.array([class_to_idx.get(str(label), -1) for label in y_true])
            unknown_count = np.sum(y_true == -1)
            if unknown_count > 0:
                print(f"Warning: {unknown_count} labels not found in class_names")
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name} (n={len(df)})")
    print(f"{'='*60}")
    
    model = get_model(model_name, device=device, **kwargs)
    
    model_config = MODELS[model_name]
    y_pred = None
    probs = None
    raw_outputs: List[str] = []
    prompts_used: Optional[List[str]] = None
    
    if dataset_config.label_mode == "vqa":
        if model_config["type"] != "vqa":
            raise ValueError(f"Dataset {dataset_name} requires a VQA model. Got {model_name}.")
        
        prompts_used = []
        responses = []
        base_prompt = (dataset_config.vqa_prompt or "").strip()
        question_col = getattr(dataset_config, "question_col", None)
        question_type_col = getattr(dataset_config, "question_type_col", None)
        option_cols = getattr(dataset_config, "option_cols", []) or []
        rows = list(df.itertuples(index=False))
        
        for image_path, row in zip(image_paths, rows):
            sections: List[str] = []
            
            # Special prompt for LLaVA-Med to enforce formatting
            if model_name == "llavamed":
                current_base_prompt = (
                    "Task: Multiple Choice Question.\n"
                    "Instruction: Output ONLY the option letter (e.g. A). Do not output text."
                )
            else:
                current_base_prompt = base_prompt
            
            if current_base_prompt:
                sections.append(current_base_prompt)
            
            if question_type_col and hasattr(row, question_type_col):
                q_type = getattr(row, question_type_col)
                q_type_str = str(q_type).strip()
                if q_type_str and q_type_str.lower() != "nan":
                    sections.append(f"Question type: {q_type_str}")
            
            question_text = ""
            if question_col and hasattr(row, question_col):
                question_text = str(getattr(row, question_col) or "").strip()
            if question_text:
                sections.append(f"Question: {question_text}")
            
            option_lines = []
            for col in option_cols:
                if not hasattr(row, col):
                    continue
                value = getattr(row, col)
                value_str = str(value).strip() if value is not None else ""
                if not value_str or value_str.lower() == "nan":
                    continue
                label = col.split("_")[-1].upper()
                option_lines.append(f"{label}: {value_str}")
            if option_lines:
                sections.append("Options:\n" + "\n".join(option_lines))
            
            if model_name == "llavamed":
                sections.append("Answer:")

            prompt = "\n\n".join(sections) if sections else ""
            # Pass task_type for task-aware fallback prompts (GPT-5)
            # Check if the model's ask method supports task_type parameter
            sig = inspect.signature(model.ask)
            if 'task_type' in sig.parameters:
                response = model.ask(image_path, prompt, task_type=dataset_config.task)
            else:
                response = model.ask(image_path, prompt)
            responses.append(response)
            prompts_used.append(prompt)
        
        raw_outputs = responses
        probs = np.array([])
    elif model_config["type"] == "zero-shot":
        # Zero-shot model uses templates
        y_pred, probs, raw_outputs = model.predict(
            image_paths,
            dataset_config.class_names,
            prompt_templates=dataset_config.prompt_templates,
        )
    else:
        # VQA model uses shared prompt for classification/captioning tasks
        # Pass task_type for task-aware fallback prompts (GPT-5)
        y_pred, probs, raw_outputs = model.predict(
            image_paths,
            dataset_config.class_names,
            prompt=dataset_config.vqa_prompt,
            task_type=dataset_config.task,
        )
    
    # Compute metrics
    if dataset_config.label_mode == "multilabel":
        y_true_multi = np.array(y_true)
        if model_config["type"] == "zero-shot":
            y_pred_multi = probabilities_to_multilabel(probs, multilabel_threshold)
        else:
            y_pred_multi = np.vstack([
                parse_response_to_multilabel(resp, dataset_config.class_names)
                for resp in raw_outputs
            ])
        metrics = compute_multilabel_metrics(y_true_multi, y_pred_multi, dataset_config.class_names)
        print_multilabel_report(y_true_multi, y_pred_multi, dataset_config.class_names, metrics)
    elif dataset_config.label_mode == "captioning":
        metrics = compute_captioning_metrics(y_true, raw_outputs)
        print("\n" + "=" * 60)
        print("CAPTIONING METRICS")
        print("=" * 60)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    elif dataset_config.label_mode == "vqa":
        gt_answers = [str(x) for x in np.array(y_true).tolist()]
        metrics = compute_vqa_metrics(gt_answers, raw_outputs)
        print("\n" + "=" * 60)
        print("VQA METRICS")
        print("=" * 60)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    else:
        metrics = compute_classification_metrics(y_true, y_pred, dataset_config.class_names)
        print_classification_report(y_true, y_pred, dataset_config.class_names, metrics)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_suffix = f"_{split}" if split else "_all"
    
    # Save metrics JSON
    result = {
        "model": model_name,
        "model_id": model.model_id,
        "dataset": dataset_name,
        "split": split,
        "num_samples": len(df),
        "timestamp": timestamp,
        "metrics": metrics,
        "class_names": dataset_config.class_names,
        "label_mode": dataset_config.label_mode,
    }
    if dataset_config.label_mode == "multilabel":
        result["multilabel_threshold"] = multilabel_threshold
    if dataset_config.label_mode == "vqa":
        result["vqa_prompt"] = dataset_config.vqa_prompt
    if retry_filtered_latest:
        result["retry_filtered_latest"] = True
        result["retry_filter_text"] = retry_filter_text
        result["retry_filtered_count"] = len(df)
        if retry_meta is not None:
            result["retry_source_predictions"] = retry_meta["latest_pred_csv"]
    
    json_path = output_path / f"{model_name}_{dataset_name}{split_suffix}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nMetrics: {json_path}")
    
    # Save predictions CSV
    pred_df = df.copy()
    if dataset_config.label_mode == "multilabel":
        y_true_multi = np.array(y_true)
        pred_vectors = y_pred_multi.astype(int)
        for idx, name in enumerate(dataset_config.class_names):
            pred_df[f"label_{name}"] = y_true_multi[:, idx]
            pred_df[f"pred_{name}"] = pred_vectors[:, idx]
        pred_df["true_concepts"] = [
            ", ".join([name for idx, name in enumerate(dataset_config.class_names) if row[idx]]) or "none"
            for row in y_true_multi
        ]
        pred_df["predicted_concepts"] = [
            ", ".join([name for idx, name in enumerate(dataset_config.class_names) if row[idx]]) or "none"
            for row in pred_vectors
        ]
        pred_df["correct"] = (y_true_multi == pred_vectors).all(axis=1)
        pred_df["raw_response"] = raw_outputs
    elif dataset_config.label_mode == "captioning":
        pred_df['true_caption'] = y_true
        pred_df['generated_caption'] = raw_outputs
    elif dataset_config.label_mode == "vqa":
        question_col = getattr(dataset_config, "question_col", "question")
        question_type_col = getattr(dataset_config, "question_type_col", "question_type")
        option_cols = getattr(dataset_config, "option_cols", []) or []
        if question_col in pred_df.columns:
            pred_df['question'] = pred_df[question_col]
        if question_type_col in pred_df.columns:
            pred_df['question_type'] = pred_df[question_type_col]
        for col in option_cols:
            if col in pred_df.columns:
                pred_df[col] = pred_df[col]
        pred_df['gt_answer'] = y_true.tolist()
        pred_df['model_answer'] = raw_outputs
        if prompts_used:
            pred_df['prompt'] = prompts_used
    else:
        pred_df['predicted_label'] = y_pred
        pred_df['predicted_class'] = [
            dataset_config.class_names[i] if 0 <= i < len(dataset_config.class_names) else "unknown"
            for i in y_pred
        ]
        pred_df['correct'] = (y_true == y_pred)
        pred_df['raw_response'] = raw_outputs
    
    # Add probability columns if available
    if isinstance(probs, np.ndarray) and probs.ndim == 2:
        for i, name in enumerate(dataset_config.class_names):
            pred_df[f'prob_{name}'] = probs[:, i]
    
    csv_path = output_path / f"predictions_{model_name}_{dataset_name}{split_suffix}_{timestamp}.csv"
    pred_df.to_csv(csv_path, index=False)
    print(f"Predictions: {csv_path}")
    
    return metrics


def run_all_models(
    dataset_name: str,
    output_dir: str,
    **kwargs
) -> dict:
    """Run all available models on a dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    dataset_config = DATASETS[dataset_name]
    whitelist = TASK_MODEL_WHITELIST.get(dataset_config.task)
    if whitelist:
        model_names = [m for m in whitelist if m in MODELS]
    else:
        model_names = list(MODELS.keys())
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'#'*60}")
        print(f"# Running: {model_name}")
        print(f"{'#'*60}")
        
        try:
            metrics = run_benchmark(
                model_name=model_name,
                dataset_name=dataset_name,
                output_dir=output_dir,
                **kwargs
            )
            results[model_name] = metrics
            
            # Clear GPU memory between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error running {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    # Save summary
    summary_path = Path(output_dir) / f"summary_{dataset_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark/run.py --model dermlip-panderm --dataset PAD
  python benchmark/run.py --model gpt4o --dataset PAD --max-samples 10
  python benchmark/run.py --model all --dataset PAD
  
Available models: """ + ", ".join(MODELS.keys()) + """
Available datasets: """ + ", ".join(DATASETS.keys())
    )
    
    parser.add_argument(
        "--model", type=str, required=True,
        help=f"Model name or 'all'. Available: {', '.join(MODELS.keys())}"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help=f"Dataset name. Available: {', '.join(DATASETS.keys())}"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="./results/benchmark",
        help="Output directory for results"
    )
    parser.add_argument(
        "--split", type=str, default=None,
        choices=["train", "val", "test"],
        help="Evaluate on specific split only"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum samples to process"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--multilabel-threshold", type=float, default=0.5,
        help="Probability threshold for multi-label predictions"
    )
    parser.add_argument(
        "--retry-filtered-latest",
        action="store_true",
        help=(
            "Retry only rows filtered in the latest predictions file for this "
            "model/dataset/split."
        ),
    )
    parser.add_argument(
        "--retry-filter-text",
        type=str,
        default="[FILTERED]",
        help="Case-insensitive substring used to detect filtered rows in latest predictions.",
    )
    
    args = parser.parse_args()
    
    # Check GPU
    if "cuda" in args.device:
        check_gpu()
    
    # Build output path: results/benchmark/task/dataset/
    dataset_config = DATASETS.get(args.dataset)
    if dataset_config:
        task = dataset_config.task
        if task == "concepts":
            task_folder = f"task2_{task}"
        elif task == "captioning":
            task_folder = f"task3_{task}"
        elif task == "vqa":
            task_folder = "task4_vqa"
        else:
            task_folder = f"task1_{task}"
        output_dir = Path(args.output_dir) / task_folder / args.dataset
    else:
        output_dir = Path(args.output_dir) / args.dataset
    
    # Run benchmark
    if args.model == "all":
        run_all_models(
            dataset_name=args.dataset,
            output_dir=str(output_dir),
            split=args.split,
            max_samples=args.max_samples,
            device=args.device,
            multilabel_threshold=args.multilabel_threshold,
            retry_filtered_latest=args.retry_filtered_latest,
            retry_filter_text=args.retry_filter_text,
        )
    else:
        run_benchmark(
            model_name=args.model,
            dataset_name=args.dataset,
            output_dir=str(output_dir),
            split=args.split,
            max_samples=args.max_samples,
            device=args.device,
            multilabel_threshold=args.multilabel_threshold,
            retry_filtered_latest=args.retry_filtered_latest,
            retry_filter_text=args.retry_filter_text,
        )


if __name__ == "__main__":
    main()

