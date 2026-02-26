"""
DermAgent Benchmark Framework

Unified evaluation framework for running single-model MLLM baselines and
computing metrics across all benchmark tasks (diagnosis, concept annotation,
captioning, VQA).

Usage:
    # Command line (run from benchmark/)
    python run.py --model dermlip-panderm --dataset PAD

    # Python API
    from benchmark.datasets import DATASETS, load_dataset
    from benchmark.models import get_model, MODELS
    from benchmark.metrics import compute_classification_metrics

    config = DATASETS["HAM10000_500"]
    df, paths, labels = load_dataset(config)
    model = get_model("dermlip-panderm")
    preds, probs, outputs = model.predict(paths, config.class_names)
    metrics = compute_classification_metrics(labels, preds, config.class_names)
"""

from .datasets import DATASETS, load_dataset, DatasetConfig
from .models import MODELS, get_model

try:
    from .metrics import (
        compute_classification_metrics,
        compute_multilabel_metrics,
        compute_captioning_metrics,
        compute_vqa_metrics,
        print_classification_report,
        print_multilabel_report,
    )
except Exception:  # pragma: no cover
    compute_classification_metrics = None  # type: ignore[assignment]
    compute_multilabel_metrics = None  # type: ignore[assignment]
    compute_captioning_metrics = None  # type: ignore[assignment]
    compute_vqa_metrics = None  # type: ignore[assignment]
    print_classification_report = None  # type: ignore[assignment]
    print_multilabel_report = None  # type: ignore[assignment]

__version__ = "0.1.0"

__all__ = [
    "compute_classification_metrics",
    "compute_multilabel_metrics",
    "compute_captioning_metrics",
    "compute_vqa_metrics",
    "print_classification_report",
    "print_multilabel_report",
    "DATASETS",
    "load_dataset",
    "DatasetConfig",
    "MODELS",
    "get_model",
]
