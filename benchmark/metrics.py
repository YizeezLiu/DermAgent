"""
Common metrics computation for all benchmark tasks.
"""

from typing import Dict, List, Any
from collections import Counter
import re
import numpy as np
import nltk
from rouge_score import rouge_scorer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    hamming_loss,
    jaccard_score,
)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


# Text normalization helpers
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_text(text: str) -> str:
    """Lowercase and remove punctuation for fair VQA comparisons."""
    text = str(text or "").lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return " ".join(text.split())


def _token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between two answers."""
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_vqa_metrics(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, float]:
    """
    Compute VQA metrics (Exact Match, token F1, BLEU-1).
    
    Args:
        y_true: Ground-truth answers.
        y_pred: Model responses.
    
    Returns:
        Dictionary of aggregated scores.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Ground-truth and prediction lengths must match for VQA metrics.")
    
    exact_matches = []
    f1_scores = []
    references = []
    hypotheses = []
    
    for ref, hyp in zip(y_true, y_pred):
        ref_norm = _normalize_text(ref)
        hyp_norm = _normalize_text(hyp)
        exact_matches.append(1.0 if ref_norm == hyp_norm else 0.0)
        f1_scores.append(_token_f1(hyp, ref))
        references.append([nltk.word_tokenize(ref.lower())])
        hypotheses.append(nltk.word_tokenize(hyp.lower()))
    
    bleu1 = nltk.translate.bleu_score.corpus_bleu(
        references, hypotheses, weights=(1.0, 0, 0, 0)
    )
    
    return {
        "exact_match": float(np.mean(exact_matches)),
        "token_f1": float(np.mean(f1_scores)),
        "bleu1": float(bleu1),
    }


def compute_captioning_metrics(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, float]:
    """
    Compute captioning metrics (BLEU, ROUGE).
    
    Args:
        y_true: List of ground truth captions
        y_pred: List of generated captions
        
    Returns:
        Dictionary containing BLEU-1, BLEU-4, ROUGE-L scores
    """
    # BLEU Scores
    # Prepare references for nltk: list of lists of references (here 1 ref per sample)
    references = [[nltk.word_tokenize(ref.lower())] for ref in y_true]
    hypotheses = [nltk.word_tokenize(hyp.lower()) for hyp in y_pred]
    
    bleu1 = nltk.translate.bleu_score.corpus_bleu(
        references, hypotheses, weights=(1.0, 0, 0, 0)
    )
    bleu4 = nltk.translate.bleu_score.corpus_bleu(
        references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)
    )
    
    # ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []
    
    for ref, hyp in zip(y_true, y_pred):
        scores = scorer.score(ref, hyp)
        rouge_l_scores.append(scores['rougeL'].fmeasure)
        
    metrics = {
        "bleu1": float(bleu1),
        "bleu4": float(bleu4),
        "rougeL": float(np.mean(rouge_l_scores)),
    }
    
    return metrics



def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary containing various metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics["per_class"] = {}
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics["per_class"][class_name] = {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i])
            }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    metrics["confusion_matrix"] = cm.tolist()
    
    return metrics


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict:
    """
    Compute metrics for multi-label classification.
    
    Args:
        y_true: Ground truth multi-hot labels (N x C)
        y_pred: Predicted multi-hot labels (N x C)
        class_names: Concept names
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "overall_accuracy": float((y_true == y_pred).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard_micro": float(jaccard_score(y_true, y_pred, average="micro", zero_division=0)),
        "jaccard_macro": float(jaccard_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_samples": float(precision_score(y_true, y_pred, average="samples", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_samples": float(recall_score(y_true, y_pred, average="samples", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
    }
    
    metrics["per_class"] = {}
    for idx, name in enumerate(class_names):
        metrics["per_class"][name] = {
            "precision": float(precision_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
            "recall": float(recall_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
            "f1": float(f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
        }
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    metrics: Dict
) -> None:
    """Print formatted classification results."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted):     {metrics['f1_weighted']:.4f}")
    print("-" * 60)
    print("\nPer-class F1 scores:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name:30s}: {class_metrics['f1']:.4f}")
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    labels = list(range(len(class_names)))
    print(classification_report(y_true, y_pred, labels=labels, 
                                target_names=class_names, zero_division=0))


def print_multilabel_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    metrics: Dict
) -> None:
    """Print formatted multi-label results."""
    print("\n" + "=" * 60)
    print("MULTI-LABEL RESULTS")
    print("=" * 60)
    print(f"Subset Accuracy:   {metrics['subset_accuracy']:.4f}")
    print(f"Overall Accuracy:  {metrics['overall_accuracy']:.4f}")
    print(f"Hamming Loss:      {metrics['hamming_loss']:.4f}")
    print(f"Jaccard (micro):   {metrics['jaccard_micro']:.4f}")
    print(f"Jaccard (macro):   {metrics['jaccard_macro']:.4f}")
    print(f"F1 (micro):        {metrics['f1_micro']:.4f}")
    print(f"F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"F1 (samples):      {metrics['f1_samples']:.4f}")
    print("-" * 60)
    print("\nPer-concept F1 scores:")
    for class_name, class_metrics in metrics["per_class"].items():
        print(f"  {class_name:30s}: {class_metrics['f1']:.4f}")
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )


