# SkinVL-PubMM Baseline

Placeholder for the SkinVL-PubMM dermatology MLLM baseline.

## Setup

Place the SkinVL-PubMM model code and weights in this directory. The expected structure:

```
SkinVL/
├── run_benchmark.py        # Benchmark runner script
├── model.py                # Model wrapper
└── README.md               # This file
```

The benchmark runner should accept the same interface as other baselines:
- Input: image path + task-specific prompt
- Output: prediction string

## Datasets

Use the same dataset CSVs and images as other baselines (see `datasets/` in the project root).

## Expected Results (Paper Table 1)

| Dataset | Metric | Score |
|---------|--------|-------|
| HAM10000 | Accuracy | 0.4517 |
| SNU | Accuracy | 0.0340 |
| Derm7pt | F1-Macro | 0.5314 |
| SkinCon | F1-Macro | 0.1320 |
| SkinCAP | ROUGE-L | 0.1444 |
