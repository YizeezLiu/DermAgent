# Baseline Reproduction

This directory contains agent-based baseline implementations referenced in the paper's Table 1.
Single-model MLLM baselines (GPT-4o, LLaVA-Med, etc.) are in the shared `benchmark/` framework
at the project root — see [benchmark/README section in main README](../README.md).

## Directory Structure

```
baselines/
├── MDAgents/           # MDAgents agent baseline (NeurIPS 2024)
├── MedAgent-Pro/       # MedAgent-Pro agent baseline
└── SkinVL/             # SkinVL-PubMM baseline (placeholder)
```

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt  # Main project requirements
   ```

2. Set API keys:
   ```bash
   export OPENAI_API_KEY=your-key        # For GPT-4o, MDAgents, MedAgent-Pro
   ```

3. Download datasets (see main README for links) and place images in `datasets/`.

## 1. MDAgents Baseline

Run from `baselines/MDAgents/`:

```bash
cd baselines/MDAgents

# Task 1: Diagnosis
python run_derm_benchmark.py --dataset HAM10000 --difficulty basic --model gpt-4o --num_samples 500
python run_derm_benchmark.py --dataset SNU_500 --difficulty basic --model gpt-4o --num_samples 500

# Task 2: Concept Annotation
python run_derm_benchmark.py --dataset derm7pt_100 --difficulty basic --model gpt-4o
python run_derm_benchmark.py --dataset skincon_100 --difficulty basic --model gpt-4o

# Task 3: Image Captioning
python run_derm_benchmark.py --dataset skin_cap --difficulty basic --model gpt-4o
```

## 2. MedAgent-Pro Baseline

Run from `baselines/MedAgent-Pro/`:

```bash
cd baselines/MedAgent-Pro

# Step 1: Generate task-level plans
python Derm_Task_level.py --tasks 1,2,3 --model chatgpt-4o-latest

# Step 2: Run case-level execution
# Task 1: Diagnosis (HAM10000)
python Derm_Case_level.py --task 1 \
    --csv-path ../../datasets/ham10000/HAM10000_benchmark_500.csv \
    --image-dir ../../datasets/ham10000 --max-samples 500

# Task 2: Concept annotation (Derm7pt)
python Derm_Case_level.py --task 2 \
    --csv-path ../../datasets/derm7pt/meta_task2_sample_100.csv \
    --image-dir ../../datasets/derm7pt/final_images --max-samples 100

# Task 3: Image captioning (SkinCAP)
python Derm_Case_level.py --task 3 \
    --csv-path ../../datasets/skin_cap/skin_cap_meta_100.csv \
    --image-dir ../../datasets/skin_cap --max-samples 100

# Step 3: Evaluate
python Derm_Evaluator.py --task 1 \
    --record-dir Dermatology/task1_diagnosis/record/<timestamp> \
    --csv-path ../../datasets/ham10000/HAM10000_benchmark_500.csv
```

## 3. SkinVL-PubMM Baseline

See `SkinVL/README.md` for setup instructions. Code and weights need to be placed in `baselines/SkinVL/` from the external server.
