# MedAgent-Pro Baseline

Adaptation of the MedAgent-Pro framework for dermatology benchmarks (Task 1-3).

MedAgent-Pro uses a two-stage pipeline: (1) a Planner generates a task-level plan using web-based RAG, then (2) a Case-level executor runs quantitative tools (PanDerm, MAKE, RAG) and qualitative analysis (GPT-4o) per sample, with a Decider producing the final answer.

Reference: Wang et al., "MedAgent-Pro: A Multi-Modal Evidence-Based Agent for Medical Diagnosis", 2025.

## Usage

```bash
# From baselines/MedAgent-Pro/

# Step 1: Generate task-level plan
python Derm_Task_level.py --tasks 1,2,3 --model chatgpt-4o-latest

# Step 2: Run case-level execution
# Task 1: Zero-shot classification (HAM10000)
python Derm_Case_level.py --task 1 \
    --csv-path ../../datasets/ham10000/HAM10000_benchmark_500.csv \
    --image-dir ../../datasets/ham10000 --max-samples 500

# Task 2: Concept annotation
python Derm_Case_level.py --task 2 \
    --csv-path ../../datasets/derm7pt/meta_task2_sample_100.csv \
    --image-dir ../../datasets/derm7pt/final_images --max-samples 100

# Task 3: Image captioning
python Derm_Case_level.py --task 3 \
    --csv-path ../../datasets/skin_cap/skin_cap_meta_100.csv \
    --image-dir ../../datasets/skin_cap --max-samples 100

# Step 3: Evaluate
python Derm_Evaluator.py --task 1 \
    --record-dir Dermatology/task1_diagnosis/record/<timestamp> \
    --csv-path ../../datasets/ham10000/HAM10000_benchmark_500.csv
```

## Requirements

- OpenAI API key (set `OPENAI_API_KEY` environment variable)
- DermAgent tools (PanDerm, MAKE, RAG) must be importable from parent project
- GPU for running quantitative tools (~20GB VRAM)
