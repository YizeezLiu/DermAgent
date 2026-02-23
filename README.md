# DermAgent

A collaborative multi-tool AI agent for comprehensive dermatological image analysis, built on LangChain/LangGraph.

DermAgent orchestrates seven specialist tools (PanDerm, MAKE, DermoGPT, Qwen3-VL, Image RAG, Text RAG, Ontology) via a **Plan-Execute-Reflect** loop, using GPT-4o as the reasoning backbone. A deterministic **Critic** module evaluates evidence chains and triggers self-correction when deficiencies are detected.

## Architecture

```
User Query + Image
       |
       v
  LLM Controller (GPT-4o)
       |
   [Plan] --> [Execute] --> [Reflect/Critic]
       |           |               |
       |     +-----+-----+        |
       |     |     |     |        |
       v     v     v     v        v
   PanDerm MAKE DermoGPT RAG   Critic
   (classify)(concepts)(VQA)(retrieve) (verify)
       |     |     |     |        |
       +-----+-----+-----+--------+
       |
       v
  Final Response with Evidence Chain
```

## Project Structure

```
DermAgent/
├── skin_agent/                  # Core agent framework
│   ├── benchmark_agent.py       # Benchmark agent + Critic + AnswerParser
│   ├── configs.py               # Dataset task configurations
│   ├── tracing.py               # TraceLogger, TracingCallback
│   ├── profiler.py              # Performance profiling
│   ├── resume.py                # Checkpoint/resume for long runs
│   ├── prompts.md               # System prompts
│   ├── tools/
│   │   ├── base.py              # BaseSkinTool, input schemas
│   │   ├── skin_tools.py        # All 7 tool implementations
│   │   ├── executor.py          # Tool execution orchestration
│   │   └── derm_knowledge_tree/ # Disease ontology JSONs
│   └── utils/
│       ├── retry.py             # Rate-limit retry logic
│       └── image_utils.py       # Image path handling
├── scripts/                     # Experiment runners
│   ├── run_task1_ham10000_500_agent_dermogpt_full_critic.sh
│   ├── run_task1_snu_500_critic.sh
│   ├── run_task2_task3_agent_critic.sh
│   ├── run_task3_loo_ablation.sh
│   └── *.py                     # Python runner scripts
├── data/                        # Benchmark CSV metadata
├── configs/                     # Experiment configuration
├── build_qdrant_db.py           # Build image RAG vector database
├── build_qdrant_rag.py          # Alternative RAG builder
├── build_qdrant_guidelines.py   # Build text RAG (guidelines)
├── requirements.txt
└── .env.example
```

## Setup

### 1. Environment

```bash
# Create conda environment
conda create -n dermagent python=3.10
conda activate dermagent

# Install PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Datasets

Download the following datasets and place images in the expected directories:

| Dataset | Task | Download | Image Directory |
|---------|------|----------|-----------------|
| HAM10000 | Diagnosis (7 classes, 642 imgs) | [ISIC Archive](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) | `datasets/ham10000/images/` |
| SNU | Diagnosis (134 classes, 500 imgs) | [SNU Quiz](https://figshare.com/articles/dataset/SNU_dataset/6454802) | `MAKE_Downstreams/SNU/` |
| Derm7pt | Concept Annotation (7 concepts) | [Derm7pt](https://derm.cs.sfu.ca/Welcome.html) | `MAKE_Downstreams/derm7pt/final_images/` |
| SkinCon | Concept Annotation (32 concepts) | [SkinCon](https://skincon-dataset.github.io/) | `MAKE_Downstreams/skincon/final_images/` |
| SkinCAP | Captioning (100 imgs) | [SkinCAP](https://huggingface.co/datasets/joshuachou/SkinCAP) | `MAKE_Downstreams/skin_cap/images/` |

Benchmark CSV metadata (split definitions) are included in `data/`.

### 4. RAG Vector Database

Build the Qdrant vector databases for retrieval:

```bash
# Image-based case retrieval (requires Derm1M dataset)
python build_qdrant_db.py

# Guideline-grounded text retrieval
python build_qdrant_guidelines.py
```

### 5. Model Weights

Tool models are auto-downloaded from HuggingFace on first use:
- **PanDerm**: `redlessone/DermLIP_PanDerm-base-w-PubMed-256` (~2GB VRAM)
- **MAKE**: `xieji-x/MAKE` (~2GB VRAM)
- **DermoGPT**: DermoGPT-RL (~16GB VRAM, bfloat16)
- **Qwen3-VL**: `Qwen/Qwen3-VL-8B-Instruct` (~16GB VRAM, bfloat16)

Total GPU requirement: ~20-22GB for all tools loaded simultaneously.

## Reproducing Paper Results

### Table 1: Main Results

```bash
# Task 1: HAM10000 Diagnosis (Acc: 0.6121)
bash scripts/run_task1_ham10000_500_agent_dermogpt_full_critic.sh

# Task 1: SNU Diagnosis (Acc: 0.3260)
bash scripts/run_task1_snu_500_critic.sh

# Task 2 + Task 3: Derm7pt, SkinCon, SkinCAP
bash scripts/run_task2_task3_agent_critic.sh
```

### Table 2: Ablation Study (Leave-One-Out on SkinCAP)

```bash
bash scripts/run_task3_loo_ablation.sh
```

### Configuration

All scripts use GPT-4o as the LLM controller with:
- 6 tools: `panderm, make, dermogpt_vqa, rag, text_rag, ontology`
- Critic: enabled with max 2 retries
- Image reinjection: enabled (re-injects image on critic retry)

Override via environment variables:
```bash
CUDA_VISIBLE_DEVICES=0,1 MAX_SAMPLES=10 bash scripts/run_task1_ham10000_500_agent_dermogpt_full_critic.sh
```

## Tool Descriptions

| Tool | Model | Purpose |
|------|-------|---------|
| PanDerm Classifier | DermLIP | Zero-shot disease classification via CLIP similarity |
| MAKE Annotator | MAKE (OpenCLIP) | Dermoscopic concept extraction |
| DermoGPT VQA | DermoGPT-RL | Dermatology-specialized visual QA |
| Qwen3-VL VQA | Qwen3-VL-8B | General visual question answering |
| Image RAG | DermLIP + Qdrant | Case retrieval from 413K diagnosed cases |
| Text RAG | Qwen3-Embedding + Qdrant | Guideline retrieval from 3,199 document chunks |
| Ontology | Knowledge Graph | Disease hierarchy and taxonomy queries |

## Citation

```bibtex
@inproceedings{dermagent2026,
  title={DermAgent: A Collaborative Agent for Comprehensive Dermatological Image Analysis},
  author={Anonymous},
  booktitle={MICCAI},
  year={2026}
}
```
