# MDAgents Baseline

Adaptation of the MDAgents adaptive multi-agent collaboration framework for dermatology benchmarks.

MDAgents uses difficulty-based routing (basic / intermediate / advanced) with LLM-powered multi-agent collaboration for medical decision-making.

Reference: Kim et al., "MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making", NeurIPS 2024 (Oral).

## Usage

```bash
# From baselines/MDAgents/
python run_derm_benchmark.py --dataset HAM10000 --difficulty basic --model gpt-4o --num_samples 500
python run_derm_benchmark.py --dataset SNU_500 --difficulty basic --model gpt-4o --num_samples 500
python run_derm_benchmark.py --dataset derm7pt_100 --difficulty basic --model gpt-4o
python run_derm_benchmark.py --dataset skincon_100 --difficulty basic --model gpt-4o
python run_derm_benchmark.py --dataset skin_cap --difficulty basic --model gpt-4o
```

## Requirements

- OpenAI API key (set `OPENAI_API_KEY` environment variable)
- GPT-4o access
