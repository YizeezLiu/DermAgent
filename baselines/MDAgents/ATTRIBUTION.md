# Attribution — MDAgents Baseline

This directory contains code adapted from the **MDAgents** project for the
purpose of reproducing the MDAgents agent baseline reported in the DermAgent
paper (Table 1).

## Upstream Source

- **Repository:** https://github.com/mitmedialab/MDAgents
- **Paper:** Yubin Kim, Chanwoo Park, Hyewon Jeong, Yik Siu Chan, Xuhai Xu,
  Daniel McDuff, Hyeonhoon Lee, Marzyeh Ghassemi, Cynthia Breazeal, Hae Won Park.
  *"MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making."*
  NeurIPS 2024 (Oral).
- **Upstream license:** No `LICENSE` file is present in the upstream repository
  at the time of adaptation (2026). The code is used here **under attribution
  only**, not under any open-source license grant.

## Files

| File | Origin |
|------|--------|
| `main.py` | Adapted from upstream `main.py` |
| `utils.py` | Adapted from upstream `utils.py` |
| `run_derm_benchmark.py` | New file by the DermAgent authors; uses the MDAgents framework for dermatology benchmarks |

Each adapted file carries an attribution comment block at the top pointing to
this document.

## Usage Notice

The MDAgents framework is reproduced here **solely for the purpose of
scientific reproducibility** of the comparison reported in the DermAgent paper.
If you use this baseline in your own work, please cite the original MDAgents
paper directly:

```bibtex
@inproceedings{kim2024mdagents,
  title={MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making},
  author={Kim, Yubin and Park, Chanwoo and Jeong, Hyewon and Chan, Yik Siu and
          Xu, Xuhai and McDuff, Daniel and Lee, Hyeonhoon and Ghassemi, Marzyeh and
          Breazeal, Cynthia and Park, Hae Won},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

## Removal Request

If you are an author or rights-holder of MDAgents and wish to have this code
removed or relicensed, please open an issue on the DermAgent repository and
we will respond promptly.
