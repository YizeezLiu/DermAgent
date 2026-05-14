# Attribution — MedAgent-Pro Baseline

This directory contains code adapted from the **MedAgent-Pro** project for the
purpose of reproducing the MedAgent-Pro agent baseline reported in the
DermAgent paper (Table 1).

## Upstream Source

- **Repository:** https://github.com/jinlab-imvr/MedAgent-Pro
- **Paper:** Ziyue Wang et al. *"MedAgent-Pro: Towards Evidence-Based
  Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow."* ICLR 2026.
- **Upstream license:** No `LICENSE` file is present in the upstream repository
  at the time of adaptation (2026). The code is used here **under attribution
  only**, not under any open-source license grant.

## Files

| File | Origin |
|------|--------|
| `Planner.py` | Adapted from upstream |
| `CodingAgent.py` | Adapted from upstream |
| `Summary_Module.py` | Adapted from upstream |
| `RAG.py` | Adapted from upstream |
| `utils.py` | Adapted from upstream |
| `Decider/GPT_Decider.py` | Adapted from upstream |
| `Decider/MultiClass_Decider.py` | Adapted from upstream |
| `Decider/Pro_Decider.py` | Adapted from upstream |
| `Decider/__init__.py` | Adapted from upstream |
| `Derm_Case_level.py` | New file by the DermAgent authors; uses the MedAgent-Pro framework for dermatology case-level evaluation |
| `Derm_Task_level.py` | New file by the DermAgent authors; uses the MedAgent-Pro framework for dermatology task-level planning |
| `Derm_Evaluator.py` | New file by the DermAgent authors; dermatology-specific evaluator |
| `Dermatology/tools/skin_tools.py` | New file by the DermAgent authors; dermatology tool wrappers |
| `Dermatology/tools/__init__.py` | New file by the DermAgent authors |
| `Dermatology/task*/task.json`, `toolset.json` | New configs by the DermAgent authors |

Each adapted file carries an attribution comment block at the top pointing to
this document.

## Usage Notice

The MedAgent-Pro framework is reproduced here **solely for the purpose of
scientific reproducibility** of the comparison reported in the DermAgent paper.
If you use this baseline in your own work, please cite the original
MedAgent-Pro paper directly:

```bibtex
@inproceedings{wang2026medagentpro,
  title={MedAgent-Pro: Towards Evidence-Based Multi-Modal Medical Diagnosis via
         Reasoning Agentic Workflow},
  author={Wang, Ziyue and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Removal Request

If you are an author or rights-holder of MedAgent-Pro and wish to have this
code removed or relicensed, please open an issue on the DermAgent repository
and we will respond promptly.
