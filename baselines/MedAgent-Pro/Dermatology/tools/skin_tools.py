"""
Bridge layer: wraps DermAgent's LangChain tools into MedAgent-Pro's
fn(image_path, save_dir, save_name) calling convention.

Each wrapper:
  1. Lazily initialises the underlying tool (singleton per process).
  2. Calls tool._run(...) to get a result string.
  3. Merges the result into save_dir/save_name (JSON).
"""

import json
import os
import sys
from pathlib import Path

# DermAgent project root: baselines/MedAgent-Pro/Dermatology/tools -> DermAgent_submission/
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3].parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_panderm_tool = None
_make_tool = None
_rag_tool = None

_DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES_TOOL", "cuda")


def _get_panderm():
    global _panderm_tool
    if _panderm_tool is None:
        from skin_agent.tools.skin_tools import PanDermTool
        _panderm_tool = PanDermTool(device=_DEVICE)
    return _panderm_tool


def _get_make():
    global _make_tool
    if _make_tool is None:
        from skin_agent.tools.skin_tools import MAKETool
        _make_tool = MAKETool(device=_DEVICE)
    return _make_tool


def _get_rag():
    global _rag_tool
    if _rag_tool is None:
        from skin_agent.tools.skin_tools import RAGTool
        _rag_tool = RAGTool(device=_DEVICE)
    return _rag_tool


# ---------------------------------------------------------------------------
# JSON merge helper
# ---------------------------------------------------------------------------
def _merge_json(save_dir: str, save_name: str, key: str, value):
    """Read existing JSON at save_dir/save_name, add key=value, write back."""
    path = os.path.join(save_dir, save_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data[key] = value
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public tool functions (MedAgent-Pro compatible signatures)
# ---------------------------------------------------------------------------

def panderm_classify(image_path: str, save_dir: str, save_name: str) -> str:
    """Zero-shot skin disease classification using PanDerm/DermLIP.

    Signature: fn(image_path, save_dir, save_name) as required by
    MedAgent-Pro's Case_level quantitative step dispatcher.
    """
    tool = _get_panderm()
    result_str = tool._run(image_path=image_path)
    _merge_json(save_dir, save_name, "panderm_classification", result_str)
    return result_str


def make_annotate(image_path: str, save_dir: str, save_name: str) -> str:
    """Dermoscopic concept extraction using MAKE/OpenCLIP.

    Returns structured concept annotations with confidence scores.
    """
    tool = _get_make()
    result_str = tool._run(image_path=image_path)
    _merge_json(save_dir, save_name, "make_concepts", result_str)
    return result_str


def rag_retrieve(image_path: str, save_dir: str, save_name: str) -> str:
    """Retrieve similar dermatology cases from Derm1M via Qdrant.

    Uses image embedding similarity to find top-5 matching cases.
    """
    tool = _get_rag()
    result_str = tool._run(image_path=image_path, top_k=5)
    _merge_json(save_dir, save_name, "rag_similar_cases", result_str)
    return result_str
