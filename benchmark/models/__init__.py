"""
Model wrappers for benchmark evaluation.
"""

from pathlib import Path

from .base import BaseModel, VQAModel, ZeroShotModel
from .dermlip import DermLIPModel
from .dermogpt import DermoGPTStructuredModel
from .gpt4o import GPT4oModel
from .gpt52_clean import GPT52CleanModel
from .huatuogpt import HuatuoGPTModel
from .hulumed import HuluMedModel
from .llavamed import LLaVAMedModel
from .qwen3vl import Qwen3VLModel
from .skinvl import SkinVLModel, SkinVLZSModel
from .make import MAKEConceptModel

# Registry of available models
MODELS = {
    # Zero-shot CLIP models
    "dermlip-vit": {
        "class": DermLIPModel,
        "model_id": "redlessone/DermLIP_ViT-B-16",
        "type": "zero-shot"
    },
    "dermlip-panderm": {
        "class": DermLIPModel,
        "model_id": "redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        "type": "zero-shot"
    },
    "make-concepts": {
        "class": MAKEConceptModel,
        "model_id": "xieji-x/MAKE",
        "type": "zero-shot"
    },
    # VQA models
    "gpt4o": {
        "class": GPT4oModel,
        "model_id": "gpt-4o",
        "type": "vqa"
    },
    "gpt52": {
        "class": GPT52CleanModel,
        "model_id": "gpt-5.2",
        "type": "vqa"
    },
    "huatuogpt": {
        "class": HuatuoGPTModel,
        "model_id": "FreedomIntelligence/HuatuoGPT-Vision-7B",
        "type": "vqa"
    },
    "llavamed": {
        "class": LLaVAMedModel,
        "model_id": "microsoft/llava-med-v1.5-mistral-7b",
        "type": "vqa"
    },
    "qwen3vl": {
        "class": Qwen3VLModel,
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "type": "vqa"
    },
    "dermogpt": {
        "class": DermoGPTStructuredModel,
        "model_id": str(Path(__file__).parent.parent.parent / "model-weights" / "DermoGPT-RL"),
        "type": "vqa"
    },
    "dermogpt-structured": {
        "class": DermoGPTStructuredModel,
        "model_id": str(Path(__file__).parent.parent.parent / "model-weights" / "DermoGPT-RL"),
        "type": "vqa"
    },
    "skinvl": {
        "class": SkinVLModel,
        "model_id": str(Path(__file__).parent.parent.parent / "model-weights" / "SkinVL-PubMM"),
        "type": "vqa"
    },
    "skinvl-zs": {
        "class": SkinVLZSModel,
        "model_id": str(Path(__file__).parent.parent.parent / "model-weights" / "SkinVL-PubMM"),
        "type": "vqa"
    },
    "hulumed": {
        "class": HuluMedModel,
        "model_id": str(Path(__file__).parent.parent.parent / "model-weights" / "Hulu-Med-7B"),
        "type": "vqa"
    },
}


def get_model(model_name: str, device: str = "cuda", **kwargs):
    """
    Get a model instance by name.

    Args:
        model_name: Model name from MODELS registry
        device: Device to load model on
        **kwargs: Additional arguments passed to model

    Returns:
        Model instance
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    config = MODELS[model_name]
    model_class = config["class"]
    model_id = kwargs.pop("model_id", config["model_id"])

    return model_class(model_id=model_id, device=device, **kwargs)


__all__ = [
    "BaseModel", "VQAModel", "ZeroShotModel",
    "DermLIPModel", "DermoGPTStructuredModel",
    "GPT4oModel", "GPT52CleanModel",
    "HuatuoGPTModel", "HuluMedModel",
    "LLaVAMedModel", "Qwen3VLModel",
    "SkinVLModel", "SkinVLZSModel",
    "MAKEConceptModel",
    "MODELS", "get_model"
]
