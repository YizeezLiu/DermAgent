"""
MAKE Concept Annotation model wrapper.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseModel


class MAKEConceptModel(BaseModel):
    """MAKE model for multi-label concept annotation."""
    
    PROMPT_TEMPLATES = [
        "This is skin image of {}",
        "This is dermatology image of {}",
        "This is image of {}",
        "A skin image of {}",
        "An image of {}, a skin condition.",
        "{}, a skin disorder, is shown in this image.",
    ]
    PROMPT_REFERENCES = [
        ["This is skin image"],
        ["This is dermatology image"],
        ["This is image"],
        ["A skin image"],
        ["An image of a skin condition."],
        ["a skin disorder is shown in this image."],
    ]
    TEMPERATURE = 1 / math.exp(4.5944)
    
    def _load_model(self, **kwargs) -> None:
        from open_clip import create_model_from_pretrained, get_tokenizer
        
        cache_dir = kwargs.pop(
            "cache_dir",
            Path(__file__).resolve().parents[2] / "model-weights"
        )
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        handle = self._resolve_model_handle(kwargs.pop("model_handle", None))
        self.model, self.preprocess = create_model_from_pretrained(
            handle,
            device=self.device,
            cache_dir=str(cache_dir),
        )
        self.tokenizer = get_tokenizer(handle)
        self.model.eval()
        
        self.make_root = Path(__file__).resolve().parents[2] / "MAKE"
        default_terms = self.make_root / "concept_annotation" / "term_lists" / "ConceptTerms.json"
        self.concept_terms_path = Path(kwargs.pop("concept_terms_path", default_terms))
        if not self.concept_terms_path.exists():
            raise FileNotFoundError(f"Concept terms file not found: {self.concept_terms_path}")
        
        self.reference_embeddings = self._build_reference_embeddings()
    
    def predict(
        self,
        image_paths: List[str],
        class_names: List[str],
        batch_size: int = 16,
        **kwargs
    ):
        """Run MAKE concept annotation and return probabilities per concept."""
        if not class_names:
            raise ValueError("class_names must be provided for MAKE concept annotation.")
        
        batch_size = kwargs.get("batch_size", batch_size)
        image_features = self._encode_images(image_paths, batch_size)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        concept_terms = self._load_concept_terms(class_names)
        concept_embeddings = self._build_concept_embeddings(concept_terms)
        
        scores = []
        for name in class_names:
            embeddings = concept_embeddings[name]
            score = self._calculate_presence_scores(image_features, embeddings)
            scores.append(score.unsqueeze(0))
        
        if not scores:
            raise ValueError("No concept scores computed. Check concept configuration.")
        
        score_tensor = torch.cat(scores, dim=0).T  # (num_images, num_concepts)
        probs = score_tensor.clamp(0.0, 1.0).cpu().numpy()
        preds = probs.argmax(axis=1)
        
        raw_outputs = [
            "; ".join(f"{class_names[j]}:{probs[i, j]:.4f}" for j in range(len(class_names)))
            for i in range(len(image_paths))
        ]
        
        return preds.astype(int), probs, raw_outputs
    
    def _resolve_model_handle(self, override: str | None) -> str:
        handle = override or self.model_id
        if handle.startswith("hf-hub:"):
            return handle
        if Path(handle).exists():
            raise ValueError(
                "Local MAKE checkpoints are not supported directly. "
                "Please provide a Hugging Face identifier (e.g., 'xieji-x/MAKE')."
            )
        return f"hf-hub:{handle}"
    
    def _encode_images(self, image_paths: List[str], batch_size: int) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        for idx in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[idx:idx + batch_size]
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception as exc:
                    warnings.warn(f"Failed to load image {path}: {exc}")
                    images.append(torch.zeros(3, 224, 224))
            batch = torch.stack(images).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch)
            tensors.append(feats)
        return torch.cat(tensors, dim=0)
    
    def _load_concept_terms(self, class_names: List[str]) -> "OrderedDict[str, List[str]]":
        with open(self.concept_terms_path, "r", encoding="utf-8") as f:
            terms_dict = json.load(f)
        
        ordered = OrderedDict()
        for name in class_names:
            key = self._normalize_concept_key(name)
            synonyms = terms_dict.get(key)
            if not synonyms:
                warnings.warn(f"No concept terms found for '{name}'. Falling back to literal name.")
                synonyms = [name]
            ordered[name] = synonyms
        return ordered
    
    def _build_reference_embeddings(self) -> torch.Tensor:
        refs = []
        for phrases in self.PROMPT_REFERENCES:
            tokens = self.tokenizer(phrases).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            refs.append(feats)
        return torch.stack(refs, dim=0)
    
    def _build_concept_embeddings(
        self,
        concept_terms: "OrderedDict[str, List[str]]"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        embeddings = {}
        for name, synonyms in concept_terms.items():
            prompt_stack = []
            for template in self.PROMPT_TEMPLATES:
                prompts = [template.format(term) for term in synonyms]
                tokens = self.tokenizer(prompts).to(self.device)
                with torch.no_grad():
                    feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                prompt_stack.append(feats)
            target_tensor = torch.stack(prompt_stack, dim=0)
            embeddings[name] = {
                "target": target_tensor,
                "ref": self.reference_embeddings,
            }
        return embeddings
    
    def _calculate_presence_scores(
        self,
        image_features_norm: torch.Tensor,
        embedding_bundle: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        target = embedding_bundle["target"]
        reference = embedding_bundle["ref"]
        
        target_similarity = torch.matmul(target.float(), image_features_norm.T.float())
        target_mean = target_similarity.mean(dim=1)
        
        ref_similarity = torch.matmul(reference.float(), image_features_norm.T.float())
        ref_mean = ref_similarity.mean(dim=1)
        
        stacked = torch.stack(
            (target_mean / self.TEMPERATURE, ref_mean / self.TEMPERATURE),
            dim=0,
        )
        weights = F.softmax(stacked, dim=0)[0]
        return weights.mean(dim=0)
    
    @staticmethod
    def _normalize_concept_key(name: str) -> str:
        return " ".join(name.replace("_", " ").lower().split())



















