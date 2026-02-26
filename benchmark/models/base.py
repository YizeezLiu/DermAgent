"""
Base model classes for benchmark evaluation.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm


class BaseModel(ABC):
    """Base class for all benchmark models."""
    
    def __init__(self, model_id: str, device: str = "cuda", **kwargs):
        self.model_id = model_id
        self.device = device
        self.model = None
        self._load_model(**kwargs)
    
    @abstractmethod
    def _load_model(self, **kwargs) -> None:
        """Load model weights."""
        pass
    
    @abstractmethod
    def predict(
        self, 
        image_paths: List[str], 
        class_names: List[str],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Run inference on images.
        
        Args:
            image_paths: List of image paths
            class_names: List of class names
            **kwargs: Model-specific arguments
            
        Returns:
            predictions: Predicted class indices
            confidences: Prediction confidences
            raw_outputs: Raw model outputs (for debugging)
        """
        pass
    
    @property
    def name(self) -> str:
        """Short name for the model."""
        return self.model_id.split("/")[-1]


class ZeroShotModel(BaseModel):
    """Base class for zero-shot classification models (CLIP-like)."""
    
    @abstractmethod
    def build_text_features(
        self, 
        class_names: List[str], 
        templates: List[str]
    ) -> Any:
        """Build text embeddings for zero-shot classification."""
        pass
    
    @abstractmethod
    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> Any:
        """Encode images to feature vectors."""
        pass


class VQAModel(BaseModel):
    """Base class for VQA-based classification models."""
    
    @abstractmethod
    def ask(self, image_path: str, question: str) -> str:
        """Ask a question about an image."""
        pass
    
    def predict(
        self,
        image_paths: List[str],
        class_names: List[str],
        prompt: str = "",
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run VQA-based classification."""
        predictions = []
        confidences = []
        raw_outputs = []
        
        for path in tqdm(image_paths, desc=f"[{self.name}] Processing"):
            response = self.ask(path, prompt)
            pred_idx, conf = self._parse_response(response, class_names)
            
            # Default fallback for unparseable responses
            if pred_idx == -1:
                pred_idx = 0
                conf = 0.0
            
            predictions.append(pred_idx)
            confidences.append(conf)
            raw_outputs.append(response)
        
        return np.array(predictions), np.array(confidences), raw_outputs
    
    def _parse_response(
        self, 
        response: str, 
        class_names: List[str]
    ) -> Tuple[int, float]:
        """Parse VQA response to extract predicted class."""
        response_lower = response.lower().strip()
        
        # Exact match
        for i, name in enumerate(class_names):
            if name.lower() in response_lower:
                return i, 1.0
        
        # Keyword fallback (skin disease specific)
        keywords = {
            # PAD classes
            "nevus": ["nevus", "nevi", "mole"],
            "basal cell carcinoma": ["basal", "bcc"],
            "actinic keratosis": ["actinic", "ak", "akiec"],
            "seborrheic keratosis": ["seborrheic", "sk"],
            "squamous cell carcinoma": ["squamous", "scc"],
            "melanoma": ["melanoma", "malignant"],
            # HAM10000 classes
            "melanocytic nevi": ["melanocytic", "nevi", "nevus", "mole"],
            "benign keratosis": ["benign keratosis", "bkl", "seborrheic"],
            "dermatofibroma": ["dermatofibroma", "fibroma", "df"],
            "vascular lesion": ["vascular", "vasc", "angioma", "hemangioma"],
        }
        
        for i, name in enumerate(class_names):
            if name in keywords:
                for kw in keywords[name]:
                    if kw in response_lower:
                        return i, 0.8
        
        return -1, 0.0

