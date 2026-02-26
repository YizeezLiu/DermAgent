"""
DermLIP Zero-Shot Classification Model.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Any

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from .base import ZeroShotModel

# Add Derm1M source to path
DERM1M_SRC = Path(__file__).parent.parent.parent / "Derm1M" / "src"
sys.path.insert(0, str(DERM1M_SRC))


class DermLIPModel(ZeroShotModel):
    """DermLIP zero-shot classification model."""
    
    def _load_model(self, **kwargs) -> None:
        from open_clip import create_model_from_pretrained, get_tokenizer
        
        print(f"Loading DermLIP: {self.model_id}")
        self.model, self.preprocess = create_model_from_pretrained(
            f'hf-hub:{self.model_id}',
            device=self.device
        )
        self.tokenizer = get_tokenizer(f'hf-hub:{self.model_id}')
        self.model.eval()
        print("Model loaded!")
    
    def build_text_features(
        self, 
        class_names: List[str], 
        templates: List[str]
    ) -> torch.Tensor:
        """Build text embeddings using prompt templates."""
        text_features_list = []
        
        with torch.no_grad():
            for class_name in class_names:
                texts = [t.format(class_name) for t in templates]
                tokens = self.tokenizer(texts).to(self.device)
                features = self.model.encode_text(tokens)
                features = features.mean(dim=0)
                features = features / features.norm(dim=-1, keepdim=True)
                text_features_list.append(features)
        
        return torch.stack(text_features_list, dim=0)
    
    def encode_images(
        self, 
        image_paths: List[str], 
        batch_size: int = 32
    ) -> torch.Tensor:
        """Encode images to feature vectors."""
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    images.append(torch.zeros(3, 224, 224))
            
            batch = torch.stack(images).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
            
            all_features.append(features)
        
        return torch.cat(all_features, dim=0)
    
    def predict(
        self,
        image_paths: List[str],
        class_names: List[str],
        prompt_templates: List[str] = None,
        batch_size: int = 32,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run zero-shot classification."""
        if prompt_templates is None:
            prompt_templates = [
                "a dermoscopy image of {}",
                "a clinical photo of {}",
                "a skin lesion of {}",
            ]
        
        # Build text features
        text_features = self.build_text_features(class_names, prompt_templates)
        
        predictions = []
        probabilities = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"[{self.name}] Processing"):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.preprocess(img))
                except:
                    images.append(torch.zeros(3, 224, 224))
            
            batch = torch.stack(images).to(self.device)
            
            with torch.no_grad():
                img_features = self.model.encode_image(batch)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * img_features @ text_features.T).softmax(dim=-1)
                
                preds = similarity.argmax(dim=-1).cpu().numpy()
                probs = similarity.cpu().numpy()
            
            predictions.extend(preds)
            probabilities.extend(probs)
        
        # Raw outputs: formatted probability strings
        raw_outputs = [
            "; ".join(f"{class_names[i]}:{p[i]:.3f}" for i in range(len(class_names)))
            for p in probabilities
        ]
        
        return np.array(predictions), np.array(probabilities), raw_outputs




















