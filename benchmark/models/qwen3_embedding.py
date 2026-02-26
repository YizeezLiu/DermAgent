"""
Qwen3-Embedding-8B Text Embedding Model for RAG.
"""

from typing import List, Tuple, Union
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from .base import BaseModel

class Qwen3EmbeddingModel(BaseModel):
    """Qwen3-Embedding-8B text embedding model for RAG."""
    
    def _load_model(self, **kwargs) -> None:
        """Load Qwen3-Embedding model from HuggingFace."""
        from pathlib import Path
        
        # Determine cache directory
        cache_dir = kwargs.pop(
            "cache_dir",
            Path(__file__).resolve().parents[2] / "model-weights"
        )
        
        print(f"Loading Qwen3-Embedding: {self.model_id}")
        print(f"Using cache directory: {cache_dir}")
        
        # trust_remote_code=True is often needed for new models like Qwen
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            cache_dir=cache_dir,
            **kwargs
        )
        self.model.eval()
        print("Model loaded!")
    
    def encode_texts(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,  # Reasonable limit for RAG chunks
                return_tensors="pt"
            ).to(self.model.device)
            
            # Encode
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Mean Pooling with attention mask
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs.attention_mask
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
                if normalize:
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu())
        
        if not all_embeddings:
            return np.array([])
            
        return torch.cat(all_embeddings, dim=0).float().numpy()

    def predict(
        self, 
        image_paths: List[str], 
        class_names: List[str],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Not applicable for embedding model."""
        raise NotImplementedError("Qwen3EmbeddingModel is text-only.")
