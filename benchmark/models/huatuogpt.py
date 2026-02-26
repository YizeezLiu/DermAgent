"""
HuatuoGPT-Vision VQA Model.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from .base import VQAModel

# Add HuatuoGPT-Vision to path (must be first to override any other llava)
HUATUO_PATH = Path(__file__).parent.parent.parent / "HuatuoGPT-Vision"


class HuatuoGPTModel(VQAModel):
    """HuatuoGPT-Vision VQA model."""
    
    def _load_model(self, **kwargs) -> None:
        # Remove any conflicting llava modules from cache
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('llava')]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        # Ensure HuatuoGPT path is at the front
        huatuo_str = str(HUATUO_PATH)
        # Remove LLaVA-Med path if present
        llavamed_str = str(Path(__file__).parent.parent.parent / "LLaVA-Med")
        if llavamed_str in sys.path:
            sys.path.remove(llavamed_str)
        if huatuo_str not in sys.path:
            sys.path.insert(0, huatuo_str)
        
        from cli import HuatuoChatbot
        
        print(f"Loading HuatuoGPT: {self.model_id}")
        self.bot = HuatuoChatbot(self.model_id, device=self.device)
        self.model = self.bot.model
        print("Model loaded!")
    
    def ask(self, image_path: str, question: str) -> str:
        """Ask HuatuoGPT about an image."""
        try:
            responses = self.bot.inference(question, [image_path])
            return responses[0] if responses else ""
        except Exception as e:
            return f"Error: {e}"
    
    def predict(
        self,
        image_paths: List[str],
        class_names: List[str],
        prompt: str = "",
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run HuatuoGPT classification."""
        if not prompt:
            prompt = self._default_prompt(class_names)
        
        predictions = []
        confidences = []
        raw_outputs = []
        
        for path in tqdm(image_paths, desc=f"[{self.name}] Processing"):
            response = self.ask(path, prompt)
            pred_idx, conf = self._parse_response(response, class_names)
            
            if pred_idx == -1:
                pred_idx = 1
                conf = 0.0
            
            predictions.append(pred_idx)
            confidences.append(conf)
            raw_outputs.append(response)
        
        return np.array(predictions), np.array(confidences), raw_outputs
    
    def _default_prompt(self, class_names: List[str]) -> str:
        options = "\n".join(f"{i+1}. {name}" for i, name in enumerate(class_names))
        return f"""Look at this skin lesion image carefully. 
What is the most likely diagnosis? Choose exactly one from the following options:
{options}

Answer with only the diagnosis name, nothing else."""

