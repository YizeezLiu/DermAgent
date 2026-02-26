"""
Qwen3-VL VQA Model.
"""

from typing import List, Tuple

import torch
import numpy as np
from tqdm import tqdm

from .base import VQAModel

try:  # Transformers >= 4.38
    from transformers import AutoModelForImageTextToText
except ImportError:  # Older releases expose the same class under a different name
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
except Exception:  # pragma: no cover
    AutoModelForImageTextToText = None

from transformers import AutoProcessor


class Qwen3VLModel(VQAModel):
    """Qwen3-VL VQA model."""
    
    def _load_model(self, use_flash_attn: bool = False, **kwargs) -> None:
        if AutoModelForImageTextToText is None:
            raise ImportError(
                "AutoModelForImageTextToText is unavailable in the current transformers install. "
                "Please upgrade transformers (>=4.38) or install a build that exposes "
                "AutoModelForImageTextToText / AutoModelForVision2Seq."
            )
        
        print(f"Loading Qwen3-VL: {self.model_id}")
        
        if use_flash_attn:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print("Model loaded!")
    
    def ask(self, image_path: str, question: str) -> str:
        """Ask Qwen3-VL about an image."""
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }]
            
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            
            # Trim input tokens
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            response = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            return response
            
        except Exception as e:
            return f"Error: {e}"
    
    def predict(
        self,
        image_paths: List[str],
        class_names: List[str],
        prompt: str = "",
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run Qwen3-VL classification."""
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


