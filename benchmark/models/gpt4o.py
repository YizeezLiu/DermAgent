"""
GPT-4o VQA Model via OpenAI API.
"""

import os
import base64
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from .base import VQAModel


class GPT4oModel(VQAModel):
    """GPT-4o VQA model using OpenAI API."""
    
    def _load_model(self, **kwargs) -> None:
        from openai import OpenAI
        from dotenv import load_dotenv
        
        # Load .env from project root (benchmark/models/ -> project root)
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded .env from {env_path}")
        else:
            load_dotenv()  # Fallback to default behavior
        
        print(f"Initializing OpenAI client for model: {self.model_id}")
        print(f"  base_url: {os.environ.get('OPENAI_BASE_URL', 'default')}")
        
        # OpenAI client automatically reads OPENAI_API_KEY and OPENAI_BASE_URL from env
        self.client = OpenAI()
        self.model = self.model_id
        print("Client ready!")
    
    def _encode_image(self, path: str) -> str:
        """Encode image to base64."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_media_type(self, path: str) -> str:
        """Get media type from file extension."""
        suffix = Path(path).suffix.lower()
        types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", 
                 ".png": "image/png", ".gif": "image/gif"}
        return types.get(suffix, "image/jpeg")
    
    def ask(self, image_path: str, question: str) -> str:
        """Ask GPT-4o about an image."""
        try:
            b64 = self._encode_image(image_path)
            media_type = self._get_media_type(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64}",
                                "detail": "high"
                            }
                        },
                        {"type": "text", "text": question}
                    ]
                }],
                max_tokens=500,
                temperature=0
            )
            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                return f"Error: Empty response from API for model {self.model}."
            return content.strip()
        except Exception as e:
            import traceback
            print(f"[GPT4oModel] Error calling API: {e}")
            traceback.print_exc()
            return f"Error: {e}"
    
    def predict(
        self,
        image_paths: List[str],
        class_names: List[str],
        prompt: str = "",
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run GPT-4o classification with simplified prompt."""
        if not prompt:
            # Use short prompt for token efficiency
            prompt = f"Classify this skin lesion. Reply with ONLY one of: {', '.join(class_names)}"
        
        predictions = []
        confidences = []
        raw_outputs = []
        
        for path in tqdm(image_paths, desc=f"[{self.name}] Processing"):
            response = self.ask(path, prompt)
            pred_idx, conf = self._parse_response(response, class_names)
            
            if pred_idx == -1:
                pred_idx = 1  # Default to most common class
                conf = 0.0
            
            predictions.append(pred_idx)
            confidences.append(conf)
            raw_outputs.append(response)
        
        return np.array(predictions), np.array(confidences), raw_outputs
