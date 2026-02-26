"""
LLaVA-Med VQA Model for benchmark evaluation.

This implementation follows the official LLaVA-Med v1.5 codebase.
Requires the llavamed conda environment to be activated.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from .base import VQAModel

# LLaVA-Med path - add to sys.path for imports
LLAVAMED_PATH = Path(__file__).parent.parent.parent / "LLaVA-Med"


class LLaVAMedModel(VQAModel):
    """LLaVA-Med VQA model following official implementation."""
    
    def _load_model(self, **kwargs) -> None:
        """Load LLaVA-Med model using official loading method."""
        # Clean up any existing llava modules to avoid conflicts
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('llava')]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        # Add LLaVA-Med to path (at front to take priority)
        llavamed_str = str(LLAVAMED_PATH)
        if llavamed_str in sys.path:
            sys.path.remove(llavamed_str)
        sys.path.insert(0, llavamed_str)
        
        # Now import from LLaVA-Med
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import (
            process_images, 
            tokenizer_image_token, 
            get_model_name_from_path,
            KeywordsStoppingCriteria
        )
        from llava.constants import (
            IMAGE_TOKEN_INDEX, 
            DEFAULT_IMAGE_TOKEN, 
            DEFAULT_IM_START_TOKEN, 
            DEFAULT_IM_END_TOKEN
        )
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.utils import disable_torch_init
        
        # Store imports for later use
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self._conv_templates = conv_templates
        self._SeparatorStyle = SeparatorStyle
        
        # Disable torch init for faster loading
        disable_torch_init()
        
        print(f"Loading LLaVA-Med: {self.model_id}")
        
        # Get model name from path (used for determining model type)
        model_name = get_model_name_from_path(self.model_id)
        print(f"Model name: {model_name}")
        
        # Load model using official method
        # model_base=None for directly loading from HuggingFace
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(
                model_path=self.model_id,
                model_base=None,
                model_name=model_name,
                device=self.device
            )
        
        # Determine conversation mode based on model
        if 'mistral' in model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif 'llama-2' in model_name.lower():
            self.conv_mode = "llava_llama_2"
        else:
            self.conv_mode = "vicuna_v1"
        
        print(f"Using conversation mode: {self.conv_mode}")
        print("LLaVA-Med loaded successfully!")
    
    def ask(self, image_path: str, question: str) -> str:
        """
        Ask LLaVA-Med about an image.
        
        Args:
            image_path: Path to the image file
            question: Question to ask about the image
            
        Returns:
            Model's response as string
        """
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self._process_images(
                [image], 
                self.image_processor, 
                self.model.config
            )
            if isinstance(image_tensor, list):
                image_tensor = image_tensor[0]
            
            # Build prompt with image token
            # Following official model_vqa.py logic
            qs = question.replace(self._DEFAULT_IMAGE_TOKEN, '').strip()
            
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                qs = (self._DEFAULT_IM_START_TOKEN + self._DEFAULT_IMAGE_TOKEN + 
                      self._DEFAULT_IM_END_TOKEN + '\n' + qs)
            else:
                qs = self._DEFAULT_IMAGE_TOKEN + '\n' + qs
            
            # Create conversation
            conv = self._conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize with image token handling
            input_ids = self._tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                self._IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            
            # Set up stopping criteria
            # For mistral_instruct, sep is empty and sep2 is "</s>"
            # We should use "</s>" as the stop string
            if conv.sep_style == self._SeparatorStyle.LLAMA_2:
                stop_str = conv.sep2  # "</s>"
            elif conv.sep_style != self._SeparatorStyle.TWO:
                stop_str = conv.sep
            else:
                stop_str = conv.sep2
            
            # Only use stopping criteria if we have a valid stop string
            stopping_criteria = None
            if stop_str:
                keywords = [stop_str]
                stopping_criteria = self._KeywordsStoppingCriteria(
                    keywords, self.tokenizer, input_ids
                )
            
            # Generate response
            with torch.inference_mode():
                generate_kwargs = dict(
                    images=image_tensor.unsqueeze(0).half().to(self.device),
                    do_sample=True,  # Enable sampling for more varied responses
                    temperature=0.5,  # Lower temperature for more consistency
                    top_p=0.95,
                    max_new_tokens=300,
                    use_cache=True,
                )
                if stopping_criteria:
                    generate_kwargs["stopping_criteria"] = [stopping_criteria]
                
                output_ids = self.model.generate(input_ids, **generate_kwargs)
            
            # Decode output
            outputs = self.tokenizer.batch_decode(
                output_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            return outputs
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {e}"
    
    def predict(
        self,
        image_paths: List[str],
        class_names: List[str],
        prompt: str = "",
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Run LLaVA-Med classification on a batch of images.
        
        Args:
            image_paths: List of image paths
            class_names: List of possible class names
            prompt: Question prompt (uses default if empty)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (predictions, confidences, raw_outputs)
        """
        if not prompt:
            prompt = self._default_prompt(class_names)
        
        predictions = []
        confidences = []
        raw_outputs = []
        
        for path in tqdm(image_paths, desc=f"[{self.name}] Processing"):
            response = self.ask(path, prompt)
            pred_idx, conf = self._parse_response(response, class_names)
            
            # Use index 0 as default fallback
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
        """
        Parse LLaVA-Med response with strong format matching.
        
        Handles formats like:
        - "Answer: 1. melanocytic nevi"
        - "1. melanocytic nevi"
        - "melanocytic nevi"
        """
        import re
        response_lower = response.lower().strip()
        
        # Try to match "Answer: X. category" or "X. category" format
        # Pattern matches: optional "answer:", then number, then dot, then text
        pattern = r'(?:answer:\s*)?(\d+)\.\s*([a-z\s]+)'
        match = re.search(pattern, response_lower)
        
        if match:
            num = int(match.group(1))
            # Check if number is valid (1-indexed in prompt)
            if 1 <= num <= len(class_names):
                return num - 1, 1.0  # Convert to 0-indexed
        
        # Fallback: exact class name match
        for i, name in enumerate(class_names):
            if name.lower() in response_lower:
                return i, 0.9
        
        # Extended keyword fallback for dermoscopy
        keywords = {
            "melanocytic nevi": ["melanocytic", "nevi", "nevus", "mole", "benign mole"],
            "melanoma": ["melanoma", "malignant melanoma", "mm"],
            "basal cell carcinoma": ["basal cell", "bcc", "basal"],
            "actinic keratosis": ["actinic", "ak", "solar keratosis"],
            "benign keratosis": ["benign keratosis", "seborrheic", "sk", "bkl"],
            "dermatofibroma": ["dermatofibroma", "fibroma", "df"],
            "vascular lesion": ["vascular", "vasc", "angioma", "hemangioma", "cherry", "blood vessel"],
        }
        
        for i, name in enumerate(class_names):
            if name.lower() in keywords:
                for kw in keywords[name.lower()]:
                    if kw in response_lower:
                        return i, 0.7
        
        # No match found
        return -1, 0.0
    
    def _default_prompt(self, class_names: List[str]) -> str:
        """Generate concise classification prompt with disclaimer."""
        options = ", ".join(class_names)
        return f"""[For research testing only, not clinical use]

This is a dermoscopy image. Classify it into one of these categories: {options}.

Look at the color and pattern:
- Brown/tan with network = likely nevi or melanoma
- Red/purple = likely vascular lesion  
- Pearly/pink with vessels = likely basal cell carcinoma
- Scaly/rough = likely actinic keratosis or benign keratosis
- Brown with central white = likely dermatofibroma

What is the most likely diagnosis? State only the category name."""
