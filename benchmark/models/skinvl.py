"""
SkinVL VQA Model for benchmark evaluation.

Based on MM-Skin (LLaVA-Med v1.5 + Mistral-7B) trained on dermatology data.
Requires MM-Skin repo at project root: MM-Skin/llava/
Model weights: model-weights/SkinVL-PubMM
"""

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from .base import VQAModel

# MM-Skin repo path — contains the llava package used by SkinVL
MMSKIN_PATH = Path(__file__).parent.parent.parent / "MM-Skin"


class SkinVLModel(VQAModel):
    """SkinVL-PubMM VQA model (LLaVA-Med + Mistral-7B, dermatology fine-tuned)."""

    def _load_model(self, **kwargs) -> None:
        """Load SkinVL model using MM-Skin's llava package."""
        # Remove any existing llava modules to avoid conflicts with LLaVA-Med
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("llava")]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Insert MM-Skin at front of sys.path so its llava package takes priority
        mmskin_str = str(MMSKIN_PATH)
        if mmskin_str in sys.path:
            sys.path.remove(mmskin_str)
        sys.path.insert(0, mmskin_str)

        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            get_model_name_from_path,
            KeywordsStoppingCriteria,
        )
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.utils import disable_torch_init

        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self._conv_templates = conv_templates
        self._SeparatorStyle = SeparatorStyle

        disable_torch_init()

        print(f"Loading SkinVL: {self.model_id}")

        # SkinVL-PubMM is LlavaMistralForCausalLM — the builder dispatches on
        # substrings in model_name: "llava" → LLaVA path, "mistral" → Mistral class.
        # The actual directory name "SkinVL-PubMM" lacks both, so we override.
        model_name = "llava-v1.5-mistral-7b-skinvl"
        print(f"Model name (override): {model_name}")

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=self.model_id,
                model_base=None,
                model_name=model_name,
                device=self.device,
            )
        )

        self.conv_mode = "mistral_instruct"
        print(f"Conversation mode: {self.conv_mode}")
        print("SkinVL loaded successfully!")

    def ask(self, image_path: str, question: str) -> str:
        """
        Ask SkinVL about an image.

        Args:
            image_path: Path to the image file.
            question: Question to ask about the image.

        Returns:
            Model's response as a string.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self._process_images(
                [image], self.image_processor, self.model.config
            )
            if isinstance(image_tensor, list):
                image_tensor = image_tensor[0]

            # Build prompt with image token (following MM-Skin VQA_test.py)
            qs = question.replace(self._DEFAULT_IMAGE_TOKEN, "").strip()
            if getattr(self.model.config, "mm_use_im_start_end", False):
                qs = (
                    self._DEFAULT_IM_START_TOKEN
                    + self._DEFAULT_IMAGE_TOKEN
                    + self._DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = self._DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = self._conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                self._tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    self._IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(self.device)
            )

            stop_str = conv.sep if conv.sep_style != self._SeparatorStyle.TWO else conv.sep2
            stopping_criteria = None
            if stop_str:
                stopping_criteria = self._KeywordsStoppingCriteria(
                    [stop_str], self.tokenizer, input_ids
                )

            with torch.inference_mode(), torch.cuda.amp.autocast():
                generate_kwargs = dict(
                    images=image_tensor.unsqueeze(0).half().to(self.device),
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.9,
                    num_beams=5,
                    max_new_tokens=256,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
                if stopping_criteria:
                    generate_kwargs["stopping_criteria"] = [stopping_criteria]

                output_ids = self.model.generate(input_ids, **generate_kwargs)

            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return output

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {e}"

    def predict(
        self,
        image_paths: List[str],
        class_names: List[str],
        prompt: str = "",
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run SkinVL prediction, routing by task_type.

        For diagnosis tasks, always builds a prompt with numbered candidate
        diseases regardless of the dataset's vqa_prompt.  Other tasks
        (concepts, captioning, vqa) fall through to the base class which
        forwards the dataset prompt to ask().
        """
        task_type = kwargs.get("task_type", "diagnosis")

        if task_type == "diagnosis":
            if len(class_names) > 30:
                diagnosis_prompt = (
                    "This is a skin lesion image. "
                    "What is the most likely dermatological diagnosis? "
                    "Answer with only the diagnosis name, nothing else."
                )
            else:
                diagnosis_prompt = self._build_diagnosis_prompt(class_names)
            return self._run_predict(image_paths, class_names, diagnosis_prompt)

        if not prompt:
            prompt = self._build_diagnosis_prompt(class_names)
        return self._run_predict(image_paths, class_names, prompt)

    def _run_predict(
        self,
        image_paths: List[str],
        class_names: List[str],
        prompt: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Shared inference loop (uses base VQAModel._parse_response)."""
        predictions, confidences, raw_outputs = [], [], []

        for path in tqdm(image_paths, desc=f"[{self.name}] Processing"):
            response = self.ask(path, prompt)
            pred_idx, conf = self._parse_response(response, class_names)
            # Preserve -1 on parse failure (saved as predicted_class="unknown").
            predictions.append(pred_idx)
            confidences.append(conf)
            raw_outputs.append(response)

        return np.array(predictions), np.array(confidences), raw_outputs

    @staticmethod
    def _build_diagnosis_prompt(class_names: List[str]) -> str:
        """Build a diagnosis prompt that always lists numbered candidate diseases."""
        options = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(class_names))
        return (
            f"Look at this skin lesion image carefully.\n"
            f"What is the most likely diagnosis? Choose exactly one from the "
            f"following options:\n{options}\n\n"
            f"Answer with only the diagnosis name, nothing else."
        )


class SkinVLZSModel(SkinVLModel):
    """SkinVL with original MM-Skin ZS_classification.py prompt and generation params.

    Differences from the base SkinVLModel (which mirrors VQA_test.py):
      - Prompt: native comma-separated category list from ZS_classification.py
      - temperature: 0.5 (ZS_classification.py default) instead of 0.2 (VQA default)
      - num_beams: 1 — beam search collapses class diversity at low temperature;
        disabling it lets the model express its full classification vocabulary.
    """

    def ask(self, image_path: str, question: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self._process_images(
                [image], self.image_processor, self.model.config
            )
            if isinstance(image_tensor, list):
                image_tensor = image_tensor[0]

            qs = question.replace(self._DEFAULT_IMAGE_TOKEN, "").strip()
            if getattr(self.model.config, "mm_use_im_start_end", False):
                qs = (
                    self._DEFAULT_IM_START_TOKEN
                    + self._DEFAULT_IMAGE_TOKEN
                    + self._DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = self._DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = self._conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                self._tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    self._IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(self.device)
            )

            stop_str = conv.sep if conv.sep_style != self._SeparatorStyle.TWO else conv.sep2
            stopping_criteria = None
            if stop_str:
                stopping_criteria = self._KeywordsStoppingCriteria(
                    [stop_str], self.tokenizer, input_ids
                )

            with torch.inference_mode(), torch.amp.autocast("cuda"):
                generate_kwargs = dict(
                    images=image_tensor.unsqueeze(0).half().to(self.device),
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    max_new_tokens=256,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
                if stopping_criteria:
                    generate_kwargs["stopping_criteria"] = [stopping_criteria]

                output_ids = self.model.generate(input_ids, **generate_kwargs)

            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return output

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {e}"

    def predict(
        self,
        image_paths: List[str],
        class_names: List[str],
        prompt: str = "",
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        task_type = kwargs.get("task_type", "diagnosis")
        if task_type == "diagnosis":
            if len(class_names) > 30:
                native_prompt = (
                    "This is a skin lesion image. "
                    "What is the most likely dermatological diagnosis? "
                    "Answer with only the diagnosis name, nothing else."
                )
            else:
                native_prompt = self._build_zs_diagnosis_prompt(class_names)
            return self._run_predict(image_paths, class_names, native_prompt)
        if not prompt:
            prompt = self._build_zs_diagnosis_prompt(class_names)
        return self._run_predict(image_paths, class_names, prompt)

    @staticmethod
    def _build_zs_diagnosis_prompt(class_names: List[str]) -> str:
        """ZS-style prompt with comma-separated candidates (native MM-Skin format)."""
        return (
            f"This is a skin lesion image. "
            f"From the following categories: {', '.join(class_names)}, "
            f"which one is the diagnosis?"
        )
