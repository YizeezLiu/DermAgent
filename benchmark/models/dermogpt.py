"""
DermoGPT-RL VQA Model.

DermoGPT-RL is a Qwen3-VL-8B model fine-tuned specifically for dermatology,
excelling at morphological descriptions and dermatology-specific VQA tasks.

Uses DermoInstruct prompt templates from the DermoGPT paper
(Appendix C of arXiv:2601.01868v1) adapted for all 4 benchmark tasks:
- Diagnosis: flat MCQA prompt (diagnosis_vqa.flat_4way_mcqa, N-way adapted)
- Concepts (SkinCon): JSON output (morphology_json.clinical_images_skincon)
- Concepts (Derm7pt): JSON output (morphology_json.dermoscopic_images_derm7pt)
- Captioning: Detailed clinical narrative description
"""

import json
import re
from pathlib import Path
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

# Default local path for DermoGPT-RL weights
_DEFAULT_MODEL_PATH = str(
    Path(__file__).parent.parent.parent / "model-weights" / "DermoGPT-RL"
)

# =============================================================================
# DermoInstruct Prompt Templates (from dermo_instruct_prompt_templates.yaml)
# Adapted with minimal changes for benchmark evaluation
# =============================================================================

# --- Task 1: Diagnosis (flat MCQA, adapted from diagnosis_vqa.flat_4way_mcqa) ---
# System prompt: adapted from chain_of_thought_reasoning.system_prompt,
# stripped of training-specific language ("confirmed diagnosis", "clinical concepts").
_DIAGNOSIS_SYSTEM = (
    "You are an expert dermatologist AI, acting as a clinical consultant. "
    "Your primary task is to analyze a skin lesion image and provide a diagnosis "
    "with clinical reasoning."
)

# User prompt: based on diagnosis_vqa.flat_4way_mcqa prompt #2,
# adapted for N-way with numbered candidate list.
_DIAGNOSIS_USER = """\
Based on the skin lesion shown in this image, please select the most \
accurate diagnosis from the options below.

{candidate_diseases}

Answer with only the diagnosis name, nothing else."""

# --- Task 2a: Concept Annotation - SkinCon (JSON output) ---
_SKINCON_SYSTEM = (
    "You are an expert in dermatology. Your task is to perform a detailed "
    "visual analysis of a provided skin lesion image (clinical or dermoscopic). "
    "You will be given an image of a skin lesion and a predefined list of "
    "standardized clinical concepts. Your task is to analyze the image, "
    "describe it clinically, and then map the observed features to the provided "
    "concepts. Any features you observe that are not on the list must be "
    "categorized separately. Your output must be a single, clean JSON object "
    "and nothing else."
)

_SKINCON_USER = """\
Analyze the provided skin lesion image using the established SkinCon vocabulary. \
First, perform a detailed, step-by-step visual assessment. Second, generate a \
single, valid JSON object as your final and ONLY output. Do not include any text, \
explanations, or markdown formatting outside of the JSON object.

### SkinCon Morphological Concepts List
Here are the standardized concepts you MUST use for classification:
{concept_list}

### Required JSON Output Structure
The JSON object MUST contain exactly three keys:
1. detailed_description: (String) A comprehensive clinical narrative of the \
lesion's morphology, including primary lesion type, color, shape, border, \
surface, and texture.
2. morphological_features_skincon: (Array of Strings) A list of all observed \
features that EXACTLY MATCH one or more terms from the concepts provided above.
3. morphological_features_others: (Array of Strings) A list of important \
observed features that are NOT found in the concept list. If all features are \
covered, this array should be empty [].

### YOUR TASK
Now, for the image I have provided, please perform the same analysis and \
generate the JSON output. Remember, the JSON object is the only thing you \
should return."""

# --- Task 2b: Concept Annotation - Derm7pt (JSON output) ---
_DERM7PT_SYSTEM = (
    "You are an expert in dermatology. Your task is to perform a detailed "
    "visual analysis of a provided dermoscopic image. You will analyze the "
    "image and classify its features according to the 7-point checklist, "
    "assigning the single most fitting morphological label to each of the "
    "seven criteria. Your output must be a single, clean JSON object and "
    "nothing else."
)

_DERM7PT_USER = """\
Analyze the provided skin lesion image using the established Derm7pt vocabulary. \
First, perform a detailed, step-by-step visual assessment. Second, for each of \
the 7 criteria, select the single most appropriate label from the lists provided \
below. Finally, generate a single, valid JSON object as your final and ONLY \
output. Do not include any text, explanations, or markdown formatting outside \
of the JSON object.

### Derm7pt Morphological Concepts and Labels
You MUST classify the lesion by selecting exactly one label for each of the 7 \
criteria:

1. pigment_network: ["absent", "typical", "atypical"]
2. blue_whitish_veil: ["absent", "present"]
3. vascular_structures: ["absent", "arborizing", "comma", "hairpin", \
"within regression", "wreath", "dotted", "linear irregular"]
4. pigmentation: ["absent", "diffuse regular", "localized regular", \
"diffuse irregular", "localized irregular"]
5. streaks: ["absent", "regular", "irregular"]
6. dots_and_globules: ["absent", "regular", "irregular"]
7. regression_structures: ["absent", "blue areas", "white areas", "combinations"]

### Required JSON Output Structure
The JSON object MUST contain exactly three keys:
1. detailed_description: (String) A comprehensive clinical narrative of the \
lesion's morphology, including primary lesion type, color, shape, border, \
surface, and texture, justifying your label choices.
2. morphological_features_Derm7pt: (Object) An object where each key is one \
of the 7 Derm7pt criteria and its value is a single (String) label selected \
from the lists above.
3. morphological_features_others: (Array of Strings) A list of important \
observed features that are NOT part of the 7-point checklist classification. \
If none, this array should be empty [].

### YOUR TASK
Now, for the image I have provided, please perform the same analysis and \
generate the JSON output. Remember, the JSON object is the only thing you \
should return."""

# --- Task 3: Captioning (clinical description) ---
_CAPTIONING_SYSTEM = (
    "You are an expert in dermatology. Your task is to perform a detailed "
    "visual analysis of a provided skin lesion image (clinical or dermoscopic) "
    "and generate a comprehensive clinical description."
)

_CAPTIONING_USER = """\
Analyze the provided skin lesion image. Perform a detailed, step-by-step \
visual assessment and provide a comprehensive clinical description.

Your description should cover:
- Primary lesion type and morphology
- Color and pigmentation patterns
- Shape and border characteristics
- Surface characteristics and texture
- Any notable dermoscopic features (if applicable)
- Clinical impression

Provide a single, detailed clinical narrative as your response. Use \
professional dermatological terminology."""


# =============================================================================
# DermoGPTStructuredModel - DermoInstruct prompt templates for all 4 tasks
# =============================================================================

class DermoGPTStructuredModel(VQAModel):
    """DermoGPT-RL with DermoInstruct prompt templates from the original paper.

    Uses task-specific structured prompts (Appendix C, arXiv:2601.01868v1):
    - Diagnosis: flat MCQA prompt (diagnosis_vqa.flat_4way_mcqa, N-way adapted)
    - Concepts (SkinCon): JSON output with morphological_features_skincon array
    - Concepts (Derm7pt): JSON output with morphological_features_Derm7pt object
    - Captioning: Detailed clinical narrative description
    - VQA: Falls back to simple question-answering with system prompt
    """

    # System prompt used for the ask() fallback path (VQA tasks)
    _SYSTEM_PROMPT: str = _DIAGNOSIS_SYSTEM

    def _load_model(self, use_flash_attn: bool = False, **kwargs) -> None:
        if AutoModelForImageTextToText is None:
            raise ImportError(
                "AutoModelForImageTextToText is unavailable in the current "
                "transformers install. Please upgrade transformers (>=4.38) or "
                "install a build that exposes AutoModelForImageTextToText / "
                "AutoModelForVision2Seq."
            )

        # Resolve model path: use provided model_id or fall back to local weights
        model_path = self.model_id
        if not Path(model_path).exists():
            # Try the default local path if model_id looks like a HF repo ID
            if Path(_DEFAULT_MODEL_PATH).exists():
                print(
                    f"[DermoGPT] model_id '{model_path}' not found locally, "
                    f"falling back to {_DEFAULT_MODEL_PATH}"
                )
                model_path = _DEFAULT_MODEL_PATH
            # Otherwise let from_pretrained handle it (may download from HF)

        print(f"Loading DermoGPT-RL: {model_path}")

        if use_flash_attn:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        self.processor = AutoProcessor.from_pretrained(model_path)
        print("DermoGPT-RL loaded!")

    def ask(self, image_path: str, question: str) -> str:
        """Ask DermoGPT-RL about an image (used for VQA fallback)."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question},
                    ],
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            # Trim input tokens
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, output_ids)
            ]
            response = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            return response

        except Exception as e:
            return f"Error: {e}"

    def _ask_structured(
        self,
        image_path: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Ask with a task-specific system prompt and configurable token limit."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, output_ids)
            ]
            return self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        except Exception as e:
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # predict() dispatcher -- routes to task-specific methods
    # ------------------------------------------------------------------

    def predict(
        self,
        image_paths: List[str],
        class_names: List[str],
        prompt: str = "",
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Route to task-specific prediction with DermoInstruct prompts."""
        task_type = kwargs.get("task_type", "diagnosis")

        if task_type == "diagnosis":
            return self._predict_diagnosis(image_paths, class_names)
        elif task_type == "concepts":
            return self._predict_concepts(image_paths, class_names)
        elif task_type == "captioning":
            return self._predict_captioning(image_paths)
        else:
            # VQA / unknown -- fall back to ask() with dataset prompt
            predictions = []
            confidences = []
            raw_outputs = []
            if not prompt:
                prompt = self._default_prompt(class_names)
            for path in tqdm(image_paths, desc=f"[{self.name}] Processing"):
                response = self.ask(path, prompt)
                pred_idx, conf = self._parse_response(response, class_names)
                if pred_idx == -1:
                    pred_idx, conf = 0, 0.0
                predictions.append(pred_idx)
                confidences.append(conf)
                raw_outputs.append(response)
            return np.array(predictions), np.array(confidences), raw_outputs

    # ------------------------------------------------------------------
    # Task 1: Diagnosis (flat MCQA, N-way adapted)
    # ------------------------------------------------------------------

    def _predict_diagnosis(
        self, image_paths: List[str], class_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        candidate_str = "\n".join(
            f"{i + 1}. {name}" for i, name in enumerate(class_names)
        )
        user_prompt = _DIAGNOSIS_USER.format(
            candidate_diseases=candidate_str
        )

        predictions: List[int] = []
        confidences: List[float] = []
        raw_outputs: List[str] = []

        for path in tqdm(image_paths, desc=f"[{self.name}] Diagnosis"):
            response = self._ask_structured(
                path, _DIAGNOSIS_SYSTEM, user_prompt, max_tokens=512
            )
            raw_outputs.append(response)

            # Simple keyword parsing (flat MCQA output, no XML)
            pred_idx, conf = self._parse_response(response, class_names)
            if pred_idx == -1:
                pred_idx, conf = 0, 0.0

            predictions.append(pred_idx)
            confidences.append(conf)

        return np.array(predictions), np.array(confidences), raw_outputs

    # ------------------------------------------------------------------
    # Task 2: Concept Annotation (SkinCon / Derm7pt)
    # ------------------------------------------------------------------

    def _predict_concepts(
        self, image_paths: List[str], class_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        is_derm7pt = "pigment_network" in class_names

        if is_derm7pt:
            system = _DERM7PT_SYSTEM
            user = _DERM7PT_USER
        else:
            concept_list = "\n".join(
                f"{i + 1}. {name}" for i, name in enumerate(class_names)
            )
            system = _SKINCON_SYSTEM
            user = _SKINCON_USER.format(concept_list=concept_list)

        predictions: List[int] = []
        confidences: List[float] = []
        raw_outputs: List[str] = []

        for path in tqdm(image_paths, desc=f"[{self.name}] Concepts"):
            response = self._ask_structured(
                path, system, user, max_tokens=1024
            )

            # Parse JSON to extract present concepts as clean text
            if is_derm7pt:
                clean = self._parse_derm7pt_json(response)
            else:
                clean = self._parse_skincon_json(response, class_names)

            raw_outputs.append(clean)
            predictions.append(0)
            confidences.append(0.0)

        return np.array(predictions), np.array(confidences), raw_outputs

    # ------------------------------------------------------------------
    # Task 3: Captioning
    # ------------------------------------------------------------------

    def _predict_captioning(
        self, image_paths: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        predictions: List[int] = []
        confidences: List[float] = []
        raw_outputs: List[str] = []

        for path in tqdm(image_paths, desc=f"[{self.name}] Captioning"):
            response = self._ask_structured(
                path,
                _CAPTIONING_SYSTEM,
                _CAPTIONING_USER,
                max_tokens=512,
            )

            # Try to extract detailed_description from JSON if model wraps it
            clean = self._extract_description(response)
            raw_outputs.append(clean)
            predictions.append(0)
            confidences.append(0.0)

        return np.array(predictions), np.array(confidences), raw_outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_prompt(class_names: List[str]) -> str:
        options = "\n".join(f"{i+1}. {name}" for i, name in enumerate(class_names))
        return (
            f"Based on the skin lesion shown in this image, please select the "
            f"most accurate diagnosis from the options below.\n\n"
            f"{options}\n\n"
            f"Answer with only the diagnosis name, nothing else."
        )

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_skincon_json(response: str, class_names: List[str]) -> str:
        """Parse SkinCon JSON -> comma-separated present concepts."""
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                features = data.get("morphological_features_skincon", [])
                if isinstance(features, list) and features:
                    # Only keep features that are valid class_names
                    names_lower = {n.lower(): n for n in class_names}
                    valid = [
                        names_lower[f.lower()]
                        for f in features
                        if f.lower() in names_lower
                    ]
                    if valid:
                        return ", ".join(valid)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        # Fallback: return raw response (keyword parser will do its best)
        return response

    @staticmethod
    def _parse_derm7pt_json(response: str) -> str:
        """Parse Derm7pt JSON -> comma-separated present (non-absent) concepts."""
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                features = data.get("morphological_features_Derm7pt", {})
                if isinstance(features, dict):
                    present = [
                        k for k, v in features.items()
                        if isinstance(v, str) and v.lower() != "absent"
                    ]
                    return ", ".join(present) if present else "none"
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return response

    @staticmethod
    def _extract_description(response: str) -> str:
        """Extract detailed_description from JSON if present."""
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                desc = data.get("detailed_description", "")
                if desc and isinstance(desc, str):
                    return desc
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return response
