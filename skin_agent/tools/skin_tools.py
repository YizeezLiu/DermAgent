"""
Dermatology tools for skin disease analysis.

Contains implementations for:
- PanDermTool: Zero-shot skin disease classification using DermLIP
- MAKETool: Concept annotation for dermoscopic features
- Qwen3VLTool: Visual Question Answering using Qwen3-VL
- DermoGPTTool: Dermatology-specialized VQA using DermoGPT-RL
- RAGTool: Multimodal retrieval from Derm1M knowledge base
- TextRAGTool: Text retrieval from dermatology literature (DermNet, Mayo)
- OntologyTool: Skin disease hierarchy knowledge tree query with fuzzy matching
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base import (
    BaseSkinTool,
    DermoGPTDiagnosisInput,
    ImageInput,
    ImageQueryInput,
    RAGQueryInput,
    TextQueryInput,
    validate_image_path,
)

# Import Qwen3EmbeddingModel for TextRAGTool
try:
    from benchmark.models.qwen3_embedding import Qwen3EmbeddingModel
    _QWEN3_EMBEDDING_AVAILABLE = True
except ImportError:
    _QWEN3_EMBEDDING_AVAILABLE = False
from ..utils.image_utils import resolve_image_path, get_display_id


# =============================================================================
# PanDerm Tool - Zero-shot Classification
# =============================================================================

class PanDermInput(BaseModel):
    """Input schema for PanDerm tool."""
    
    image_path: str = Field(
        description="Path to the skin/dermoscopic image file"
    )
    dataset: Optional[str] = Field(
        default=None,
        description=(
            "Dataset name (e.g., 'SNU_500', 'PAD', 'HAM10000'). "
            "If provided, candidate diseases are loaded from dataset config automatically."
        )
    )
    candidate_diseases: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated list of candidate disease names for classification. "
            "Takes priority over dataset parameter. If neither provided, uses default diseases."
        )
    )


class PanDermTool(BaseSkinTool):
    """
    Zero-shot skin disease classification using DermLIP/PanDerm model.
    
    Classifies dermoscopic or clinical skin images into disease categories
    without requiring task-specific fine-tuning.
    """
    
    name: str = "panderm_classifier"
    description: str = (
        "Classify skin lesions in dermoscopic or clinical images using zero-shot learning. "
        "Input an image path and specify either 'dataset' (e.g., 'SNU_500') to load options "
        "from config, or 'candidate_diseases' for custom disease list. "
        "Returns top predicted diseases with confidence scores."
    )
    args_schema: Type[BaseModel] = PanDermInput
    
    model_id: str = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"
    model: Any = None
    preprocess: Any = None
    tokenizer: Any = None
    
    # Default disease classes for classification
    default_diseases: List[str] = [
        "melanoma",
        "basal cell carcinoma",
        "squamous cell carcinoma",
        "nevus",
        "seborrheic keratosis",
        "actinic keratosis",
        "dermatofibroma",
        "vascular lesion",
        "pigmented benign keratosis",
        "atypical nevus",
    ]
    
    # Store reference to Derm1M open_clip module for runtime use
    _derm1m_open_clip: Any = None
    
    def _load_model(self) -> None:
        """Load DermLIP model from HuggingFace Hub."""
        print(f"[PanDermTool] Loading model: {self.model_id}")
        
        # Add Derm1M source to path for open_clip
        # Use isolated loading to avoid conflicts with MAKE's open_clip
        derm1m_src = Path(__file__).parent.parent.parent / "Derm1M" / "src"
        
        # Save current sys.path state
        original_path = sys.path.copy()
        
        # Temporarily modify sys.path to prioritize Derm1M's open_clip
        # Remove any paths containing 'MAKE' to avoid loading wrong version
        sys.path = [p for p in sys.path if 'MAKE' not in p]
        if str(derm1m_src) not in sys.path:
            sys.path.insert(0, str(derm1m_src))
        
        # Force reimport of open_clip from Derm1M
        import importlib
        modules_to_remove = [key for key in list(sys.modules.keys()) if key.startswith('open_clip')]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        # Import and store reference for runtime use
        import open_clip as derm1m_open_clip
        self._derm1m_open_clip = derm1m_open_clip
        
        self.model, self.preprocess = derm1m_open_clip.create_model_from_pretrained(
            f'hf-hub:{self.model_id}',
            device=self.device
        )
        self.tokenizer = derm1m_open_clip.get_tokenizer(f'hf-hub:{self.model_id}')
        self.model.eval()
        
        # Restore original sys.path
        sys.path = original_path
        
        print("[PanDermTool] Model loaded successfully!")
    
    def _build_text_features(
        self, 
        class_names: List[str],
        templates: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Build text embeddings for candidate diseases."""
        if templates is None:
            templates = [
                "a dermoscopy image of {}",
                "a clinical photo of {}",
                "a skin lesion of {}",
            ]
        
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
    
    def _run(
        self, 
        image_path: str,
        dataset: Optional[str] = None,
        candidate_diseases: Optional[str] = None
    ) -> str:
        """
        Classify the skin image.
        
        Args:
            image_path: Path to the image file (can be anonymous ID like IMG_xxx)
            dataset: Optional dataset name (e.g., 'SNU_500', 'PAD', 'HAM10000').
                     If provided, candidate diseases are loaded from config.
            candidate_diseases: Optional comma-separated disease names.
                     Takes priority over dataset parameter.
            
        Returns:
            Classification results as formatted string
            
        Priority:
            candidate_diseases > dataset > default_diseases
        """
        try:
            # Resolve anonymous ID to actual path if needed
            actual_path = resolve_image_path(image_path)
            # Validate image path
            img_path = validate_image_path(actual_path)
            # Get anonymous display ID (prevents filename leakage)
            display_id = get_display_id(image_path)
            
            # Ensure model is loaded
            self._ensure_initialized()
            
            # Parse candidate diseases with priority: candidate_diseases > dataset > default
            if candidate_diseases:
                diseases = [d.strip() for d in candidate_diseases.split(",")]
            elif dataset:
                # Load from config to avoid circular import
                from ..configs import DATASET_CONFIGS
                config = DATASET_CONFIGS.get(dataset)
                if config and "options" in config:
                    diseases = config["options"]
                else:
                    return f"Error: Unknown dataset '{dataset}' or no options defined for it."
            else:
                diseases = self.default_diseases
            
            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Build text features for diseases
            text_features = self._build_text_features(diseases)
            
            # Compute similarity
            with torch.no_grad():
                img_features = self.model.encode_image(image_tensor)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * img_features @ text_features.T).softmax(dim=-1)
                probs = similarity.cpu().numpy()[0]
            
            # Sort by probability
            sorted_indices = np.argsort(probs)[::-1]
            
            # Format results (use anonymous ID to prevent filename leakage)
            results = ["Zero-shot Classification Results:"]
            results.append(f"Image: {display_id}")
            results.append("-" * 40)
            
            for i, idx in enumerate(sorted_indices[:5]):  # Top 5
                disease = diseases[idx]
                prob = probs[idx]
                results.append(f"{i+1}. {disease}: {prob:.2%}")
            
            return "\n".join(results)
            
        except FileNotFoundError as e:
            return self._handle_error(e, "image loading")
        except Exception as e:
            return self._handle_error(e, "classification")
    
    def unload(self) -> None:
        """Unload PanDerm model and associated resources."""
        if self.preprocess is not None:
            del self.preprocess
            self.preprocess = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._derm1m_open_clip = None
        super().unload()
        print("[PanDermTool] Model unloaded.")


# =============================================================================
# MAKE Tool - Concept Annotation
# =============================================================================

class MAKEInput(BaseModel):
    """Input schema for MAKE concept annotation tool."""
    
    image_path: str = Field(
        description="Path to the skin/dermoscopic image file"
    )
    top_k: int = Field(
        default=5,
        description="Number of top concepts to return"
    )
    target_concepts: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated list of target concepts to evaluate. "
            "When provided, returns scores ONLY for these specified concepts "
            "(used for Task2 concept annotation). Uses ConceptTerms.json for synonym lookup. "
            "Example: 'pigment_network, pigmentation, vascular_structures, dots_and_globules'"
        )
    )


class MAKETool(BaseSkinTool):
    """
    Medical concept extraction and annotation from skin images.
    
    Uses MAKE (Medical-Aware Knowledge Extraction) model to identify
    dermoscopic features, lesion characteristics, and clinical concepts.
    
    Supports two modes:
    - Default mode: Returns top-k scores from default dermoscopic concepts
    - Target mode: When target_concepts is provided, evaluates ONLY those concepts
    """
    
    name: str = "make_concept_annotator"
    description: str = (
        "Extract medical concepts and dermoscopic features from skin images. "
        "Identifies characteristics like pigment network, dots, globules, "
        "asymmetry, border irregularity, and color patterns. "
        "Returns relevant concepts with presence scores. "
        "For concept annotation tasks, pass target_concepts parameter with the "
        "comma-separated list of concepts to evaluate (e.g., 'pigment_network, pigmentation')."
    )
    args_schema: Type[BaseModel] = MAKEInput
    
    model_id: str = "hf-hub:xieji-x/MAKE"
    model: Any = None
    preprocess: Any = None
    concept_embeddings: dict = None
    concept_terms_path: Path = None
    _make_open_clip: Any = None  # Store reference to MAKE's open_clip module
    
    # Default dermoscopic concepts
    default_concepts: List[str] = [
        "pigment network",
        "dots",
        "globules",
        "streaks",
        "blue-whitish veil",
        "regression structures",
        "blotches",
        "atypical vessels",
        "asymmetry",
        "border irregularity",
        "color variegation",
        "diameter large",
        "ulceration",
        "scaling",
        "erythema",
    ]
    
    def _load_model(self) -> None:
        """Load MAKE model and prepare concept embeddings."""
        print(f"[MAKETool] Loading model: {self.model_id}")
        
        # Add MAKE source to path
        # Use isolated loading to avoid conflicts with Derm1M's open_clip
        make_src = Path(__file__).parent.parent.parent / "MAKE" / "src"
        
        # Save current sys.path state
        original_path = sys.path.copy()
        
        # Temporarily modify sys.path to prioritize MAKE's open_clip
        # Remove any paths containing 'Derm1M' to avoid loading wrong version
        sys.path = [p for p in sys.path if 'Derm1M' not in p]
        if str(make_src) not in sys.path:
            sys.path.insert(0, str(make_src))
        
        # Force reimport of open_clip from MAKE
        import importlib
        modules_to_remove = [key for key in list(sys.modules.keys()) if key.startswith('open_clip')]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        import open_clip as make_open_clip
        self._make_open_clip = make_open_clip  # Store reference for later use
        
        self.model, _, self.preprocess = make_open_clip.create_model_and_transforms(
            self.model_id
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Restore original sys.path
        sys.path = original_path
        
        # Path to concept terms JSON
        self.concept_terms_path = (
            Path(__file__).parent.parent.parent / 
            "MAKE" / "concept_annotation" / "term_lists" / "ConceptTerms.json"
        )
        
        # Pre-compute concept embeddings
        print("[MAKETool] Building concept embeddings...")
        self._build_concept_embeddings()
        print("[MAKETool] Model loaded successfully!")
    
    def _build_concept_embeddings(self) -> None:
        """Pre-compute embeddings for all concepts."""
        # Use the stored reference to MAKE's open_clip
        tokenizer = self._make_open_clip.get_tokenizer(self.model_id)
        
        # Load concept terms if available
        concept_terms_dict = {}
        if self.concept_terms_path.exists():
            with open(self.concept_terms_path, 'r') as f:
                concept_terms_dict = json.load(f)
        
        prompt_templates = [
            "This is skin image of {}",
            "This is dermatology image of {}",
            "A skin image of {}",
            "An image of {}, a skin condition.",
        ]
        
        prompt_refs = [
            "This is skin image",
            "This is dermatology image",
            "A skin image",
            "An image of a skin condition.",
        ]
        
        self.concept_embeddings = {}
        
        for concept in self.default_concepts:
            # Get synonyms if available
            terms = concept_terms_dict.get(concept, [concept])
            if not terms:
                terms = [concept]
            
            # Compute target embeddings
            target_embeds = []
            for template in prompt_templates:
                prompts = [template.format(term) for term in terms]
                tokens = tokenizer(prompts).to(self.device)
                with torch.no_grad():
                    embeds = self.model.encode_text(tokens)
                    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                    target_embeds.append(embeds.mean(dim=0))
            
            target_embed = torch.stack(target_embeds).mean(dim=0)
            
            # Compute reference embeddings
            ref_tokens = tokenizer(prompt_refs).to(self.device)
            with torch.no_grad():
                ref_embeds = self.model.encode_text(ref_tokens)
                ref_embeds = ref_embeds / ref_embeds.norm(dim=-1, keepdim=True)
                ref_embed = ref_embeds.mean(dim=0)
            
            self.concept_embeddings[concept] = {
                'target': target_embed.cpu(),
                'reference': ref_embed.cpu(),
            }
    
    @staticmethod
    def _normalize_concept_key(name: str) -> str:
        """Normalize concept name to match ConceptTerms.json keys.
        
        Converts underscores to spaces and lowercases.
        Example: 'pigment_network' -> 'pigment network'
        """
        return " ".join(name.replace("_", " ").lower().split())
    
    def _build_concept_embeddings_for(self, concepts: List[str]) -> dict:
        """Build embeddings for specified concepts using ConceptTerms.json.
        
        This method dynamically builds embeddings for any list of concepts,
        using ConceptTerms.json for synonym lookup.
        
        Args:
            concepts: List of concept names (can use underscores, e.g., 'pigment_network')
            
        Returns:
            Dictionary mapping concept names to their embeddings
        """
        tokenizer = self._make_open_clip.get_tokenizer(self.model_id)
        
        # Load concept terms
        concept_terms_dict = {}
        if self.concept_terms_path.exists():
            with open(self.concept_terms_path, 'r') as f:
                concept_terms_dict = json.load(f)
        
        prompt_templates = [
            "This is skin image of {}",
            "This is dermatology image of {}",
            "A skin image of {}",
            "An image of {}, a skin condition.",
        ]
        
        prompt_refs = [
            "This is skin image",
            "This is dermatology image",
            "A skin image",
            "An image of a skin condition.",
        ]
        
        embeddings = {}
        
        for concept in concepts:
            # Normalize concept key to match JSON format
            normalized_key = self._normalize_concept_key(concept)
            
            # Get synonyms from ConceptTerms.json
            terms = concept_terms_dict.get(normalized_key)
            if not terms:
                # Fallback: try the original concept name
                terms = concept_terms_dict.get(concept, [concept])
            if not terms:
                terms = [concept]
            
            # Compute target embeddings
            target_embeds = []
            for template in prompt_templates:
                prompts = [template.format(term) for term in terms]
                tokens = tokenizer(prompts).to(self.device)
                with torch.no_grad():
                    embeds = self.model.encode_text(tokens)
                    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                    target_embeds.append(embeds.mean(dim=0))
            
            target_embed = torch.stack(target_embeds).mean(dim=0)
            
            # Compute reference embeddings
            ref_tokens = tokenizer(prompt_refs).to(self.device)
            with torch.no_grad():
                ref_embeds = self.model.encode_text(ref_tokens)
                ref_embeds = ref_embeds / ref_embeds.norm(dim=-1, keepdim=True)
                ref_embed = ref_embeds.mean(dim=0)
            
            # Use original concept name as key (preserve caller's naming)
            embeddings[concept] = {
                'target': target_embed.cpu(),
                'reference': ref_embed.cpu(),
            }
        
        return embeddings
    
    def _compute_concept_scores(self, image_features: torch.Tensor, embeddings: dict = None) -> dict:
        """Compute concept presence scores for an image.
        
        Args:
            image_features: Image feature tensor from MAKE encoder
            embeddings: Optional custom embeddings dict. If None, uses self.concept_embeddings
        """
        import scipy.special
        
        temp = 1 / np.exp(4.5944)  # Temperature from MAKE paper
        scores = {}
        
        # Use custom embeddings if provided, otherwise use default
        concept_embeds = embeddings if embeddings is not None else self.concept_embeddings
        
        image_features = image_features.cpu()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Flatten to 1D if needed
        if image_features.dim() > 1:
            image_features = image_features.squeeze()
        
        for concept, embeds in concept_embeds.items():
            target_embed = embeds['target']
            ref_embed = embeds['reference']
            
            # Ensure 1D vectors for dot product
            if target_embed.dim() > 1:
                target_embed = target_embed.squeeze()
            if ref_embed.dim() > 1:
                ref_embed = ref_embed.squeeze()
            
            target_sim = torch.dot(target_embed, image_features).item()
            ref_sim = torch.dot(ref_embed, image_features).item()
            
            # Softmax to get presence probability
            score = scipy.special.softmax([target_sim / temp, ref_sim / temp])[0]
            scores[concept] = score
        
        return scores
    
    def _run(self, image_path: str, top_k: int = 5, target_concepts: Optional[str] = None) -> str:
        """
        Extract medical concepts from the skin image.
        
        Args:
            image_path: Path to the image file (can be anonymous ID like IMG_xxx)
            top_k: Number of top concepts to return (used when target_concepts is None)
            target_concepts: Optional comma-separated list of target concepts.
                When provided, evaluates ONLY these concepts and returns all their scores.
                Uses ConceptTerms.json for synonym lookup.
            
        Returns:
            Concept annotation results as formatted string
        """
        try:
            # Resolve anonymous ID to actual path if needed
            actual_path = resolve_image_path(image_path)
            # Validate image path
            img_path = validate_image_path(actual_path)
            # Get anonymous display ID (prevents filename leakage)
            display_id = get_display_id(image_path)
            
            # Ensure model is loaded
            self._ensure_initialized()
            
            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Get image features
            with torch.no_grad():
                output = self.model.encode_image(image_tensor)
                if isinstance(output, tuple):
                    image_features = output[0]
                else:
                    image_features = output
            
            # Determine which embeddings to use
            if target_concepts:
                # Task2 mode: Build embeddings for specified target concepts
                concepts_list = [c.strip() for c in target_concepts.split(",") if c.strip()]
                custom_embeddings = self._build_concept_embeddings_for(concepts_list)
                scores = self._compute_concept_scores(image_features[0], embeddings=custom_embeddings)
                
                # For target mode, return ALL concepts with scores (no top_k filtering)
                sorted_concepts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                # Format results for target concepts (use anonymous ID to prevent filename leakage)
                results = ["Concept Annotation Results (Target Mode):"]
                results.append(f"Image: {display_id}")
                results.append(f"Evaluated concepts: {len(concepts_list)}")
                results.append("-" * 40)
                
                for i, (concept, score) in enumerate(sorted_concepts):
                    results.append(f"{i+1}. {concept}: {score:.4f}")
                
                # Add detected concepts (score > 0.5)
                detected = [c for c, s in sorted_concepts if s > 0.5]
                results.append("")
                if detected:
                    results.append(f"Detected concepts (score > 0.5): {', '.join(detected)}")
                else:
                    results.append("No concepts detected with score > 0.5")
            else:
                # Default mode: Use pre-computed default concept embeddings
                scores = self._compute_concept_scores(image_features[0])
                
                # Sort by score
                sorted_concepts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                # Format results (use anonymous ID to prevent filename leakage)
                results = ["Concept Annotation Results:"]
                results.append(f"Image: {display_id}")
                results.append("-" * 40)
                
                for i, (concept, score) in enumerate(sorted_concepts[:top_k]):
                    results.append(f"{i+1}. {concept}: {score:.4f}")
                
                # Add interpretation
                top_concepts = [c for c, _ in sorted_concepts[:3]]
                results.append("")
                results.append(f"Key features detected: {', '.join(top_concepts)}")
            
            return "\n".join(results)
            
        except FileNotFoundError as e:
            return self._handle_error(e, "image loading")
        except Exception as e:
            return self._handle_error(e, "concept annotation")
    
    def unload(self) -> None:
        """Unload MAKE model and associated resources."""
        if self.preprocess is not None:
            del self.preprocess
            self.preprocess = None
        if self.concept_embeddings is not None:
            del self.concept_embeddings
            self.concept_embeddings = None
        self._make_open_clip = None
        super().unload()
        print("[MAKETool] Model unloaded.")


# =============================================================================
# Qwen3VL Tool - Visual Question Answering
# =============================================================================

class Qwen3VLTool(BaseSkinTool):
    """
    Visual Question Answering for dermatological images.
    
    Uses Qwen3-VL model to answer natural language questions
    about skin lesions and dermatological conditions.
    
    Acceleration options:
    - Flash Attention 2: Set use_flash_attn=True
    - Uses bfloat16 for efficient inference
    """
    
    name: str = "qwen_vqa"
    description: str = (
        "Answer questions about skin images using visual understanding. "
        "Provide an image path and a question in natural language. "
        "Can describe lesion appearance, suggest diagnoses, or explain features."
    )
    args_schema: Type[BaseModel] = ImageQueryInput
    
    # Default to locally downloaded model
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    model: Any = None
    processor: Any = None
    _load_failed: bool = False
    
    # Acceleration options
    use_flash_attn: bool = False   # Flash Attention 2 (requires flash-attn)
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    def __init__(
        self, 
        device: str = "cuda",
        model_id: str = None,
        use_flash_attn: bool = False,
        **kwargs
    ):
        super().__init__(device=device, **kwargs)
        if model_id:
            self.model_id = model_id
        self.use_flash_attn = use_flash_attn
    
    def _load_model(self) -> None:
        """Load Qwen VL model (cf. benchmark/models/qwen3vl.py)."""
        try:
            # Compatible with different transformers versions
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
            
            from transformers import AutoProcessor
            
            print(f"[Qwen3VLTool] Loading model: {self.model_id}")
            print(f"[Qwen3VLTool] Flash Attention: {self.use_flash_attn}")
            
            # Load model - use bfloat16, consistent with benchmark version
            if self.use_flash_attn:
                print("[Qwen3VLTool] Using Flash Attention 2")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            print("[Qwen3VLTool] Model loaded successfully!")
            
        except Exception as e:
            print(f"[Qwen3VLTool] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            print("[Qwen3VLTool] VQA tool will return placeholder responses.")
            self._load_failed = True
    
    def _run(self, image_path: str, query: str, target_concepts: Optional[str] = None) -> str:
        """
        Answer a question about the skin image.
        
        Args:
            image_path: Path to the image file (can be anonymous ID like IMG_xxx)
            query: Natural language question about the image
            target_concepts: Optional comma-separated list of concepts for concept annotation.
                When provided, enhances the query to specifically evaluate these concepts.
            
        Returns:
            Generated answer as string
        """
        try:
            # Resolve anonymous ID to actual path if needed
            actual_path = resolve_image_path(image_path)
            # Validate image path
            img_path = validate_image_path(actual_path)
            
            # Ensure model is loaded
            self._ensure_initialized()
            
            # If model failed to load, return placeholder
            if self._load_failed:
                return (
                    "VQA Response:\n"
                    "[Model not available] Unable to perform visual analysis. "
                    "Please rely on other tools (panderm_classifier, make_concept_annotator) "
                    "for image analysis."
                )
            
            # Enhance query with target concepts if provided (Task2 concept annotation)
            if target_concepts:
                enhanced_query = (
                    f"{query}\n\n"
                    f"Please specifically evaluate the presence of these concepts in the image: "
                    f"{target_concepts}\n"
                    f"For each concept, indicate whether it is present or absent based on visual evidence."
                )
            else:
                enhanced_query = query
            
            # Build message (consistent with benchmark/models/qwen3vl.py)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": enhanced_query},
                ],
            }]
            
            # Process input using apply_chat_template
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            
            # Trim input tokens and decode
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            response = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            return f"VQA Response:\n{response}"
            
        except FileNotFoundError as e:
            return self._handle_error(e, "image loading")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._handle_error(
                RuntimeError("GPU out of memory"),
                "Try using a smaller model or reducing batch size"
            )
        except Exception as e:
            return self._handle_error(e, "visual question answering")
    
    def unload(self) -> None:
        """Unload Qwen3VL model and associated resources."""
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._load_failed = False
        super().unload()
        print("[Qwen3VLTool] Model unloaded.")


# =============================================================================
# DermoGPT Tool - Dermatology-specialized VQA
# =============================================================================

# -----------------------------------------------------------------------------
# DermoGPT Diagnosis Prompt Templates
# -----------------------------------------------------------------------------

DERMOGPT_DIAGNOSIS_SYSTEM_PROMPT = """You are DermoGPT, a versatile expert AI dermatologist and pathologist. You are capable of analyzing skin lesions across different imaging modalities, including standard clinical photography (macroscopic) and dermoscopy.

### Core Capabilities & Protocol
1. **Modality Recognition:** You must first strictly identify whether the input image is a **Clinical Image** (naked eye view) or a **Dermoscopy Image** (magnified, polarized/fluid interface).
2. **Adaptive Morphological Analysis:**
   - **For Clinical Images:** Focus on the ABCD rule (Asymmetry, Border, Color, Diameter), evolution, elevation, surface texture (crust, scale, ulceration), and surrounding skin context.
   - **For Dermoscopy:** Focus on specific micro-structures (e.g., pigment networks, globules, streaks, blue-white veil, vascular patterns, leaf-like areas).
3. **Diagnostic Logic:** Synthesize the visual features to select the single most probable diagnosis from the provided options.

### Operational Constraints
- **Strict XML Output:** Your response must be contained ENTIRELY within <reasoning> and <final_diagnosis> tags.
- **Closed-Set Selection:** You must choose the diagnosis STRICTLY from the provided `candidate_diseases` list. Do not output synonyms or abbreviations not present in the list.
- **Objective & Precise:** Use professional medical terminology appropriate for the identified image modality.
"""

DERMOGPT_DIAGNOSIS_USER_TEMPLATE = """Analyze the provided skin lesion image and determine the most likely diagnosis.
You are strictly limited to the following diagnostic options: {candidate_diseases}

### Analysis Instructions
Generate a structured response following these steps inside the <reasoning> block:

1. **Image Modality Identification:**
   - Explicitly state if the image is **Clinical** or **Dermoscopic**.

2. **Morphological Decoding (Adaptive):**
   - **If Clinical:** Describe the lesion's gross pathology. Analyze Asymmetry, Border irregularity, Color variegation (red, white, brown, black), and Diameter/Dimension. Note any elevation, ulceration, or scaling.
   - **If Dermoscopic:** Analyze the specific dermoscopic patterns. Look for pigment networks (typical/atypical), structural patterns (globules, streaks, homogeneous areas), and vascular features.

3. **Diagnostic Synthesis:**
   - Correlate the observed features with the candidate diseases.
   - Explain why the visual evidence supports your chosen diagnosis over the others in the list.

### Required Output Format
<reasoning>
[Step 1: Modality Identification]
[Step 2: Detailed Morphological Description]
[Step 3: Diagnostic Deduction]
</reasoning>
<final_diagnosis>
[The Exact Diagnostic Option from the list]
</final_diagnosis>
"""


class DermoGPTTool(BaseSkinTool):
    """
    Visual Question Answering for dermatological images using DermoGPT-RL.
    
    DermoGPT-RL is a Qwen3-VL-8B model fine-tuned specifically for dermatology,
    excelling at morphological descriptions and dermatology-specific VQA tasks.
    
    Supports two modes:
    - Diagnosis mode: When candidate_diseases is provided, uses structured prompt
      template with XML output (<reasoning> and <final_diagnosis> tags)
    - VQA mode: When only query is provided, uses general VQA behavior
    
    Acceleration options:
    - Flash Attention 2: Set use_flash_attn=True
    - Uses bfloat16 for efficient inference
    """
    
    name: str = "dermogpt_vqa"
    description: str = (
        "Dermatology-specialized vision-language model for diagnosis and VQA. "
        "For DIAGNOSIS: Pass 'candidate_diseases' (comma-separated list) to get structured "
        "XML output with <reasoning> and <final_diagnosis> tags. "
        "For general VQA: Pass 'query' for free-form questions about the image. "
        "This model excels at morphological descriptions and clinical assessment."
    )
    args_schema: Type[BaseModel] = DermoGPTDiagnosisInput
    
    # DermoGPT-RL model (fine-tuned from Qwen3-VL-8B)
    # Use local path to avoid HuggingFace authentication issues
    model_id: str = str(Path(__file__).parent.parent.parent / "model-weights" / "DermoGPT-RL")
    model: Any = None
    processor: Any = None
    _load_failed: bool = False
    
    # Acceleration options
    use_flash_attn: bool = False
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    def __init__(
        self, 
        device: str = "cuda",
        model_id: str = None,
        use_flash_attn: bool = False,
        **kwargs
    ):
        super().__init__(device=device, **kwargs)
        if model_id:
            self.model_id = model_id
        self.use_flash_attn = use_flash_attn
    
    def _load_model(self) -> None:
        """Load DermoGPT-RL model (based on Qwen3-VL architecture)."""
        try:
            # Compatible with different transformers versions
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
            
            from transformers import AutoProcessor
            
            print(f"[DermoGPTTool] Loading model: {self.model_id}")
            print(f"[DermoGPTTool] Flash Attention: {self.use_flash_attn}")
            
            # Load model with bfloat16
            # Use "auto" instead of "cuda" to respect CUDA_VISIBLE_DEVICES
            if self.use_flash_attn:
                print("[DermoGPTTool] Using Flash Attention 2")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            print("[DermoGPTTool] Model loaded successfully!")
            
        except Exception as e:
            print(f"[DermoGPTTool] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            print("[DermoGPTTool] VQA tool will return placeholder responses.")
            self._load_failed = True
    
    def _run(
        self,
        image_path: str,
        candidate_diseases: str = None,
        query: str = None,
        target_concepts: str = None,
    ) -> str:
        """
        Analyze a dermatology image using DermoGPT.
        
        Supports three modes based on input parameters:
        - Diagnosis mode: When candidate_diseases is provided, uses structured
          prompt template with XML output
        - Concept mode: When target_concepts is provided, evaluates specific concepts
        - VQA mode: When only query is provided, uses general VQA behavior
        
        Args:
            image_path: Path to the image file (can be anonymous ID like IMG_xxx)
            candidate_diseases: Comma-separated list of candidate diseases for diagnosis.
                When provided, uses structured diagnosis prompt template.
            query: Natural language question for general VQA mode.
                Used when candidate_diseases is not provided.
            target_concepts: Comma-separated list of concepts for concept annotation.
                When provided, evaluates presence of these specific concepts.
            
        Returns:
            Generated answer as string (XML format for diagnosis, free-form for VQA)
        """
        try:
            # Resolve anonymous ID to actual path if needed
            actual_path = resolve_image_path(image_path)
            # Validate image path
            img_path = validate_image_path(actual_path)
            
            # Ensure model is loaded
            self._ensure_initialized()
            
            # If model failed to load, return placeholder
            if self._load_failed:
                return (
                    "DermoGPT Response:\n"
                    "[Model not available] Unable to perform visual analysis. "
                    "Please rely on other tools (panderm_classifier, make_concept_annotator) "
                    "for image analysis."
                )
            
            # Route based on input parameters
            if candidate_diseases:
                # Diagnosis mode: Use structured prompt template
                # Combine system prompt and user prompt for Qwen3-VL compatibility
                system_prompt = DERMOGPT_DIAGNOSIS_SYSTEM_PROMPT
                user_prompt = DERMOGPT_DIAGNOSIS_USER_TEMPLATE.format(
                    candidate_diseases=candidate_diseases
                )
                # Prepend system instructions to user message
                full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path)},
                        {"type": "text", "text": full_prompt},
                    ],
                }]
                response_prefix = "DermoGPT Diagnosis Response:\n"
            elif target_concepts:
                # Concept mode: Evaluate specific dermoscopic concepts
                concept_prompt = (
                    "You are an expert dermatologist analyzing a skin lesion image.\n\n"
                    f"Please evaluate the presence of each of the following dermoscopic concepts in this image:\n"
                    f"{target_concepts}\n\n"
                    "For each concept, indicate:\n"
                    "1. Whether it is PRESENT or ABSENT\n"
                    "2. Brief visual evidence supporting your assessment\n\n"
                    "Format your response as a list of concepts with their status."
                )
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path)},
                        {"type": "text", "text": concept_prompt},
                    ],
                }]
                response_prefix = "DermoGPT Concept Analysis:\n"
            else:
                # VQA mode: Use general question-answering
                user_text = query or "Describe this skin lesion in detail."
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path)},
                        {"type": "text", "text": user_text},
                    ],
                }]
                response_prefix = "DermoGPT VQA Response:\n"
            
            # Process input using apply_chat_template
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response (use more tokens for diagnosis mode due to XML)
            max_tokens = self.max_new_tokens if not candidate_diseases else max(self.max_new_tokens, 1024)
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            
            # Trim input tokens and decode
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            response = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            return f"{response_prefix}{response}"
            
        except FileNotFoundError as e:
            return self._handle_error(e, "image loading")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._handle_error(
                RuntimeError("GPU out of memory"),
                "Try using a smaller model or reducing batch size"
            )
        except Exception as e:
            return self._handle_error(e, "visual question answering")
    
    def unload(self) -> None:
        """Unload DermoGPT model and associated resources."""
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._load_failed = False
        super().unload()
        print("[DermoGPTTool] Model unloaded.")


# =============================================================================
# RAG Tool - Retrieval-Augmented Generation
# =============================================================================

class RAGTool(BaseSkinTool):
    """
    Multimodal Retrieval-Augmented Generation from dermatology knowledge base.
    
    Supports two retrieval modes:
    1. Image-based: Find visually similar cases using image embeddings
    2. Text-based: Find cases matching morphological feature descriptions
    
    Uses DermLIP (CLIP-like) embeddings for cross-modal retrieval.
    """
    
    name: str = "rag_retrieval"
    description: str = (
        "Search the dermatology knowledge base for similar cases. "
        "Supports TWO modes:\n"
        "1. IMAGE MODE: Provide 'image_path' to find visually similar cases\n"
        "2. TEXT MODE: Provide 'text_query' with morphological features "
        "(e.g., 'irregular borders, dark pigmentation, asymmetric lesion')\n"
        "You can use both modes together for comprehensive retrieval. "
        "Returns similar cases with disease labels and clinical descriptions."
    )
    args_schema: Type[BaseModel] = RAGQueryInput
    
    qdrant_path: str = "./qdrant_storage"
    collection_name: str = "derm1m"
    encoder_model_id: str = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"
    
    client: Any = None
    encoder: Any = None
    tokenizer: Any = None
    preprocess: Any = None  # Image preprocessing function
    _load_failed: bool = False
    _derm1m_open_clip: Any = None  # Store reference to Derm1M open_clip module
    
    # VQA Enhancement: use DermoGPT to describe retrieved images
    vqa_enhancement: bool = False  # Whether to enable VQA enhancement
    vqa_tool: Any = None  # DermoGPTTool instance reference
    image_root: str = "datasets/Derm1M"  # Root directory for Derm1M images
    
    # Leakage Prevention: exclude current query image from retrieval results
    # Enable this for Derm1M VQA benchmarks to prevent data leakage
    exclude_current_image: bool = False
    
    def _load_model(self) -> None:
        """Load Qdrant client and DermLIP encoder (supports both image and text)."""
        print("[RAGTool] Initializing Qdrant client and encoder...")
        
        try:
            from qdrant_client import QdrantClient
            
            # Try to connect to Qdrant server first (localhost:6333)
            use_local_storage = False
            try:
                print("[RAGTool] Trying to connect to Qdrant server at localhost:6333...")
                server_client = QdrantClient(host="localhost", port=6333, timeout=120)
                # Test connection and check if collection exists
                collections = server_client.get_collections().collections
                collection_names = [c.name for c in collections]
                if self.collection_name in collection_names:
                    self.client = server_client
                    print(f"[RAGTool] Successfully connected to Qdrant server with collection '{self.collection_name}'")
                else:
                    print(f"[RAGTool] Server connected but collection '{self.collection_name}' not found")
                    print(f"[RAGTool] Available collections: {collection_names}")
                    use_local_storage = True
            except Exception as e:
                print(f"[RAGTool] Failed to connect to Qdrant server: {e}")
                use_local_storage = True
            
            # Fallback to file-based Qdrant client
            if use_local_storage:
                print("[RAGTool] Falling back to local file storage...")
                qdrant_path = Path(__file__).parent.parent.parent / self.qdrant_path
                if not qdrant_path.exists():
                    raise FileNotFoundError(f"Qdrant storage not found at {qdrant_path}")
                
                print(f"[RAGTool] Connecting to Qdrant at {qdrant_path}")
                self.client = QdrantClient(path=str(qdrant_path))
            
            # Load encoder using the same approach as DermLIPModel in benchmark/models/dermlip.py
            # This ensures we get the correct Derm1M open_clip with timm/BEiT architecture support
            print(f"[RAGTool] Loading encoder: {self.encoder_model_id}")
            
            # Add Derm1M source to path (must be first to take precedence)
            derm1m_src = Path(__file__).parent.parent.parent / "Derm1M" / "src"
            
            # Save current sys.path state
            original_path = sys.path.copy()
            
            # Temporarily modify sys.path to prioritize Derm1M's open_clip
            # Remove any existing open_clip paths first
            sys.path = [p for p in sys.path if 'MAKE' not in p]
            if str(derm1m_src) not in sys.path:
                sys.path.insert(0, str(derm1m_src))
            
            # Force reimport of open_clip from Derm1M
            import importlib
            modules_to_remove = [key for key in list(sys.modules.keys()) if key.startswith('open_clip')]
            for mod in modules_to_remove:
                del sys.modules[mod]
            
            # Now import - should get Derm1M's version
            import open_clip as derm1m_open_clip
            self._derm1m_open_clip = derm1m_open_clip  # Store reference for runtime use
            
            self.encoder, self.preprocess = derm1m_open_clip.create_model_from_pretrained(
                f'hf-hub:{self.encoder_model_id}',
                device=self.device
            )
            self.tokenizer = derm1m_open_clip.get_tokenizer(f'hf-hub:{self.encoder_model_id}')
            self.encoder.eval()
            
            # Restore original sys.path
            sys.path = original_path
            
            print("[RAGTool] Initialized successfully (supports image & text retrieval)!")
            
        except Exception as e:
            print(f"[RAGTool] Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            self._load_failed = True
    
    def _encode_text(self, query: str) -> np.ndarray:
        """Encode text query to embedding vector (512-dim)."""
        tokens = self.tokenizer([query])
        if isinstance(tokens, dict):
            text_tokens = tokens['input_ids'].to(self.device)
        else:
            text_tokens = tokens.to(self.device)
        
        with torch.no_grad():
            text_features = self.encoder.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0]
    
    def _encode_image(self, image_path: str) -> np.ndarray:
        """Encode image to embedding vector (512-dim)."""
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.encoder.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def _search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        vector_name: str = "image_embedding",
        exclude_filename: Optional[str] = None,
    ) -> list:
        """Search Qdrant with the given query vector.
        
        Args:
            query_vector: The embedding vector for the query
            top_k: Number of results to return
            vector_name: Name of the vector field to search
            exclude_filename: Optional filename to exclude from results (for leakage prevention)
            
        Returns:
            List of search result points
        """
        # Build filter to exclude current image (leakage prevention)
        query_filter = None
        if exclude_filename:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(
                must_not=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=exclude_filename)
                    )
                ]
            )
            print(f"[RAGTool] Leakage prevention: excluding filename '{exclude_filename}'")
        
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            using=vector_name,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        ).points
    
    def _describe_image(self, image_path: str) -> str:
        """
        Call DermoGPT to describe the retrieved image.
        
        Used when vqa_enhancement is enabled to replace low-quality
        database captions with VLM-generated descriptions.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            VLM-generated description or empty string on error
        """
        if self.vqa_tool is None:
            return ""
        if not Path(image_path).exists():
            return ""
        
        try:
            result = self.vqa_tool._run(
                image_path=image_path,
                query="Describe the morphological features of this skin lesion, including color, shape, border, and surface characteristics."
            )
            # Extract content after "VQA Response:" prefix
            if "VQA Response:" in result:
                return result.split("VQA Response:")[-1].strip()
            return result.strip()
        except Exception as e:
            return f"[VQA Error: {e}]"
    
    def _format_results(self, results: list, query_type: str, query_info: str) -> str:
        """Format search results for display."""
        if not results:
            return f"No results found for {query_type}: '{query_info}'"
        
        output = [f"RAG Retrieval Results ({query_type}): '{query_info}'"]
        output.append("-" * 60)
        
        for i, hit in enumerate(results):
            output.append(f"\nResult {i+1}:")
            output.append(f"  Similarity Score: {hit.score:.4f}")
            
            payload = hit.payload or {}
            output.append(f"  Disease Label: {payload.get('disease_label', 'N/A')}")
            
            # Add hierarchical label if available
            hier_label = payload.get('hierarchical_disease_label', '')
            if hier_label and hier_label != payload.get('disease_label', ''):
                output.append(f"  Category: {hier_label}")
            
            # VQA Enhancement: use DermoGPT to describe retrieved images
            # instead of using low-quality database captions
            vqa_description_added = False
            if self.vqa_enhancement and self.vqa_tool:
                filename = payload.get('filename', '')
                if filename:
                    # Construct full path to the image
                    full_path = Path(self.image_root) / filename
                    if full_path.exists():
                        print(f"    [RAG VQA] Describing retrieved image {i+1}...")
                        vqa_desc = self._describe_image(str(full_path))
                        if vqa_desc and not vqa_desc.startswith("[VQA Error"):
                            output.append(f"  VLM Description: {vqa_desc}")
                            vqa_description_added = True
            
            # Fallback to original caption if VQA not used or failed
            if not vqa_description_added:
                caption = payload.get('truncated_caption', payload.get('original_caption', ''))
                if len(caption) > 200:
                    caption = caption[:200] + "..."
                if caption:
                    output.append(f"  Description: {caption}")
            
            # Add skin concept if available
            skin_concept = payload.get('skin_concept', '')
            if skin_concept:
                output.append(f"  Skin Concept: {skin_concept}")
        
        return "\n".join(output)
    
    def _run(
        self, 
        image_path: Optional[str] = None, 
        text_query: Optional[str] = None, 
        top_k: int = 5
    ) -> str:
        """
        Search the knowledge base using image and/or text query.
        
        Args:
            image_path: Path to image for visual similarity search
            text_query: Text description of morphological features
            top_k: Number of results to return per query mode
            
        Returns:
            Retrieved cases as formatted string
        """
        try:
            # Validate input
            if not image_path and not text_query:
                return (
                    "RAG Retrieval Error:\n"
                    "Please provide either 'image_path' for image-based retrieval "
                    "or 'text_query' for text-based retrieval (or both)."
                )
            
            # Ensure resources are loaded
            self._ensure_initialized()
            
            # If initialization failed, return error
            if self._load_failed:
                return (
                    "RAG Retrieval Error:\n"
                    "[Knowledge base not available] Unable to search for similar cases. "
                    "Please ensure Qdrant storage is properly configured."
                )
            
            # Check collection exists
            try:
                collection_info = self.client.get_collection(self.collection_name)
                if collection_info.points_count == 0:
                    return "Error: Knowledge base is empty. Please build the index first."
            except Exception as e:
                return f"Error: Collection '{self.collection_name}' not found: {str(e)}"
            
            results_parts = []
            
            # Leakage prevention: extract relative filename to exclude from results
            exclude_filename = None
            if self.exclude_current_image and image_path:
                try:
                    actual_path_for_exclude = resolve_image_path(image_path)
                    # Extract relative path: /full/path/datasets/Derm1M/IIYI/xxx.png -> IIYI/xxx.png
                    # This matches the 'filename' field stored in Qdrant
                    image_root_resolved = str(Path(self.image_root).resolve())
                    actual_path_str = str(Path(actual_path_for_exclude).resolve())
                    
                    if image_root_resolved in actual_path_str:
                        # Extract relative path after image_root
                        rel_start = actual_path_str.find(image_root_resolved) + len(image_root_resolved)
                        exclude_filename = actual_path_str[rel_start:].lstrip('/')
                    else:
                        # Fallback: try splitting by the image_root pattern
                        parts = actual_path_str.split(self.image_root.rstrip('/') + '/')
                        if len(parts) > 1:
                            exclude_filename = parts[-1]
                        else:
                            # Last resort: use just the filename (may not match exactly)
                            exclude_filename = Path(actual_path_for_exclude).name
                            print(f"[RAGTool] Warning: Could not extract relative path, using filename: {exclude_filename}")
                except Exception as e:
                    print(f"[RAGTool] Warning: Failed to extract filename for leakage prevention: {e}")
            
            # Image-based retrieval
            if image_path:
                try:
                    # Resolve anonymous ID to actual path if needed
                    import time
                    t0 = time.time()
                    actual_path = resolve_image_path(image_path)
                    validated_path = validate_image_path(actual_path)
                    # Get anonymous display ID (prevents filename leakage)
                    display_id = get_display_id(image_path)
                    print(f"[RAG DEBUG] Time taken to resolve image path: {time.time() - t0:.2f} seconds")
                    
                    t0 = time.time()
                    image_vector = self._encode_image(str(validated_path))
                    print(f"[RAG DEBUG] Time taken to encode image: {time.time() - t0:.2f} seconds")

                    t0 = time.time()
                    image_results = self._search(image_vector, top_k, "image_embedding", exclude_filename)
                    print(f"[RAG DEBUG] Time taken to search: {time.time() - t0:.2f} seconds")

                    results_parts.append(self._format_results(
                        image_results, 
                        "Image Similarity", 
                        display_id  # Use anonymous ID instead of filename
                    ))
                except FileNotFoundError as e:
                    results_parts.append(f"Image retrieval error: {e}")
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    results_parts.append(
                        "Image retrieval failed: GPU out of memory. "
                        "Try closing other GPU processes or reducing batch size."
                    )
                except Exception as e:
                    error_msg = str(e).lower()
                    if "timed out" in error_msg or "timeout" in error_msg:
                        results_parts.append(
                            f"Image encoding error: {e}. "
                            "Likely cause: GPU resource contention from parallel processes. "
                            "Consider running one benchmark at a time or using separate GPUs."
                        )
                    else:
                        results_parts.append(f"Image encoding error: {e}")
            
            # Text-based retrieval
            if text_query:
                try:
                    text_vector = self._encode_text(text_query)
                    # Use image_embedding for cross-modal search (CLIP-style)
                    # Also apply exclude_filename filter for leakage prevention
                    text_results = self._search(text_vector, top_k, "image_embedding", exclude_filename)
                    results_parts.append(self._format_results(
                        text_results,
                        "Text Query",
                        text_query[:80] + "..." if len(text_query) > 80 else text_query
                    ))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    results_parts.append("Text retrieval failed: GPU out of memory.")
                except Exception as e:
                    error_msg = str(e).lower()
                    if "timed out" in error_msg or "timeout" in error_msg:
                        results_parts.append(
                            f"Text encoding error: {e}. GPU resource contention likely."
                        )
                    else:
                        results_parts.append(f"Text encoding error: {e}")
            
            return "\n\n" + "=" * 60 + "\n\n".join(results_parts)
            
        except Exception as e:
            return self._handle_error(e, "retrieval")
    
    def unload(self) -> None:
        """Unload RAG encoder and associated resources."""
        if self.encoder is not None:
            del self.encoder
            self.encoder = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.preprocess is not None:
            del self.preprocess
            self.preprocess = None
        # Note: We don't close the Qdrant client as it's lightweight
        self._derm1m_open_clip = None
        self._load_failed = False
        # Don't unload vqa_tool as it's a reference, not owned by us
        super().unload()
        print("[RAGTool] Model unloaded.")


# =============================================================================
# Text RAG Tool - Pure Text Retrieval from Dermatology Literature
# =============================================================================

# Stopwords for medical/dermatology RAG queries to reduce noise
TEXT_RAG_STOPWORDS = {
    # --- Query Artifacts ---
    "what", "is", "the", "of", "in", "this", "image", "picture", "photo", 
    "depicted", "shown", "demonstrated", "observed", "present", "abnormality",
    "anomaly", "lesion", "growth", "case", "patient", "regarding", "concerning",
    "for", "to", "and", "or", "between", "vs", "versus",
    
    # --- Medical Generics (cause matching to irrelevant sections) ---
    "disease", "condition", "disorder", "syndrome", "issue", "problem",
    "treatment", "treatments", "therapy", "management", "cure",
    "diagnosis", "diagnostic", "diagnose", "diagnosed", "criteria",
    "symptoms", "signs", "features", "characteristics", "manifestations", "clinical",
    "appearance", "look", "looks", "like",
    "cause", "causes", "etiology", "risk", "factors",
    "overview", "introduction", "definition", "summary",
    "score", "index", "scale", "grading", "grade", "stage", "staging",
    
    # --- Comparison and Judgment Terms ---
    "differentiating", "differentiate", "distinguish", "difference", "differences",
    "indicate", "indicative", "suggest", "suggestive",
    "associated", "related", "linked"
}


class TextRAGTool(BaseSkinTool):
    """
    Pure text Retrieval-Augmented Generation from dermatology knowledge base.

    Retrieves relevant medical knowledge from dermatology literature (DermNet, Mayo)
    based on text queries. Uses Qwen3-Embedding-8B for text embeddings and
    Qdrant 'derm_rag' collection for vector storage.

    Supports three search modes:
    - vector: Pure semantic search using embeddings
    - keyword: Full-text keyword search using Qdrant text index
    - hybrid: Combines both with Reciprocal Rank Fusion (RRF)

    Features:
    - Optional BGE Reranker for improved result relevance
    - Filters out irrelevant results that match generic keywords but not semantic intent

    Unlike the multimodal RAGTool (which uses DermLIP for image similarity),
    this tool is optimized for pure text retrieval of medical knowledge.
    """

    name: str = "text_rag_retrieval"
    description: str = (
        "Search the dermatology medical literature knowledge base using text queries. "
        "Provide a medical query (e.g., 'What is melanoma?', 'symptoms of psoriasis', "
        "'diagnostic criteria for basal cell carcinoma') to retrieve relevant medical "
        "knowledge from DermNet and Mayo Clinic sources. "
        "Supports search_mode: 'hybrid' (default, best results), 'vector' (semantic), "
        "or 'keyword' (exact match). Returns disease information, clinical descriptions, "
        "and source URLs for reference."
    )
    args_schema: Type[BaseModel] = TextQueryInput

    # Qdrant configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "derm_rag"

    # Embedding model
    encoder_model_id: str = "Qwen/Qwen3-Embedding-8B"

    # Reranker configuration (Qwen3-Reranker-0.6B)
    reranker_model_id: str = "Qwen/Qwen3-Reranker-0.6B"
    use_reranker: bool = True  # Enable reranker by default for better relevance
    reranker_top_k: int = 20   # Retrieve more candidates for reranking
    reranker_max_length: int = 8192  # Max input length for reranker

    client: Any = None
    encoder: Any = None
    tokenizer: Any = None
    reranker: Any = None
    reranker_tokenizer: Any = None
    _reranker_prefix_tokens: List = None
    _reranker_suffix_tokens: List = None
    _token_true_id: int = None
    _token_false_id: int = None
    _load_failed: bool = False
    _reranker_failed: bool = False
    
    def _load_model(self) -> None:
        """Load Qdrant client, Qwen3-Embedding model, and Qwen3-Reranker."""
        print("[TextRAGTool] Initializing Qdrant client and Qwen3-Embedding encoder...")

        try:
            from qdrant_client import QdrantClient
            from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

            # Connect to Qdrant server
            print(f"[TextRAGTool] Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}...")
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=120)

            # Verify collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            if self.collection_name not in collection_names:
                raise ValueError(
                    f"Collection '{self.collection_name}' not found. "
                    f"Available collections: {collection_names}"
                )

            collection_info = self.client.get_collection(self.collection_name)
            print(f"[TextRAGTool] Connected to collection '{self.collection_name}' "
                  f"with {collection_info.points_count} points")

            # Determine cache directory
            cache_dir = Path(__file__).parent.parent.parent / "model-weights"

            # Load Qwen3-Embedding model
            print(f"[TextRAGTool] Loading encoder: {self.encoder_model_id}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.encoder_model_id,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            self.encoder = AutoModel.from_pretrained(
                self.encoder_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            self.encoder.eval()

            print("[TextRAGTool] Encoder initialized successfully!")

            # Load Qwen3-Reranker if enabled
            if self.use_reranker:
                print(f"[TextRAGTool] Loading reranker: {self.reranker_model_id}")
                try:
                    self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                        self.reranker_model_id,
                        padding_side='left',
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                        local_files_only=True,
                    )
                    self.reranker = AutoModelForCausalLM.from_pretrained(
                        self.reranker_model_id,
                        torch_dtype=torch.bfloat16,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                        local_files_only=True,
                    )
                    self.reranker.eval()

                    # Setup reranker tokens and prompts
                    self._token_false_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
                    self._token_true_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")

                    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
                    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    self._reranker_prefix_tokens = self.reranker_tokenizer.encode(prefix, add_special_tokens=False)
                    self._reranker_suffix_tokens = self.reranker_tokenizer.encode(suffix, add_special_tokens=False)

                    print("[TextRAGTool] Reranker initialized successfully!")
                except Exception as e:
                    print(f"[TextRAGTool] Failed to load reranker: {e}")
                    import traceback
                    traceback.print_exc()
                    self._reranker_failed = True
                    print("[TextRAGTool] Continuing without reranker...")

            print("[TextRAGTool] Initialized successfully!")

        except Exception as e:
            print(f"[TextRAGTool] Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            self._load_failed = True
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector using Qwen3-Embedding (4096-dim)."""
        # Tokenize
        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.encoder.device)
        
        # Encode with mean pooling
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            
            # Mean Pooling with attention mask
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs.attention_mask
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # Normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().float().numpy()[0]
    
    def _clean_query(self, query: str) -> str:
        """
        Remove stopwords from the query to improve retrieval precision.
        
        Args:
            query: Original query string
            
        Returns:
            Cleaned query string with stopwords removed
        """
        if not query:
            return query
            
        # Replace common punctuation with space to handle "word1,word2"
        for char in ",.!?;:()[]{}":
            query = query.replace(char, " ")
            
        words = query.split()
        cleaned_words = [
            w for w in words 
            if w.lower() not in TEXT_RAG_STOPWORDS
        ]
        
        cleaned = " ".join(cleaned_words)
        
        # If all words were stopwords, return original to avoid empty query
        if not cleaned.strip():
            return query
            
        return cleaned
    
    def _vector_search(self, query_vector: np.ndarray, top_k: int) -> list:
        """Vector-based semantic search using Qdrant."""
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        ).points
    
    def _keyword_search(self, query: str, top_k: int) -> list:
        """Full-text keyword search using Qdrant text index."""
        from qdrant_client.models import Filter, FieldCondition, MatchText
        
        # Extract meaningful keywords (filter out short words and common terms)
        keywords = [
            kw.strip().lower() 
            for kw in query.split() 
            if len(kw.strip()) >= 3
        ]
        
        if not keywords:
            return []
        
        try:
            # Search using text match filter with OR logic (should)
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    should=[
                        FieldCondition(
                            key="text_for_embedding",
                            match=MatchText(text=kw)
                        ) for kw in keywords[:10]  # Limit to top 10 keywords
                    ]
                ),
                limit=top_k,
                with_payload=True,
            )
            return results
        except Exception as e:
            print(f"[TextRAGTool] Keyword search error: {e}")
            return []
    
    def _rrf_fusion(
        self, 
        vector_results: list, 
        keyword_results: list, 
        top_k: int, 
        k: int = 60
    ) -> list:
        """
        Reciprocal Rank Fusion to combine vector and keyword search results.
        
        RRF score = sum(1 / (k + rank)) for each result list
        Higher k gives more weight to lower-ranked results.
        """
        # Build score map and payload map
        scores = {}
        payloads = {}
        
        # Score from vector search
        for rank, hit in enumerate(vector_results):
            point_id = hit.id
            scores[point_id] = scores.get(point_id, 0) + 1 / (k + rank + 1)
            payloads[point_id] = {
                "payload": hit.payload,
                "vector_rank": rank + 1,
                "vector_score": getattr(hit, 'score', 0),
            }
        
        # Score from keyword search
        for rank, point in enumerate(keyword_results):
            point_id = point.id
            scores[point_id] = scores.get(point_id, 0) + 1 / (k + rank + 1)
            if point_id not in payloads:
                payloads[point_id] = {
                    "payload": point.payload,
                    "keyword_rank": rank + 1,
                }
            else:
                payloads[point_id]["keyword_rank"] = rank + 1
        
        # Sort by RRF score and take top-k
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        
        # Build result objects with combined scores
        class HybridResult:
            def __init__(self, id, score, payload, vector_rank=None, keyword_rank=None):
                self.id = id
                self.score = score
                self.payload = payload
                self.vector_rank = vector_rank
                self.keyword_rank = keyword_rank
        
        results = []
        for point_id in sorted_ids:
            info = payloads[point_id]
            results.append(HybridResult(
                id=point_id,
                score=scores[point_id],
                payload=info["payload"],
                vector_rank=info.get("vector_rank"),
                keyword_rank=info.get("keyword_rank"),
            ))
        
        return results
    
    def _hybrid_search(self, query: str, top_k: int) -> list:
        """Combine vector and keyword search with RRF fusion."""
        # 1. Vector search (get more results for fusion)
        query_vector = self._encode_text(query)
        vector_results = self._vector_search(query_vector, top_k * 2)

        # 2. Keyword search
        keyword_results = self._keyword_search(query, top_k * 2)

        # 3. RRF Fusion
        if not keyword_results:
            # Fallback to vector-only if keyword search fails
            return vector_results[:top_k]

        return self._rrf_fusion(vector_results, keyword_results, top_k)

    def _format_reranker_input(self, query: str, doc: str) -> str:
        """Format input for Qwen3-Reranker with instruction, query, and document."""
        instruction = (
            "Given a medical query about dermatology or skin conditions, "
            "retrieve relevant passages that provide accurate diagnostic criteria, "
            "clinical features, or treatment information for the queried condition."
        )
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _rerank(self, query: str, results: list, top_k: int) -> list:
        """
        Rerank results using Qwen3-Reranker-4B.

        Args:
            query: Original search query
            results: List of search results with payloads
            top_k: Number of top results to return after reranking

        Returns:
            Reranked list of results
        """
        if not results or not self.reranker or self._reranker_failed:
            return results[:top_k]

        try:
            # Prepare query-document pairs
            pairs = []
            for hit in results:
                payload = hit.payload or {}
                # Get document content
                doc_content = payload.get("original_content", "")
                if not doc_content:
                    doc_content = payload.get("text_for_embedding", "")

                # Add metadata context
                metadata = payload.get("metadata", {})
                disease = metadata.get("disease", "")
                header = metadata.get("header", "")
                if disease or header:
                    doc_content = f"{disease} - {header}: {doc_content}" if header else f"{disease}: {doc_content}"

                pairs.append(self._format_reranker_input(query, doc_content[:2000]))  # Truncate long docs

            # Tokenize inputs
            max_content_length = self.reranker_max_length - len(self._reranker_prefix_tokens) - len(self._reranker_suffix_tokens)
            inputs = self.reranker_tokenizer(
                pairs,
                padding=False,
                truncation=True,
                return_attention_mask=False,
                max_length=max_content_length
            )

            # Add prefix and suffix tokens
            for i, ele in enumerate(inputs['input_ids']):
                inputs['input_ids'][i] = self._reranker_prefix_tokens + ele + self._reranker_suffix_tokens

            # Pad and convert to tensors
            inputs = self.reranker_tokenizer.pad(
                inputs,
                padding=True,
                return_tensors="pt",
                max_length=self.reranker_max_length
            )
            inputs = {k: v.to(self.reranker.device) for k, v in inputs.items()}

            # Compute relevance scores
            with torch.no_grad():
                batch_logits = self.reranker(**inputs).logits[:, -1, :]
                true_scores = batch_logits[:, self._token_true_id]
                false_scores = batch_logits[:, self._token_false_id]
                stacked = torch.stack([false_scores, true_scores], dim=1)
                log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
                scores = log_probs[:, 1].exp().tolist()  # Probability of "yes"

            # Create reranked results with new scores
            class RerankedResult:
                def __init__(self, original_hit, rerank_score, original_rank):
                    self.id = original_hit.id
                    self.payload = original_hit.payload
                    self.score = rerank_score
                    self.original_score = getattr(original_hit, 'score', 0)
                    self.original_rank = original_rank

            reranked = [
                RerankedResult(hit, score, i + 1)
                for i, (hit, score) in enumerate(zip(results, scores))
            ]

            # Sort by rerank score (descending) and take top_k
            reranked.sort(key=lambda x: x.score, reverse=True)
            return reranked[:top_k]

        except Exception as e:
            print(f"[TextRAGTool] Reranking failed: {e}")
            import traceback
            traceback.print_exc()
            return results[:top_k]
    
    def _format_results(self, results: list, query: str, search_mode: str = "hybrid", reranked: bool = False) -> str:
        """Format search results for display."""
        if not results:
            return f"No results found for query: '{query}'"

        mode_str = f"{search_mode}+rerank" if reranked else search_mode
        output = [f"Text RAG Retrieval Results (mode: {mode_str}):"]
        output.append(f"Query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        output.append("-" * 60)

        for i, hit in enumerate(results):
            output.append(f"\nResult {i+1}:")

            # Format score based on result type
            if reranked and hasattr(hit, 'original_rank'):
                # Reranked result
                output.append(f"  Rerank Score: {hit.score:.4f}")
                output.append(f"  Original Rank: #{hit.original_rank} (score: {hit.original_score:.4f})")
            elif search_mode == "hybrid" and hasattr(hit, 'vector_rank'):
                output.append(f"  RRF Score: {hit.score:.4f}")
                ranks = []
                if hit.vector_rank:
                    ranks.append(f"vector #{hit.vector_rank}")
                if hasattr(hit, 'keyword_rank') and hit.keyword_rank:
                    ranks.append(f"keyword #{hit.keyword_rank}")
                if ranks:
                    output.append(f"  Ranks: {', '.join(ranks)}")
            else:
                score = getattr(hit, 'score', 0)
                if score:
                    output.append(f"  Similarity Score: {score:.4f}")

            payload = hit.payload or {}
            metadata = payload.get("metadata", {})

            # Disease name from metadata
            disease = metadata.get("disease", "N/A")
            output.append(f"  Disease: {disease}")

            # Section header
            header = metadata.get("header", "")
            if header:
                output.append(f"  Section: {header}")

            # Category
            category = metadata.get("category", "")
            if category:
                output.append(f"  Category: {category}")

            # Original content (truncate for display)
            content = payload.get("original_content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            if content:
                output.append(f"  Content: {content}")

            # Source URL
            source_url = metadata.get("source_url", "")
            if source_url:
                output.append(f"  Source: {source_url}")

            # Source file
            source_file = payload.get("source_file", "")
            if source_file:
                output.append(f"  Data Source: {source_file}")

        return "\n".join(output)
    
    def _run(self, query: str, top_k: int = 5, search_mode: str = "hybrid") -> str:
        """
        Search the knowledge base using text query.

        Args:
            query: Text query for medical knowledge retrieval
            top_k: Number of results to return
            search_mode: Search mode - 'vector', 'keyword', or 'hybrid' (default)

        Returns:
            Retrieved medical knowledge as formatted string
        """
        try:
            # Validate input
            if not query or not query.strip():
                return (
                    "Text RAG Retrieval Error:\n"
                    "Please provide a non-empty 'query' for text-based retrieval."
                )

            # Validate search mode
            valid_modes = {"vector", "keyword", "hybrid"}
            if search_mode not in valid_modes:
                search_mode = "hybrid"

            # Clean query by removing stopwords for better retrieval precision
            original_query = query
            query = self._clean_query(query)

            # Ensure resources are loaded
            self._ensure_initialized()

            # If initialization failed, return error
            if self._load_failed:
                return (
                    "Text RAG Retrieval Error:\n"
                    "[Knowledge base not available] Unable to search for medical knowledge. "
                    "Please ensure Qdrant server is running and 'derm_rag' collection exists."
                )

            # Check collection exists and has data
            try:
                collection_info = self.client.get_collection(self.collection_name)
                if collection_info.points_count == 0:
                    return "Error: Knowledge base is empty. Please build the index first."
            except Exception as e:
                return f"Error: Collection '{self.collection_name}' not found: {str(e)}"

            # Execute search based on mode
            try:
                # Determine how many candidates to retrieve for reranking
                retrieve_k = self.reranker_top_k if (self.use_reranker and not self._reranker_failed) else top_k

                if search_mode == "vector":
                    query_vector = self._encode_text(query)
                    results = self._vector_search(query_vector, retrieve_k)
                elif search_mode == "keyword":
                    results = self._keyword_search(query, retrieve_k)
                else:  # hybrid
                    results = self._hybrid_search(query, retrieve_k)

                # Apply reranking if enabled and available
                reranked = False
                if self.use_reranker and not self._reranker_failed and self.reranker is not None:
                    results = self._rerank(query, results, top_k)
                    reranked = True
                else:
                    results = results[:top_k]

                return self._format_results(results, query, search_mode, reranked=reranked)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return (
                    "Text RAG Retrieval Error:\n"
                    "GPU out of memory. Try reducing batch size or closing other GPU processes."
                )
            except Exception as e:
                error_msg = str(e).lower()
                if "timed out" in error_msg or "timeout" in error_msg:
                    return (
                        f"Text RAG Retrieval Error:\n"
                        f"Query encoding timed out: {e}. "
                        "GPU resource contention likely."
                    )
                else:
                    return f"Text RAG Retrieval Error:\nQuery encoding failed: {e}"

        except Exception as e:
            return self._handle_error(e, "text retrieval")
    
    def unload(self) -> None:
        """Unload TextRAG encoder, reranker, and associated resources."""
        if self.encoder is not None:
            del self.encoder
            self.encoder = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.reranker is not None:
            del self.reranker
            self.reranker = None
        if self.reranker_tokenizer is not None:
            del self.reranker_tokenizer
            self.reranker_tokenizer = None
        self._reranker_prefix_tokens = None
        self._reranker_suffix_tokens = None
        self._token_true_id = None
        self._token_false_id = None
        self._load_failed = False
        self._reranker_failed = False
        # Note: We don't close the Qdrant client as it's lightweight
        super().unload()
        print("[TextRAGTool] Model unloaded.")


# =============================================================================
# Ontology Tool - Skin Disease Hierarchy Knowledge Tree
# =============================================================================

# Try to import rapidfuzz for faster fuzzy matching
try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from difflib import SequenceMatcher


class OntologyQueryInput(BaseModel):
    """Input schema for ontology query tool."""
    
    disease_name: str = Field(
        description=(
            "The disease or category name to query. Supports fuzzy matching, "
            "so partial names or misspellings will still find matches. "
            "Examples: 'melanoma', 'psoriasis', 'tinea', 'eczema'"
        )
    )
    query_type: str = Field(
        default="hierarchy",
        description=(
            "Type of query to perform:\n"
            "- 'hierarchy': Get the full path from root to the disease (default)\n"
            "- 'children': Get direct children of a disease category\n"
            "- 'siblings': Get diseases at the same level (same parent)\n"
            "- 'search': Find all diseases matching the query name"
        )
    )
    fuzzy_threshold: int = Field(
        default=70,
        description=(
            "Minimum similarity score (0-100) for fuzzy matching. "
            "Lower values return more results but may include less relevant matches. "
            "Default is 70."
        )
    )
    top_k: int = Field(
        default=10,
        description="Maximum number of results to return for search queries."
    )


class OntologyTool(BaseTool):
    """
    Skin disease ontology knowledge tree query tool.
    
    Enables querying the hierarchical taxonomy of skin diseases with fuzzy
    name matching. Supports multiple query types:
    - hierarchy: Get full path from root to disease
    - children: Get direct children of a category
    - siblings: Get diseases at the same level
    - search: Find diseases by name with fuzzy matching
    
    This tool does not require GPU and loads static JSON data.
    """
    
    name: str = "ontology_query"
    description: str = (
        "Query the skin disease knowledge tree to understand disease relationships. "
        "Use this tool to:\n"
        "1. Find the hierarchical category of a disease (query_type='hierarchy')\n"
        "2. Get subtypes/children of a disease category (query_type='children')\n"
        "3. Find related diseases at the same level (query_type='siblings')\n"
        "4. Search for diseases by name with fuzzy matching (query_type='search')\n"
        "Supports fuzzy matching, so partial names like 'melanoma' or 'psoria' will work."
    )
    args_schema: Type[BaseModel] = OntologyQueryInput
    
    # Data storage
    ontology_flat: Dict[str, List[str]] = {}
    child_to_parent: Dict[str, str] = {}
    all_diseases: List[str] = []
    _initialized: bool = False
    
    # Path to ontology files
    _data_dir: Path = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data_dir = Path(__file__).parent / "derm_knowledge_tree"
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Lazy initialization of ontology data."""
        if not self._initialized:
            self._load_ontology()
            self._initialized = True
    
    def _load_ontology(self) -> None:
        """Load ontology data and build reverse index."""
        print("[OntologyTool] Loading ontology data...")
        
        # Load flat ontology
        flat_path = self._data_dir / "ontology.json"
        if not flat_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {flat_path}")
        
        with open(flat_path, 'r', encoding='utf-8') as f:
            self.ontology_flat = json.load(f)
        
        # Build child -> parent reverse index
        self.child_to_parent = {}
        for parent, children in self.ontology_flat.items():
            for child in children:
                self.child_to_parent[child] = parent
        
        # Collect all disease names (excluding 'root' key)
        self.all_diseases = [
            name for name in self.ontology_flat.keys() 
            if name != "root"
        ]
        
        print(f"[OntologyTool] Loaded {len(self.all_diseases)} disease entries.")
    
    # =========================================================================
    # Fuzzy Matching
    # =========================================================================
    
    def _compute_similarity(self, query: str, target: str) -> int:
        """
        Compute similarity score between query and target string.
        
        Uses rapidfuzz if available, otherwise falls back to difflib.
        
        Args:
            query: Query string (lowercase)
            target: Target string to compare against
            
        Returns:
            Similarity score from 0 to 100
        """
        target_lower = target.lower()
        
        if RAPIDFUZZ_AVAILABLE:
            # Use token_set_ratio for better matching with word reordering
            return int(rapidfuzz_fuzz.token_set_ratio(query, target_lower))
        else:
            # Fallback to difflib SequenceMatcher
            ratio = SequenceMatcher(None, query, target_lower).ratio()
            return int(ratio * 100)
    
    def _fuzzy_search(
        self, 
        query: str, 
        threshold: int = 70,
        top_k: int = 10
    ) -> List[Tuple[str, int]]:
        """
        Find diseases matching query with similarity scores.
        
        Handles alias matching by splitting on comma.
        
        Args:
            query: Search query string
            threshold: Minimum similarity score (0-100)
            top_k: Maximum number of results
            
        Returns:
            List of (disease_name, score) tuples, sorted by score descending
        """
        query_lower = query.lower().strip()
        matches = []
        
        for disease_name in self.all_diseases:
            # Split aliases: "Malignant melanoma, melanoma" -> ["Malignant melanoma", "melanoma"]
            aliases = [a.strip() for a in disease_name.split(",")]
            
            # Get best score among all aliases
            best_score = max(
                self._compute_similarity(query_lower, alias) 
                for alias in aliases
            )
            
            if best_score >= threshold:
                matches.append((disease_name, best_score))
        
        # Sort by score descending, then alphabetically for ties
        matches.sort(key=lambda x: (-x[1], x[0]))
        
        return matches[:top_k]
    
    def _find_best_match(self, query: str, threshold: int = 70) -> Optional[str]:
        """
        Find the single best matching disease name.
        
        Args:
            query: Search query string
            threshold: Minimum similarity score
            
        Returns:
            Best matching disease name, or None if no match above threshold
        """
        matches = self._fuzzy_search(query, threshold=threshold, top_k=1)
        if matches:
            return matches[0][0]
        return None
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def _get_hierarchy_path(self, disease: str) -> List[str]:
        """
        Get the full path from root to the disease.
        
        Args:
            disease: Disease name (must be exact match from ontology)
            
        Returns:
            List of nodes from root to disease, e.g.,
            ['Root Ontology', 'inflammatory', 'infectious', 'bacterial', 'Cellulitis']
        """
        path = [disease]
        current = disease
        
        # Walk up the tree until we reach root
        while current in self.child_to_parent:
            parent = self.child_to_parent[current]
            path.insert(0, parent)
            current = parent
        
        # Add "Root Ontology" at the beginning if not already there
        if path and path[0] != "Root Ontology":
            path.insert(0, "Root Ontology")
        
        return path
    
    def _get_children(self, disease: str) -> List[str]:
        """
        Get direct children of a disease category.
        
        Args:
            disease: Disease/category name
            
        Returns:
            List of child disease names
        """
        return self.ontology_flat.get(disease, [])
    
    def _get_siblings(self, disease: str) -> List[str]:
        """
        Get sibling diseases (same parent).
        
        Args:
            disease: Disease name
            
        Returns:
            List of sibling disease names (excluding the query disease itself)
        """
        parent = self.child_to_parent.get(disease)
        if not parent:
            return []
        
        siblings = self.ontology_flat.get(parent, [])
        return [s for s in siblings if s != disease]
    
    # =========================================================================
    # Main Entry Point
    # =========================================================================
    
    def _run(
        self, 
        disease_name: str,
        query_type: str = "hierarchy",
        fuzzy_threshold: int = 70,
        top_k: int = 10
    ) -> str:
        """
        Execute ontology query.
        
        Args:
            disease_name: Disease name to query (supports fuzzy matching)
            query_type: Type of query (hierarchy, children, siblings, search)
            fuzzy_threshold: Minimum similarity score for fuzzy matching
            top_k: Maximum results for search queries
            
        Returns:
            Formatted query results as string
        """
        try:
            # Ensure ontology is loaded
            self._ensure_initialized()
            
            # Normalize query type
            query_type = query_type.lower().strip()
            
            # Handle search query type separately (returns multiple matches)
            if query_type == "search":
                return self._format_search_results(disease_name, fuzzy_threshold, top_k)
            
            # For other query types, find best match first
            matched_disease = self._find_best_match(disease_name, threshold=fuzzy_threshold)
            
            if not matched_disease:
                # No match found - provide helpful suggestions
                suggestions = self._fuzzy_search(disease_name, threshold=50, top_k=5)
                if suggestions:
                    suggestion_list = "\n".join(
                        f"  - {name} (score: {score})" 
                        for name, score in suggestions
                    )
                    return (
                        f"No disease found matching '{disease_name}' "
                        f"(threshold: {fuzzy_threshold}).\n\n"
                        f"Did you mean one of these?\n{suggestion_list}"
                    )
                return f"No disease found matching '{disease_name}'."
            
            # Execute the appropriate query
            if query_type == "hierarchy":
                return self._format_hierarchy_results(matched_disease, disease_name)
            elif query_type == "children":
                return self._format_children_results(matched_disease, disease_name)
            elif query_type == "siblings":
                return self._format_siblings_results(matched_disease, disease_name)
            else:
                return (
                    f"Unknown query_type: '{query_type}'. "
                    f"Valid options: hierarchy, children, siblings, search"
                )
                
        except Exception as e:
            return f"Error in ontology query: {type(e).__name__}: {str(e)}"
    
    # =========================================================================
    # Result Formatting
    # =========================================================================
    
    def _format_search_results(
        self, 
        query: str, 
        threshold: int,
        top_k: int
    ) -> str:
        """Format search query results."""
        matches = self._fuzzy_search(query, threshold=threshold, top_k=top_k)
        
        if not matches:
            return f"No diseases found matching '{query}' (threshold: {threshold})."
        
        lines = [f"Diseases matching '{query}' (fuzzy search, threshold={threshold}):"]
        lines.append("-" * 50)
        
        for i, (name, score) in enumerate(matches, 1):
            # Get parent category for context
            parent = self.child_to_parent.get(name, "root category")
            lines.append(f"{i}. {name}")
            lines.append(f"   Score: {score} | Parent: {parent}")
        
        return "\n".join(lines)
    
    def _format_hierarchy_results(self, matched: str, original_query: str) -> str:
        """Format hierarchy query results."""
        path = self._get_hierarchy_path(matched)
        
        lines = []
        if matched.lower() != original_query.lower():
            lines.append(f"Query '{original_query}' matched to: '{matched}'")
            lines.append("")
        
        lines.append(f"Ontology Hierarchy for '{matched}':")
        lines.append("-" * 50)
        
        # Compact path representation
        lines.append(" -> ".join(path))
        lines.append("")
        
        # Detailed path with indentation
        lines.append("Category Path:")
        for i, node in enumerate(path):
            indent = "  " * i
            if i == 0:
                label = "(root)"
            elif i == len(path) - 1:
                label = "(target)"
            else:
                label = "(category)"
            lines.append(f"{indent}└─ {node} {label}")
        
        # Add children if any
        children = self._get_children(matched)
        if children:
            lines.append("")
            lines.append(f"Subtypes ({len(children)}):")
            for child in children[:10]:  # Limit to 10
                lines.append(f"  - {child}")
            if len(children) > 10:
                lines.append(f"  ... and {len(children) - 10} more")
        
        return "\n".join(lines)
    
    def _format_children_results(self, matched: str, original_query: str) -> str:
        """Format children query results."""
        children = self._get_children(matched)
        
        lines = []
        if matched.lower() != original_query.lower():
            lines.append(f"Query '{original_query}' matched to: '{matched}'")
            lines.append("")
        
        if not children:
            lines.append(f"'{matched}' has no children (leaf node in ontology).")
            # Suggest looking at siblings instead
            siblings = self._get_siblings(matched)
            if siblings:
                lines.append("")
                lines.append(f"Related diseases at the same level ({len(siblings)}):")
                for sib in siblings[:5]:
                    lines.append(f"  - {sib}")
        else:
            lines.append(f"Children of '{matched}' ({len(children)}):")
            lines.append("-" * 50)
            for i, child in enumerate(children, 1):
                # Check if child has sub-children
                sub_children = self._get_children(child)
                suffix = f" [{len(sub_children)} subtypes]" if sub_children else ""
                lines.append(f"{i}. {child}{suffix}")
        
        return "\n".join(lines)
    
    def _format_siblings_results(self, matched: str, original_query: str) -> str:
        """Format siblings query results."""
        siblings = self._get_siblings(matched)
        parent = self.child_to_parent.get(matched, "unknown")
        
        lines = []
        if matched.lower() != original_query.lower():
            lines.append(f"Query '{original_query}' matched to: '{matched}'")
            lines.append("")
        
        if not siblings:
            lines.append(f"'{matched}' has no siblings (only child under '{parent}').")
        else:
            lines.append(f"Siblings of '{matched}' (under '{parent}'):")
            lines.append("-" * 50)
            for i, sib in enumerate(siblings, 1):
                lines.append(f"{i}. {sib}")
        
        return "\n".join(lines)

