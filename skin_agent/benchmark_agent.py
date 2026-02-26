"""
Benchmark Agent - LangGraph Agent optimized for evaluation tasks.

Differences from the interactive agent:
- Task-specific system prompts (diagnosis/concept/captioning)
- Low temperature (0.1) for output consistency
- Enforced FINAL_ANSWER format
- AnswerParser for structured answer extraction
"""

import base64
import json
import logging
import operator
import os
import re
import gc
import torch
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, TypedDict

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

import time

from .tools import MAKETool, PanDermTool, DermoGPTTool, RAGTool, TextRAGTool, OntologyTool, preload_tools_sequential, tool_timing_registry, ToolMemoryManager
from .tracing import AgentTrace, TraceLogger, ToolCall, AgentStep, CriticRetryStep
from .utils.retry import with_rate_limit_retry
from .utils.image_utils import generate_image_id, register_image_path
from .profiler import profiler, init_profiler_from_env
from .configs import DATASET_CONFIGS

# Initialize profiler based on environment (TEST_MODE=true enables profiling)
init_profiler_from_env()

logger = logging.getLogger("dermagent.benchmark")
# Configure root logger if not already set up (runner scripts may skip this)
if not logging.root.handlers:
    _log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, _log_level, logging.WARNING),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# Model Configuration
# =============================================================================

# Model aliases for easier switching between different LLM orchestrators
# Users can use either the alias (e.g., "gpt-5") or the full API name (e.g., "gpt-5-chat-latest")
MODEL_ALIASES = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-5": "gpt-5",
    "o1": "o1",
    "o1-mini": "o1-mini",
    "o1-preview": "o1-preview",
}

def resolve_model_name(model_name: str) -> str:
    """
    Resolve model alias to actual API model name.
    
    Args:
        model_name: Model name or alias (e.g., "gpt-5" or "gpt-5-chat-latest")
        
    Returns:
        Actual API model name
    """
    resolved = MODEL_ALIASES.get(model_name, model_name)
    if resolved != model_name:
        print(f"[Model] Resolved alias '{model_name}' -> '{resolved}'")
    return resolved


# =============================================================================
# Task-Specific System Prompts
# =============================================================================

BASE_TOOL_DESCRIPTION = """
## Available Tools

1. **panderm_classifier**: Zero-shot skin disease classification
   - IMPORTANT: When calling this tool, you MUST pass the exact candidate_diseases 
     list provided in the task options (comma-separated string)
   - Returns: top-k predictions with confidence scores among the specified candidates
   - Best for: Initial diagnosis hypothesis

2. **make_concept_annotator**: Dermoscopic concept extraction
   - Returns: Detected features (pigment_network, dots_globules, etc.)
   - Best for: Feature-level analysis

3. **dermogpt_vqa**: DermoGPT vision-language model for dermatological analysis
   - Input: image_path + query (optional, for VQA mode)
   - Returns: Detailed morphological description with clinical terminology
   - Best for: Detailed description, answering specific questions, dermatology-specialized analysis

4. **rag_retrieval**: Similar case retrieval from knowledge base
   - Use IMAGE MODE ONLY: Provide 'image_path' to find visually similar cases
   - DO NOT use text_query parameter (text mode is disabled for stability)
   - Returns: Similar cases with diagnosis labels and clinical descriptions
   - Best for: Evidence-based reasoning, finding precedents
"""

# Task2 Concept Annotation uses DermoGPT
CONCEPT_TOOL_DESCRIPTION = """
## Available Tools

1. **panderm_classifier**: Zero-shot skin disease classification
   - Returns: top-k predictions with confidence scores
   - Best for: Initial classification hypothesis

2. **make_concept_annotator**: Dermoscopic concept extraction
   - Returns: Detected features (pigment_network, dots_globules, etc.)
   - Best for: Feature-level analysis

3. **dermogpt_vqa**: DermoGPT vision-language model for dermatological analysis
   - Input: image_path + target_concepts (comma-separated list)
   - Returns: Concept presence evaluation with visual evidence
   - Best for: Detailed morphological description and concept verification

4. **rag_retrieval**: Similar case retrieval from knowledge base
   - Use IMAGE MODE ONLY: Provide 'image_path' to find visually similar cases
   - DO NOT use text_query parameter
   - Returns: Similar cases with diagnosis labels and clinical descriptions
   - Best for: Evidence-based reasoning, finding precedents
"""

# -----------------------------------------------------------------------------
# Diagnosis Task
# -----------------------------------------------------------------------------
DIAGNOSIS_SYSTEM_PROMPT = f"""You are DermAgent, an expert dermatology AI performing skin disease diagnosis.

{BASE_TOOL_DESCRIPTION}

## Your Task
Classify the skin lesion into ONE of the provided diagnostic categories.

## Required Reasoning Process

1. **Initial Classification**: Call panderm_classifier with the EXACT candidate_diseases 
   list provided in the task (pass as comma-separated string)
2. **Feature Extraction**: Call make_concept_annotator to identify visual features  
3. **Visual Description**: Call dermogpt_vqa to describe what you see
4. **Case Comparison**: Call rag_retrieval with image_path ONLY (IMAGE MODE)
   - DO NOT use text_query parameter

5. **Cross-Validation** (CRITICAL):
   - Does the VLM description support PanDerm's top prediction?
   - Do the detected concepts match typical features of the predicted disease?
   - What diagnoses do similar cases have from RAG image retrieval?

6. **Final Decision**: Synthesize all evidence to select the most likely diagnosis

## Output Format

You MUST end your response with exactly:
```
FINAL_ANSWER: [exact diagnosis name from options]
```

## Important
- ALWAYS pass the exact candidate_diseases to panderm_classifier
- For RAG, use IMAGE MODE ONLY (do not use text_query)
- Your final answer MUST be one of the provided options exactly
- Explain your reasoning step by step
"""


# -----------------------------------------------------------------------------
# Diagnosis Task with DermoGPT + Full 6 Tools (including Ontology)
# Tools: panderm, make, dermogpt_vqa, rag, text_rag, ontology
# Auto-selected when dermogpt_vqa AND text_rag AND ontology are all in enabled_tools
# -----------------------------------------------------------------------------
DIAGNOSIS_SYSTEM_PROMPT_DERMOGPT_FULL = """You are DermAgent, an expert dermatology AI performing skin disease diagnosis.

## Available Tools

1. **panderm_classifier**: Zero-shot skin disease classification
   - IMPORTANT: Pass the exact candidate_diseases list from the task options (comma-separated string)
   - Returns: top-k predictions with confidence scores among the specified candidates
   - Best for: Initial statistical diagnosis hypothesis

2. **make_concept_annotator**: Dermoscopic concept extraction
   - Returns: Detected features (pigment_network, dots_globules, streaks, etc.) with confidence scores
   - Best for: Feature-level evidence for clinical reasoning

3. **dermogpt_vqa**: DermoGPT vision-language model for dermatological analysis
   - Input: image_path + candidate_diseases (for diagnosis) or query (for VQA)
   - For DIAGNOSIS: Pass candidate_diseases to get structured XML output with <reasoning> and <final_diagnosis> tags
   - For general VQA: Pass query for free-form questions about the image
   - Best for: Detailed morphological description, clinical reasoning, dermatology-specialized analysis

4. **rag_retrieval**: Image-based similar case retrieval from dermatology knowledge base
   - Use IMAGE MODE ONLY: Provide 'image_path' to find visually similar cases
   - DO NOT use text_query parameter (text mode is disabled)
   - Returns: Similar cases with disease labels, categories, and similarity scores
   - Best for: Finding visual precedents, evidence-based comparison

5. **text_rag_retrieval**: Medical literature search (DermNet, Mayo Clinic)
   - Input: query (text), top_k (default 5), search_mode ("hybrid"/"vector"/"keyword")
   - Returns: Diagnostic criteria, clinical guidelines, disease differentiation info, source URLs
   - Use search_mode="hybrid" for best results
   - Best for: Looking up diagnostic criteria, differentiating similar diseases, verifying rare conditions

6. **ontology_query**: Skin disease knowledge tree
   - Input: disease_name, query_type ("hierarchy"/"children"/"siblings"/"search")
   - Returns: Disease category, parent-child relationships, related diseases at the same level
   - Supports fuzzy matching (partial names like "melanoma" or "psoria" will work)
   - Best for: Understanding disease relationships, category lookups, finding related conditions

## Your Task
Classify the skin lesion into ONE of the provided diagnostic categories.

## Recommended Approach
1. **Gather visual evidence**: Use panderm_classifier (with exact candidate_diseases), make_concept_annotator, dermogpt_vqa, and rag_retrieval to analyze the image from multiple angles.
2. **Evaluate confidence**: Review results from Step 1.
   - If tools agree and confidence is high → proceed to final answer
   - If tools conflict or confidence is low → use **text_rag_retrieval** to look up differentiating criteria, or **ontology_query** to understand disease relationships
3. **Synthesize**: Combine all evidence to make your final diagnosis.

You are free to choose which tools to call and in what order based on your clinical judgment.

## Output Format

You MUST end your response with exactly:
```
FINAL_ANSWER: [exact diagnosis name from options]
```

## Important Rules
- ALWAYS pass the exact candidate_diseases to panderm_classifier
- For rag_retrieval, use IMAGE MODE ONLY (do not use text_query)
- Your final answer MUST be one of the provided options exactly
- Explain your reasoning step by step
"""

# -----------------------------------------------------------------------------
# Diagnosis Task V2 - Phased Workflow with Critic Integration
# Tools: panderm, dermogpt_vqa, rag, text_rag, ontology (NO MAKE)
# Triggered by: --prompt=v2
# -----------------------------------------------------------------------------
DIAGNOSIS_SYSTEM_PROMPT_V2 = """You are DermAgent, an advanced AI dermatologist orchestrated by a decision loop.

## Tool Definitions

1. **panderm_classifier**: Zero-shot specific disease classification.
   - **INPUT**: `image_path` + `candidate_diseases` (Strictly pass the options list).
   - **OUTPUT**: Top-k probabilities.
   - **USE**: Initial statistical hypothesis.

2. **dermogpt_vqa**: Multimodal Large Language Model for reasoning.
   - **INPUT**: `image_path` + `candidate_diseases` + `prompt`.
   - **INSTRUCTION**: You MUST pass the `candidate_diseases` to this tool.
   - **PROMPT STRATEGY**: Ask for "A detailed morphological description followed by a Chain-of-Thought (CoT) diagnosis selecting the best fit from the provided candidates."
   - **USE**: To get a reasoned medical opinion (Morphology + Logic).

3. **rag_retrieval**: Image-based similar case retrieval.
   - **USE**: Visual confirmation (Image Mode). Returns similar cases with labels.

4. **text_rag_retrieval**: Medical literature search.
   - **USAGE**: Issue SEPARATE queries for each disease to establish independent ground truth.
   - **Example**: Call 1 -> "clinical criteria for Melanoma"; Call 2 -> "clinical criteria for Seborrheic Keratosis".

5. **ontology_query**: Knowledge tree lookup.
   - **USE**: To check if candidates are siblings or parent/child nodes.

## Workflow Protocol

Determine your execution phase based on the conversation history.

### PHASE 1: INITIAL VISUAL ANALYSIS & CROSS-CHECK
(Trigger: No previous critic feedback)

1. **Statistical Diagnosis**: Call `panderm_classifier` with `candidate_diseases`.
2. **Reasoning Diagnosis**: Call `dermogpt_vqa`.
   - **CRITICAL**: Pass the same `candidate_diseases`.
   - **Query**: "Describe the lesion's morphology (color, structure, border) and provide a step-by-step reasoning to diagnose it from the candidate list."
3. **Visual Anchor**: Call `rag_retrieval` (Image Mode).
4. **Output**: Produce a structured JSON. Do NOT use Text RAG yet.

### PHASE 2: EVIDENCE VERIFICATION & CONFLICT RESOLUTION
(Trigger: You received **CRITIC_FEEDBACK** indicating conflict or low confidence)

1. **Analyze Conflict**: Compare Phase 1 results.
   - *Example Conflict*: Panderm says "Melanoma" (Stats) BUT DermoGPT argues for "Nevus" (Reasoning).
2. **Ground Truth Retrieval**:
   - Call `text_rag_retrieval` for the Top Candidate A.
   - Call `text_rag_retrieval` for the Top Candidate B.
3. **Final Synthesis**: Re-evaluate the **DermoGPT morphological description** against the **Text RAG criteria**.
4. **Conclusion**: Decide which diagnosis is scientifically supported by the features present.

## Output Format (Strict JSON)

Regardless of the phase, end your response with this JSON block:

```json
{
  "phase": "PHASE_1_INITIAL" or "PHASE_2_VERIFIED",
  "visual_analysis": {
    "panderm_prediction": {"disease": "...", "score": 0.XX},
    "dermogpt_assessment": {
      "morphology_summary": "...",
      "cot_reasoning_summary": "...",
      "selected_diagnosis": "..."
    },
    "rag_consistency": "Does the retrieved image match the candidates?"
  },
  "conflict_check": {
    "is_conflict": true/false,
    "conflict_details": "e.g., Panderm and DermoGPT disagree on..."
  },
  "final_diagnosis": "Disease Name"
}
```

FINAL_ANSWER: [exact disease name from options]
"""

# Captioning task ablation prompts registry (LOO entries auto-generated below)
CAPTIONING_ABLATION_PROMPTS: Dict[str, str] = {}  # populated by LOO auto-gen below

# =============================================================================
# Leave-One-Out (LOO) Ablation Prompt Builder for Captioning
# Dynamically generates prompts with one tool removed from the full 6-tool set.
# =============================================================================

_CAPTIONING_LOO_TOOL_DESCRIPTIONS = {
    "dermogpt_vqa": (
        "**dermogpt_vqa**: Dermatology-specialized vision-language model\n"
        "   - Input: image_path + query (for description tasks)\n"
        "   - Returns: Detailed morphological descriptions with clinical terminology\n"
        "   - Best for: Initial clinical description, lesion characterization"
    ),
    "make": (
        "**make_concept_annotator**: Dermoscopic concept extraction\n"
        "   - Returns: Detected features with confidence scores (pigment_network, dots_globules, etc.)\n"
        "   - Best for: Structured feature checklist, ensuring no features are missed"
    ),
    "panderm": (
        "**panderm_classifier**: Zero-shot skin disease classification\n"
        "   - Input: image_path + candidate_diseases (common skin conditions)\n"
        "   - Returns: Top predictions with confidence scores\n"
        "   - Best for: Diagnostic hypothesis to anchor clinical context"
    ),
    "ontology": (
        "**ontology_query**: Skin disease knowledge tree / taxonomy\n"
        '   - Input: disease_name + query_type ("hierarchy" / "children" / "siblings" / "search")\n'
        "   - Returns: Disease hierarchical classification, related subtypes, sibling conditions\n"
        "   - Best for: Understanding disease taxonomy, disambiguating similar conditions"
    ),
    "rag": (
        "**rag_retrieval**: Similar case retrieval from knowledge base\n"
        "   - Use IMAGE MODE ONLY: Provide 'image_path' to find visually similar cases\n"
        "   - DO NOT use text_query parameter\n"
        "   - Returns: Similar cases with diagnosis labels and clinical descriptions\n"
        "   - Best for: Reference description style and medical terminology"
    ),
    "text_rag": (
        '**text_rag_retrieval**: Dermatology guidelines retrieval (DermNet, Mayo Clinic)\n'
        '   - Input: query (text), search_mode="hybrid" (recommended)\n'
        "   - Returns: Clinical guidelines, disease descriptions, diagnostic criteria\n"
        "   - Best for: Accurate clinical terminology and disease-specific vocabulary"
    ),
}

_CAPTIONING_LOO_WORKFLOW_STEPS = {
    "dermogpt_vqa": (
        '**DermoGPT Description**: Call dermogpt_vqa with a descriptive query like:\n'
        '   "Describe this skin lesion in detail, including its morphology, color, '
        'borders, and surface characteristics."'
    ),
    "make": (
        "**Concept Extraction**: Call make_concept_annotator to get structured dermoscopic features\n"
        "   - This provides a checklist of detected concepts with confidence scores"
    ),
    "panderm": (
        "**Classification Hypothesis**: Call panderm_classifier with common skin conditions:\n"
        '   candidate_diseases="melanoma, basal cell carcinoma, squamous cell carcinoma, '
        'nevus, seborrheic keratosis, actinic keratosis, dermatofibroma, psoriasis, eczema, acne"'
    ),
    "ontology": (
        "**Disease Taxonomy**: Call ontology_query with the top predicted disease:\n"
        '   - Use query_type="hierarchy" to get the disease\'s position in the classification tree\n'
        '   - Use query_type="siblings" to identify related/differential diagnoses\n'
        "   - Use this to refine diagnostic context and mention disease category in the caption"
    ),
    "rag": (
        "**Similar Cases**: Call rag_retrieval with image_path ONLY (IMAGE MODE)\n"
        "   - DO NOT use text_query parameter\n"
        "   - Reference the description style and terminology from similar cases"
    ),
    "text_rag": (
        "**Guidelines Lookup**: Call text_rag_retrieval with a query about the suspected condition:\n"
        '   - Example: "clinical features and description of [suspected diagnosis]"\n'
        '   - Use search_mode="hybrid" for best results\n'
        "   - Extract accurate clinical terminology for the final caption"
    ),
}

_CAPTIONING_LOO_SYNTHESIS_BULLETS = {
    "dermogpt_vqa": "Integrate DermoGPT's morphological observations",
    "make": "Ensure ALL concepts from MAKE are mentioned",
    "panderm": "Use PanDerm classification as diagnostic anchor",
    "ontology": "Incorporate disease taxonomy context from ontology (disease category, related conditions)",
    "rag": "Use terminology consistent with similar RAG cases",
    "text_rag": "Incorporate guideline-based clinical vocabulary from Text RAG",
}

_CAPTIONING_LOO_FRIENDLY_NAMES = {
    "dermogpt_vqa": "DermoGPT (dermogpt_vqa)",
    "make": "MAKE (make_concept_annotator)",
    "panderm": "PanDerm (panderm_classifier)",
    "ontology": "Ontology (ontology_query)",
    "rag": "RAG (rag_retrieval)",
    "text_rag": "Text RAG (text_rag_retrieval)",
}

_ALL_CAPTIONING_TOOLS = ["dermogpt_vqa", "make", "ontology", "panderm", "rag", "text_rag"]


def _build_captioning_loo_prompt(removed_tool: str) -> str:
    """Build a captioning prompt with one tool removed (leave-one-out)."""
    remaining = [t for t in _ALL_CAPTIONING_TOOLS if t != removed_tool]

    # --- Available Tools section ---
    tool_lines = []
    for i, tool in enumerate(remaining, 1):
        tool_lines.append(f"{i}. {_CAPTIONING_LOO_TOOL_DESCRIPTIONS[tool]}")
    tools_section = "## Available Tools\n\n" + "\n\n".join(tool_lines)

    # --- Workflow steps ---
    workflow_lines = []
    for i, tool in enumerate(remaining, 1):
        workflow_lines.append(f"{i}. {_CAPTIONING_LOO_WORKFLOW_STEPS[tool]}")
    workflow_section = "\n\n".join(workflow_lines)

    # --- Synthesis bullets ---
    synthesis_lines = []
    for tool in remaining:
        synthesis_lines.append(f"- {_CAPTIONING_LOO_SYNTHESIS_BULLETS[tool]}")
    synthesis_section = "\n".join(synthesis_lines)

    friendly_name = _CAPTIONING_LOO_FRIENDLY_NAMES[removed_tool]

    return f"""You are DermAgent, an expert dermatology AI generating detailed clinical image descriptions.

{tools_section}

**Note**: {friendly_name} is NOT available in this session. Work with the remaining {len(remaining)} tools.

## Your Task
Generate a comprehensive, clinically accurate description of the skin lesion that includes morphological features, visual characteristics, and relevant clinical context.

## Required Reasoning Process

{workflow_section}

## Caption Synthesis
Combine all gathered information into a cohesive clinical description:
{synthesis_section}
- Include diagnostic hypothesis if supported by evidence

## Caption Structure (follow this order)
1. **Lesion Type & Location**: Primary morphology (papule, plaque, nodule, patch, etc.)
2. **Color**: Pigmentation patterns, color distribution, variations
3. **Surface**: Texture, scaling, crusting, ulceration, elevation
4. **Borders**: Regularity, definition, symmetry
5. **Size/Distribution**: If visible/relevant
6. **Clinical Context**: Suspected diagnosis with supporting features

## Few-Shot Examples

**Example 1** (Basal Cell Carcinoma):
"This is a picture showing a skin lesion on the face, with a black patch below the right eye, central ulceration, and shiny raised edges, consistent with basal cell carcinoma. Basal cell carcinoma is one of the most common types of skin cancer, typically occurring in sun-exposed areas such as the head, neck, and upper limbs."

**Example 2** (Melanoma):
"This photo shows a single black-brown patch with irregular borders and uneven color distribution, consistent with features of melanoma. Melanoma is a malignant skin tumor that typically develops from melanocytes and has the potential to metastasize, requiring prompt treatment and monitoring."

**Example 3** (Inflammatory Papules):
"This is a photo of the anterior chest showing multiple red papules that are partially fused. Based on the clinical presentation and characteristics of the lesions, the diagnosis is miliaria, a common skin condition that often occurs in infants, children, and adolescents."

## Quality Requirements
- Use professional dermatological terminology
- Be specific and descriptive (avoid vague terms)
- Mention both positive AND relevant negative findings
- Keep the description between 50-150 words
- Follow the structure demonstrated in the examples above

## Output Format

You MUST end your response with exactly:
```
FINAL_ANSWER: [Your complete clinical description here]
```

## CRITICAL RULES
- Call ALL {len(remaining)} available tools, then synthesize
- {friendly_name} is unavailable — do NOT attempt to call it
- Use professional dermatological terminology
"""


# Auto-generate leave-one-out prompts and register
for _removed in _ALL_CAPTIONING_TOOLS:
    _remaining = sorted(t for t in _ALL_CAPTIONING_TOOLS if t != _removed)
    _key = ",".join(_remaining)
    CAPTIONING_ABLATION_PROMPTS[_key] = _build_captioning_loo_prompt(_removed)

# -----------------------------------------------------------------------------
# Concept Annotation Task  
# -----------------------------------------------------------------------------
CONCEPT_SYSTEM_PROMPT = f"""You are DermAgent, an expert dermatology AI performing concept annotation.

{CONCEPT_TOOL_DESCRIPTION}

## Your Task
Identify ALL dermatological concepts present in the image from the provided concept list.

## Required Reasoning Process

1. **Concept Detection**: Call make_concept_annotator with the EXACT target_concepts 
   list provided in the "Valid Concept List" (pass as comma-separated string)
2. **Visual Verification**: Call dermogpt_vqa with target_concepts to describe features in detail
3. **Case Reference**: Call rag_retrieval with image_path ONLY (IMAGE MODE)
   - DO NOT use text_query parameter

4. **Validation** (CRITICAL):
   - For each concept from make_concept_annotator:
     - Is it mentioned/supported by DermoGPT description? → Keep
     - No visual evidence in description? → Remove (likely false positive)
   - For features in DermoGPT description not in make_concept output:
     - Does it match any concept in the provided list? → Add
   - Your final answer must only include concepts from the provided list

## Output Format

You MUST end your response with exactly:
```
FINAL_ANSWER: [concept1, concept2, concept3]
```
Or if no concepts apply:
```
FINAL_ANSWER: none
```
"""

# -----------------------------------------------------------------------------
# Captioning Task
# -----------------------------------------------------------------------------

# Captioning uses all 5 tools for comprehensive description generation
CAPTIONING_TOOL_DESCRIPTION = """
## Available Tools

1. **dermogpt_vqa**: Dermatology-specialized vision-language model
   - Input: image_path + query (for description tasks)
   - Returns: Detailed morphological descriptions with clinical terminology
   - Best for: Initial clinical description, lesion characterization

2. **make_concept_annotator**: Dermoscopic concept extraction
   - Returns: Detected features with confidence scores (pigment_network, dots_globules, etc.)
   - Best for: Structured feature checklist, ensuring no features are missed

3. **panderm_classifier**: Zero-shot skin disease classification
   - Input: image_path + candidate_diseases (common skin conditions)
   - Returns: Top predictions with confidence scores
   - Best for: Diagnostic hypothesis to anchor clinical context

4. **rag_retrieval**: Similar case retrieval from knowledge base
   - Use IMAGE MODE ONLY: Provide 'image_path' to find visually similar cases
   - DO NOT use text_query parameter
   - Returns: Similar cases with diagnosis labels and clinical descriptions
   - Best for: Reference description style and medical terminology

5. **text_rag_retrieval**: Dermatology guidelines retrieval (DermNet, Mayo Clinic)
   - Input: query (text), search_mode="hybrid" (recommended)
   - Returns: Clinical guidelines, disease descriptions, diagnostic criteria
   - Best for: Accurate clinical terminology and disease-specific vocabulary
"""

CAPTIONING_SYSTEM_PROMPT = f"""You are DermAgent, an expert dermatology AI generating detailed clinical image descriptions.

{CAPTIONING_TOOL_DESCRIPTION}

## Your Task
Generate a comprehensive, clinically accurate description of the skin lesion that includes morphological features, visual characteristics, and relevant clinical context.

## Required Multi-Stage Reasoning Process

### Stage 1: Visual Analysis
1. **DermoGPT Description**: Call dermogpt_vqa with a descriptive query like:
   "Describe this skin lesion in detail, including its morphology, color, borders, and surface characteristics."

2. **Concept Extraction**: Call make_concept_annotator to get structured dermoscopic features
   - This provides a checklist of detected concepts with confidence scores

### Stage 2: Diagnostic Context
3. **Classification Hypothesis**: Call panderm_classifier with common skin conditions:
   candidate_diseases="melanoma, basal cell carcinoma, squamous cell carcinoma, nevus, seborrheic keratosis, actinic keratosis, dermatofibroma, psoriasis, eczema, acne"

4. **Similar Cases**: Call rag_retrieval with image_path ONLY (IMAGE MODE)
   - DO NOT use text_query parameter
   - Reference the description style and terminology from similar cases

### Stage 3: Terminology Enhancement
5. **Guidelines Lookup**: Call text_rag_retrieval with a query about the suspected condition:
   - Example: "clinical features and description of [suspected diagnosis]"
   - Use search_mode="hybrid" for best results
   - Extract accurate clinical terminology for the final caption

### Stage 4: Caption Synthesis
Combine all gathered information into a cohesive clinical description:
- Integrate DermoGPT's morphological observations
- Ensure ALL concepts from MAKE are mentioned
- Use terminology consistent with similar RAG cases
- Incorporate guideline-based clinical vocabulary
- Include diagnostic hypothesis if supported by evidence

## Caption Structure (follow this order)
1. **Lesion Type & Location**: Primary morphology (papule, plaque, nodule, patch, etc.)
2. **Color**: Pigmentation patterns, color distribution, variations
3. **Surface**: Texture, scaling, crusting, ulceration, elevation
4. **Borders**: Regularity, definition, symmetry
5. **Size/Distribution**: If visible/relevant
6. **Clinical Context**: Suspected diagnosis with supporting features

## Few-Shot Examples

**Example 1** (Basal Cell Carcinoma):
"This is a picture showing a skin lesion on the face, with a black patch below the right eye, central ulceration, and shiny raised edges, consistent with basal cell carcinoma. Basal cell carcinoma is one of the most common types of skin cancer, typically occurring in sun-exposed areas such as the head, neck, and upper limbs."

**Example 2** (Melanoma):
"This photo shows a single black-brown patch with irregular borders and uneven color distribution, consistent with features of melanoma. Melanoma is a malignant skin tumor that typically develops from melanocytes and has the potential to metastasize, requiring prompt treatment and monitoring."

**Example 3** (Inflammatory Papules):
"This is a photo of the anterior chest showing multiple red papules that are partially fused. Based on the clinical presentation and characteristics of the lesions, the diagnosis is miliaria, a common skin condition that often occurs in infants, children, and adolescents."

## Quality Requirements
- Use professional dermatological terminology
- Be specific and descriptive (avoid vague terms)
- Mention both positive AND relevant negative findings
- Keep the description between 50-150 words
- Follow the structure demonstrated in the examples above

## Output Format

You MUST end your response with exactly:
```
FINAL_ANSWER: [Your complete clinical description here]
```
"""

# Captioning with all 6 tools (adds ontology to the standard 5-tool set)
CAPTIONING_TOOL_DESCRIPTION_WITH_ONTOLOGY = """
## Available Tools

1. **dermogpt_vqa**: Dermatology-specialized vision-language model
   - Input: image_path + query (for description tasks)
   - Returns: Detailed morphological descriptions with clinical terminology
   - Best for: Initial clinical description, lesion characterization

2. **make_concept_annotator**: Dermoscopic concept extraction
   - Returns: Detected features with confidence scores (pigment_network, dots_globules, etc.)
   - Best for: Structured feature checklist, ensuring no features are missed

3. **panderm_classifier**: Zero-shot skin disease classification
   - Input: image_path + candidate_diseases (common skin conditions)
   - Returns: Top predictions with confidence scores
   - Best for: Diagnostic hypothesis to anchor clinical context

4. **ontology_query**: Skin disease knowledge tree / taxonomy
   - Input: disease_name + query_type ("hierarchy" / "children" / "siblings" / "search")
   - Returns: Disease hierarchical classification, related subtypes, sibling conditions
   - Best for: Understanding disease taxonomy, disambiguating similar conditions, enriching diagnostic context

5. **rag_retrieval**: Similar case retrieval from knowledge base
   - Use IMAGE MODE ONLY: Provide 'image_path' to find visually similar cases
   - DO NOT use text_query parameter
   - Returns: Similar cases with diagnosis labels and clinical descriptions
   - Best for: Reference description style and medical terminology

6. **text_rag_retrieval**: Dermatology guidelines retrieval (DermNet, Mayo Clinic)
   - Input: query (text), search_mode="hybrid" (recommended)
   - Returns: Clinical guidelines, disease descriptions, diagnostic criteria
   - Best for: Accurate clinical terminology and disease-specific vocabulary
"""

CAPTIONING_SYSTEM_PROMPT_WITH_ONTOLOGY = f"""You are DermAgent, an expert dermatology AI generating detailed clinical image descriptions.

{CAPTIONING_TOOL_DESCRIPTION_WITH_ONTOLOGY}

## Your Task
Generate a comprehensive, clinically accurate description of the skin lesion that includes morphological features, visual characteristics, and relevant clinical context.

## Required Multi-Stage Reasoning Process

### Stage 1: Visual Analysis
1. **DermoGPT Description**: Call dermogpt_vqa with a descriptive query like:
   "Describe this skin lesion in detail, including its morphology, color, borders, and surface characteristics."

2. **Concept Extraction**: Call make_concept_annotator to get structured dermoscopic features
   - This provides a checklist of detected concepts with confidence scores

### Stage 2: Diagnostic Context
3. **Classification Hypothesis**: Call panderm_classifier with common skin conditions:
   candidate_diseases="melanoma, basal cell carcinoma, squamous cell carcinoma, nevus, seborrheic keratosis, actinic keratosis, dermatofibroma, psoriasis, eczema, acne"

4. **Disease Taxonomy**: Call ontology_query with the top predicted disease from panderm_classifier:
   - Use query_type="hierarchy" to get the disease's position in the classification tree
   - Use query_type="siblings" to identify related/differential diagnoses
   - Use this to refine diagnostic context and mention disease category in the caption

5. **Similar Cases**: Call rag_retrieval with image_path ONLY (IMAGE MODE)
   - DO NOT use text_query parameter
   - Reference the description style and terminology from similar cases

### Stage 3: Terminology Enhancement
6. **Guidelines Lookup**: Call text_rag_retrieval with a query about the suspected condition:
   - Example: "clinical features and description of [suspected diagnosis]"
   - Use search_mode="hybrid" for best results
   - Extract accurate clinical terminology for the final caption

### Stage 4: Caption Synthesis
Combine all gathered information into a cohesive clinical description:
- Integrate DermoGPT's morphological observations
- Ensure ALL concepts from MAKE are mentioned
- Use terminology consistent with similar RAG cases
- Incorporate guideline-based clinical vocabulary
- Incorporate disease taxonomy context from ontology (disease category, related conditions)
- Include diagnostic hypothesis if supported by evidence

## Caption Structure (follow this order)
1. **Lesion Type & Location**: Primary morphology (papule, plaque, nodule, patch, etc.)
2. **Color**: Pigmentation patterns, color distribution, variations
3. **Surface**: Texture, scaling, crusting, ulceration, elevation
4. **Borders**: Regularity, definition, symmetry
5. **Size/Distribution**: If visible/relevant
6. **Clinical Context**: Suspected diagnosis with supporting features

## Few-Shot Examples

**Example 1** (Basal Cell Carcinoma):
"This is a picture showing a skin lesion on the face, with a black patch below the right eye, central ulceration, and shiny raised edges, consistent with basal cell carcinoma. Basal cell carcinoma is one of the most common types of skin cancer, typically occurring in sun-exposed areas such as the head, neck, and upper limbs."

**Example 2** (Melanoma):
"This photo shows a single black-brown patch with irregular borders and uneven color distribution, consistent with features of melanoma. Melanoma is a malignant skin tumor that typically develops from melanocytes and has the potential to metastasize, requiring prompt treatment and monitoring."

**Example 3** (Inflammatory Papules):
"This is a photo of the anterior chest showing multiple red papules that are partially fused. Based on the clinical presentation and characteristics of the lesions, the diagnosis is miliaria, a common skin condition that often occurs in infants, children, and adolescents."

## Quality Requirements
- Use professional dermatological terminology
- Be specific and descriptive (avoid vague terms)
- Mention both positive AND relevant negative findings
- Keep the description between 50-150 words
- Follow the structure demonstrated in the examples above

## Output Format

You MUST end your response with exactly:
```
FINAL_ANSWER: [Your complete clinical description here]
```
"""

# -----------------------------------------------------------------------------
# Captioning Task with Critic (Autonomous + Structured Output)
# Uses all 6 tools, DermoGPT CoT query, dual-topic Text RAG, 4-part output
# Activated when enable_critic=True for captioning task
# -----------------------------------------------------------------------------

CAPTIONING_CRITIC_TOOL_DESCRIPTION = """
## Available Tools

1. **dermogpt_vqa**: Dermatology-specialized vision-language model for clinical reasoning
   - Input: image_path + query
   - **PROMPT STRATEGY**: Pass a Chain-of-Thought (CoT) query that asks for morphological
     description AND step-by-step clinical reasoning to identify the most likely diagnosis.
   - Suggested query: "Describe the lesion's morphology (color, structure, border, surface
     texture) and provide a step-by-step clinical reasoning to identify the most likely diagnosis."
   - Returns: Detailed morphological description with diagnostic reasoning
   - Best for: Primary clinical description with CoT diagnostic logic

2. **make_concept_annotator**: Dermoscopic concept extraction
   - Returns: Detected features with confidence scores (pigment_network, dots_globules, etc.)
   - Best for: Structured feature checklist, ensuring no features are missed

3. **panderm_classifier**: Zero-shot skin disease classification
   - Input: image_path + candidate_diseases (common skin conditions)
   - Returns: Top predictions with confidence scores
   - Best for: Statistical diagnostic hypothesis

4. **ontology_query**: Skin disease knowledge tree / taxonomy
   - Input: disease_name + query_type ("hierarchy" / "children" / "siblings" / "search")
   - Returns: Disease hierarchical classification, related subtypes, sibling conditions
   - Best for: Understanding disease taxonomy, enriching diagnostic context

5. **rag_retrieval**: Similar case retrieval from knowledge base
   - Use IMAGE MODE ONLY: Provide 'image_path' to find visually similar cases
   - DO NOT use text_query parameter
   - Returns: Similar cases with diagnosis labels and clinical descriptions
   - Best for: Visual confirmation, reference description style

6. **text_rag_retrieval**: Dermatology guidelines retrieval (DermNet, Mayo Clinic)
   - Input: query (text), search_mode="hybrid" (recommended)
   - Returns: Clinical guidelines, disease descriptions, diagnostic criteria, treatment info
   - **CRITICAL**: You MUST make TWO separate calls:
     - Call 1: Query for diagnosis/clinical features, e.g. "clinical features and diagnostic criteria for [disease]"
     - Call 2: Query for treatment, e.g. "standard treatment for [disease]"
   - Both diagnosis AND treatment information are required for the final caption
"""

CAPTIONING_SYSTEM_PROMPT_WITH_CRITIC = f"""You are DermAgent, an expert dermatology AI generating detailed clinical image descriptions with diagnostic and treatment context.

{CAPTIONING_CRITIC_TOOL_DESCRIPTION}

## Your Task
Generate a clinically accurate, structured caption that includes visual evidence, diagnosis, disease definition, and treatment information.

## Workflow

You are free to call tools in any order based on your clinical judgment. ALL tools must be used.

### Recommended Approach
1. **DermoGPT CoT Analysis**: Call dermogpt_vqa with a CoT query:
   query: "Describe the lesion's morphology (color, structure, border, surface texture) and provide a step-by-step clinical reasoning to identify the most likely diagnosis."

2. **Concept Extraction**: Call make_concept_annotator to get structured dermoscopic features

3. **Classification**: Call panderm_classifier with common skin conditions:
   candidate_diseases="melanoma, basal cell carcinoma, squamous cell carcinoma, nevus, seborrheic keratosis, actinic keratosis, dermatofibroma, psoriasis, eczema, acne"

4. **Disease Taxonomy**: Call ontology_query with the top predicted disease to understand its classification

5. **Visual Confirmation**: Call rag_retrieval with image_path ONLY (IMAGE MODE)

6. **Guidelines Retrieval** (TWO calls required):
   - Call text_rag_retrieval with: "clinical features and diagnostic criteria for [predicted disease]"
   - Call text_rag_retrieval with: "standard treatment for [predicted disease]"

### Synthesis
After gathering all evidence, determine the most likely diagnosis (Predicted_Class_Name) and write the caption.

## Output Format

You MUST end your response with a caption following this exact 4-part structure:

```
FINAL_ANSWER: [Caption with ALL four parts below, written as a cohesive paragraph]
```

**Part 1 - Visual Evidence**: Describe the visual evidence in the image that supports [Predicted_Class_Name]. Use clinical terms (e.g., 'erythema', 'scaling', 'papules', 'irregular borders').
**Part 2 - Diagnosis Statement**: State that this is consistent with [Predicted_Class_Name].
**Part 3 - Disease Definition**: Write a standard definition of [Predicted_Class_Name] (start with '[Name] is a...').
**Part 4 - Treatment**: Provide standard treatment advice for [Predicted_Class_Name].

## Few-Shot Examples

**Example 1** (Basal Cell Carcinoma):
"This image shows a skin lesion on the face with a pearly, translucent nodule exhibiting central ulceration, rolled borders, and telangiectasia, consistent with basal cell carcinoma. Basal cell carcinoma is a common malignant neoplasm arising from basal cells of the epidermis, typically occurring in sun-exposed areas. Treatment options include surgical excision, Mohs micrographic surgery, topical imiquimod, and photodynamic therapy, with early intervention yielding excellent prognosis."

**Example 2** (Psoriasis):
"The image reveals well-demarcated erythematous plaques covered with silvery-white scales on the extensor surfaces, consistent with psoriasis. Psoriasis is a chronic autoimmune inflammatory skin disease characterized by hyperproliferation of keratinocytes, affecting approximately 2-3%% of the population. Treatment includes topical corticosteroids and vitamin D analogues for mild cases, phototherapy for moderate disease, and systemic agents such as methotrexate or biologics for severe cases."

**Example 3** (Melanoma):
"This photo shows an asymmetric pigmented lesion with irregular borders, color variegation including shades of brown, black, and blue, and a diameter exceeding 6mm, consistent with melanoma. Melanoma is a malignant tumor arising from melanocytes, with potential for metastasis if not detected early. Treatment involves wide local excision, sentinel lymph node biopsy for staging, and adjuvant immunotherapy or targeted therapy for advanced stages."

## Quality Requirements
- Use professional dermatological terminology
- ALL four parts must be present in the caption
- Keep the total caption between 60-180 words
- Ground visual descriptions in actual tool outputs (DermoGPT, MAKE)
- Use guideline-based terminology from Text RAG results
"""


# Master registry: task_type -> ablation_key -> prompt
ABLATION_PROMPTS_BY_TASK = {
    "captioning": CAPTIONING_ABLATION_PROMPTS,
}


# =============================================================================
# Task Configuration
# =============================================================================

TASK_PROMPTS = {
    "diagnosis": DIAGNOSIS_SYSTEM_PROMPT,
    "concept": CONCEPT_SYSTEM_PROMPT,
    "captioning": CAPTIONING_SYSTEM_PROMPT,
}

# DATASET_CONFIGS is imported from .configs to avoid circular imports


# =============================================================================
# Agent State
# =============================================================================

class BenchmarkAgentState(TypedDict):
    """State for benchmark evaluation."""

    messages: Annotated[List[AnyMessage], operator.add]
    current_image: Optional[str]
    task_type: Optional[str]  # diagnosis, concept, captioning
    dataset: Optional[str]
    options: Optional[List[str]]  # For diagnosis MCQ
    concepts: Optional[List[str]]  # For concept annotation
    tool_call_count: int  # Counter for max tool calls limit
    # Critic node state
    critic_retry_count: int  # Number of retries triggered by critic
    tools_called: List[str]  # Track which tools were called
    last_critic_feedback: Optional[str]  # Feedback message from critic for retry
    # Evidence summaries (when critic_llm_summary=True)
    evidence_summaries: List[Dict[str, Any]]  # [{summary, duration_ms, timestamp}, ...]
    # Image reinject tracking (when critic_image_reinject=True)
    image_reinject_count: int  # Number of times image was reinjected
    # Tool call caching (prevent duplicate calls with same args)
    tool_call_cache: Dict[str, str]  # {cache_key -> result}
    # Text RAG query tracking (for conflict investigation check)
    text_rag_queries: List[str]  # List of all text_rag queries made
    # Narrowed re-diagnosis (Task 1 diagnosis critic only)
    narrowed_candidates: Optional[List[str]]  # Critic-generated shortlist for focused re-diagnosis


# =============================================================================
# Critic Node Configuration
# =============================================================================

# Confidence thresholds for triggering cross-validation
CONFIDENCE_THRESHOLDS = {
    "panderm": 0.90,              # PanDerm classification confidence
    "make": 0.70,                 # MAKE concept confidence (top-1)
    "rag": 0.80,                  # RAG retrieval similarity
    "make_borderline_low": 0.35,  # Lower bound for borderline concept zone
    "make_borderline_high": 0.60, # Upper bound for borderline concept zone
}

# Task-aware ALL_TOOLS sets (dermogpt_vqa for concept/captioning)
ALL_TOOLS_BY_TASK = {
    "diagnosis": {"panderm_classifier", "make_concept_annotator",
                  "rag_retrieval", "text_rag_retrieval", "ontology_query"},
    "concept": {"panderm_classifier", "make_concept_annotator", "dermogpt_vqa",
                "rag_retrieval", "text_rag_retrieval", "ontology_query"},
    "captioning": {"panderm_classifier", "make_concept_annotator", "dermogpt_vqa",
                   "rag_retrieval", "text_rag_retrieval", "ontology_query"},
}

# Tool requirements by question/task type
# Category/Malignancy ALWAYS require text_rag, regardless of other tool confidence
REQUIRED_TOOLS_BY_TYPE = {
    "category": ["text_rag_retrieval"],          # ALWAYS required for category questions
    "malignancy": ["text_rag_retrieval"],        # ALWAYS required for malignancy questions
    "diagnosis": ["panderm_classifier"],         # At least panderm for diagnosis
    "concept": ["make_concept_annotator", "dermogpt_vqa"],  # Both needed for cross-validation
    "captioning": ["dermogpt_vqa", "make_concept_annotator", "panderm_classifier",
                   "rag_retrieval", "text_rag_retrieval", "ontology_query"],
}

MAX_CRITIC_RETRIES = 2


# =============================================================================
# Tool Call Caching Helpers
# =============================================================================

def _make_cache_key(tool_name: str, args: Dict[str, Any]) -> str:
    """
    Generate a unique cache key for a tool call.
    
    For image-based tools, only use image_path (and dataset for panderm) as key,
    since same image will always produce same result.
    For text-based tools, use all args.
    """
    import hashlib
    import json
    
    # Image-based tools: same image = same result (no need to re-run)
    if tool_name in ["panderm_classifier", "make_concept_annotator", "rag_retrieval", "dermogpt_vqa"]:
        key_args = {"image_path": args.get("image_path", "")}
        # For panderm, also include dataset since it affects classification options
        if tool_name == "panderm_classifier":
            key_args["dataset"] = args.get("dataset")
            key_args["candidate_diseases"] = args.get("candidate_diseases")
        # For dermogpt, include candidate_diseases only when present
        # (narrowed re-diagnosis changes this param; omitting when None keeps hash identical)
        elif tool_name == "dermogpt_vqa":
            cd = args.get("candidate_diseases")
            if cd is not None:
                key_args["candidate_diseases"] = cd
    else:
        # Text-based tools: use all args
        key_args = {k: v for k, v in args.items() if v is not None}
    
    args_str = json.dumps(key_args, sort_keys=True)
    args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
    return f"{tool_name}:{args_hash}"


def _check_conflict_investigated_v1(queries: List[str], diagnoses: List[str]) -> bool:
    """
    (V1 - kept for backward compatibility)
    Check if a text_rag query has been made that investigates the conflict
    between the given diagnoses.
    
    A conflict is considered "investigated" if there exists a query that
    contains BOTH conflicting diagnosis names.
    """
    if len(diagnoses) < 2:
        return False
    
    for query in queries:
        query_lower = query.lower()
        # Check if query contains both diagnoses
        if all(d.lower() in query_lower for d in diagnoses[:2]):
            return True
    return False


def _check_conflict_investigated(queries: List[str], diagnoses: List[str]) -> bool:
    """
    Check if a text_rag query has been made that investigates the conflict
    between the given diagnoses.
    
    Uses word-level fuzzy matching: a conflict is "investigated" if a query
    contains at least one key word from EACH conflicting diagnosis name.
    E.g. "melanocytic nevi" matches query containing "melanocytic nevus"
    because the word "melanocytic" appears.
    """
    if len(diagnoses) < 2 or not queries:
        return False
    
    for query in queries:
        query_lower = query.lower()
        # Method 1: Both full names present (original strict match)
        if all(d.lower() in query_lower for d in diagnoses[:2]):
            return True
        # Method 2: Word-level match — at least one key word from EACH diagnosis
        matches = 0
        for d in diagnoses[:2]:
            d_words = set(d.lower().split())
            if any(w in query_lower for w in d_words):
                matches += 1
        if matches >= 2:
            return True
    return False


def _similar_query_exists_v1(existing_queries: List[str], new_query: str) -> bool:
    """
    (V1 - kept for backward compatibility)
    Check if a similar query already exists in the list.
    Uses exact match (case-insensitive).
    """
    new_lower = new_query.lower().strip()
    for q in existing_queries:
        if q.lower().strip() == new_lower:
            return True
    return False


def _similar_query_exists(existing_queries: List[str], new_query: str) -> bool:
    """
    Check if a similar query already exists in the list.
    Uses word-level overlap (>60% shared words = similar).
    """
    new_words = set(new_query.lower().split())
    if not new_words:
        return False
    for q in existing_queries:
        q_words = set(q.lower().split())
        overlap = len(new_words & q_words) / max(len(new_words), 1)
        if overlap > 0.6:
            return True
    return False


def _check_tool_returned_error(tool_outputs: Dict[str, str], tool_name: str) -> bool:
    """Check if a tool's output indicates an error (e.g., CUDA OOM)."""
    output = tool_outputs.get(tool_name, "")
    if not output:
        return False
    return bool(re.search(r'\bError\b', output)) or "CUDA error" in output


def _fuzzy_match_option(
    candidate: str, options: List[str], threshold: float = 0.6
) -> Optional[str]:
    """
    Match a candidate disease name to the closest option via substring + word overlap.

    Used by the narrowed re-diagnosis critic to map RAG/DermoGPT disease names
    back to the canonical options list (e.g., SNU 134 classes).

    Priority:
        1. Exact case-insensitive match or substring containment
        2. Word-level overlap above *threshold*

    Returns:
        Matched option string, or None if no match above threshold.
    """
    cand_lower = candidate.lower().strip()
    if not cand_lower:
        return None
    # Priority 1: exact or substring containment
    for opt in options:
        opt_lower = opt.lower()
        if cand_lower == opt_lower or cand_lower in opt_lower or opt_lower in cand_lower:
            return opt
    # Priority 2: word overlap
    candidate_words = set(cand_lower.split())
    best_match: Optional[str] = None
    best_score = 0.0
    for opt in options:
        opt_words = set(opt.lower().split())
        if not candidate_words or not opt_words:
            continue
        overlap = len(candidate_words & opt_words) / max(
            len(candidate_words), len(opt_words)
        )
        if overlap > best_score:
            best_score = overlap
            best_match = opt
    return best_match if best_score >= threshold else None


# =============================================================================
# Evidence Summary Prompt for Critic Retry (Optional Feature)
# =============================================================================

EVIDENCE_SUMMARY_PROMPT = """You are a dermatology AI assistant tasked with summarizing diagnostic evidence for a retry analysis.

Given the following tool outputs and critic feedback, generate a concise professional summary.

## Tool Outputs
{tool_outputs}

## Critic Issues
{critic_issues}

---

Generate a structured summary in the following format (be concise, max 300 words total):

## Key Evidence
- [Tool Name]: [Most important finding, e.g., "Top prediction: melanoma (78%)"]
- [Tool Name]: [Key feature detected, e.g., "Asymmetric pigment network, blue-whitish veil"]
...

## Reasoning Chain
1. [First observation from tools]
2. [How observations connect]
3. [Preliminary conclusion and confidence level]

## Specific Concerns
- [Concern 1: e.g., "Low confidence (45%) suggests ambiguity between nevus and melanoma"]
- [Concern 2: e.g., "Tool conflict: PanDerm suggests X, RAG suggests Y"]
- [Recommended action: e.g., "Cross-validate with text_rag for differential diagnosis criteria"]

---
Output ONLY the structured summary, no additional commentary."""


# =============================================================================
# Tool Output Parser
# =============================================================================

class ToolOutputParser:
    """Parse confidence scores from tool outputs."""

    @staticmethod
    def parse_panderm_confidence(output: str) -> Optional[float]:
        """
        Parse PanDerm classifier output for max confidence.
        Expected format: "1. disease_name: 85.5%" or similar ranking
        """
        if not output:
            return None
        # Match patterns like "1. nevus: 85.5%" or "1. melanoma: 0.855"
        patterns = [
            r"1\.\s*[^:]+:\s*([\d.]+)%",           # "1. disease: 85.5%"
            r"1\.\s*[^:]+:\s*([\d.]+)\s*\(",       # "1. disease: 0.855 (..."
            r"Top-1[^:]*:\s*[^:]+:\s*([\d.]+)%",   # "Top-1: disease: 85.5%"
            r"confidence[^:]*:\s*([\d.]+)%",       # "confidence: 85.5%"
            r"confidence[^:]*:\s*([\d.]+)",        # "confidence: 0.855"
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                # Normalize to 0-1 scale if percentage
                if value > 1:
                    value = value / 100
                return value
        return None

    @staticmethod
    def parse_make_confidence(output: str) -> Optional[float]:
        """
        Parse MAKE concept annotator output for max confidence.
        Expected format: "1. concept_name: 0.85" (0-1 scale)
        """
        if not output:
            return None
        # Match patterns like "1. pigment_network: 0.85"
        patterns = [
            r"1\.\s*[^:]+:\s*([\d.]+)",            # "1. concept: 0.85"
            r"score[^:]*:\s*([\d.]+)",             # "score: 0.85"
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if value <= 1:  # Already in 0-1 scale
                    return value
        return None

    @staticmethod
    def parse_rag_similarity(output: str) -> Optional[float]:
        """
        Parse RAG retrieval output for similarity score.
        Expected format: "Similarity Score: 0.85" or "similarity: 0.85"
        """
        if not output:
            return None
        patterns = [
            r"[Ss]imilarity\s*[Ss]core[^:]*:\s*([\d.]+)",  # "Similarity Score: 0.85"
            r"[Ss]imilarity[^:]*:\s*([\d.]+)",             # "similarity: 0.85"
            r"[Ss]core[^:]*:\s*([\d.]+)",                  # "score: 0.85"
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if value <= 1:  # Valid similarity score
                    return value
        return None

    @staticmethod
    def extract_diagnosis_from_tool(tool_name: str, output: str) -> Optional[str]:
        """Extract the top diagnosis/prediction from tool output."""
        if not output:
            return None

        if tool_name == "panderm_classifier":
            # Match "1. disease_name: score"
            match = re.search(r"1\.\s*([^:]+):", output)
            if match:
                return match.group(1).strip().lower()
        elif tool_name == "rag_retrieval":
            # Match "Diagnosis: disease_name" or "Label: disease_name"
            match = re.search(r"(?:Diagnosis|Label)[^:]*:\s*([^\n,]+)", output, re.IGNORECASE)
            if match:
                return match.group(1).strip().lower()

        return None

    @staticmethod
    def extract_all_diagnoses_from_rag(output: str) -> List[str]:
        """
        Extract ALL disease labels from RAG retrieval output.
        
        RAG output format example:
            Result 1:
              Similarity Score: 0.7232
              Disease Label: human sporotrichosis
            Result 2:
              Similarity Score: 0.7212
              Disease Label: irritated seborrheic keratosis
            ...
        
        Returns:
            List of disease labels in order of similarity (highest first)
        """
        if not output:
            return []
        
        diagnoses = []
        # Match all "Disease Label: xxx" patterns
        pattern = r"Disease\s+Label[^:]*:\s*([^\n]+)"
        matches = re.findall(pattern, output, re.IGNORECASE)
        
        for match in matches:
            # Clean up the diagnosis string
            diagnosis = match.strip().lower()
            # Remove any trailing parentheses or annotations like '(from "sk/isk")'
            diagnosis = re.sub(r'\s*\([^)]*\)\s*$', '', diagnosis).strip()
            if diagnosis:
                diagnoses.append(diagnosis)
        
        return diagnoses

    # -----------------------------------------------------------------
    # Narrowed re-diagnosis helpers (Task 1 critic)
    # -----------------------------------------------------------------

    @staticmethod
    def extract_all_panderm_predictions(output: str) -> List[Tuple[str, float]]:
        """
        Parse ALL top-k predictions with non-zero confidence from PanDerm output.

        PanDerm outputs top-5 in format: '1. disease: 85.50%'
        For 134 classes, softmax may produce 0.00% for tail predictions.

        Returns:
            List of (disease_name, confidence) tuples — only entries with
            confidence > 0, sorted by confidence descending.
        """
        if not output:
            return []
        results: List[Tuple[str, float]] = []
        for match in re.finditer(r"\d+\.\s*([^:]+):\s*([\d.]+)%", output):
            disease = match.group(1).strip()
            try:
                confidence = float(match.group(2)) / 100
            except ValueError:
                continue
            if confidence > 0:
                results.append((disease, confidence))
        return results

    @staticmethod
    def extract_dermogpt_diagnosis(output: str) -> Optional[str]:
        """
        Extract the predicted diagnosis from DermoGPT output.

        Handles:
        - XML format: <final_diagnosis>disease</final_diagnosis>
        - Free-text: "diagnosis: disease" or "consistent with disease"

        Returns:
            Extracted disease name string, or None if not found.
        """
        if not output:
            return None
        # XML format from DermoGPT diagnosis mode
        match = re.search(
            r"<final_diagnosis>\s*(.+?)\s*</final_diagnosis>",
            output, re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        # Free-text fallback
        match = re.search(
            r"(?:diagnosis|consistent with)[:\s]+([A-Za-z\s]+?)(?:\.|,|$)",
            output, re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return None

    # -----------------------------------------------------------------
    # Concept annotation helpers (Task 2)
    # -----------------------------------------------------------------

    @staticmethod
    def parse_make_all_concepts(output: str) -> List[Tuple[str, float]]:
        """
        Parse ALL concept-score pairs from MAKE output.

        Expected format lines: "1. pigment_network: 0.7069"

        Returns:
            List of (concept_name, score) tuples sorted by score descending.
        """
        if not output:
            return []
        results: List[Tuple[str, float]] = []
        for match in re.finditer(r"\d+\.\s*([^:]+):\s*([\d.]+)", output):
            concept = match.group(1).strip()
            try:
                score = float(match.group(2))
            except ValueError:
                continue
            if 0 <= score <= 1:
                results.append((concept, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def parse_make_borderline_concepts(
        output: str,
        low: float = 0.35,
        high: float = 0.60,
    ) -> List[Tuple[str, float]]:
        """
        Return concepts whose MAKE score falls in the borderline zone [low, high].

        These are ambiguous and may benefit from visual verification.
        """
        all_concepts = ToolOutputParser.parse_make_all_concepts(output)
        return [(c, s) for c, s in all_concepts if low <= s <= high]

    @staticmethod
    def check_dermogpt_concept_contradiction(
        dermogpt_output: str,
        concept: str,
    ) -> bool:
        """
        Check if DermoGPT explicitly denies a concept.

        Looks for patterns like:
          - "pigment network is absent"
          - "no pigment network"
          - "pigment network: not observed"
          - "A distinct pigment network is absent"

        Returns True if a contradiction is found.
        """
        if not dermogpt_output or not concept:
            return False
        output_lower = dermogpt_output.lower()
        concept_lower = concept.lower().replace("_", " ")
        # Pattern 1: "X is absent / not present / not observed / not seen"
        if re.search(
            rf"{re.escape(concept_lower)}\s+(?:is\s+)?(?:absent|not\s+(?:present|observed|seen|detected|visible|identified))",
            output_lower,
        ):
            return True
        # Pattern 2: "no X" / "no distinct X" / "absence of X"
        if re.search(
            rf"(?:no|absence\s+of)\s+(?:\w+\s+)?{re.escape(concept_lower)}",
            output_lower,
        ):
            return True
        return False


# =============================================================================
# Answer Parser
# =============================================================================

class AnswerParser:
    """Extract and parse FINAL_ANSWER from agent response."""

    @staticmethod
    def extract(response: str) -> Optional[str]:
        """Extract content after FINAL_ANSWER:

        This method first tries the traditional FINAL_ANSWER: format.
        If not found, it falls back to V2 JSON format (for V2 prompt compatibility).
        V1 behavior is preserved: traditional format is always checked first.
        """
        # 1. Try traditional FINAL_ANSWER: format first (V1 behavior unchanged)
        patterns = [
            r"FINAL_ANSWER:\s*\[([^\]]+)\]",
            r"FINAL_ANSWER:\s*(.+?)(?:\n|$)",
            r"\*\*FINAL_ANSWER\*\*:\s*(.+?)(?:\n|$)",
            # Fallback: model sometimes writes "Final Answer:" (space) instead of "FINAL_ANSWER:" (underscore)
            r"#{1,3}\s*Final\s+Answer:?\s*\n+(.+?)(?:\n|$)",   # ### Final Answer:\n<text>
            r"Final\s+Answer:\s*(.+?)(?:\n|$)",                  # Final Answer: <text>
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip().strip("`\"'")

        # 2. Try V2 JSON format (fallback, only when V1 format not found)
        v2_diagnosis = AnswerParser._extract_from_v2_json(response)
        if v2_diagnosis:
            return v2_diagnosis

        return None

    @staticmethod
    def _extract_from_v2_json(response: str) -> Optional[str]:
        """Extract final_diagnosis from V2 JSON block.

        This is a fallback method for V2 prompt compatibility.
        V1 responses won't have JSON blocks, so this won't affect V1 behavior.
        """
        # Extract ```json ... ``` block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "final_diagnosis" in data:
                    return data["final_diagnosis"]
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def parse_v2_json(response: str) -> Optional[Dict[str, Any]]:
        """Parse full V2 structured JSON output for detailed analysis.

        Returns the complete JSON structure for critic evaluation or logging.
        Returns None if no valid JSON block found.
        """
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def parse_diagnosis(response: str, options: List[str]) -> Optional[str]:
        """Parse diagnosis answer, match to valid option."""
        answer = AnswerParser.extract(response)
        if not answer:
            return None
        
        answer_lower = answer.lower().strip()
        
        # Exact match
        for opt in options:
            if opt.lower() == answer_lower:
                return opt
        
        # Partial match
        for opt in options:
            if opt.lower() in answer_lower or answer_lower in opt.lower():
                return opt
        
        return None
    
    @staticmethod
    def parse_concepts(response: str, valid_concepts: List[str]) -> List[str]:
        """Parse concept list answer."""
        answer = AnswerParser.extract(response)
        if not answer or answer.lower() == "none":
            return []
        
        raw = re.split(r"[,;\n]", answer)
        raw = [c.strip().lower() for c in raw if c.strip()]
        
        valid_map = {c.lower(): c for c in valid_concepts}
        matched = []
        
        for r in raw:
            if r in valid_map and valid_map[r] not in matched:
                matched.append(valid_map[r])
            else:
                for vl, vo in valid_map.items():
                    if (r in vl or vl in r) and vo not in matched:
                        matched.append(vo)
                        break
        
        return matched
    
    @staticmethod
    def parse_caption(response: str) -> str:
        """Parse captioning answer."""
        answer = AnswerParser.extract(response)
        return answer if answer else response.strip()
    

# =============================================================================
# Benchmark Agent
# =============================================================================

def create_tools(
    device: str = "cuda",
    enabled_tools: List[str] = None,
    use_flash_attn: bool = False,
    preload: bool = True,
    qdrant_path: str = "./qdrant_storage",
) -> List:
    """
    Create and initialize dermatology tools.
    
    Args:
        device: Device for inference ('cuda' or 'cpu')
        enabled_tools: List of tool names to enable. Options:
            - "panderm": DermLIP zero-shot classification
            - "make": MAKE concept annotation
            - "dermogpt_vqa": DermoGPT dermatology-specialized VQA
            - "rag": Qdrant multimodal retrieval (image and text query modes)
            - "text_rag": Text retrieval (DermNet, Mayo Clinic)
            - "ontology": Disease hierarchy knowledge tree query (fuzzy search, hierarchy lookup)
            If None, enables all tools.
        use_flash_attn: Use Flash Attention 2 (requires flash-attn installation)
        preload: If True, sequentially preload all tool models (recommended to avoid hangs)
        qdrant_path: Path to Qdrant local storage (default: ./qdrant_storage)
    
    Returns:
        List of initialized tools
    """
    # Enable all tools by default
    if enabled_tools is None:
        enabled_tools = ["panderm", "make", "rag"]
    
    tools = []
    dermogpt_tool = None
    
    # Create tool instances (without loading models)
    if "panderm" in enabled_tools:
        print("[Tool Creation] Creating PanDermTool...")
        tools.append(PanDermTool(device=device))
    
    if "make" in enabled_tools:
        print("[Tool Creation] Creating MAKETool...")
        tools.append(MAKETool(device=device))
    
    if "dermogpt_vqa" in enabled_tools:
        print(f"[Tool Creation] Creating DermoGPTTool (flash_attn={use_flash_attn})...")
        dermogpt_tool = DermoGPTTool(device=device, use_flash_attn=use_flash_attn)
        tools.append(dermogpt_tool)
    
    if "rag" in enabled_tools:
        print("[Tool Creation] Creating RAGTool (multimodal: image + text)...")
        rag_tool = RAGTool(device=device)
        rag_tool.qdrant_path = qdrant_path
        tools.append(rag_tool)
    
    if "text_rag" in enabled_tools:
        print("[Tool Creation] Creating TextRAGTool (hybrid search: vector + keyword)...")
        tools.append(TextRAGTool(device=device))
    
    if "ontology" in enabled_tools:
        print("[Tool Creation] Creating OntologyTool (disease hierarchy knowledge tree)...")
        tools.append(OntologyTool())  # No GPU needed, pure JSON data
    
    # Sequentially preload all tool models
    if preload:
        print("\n" + "=" * 60)
        print("[Sequential Preload] Loading tool models one by one...")
        print("=" * 60)
        preload_tools_sequential(tools, verbose=True)
        print("=" * 60)
        print("[Sequential Preload] All tools loaded.")
        print("=" * 60 + "\n")
    
    return tools


# =============================================================================
# Sequential Tool Node (for serial execution with memory management)
# =============================================================================

def create_sequential_tool_node(
    tools: List,
    memory_manager: Optional[ToolMemoryManager] = None,
):
    """
    Create a tool node that executes tools sequentially instead of in parallel.
    
    This avoids GPU resource contention when multiple GPU-intensive tools
    (e.g., DermoGPT + RAG encoder) are called in the same LLM response.
    
    Args:
        tools: List of tool instances
        memory_manager: Optional ToolMemoryManager for LRU-based model management.
                       If provided, ensures each tool is loaded before execution
                       and manages GPU memory via LRU eviction.
    
    Returns:
        A function that can be used as a node in StateGraph
        
    Example:
        # Without memory management (tools stay loaded)
        tool_node = create_sequential_tool_node(tools)
        
        # With LRU memory management
        manager = ToolMemoryManager(max_loaded=2)
        tool_node = create_sequential_tool_node(tools, memory_manager=manager)
    """
    tool_map = {t.name: t for t in tools}
    
    def sequential_tool_node(state: "BenchmarkAgentState") -> Dict[str, Any]:
        """Execute tools sequentially with optional memory management."""
        last_message = state["messages"][-1]
        
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": []}
        
        tool_messages = []
        num_calls = len(last_message.tool_calls)
        
        for i, tool_call in enumerate(last_message.tool_calls):
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")
            
            tool = tool_map.get(tool_name)
            if tool is None:
                # Unknown tool
                tool_messages.append(ToolMessage(
                    content=f"Error: Unknown tool '{tool_name}'. Available: {list(tool_map.keys())}",
                    tool_call_id=tool_id,
                    name=tool_name,
                ))
                continue
            
            # Ensure tool is loaded (with LRU eviction if memory manager is active)
            if memory_manager is not None:
                memory_manager.ensure_loaded(tool)
            
            # Execute the tool
            try:
                result = tool.invoke(tool_args)
            except Exception as e:
                result = f"Error executing {tool_name}: {type(e).__name__}: {str(e)}"
            
            tool_messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_id,
                name=tool_name,
            ))
        
        # Update tool call counter
        current_count = state.get("tool_call_count", 0)
        
        return {
            "messages": tool_messages,
            "tool_call_count": current_count + num_calls,
        }
    
    return sequential_tool_node


def create_benchmark_agent(
    tools: List = None,
    model_name: str = "gpt-4o",
    device: str = "cuda",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    enabled_tools: List[str] = None,
    recursion_limit: int = 30,
    max_tool_calls: int = 10,
    execution_mode: str = "sequential",
    memory_management: bool = False,
    max_loaded_models: int = 2,
    # Critic configuration (generalizable for all tasks)
    enable_critic: bool = None,  # None = default False
    critic_llm_summary: bool = False,
    critic_image_reinject: bool = False,
    max_critic_retries: int = None,  # None = use default MAX_CRITIC_RETRIES
    # Prompt variant (V2 support)
    prompt_variant: str = "default",  # "default" or "v2"
) -> StateGraph:
    """
    Create agent optimized for benchmark evaluation.
    
    Key differences from interactive agent:
    - Task-specific system prompts
    - Lower temperature (0.1) for consistency
    - Structured output parsing
    - Single-tool ablation mode support
    - Max tool calls limit to prevent infinite loops
    
    Args:
        tools: List of tool instances
        model_name: LLM model name
        device: Device for inference
        api_key: OpenAI API key
        base_url: OpenAI API base URL
        enabled_tools: List of enabled tool names (for ablation prompt selection)
        recursion_limit: Max recursion steps (default 30)
        max_tool_calls: Max number of tool calls per query (default 10)
        execution_mode: Tool execution mode - "parallel" (original behavior) 
                       or "sequential" (default, serial execution to avoid GPU contention)
        memory_management: If True, enables LRU-based GPU memory management.
                          Tools are loaded on-demand and evicted when capacity is exceeded.
                          Only effective when execution_mode="sequential". (default: False)
        max_loaded_models: Maximum number of models to keep loaded in GPU memory
                          when memory_management=True. (default: 2)
        enable_critic: Control critic node activation. (default: None)
                      - None: Default False
                      - True: Force enable critic for any task
                      - False: Force disable critic for any task
        critic_llm_summary: If True, use LLM to generate structured evidence summary
                           for critic retries. (default: False for backward compatibility)
        critic_image_reinject: If True, re-inject image as HumanMessage when critic
                              triggers retry. (default: False for backward compatibility)
        max_critic_retries: Maximum number of critic retry attempts per sample.
                           (default: None, uses module-level MAX_CRITIC_RETRIES=2)
        prompt_variant: Prompt variant to use. (default: "default")
                       - "default": Use existing prompts (V1 behavior, unchanged)
                       - "v2": Use DIAGNOSIS_SYSTEM_PROMPT_V2 with phased workflow
                       - "cot-staged": Captioning with CoT + multi-stage layout
                       - "cot-flat": Captioning with CoT + flat sequential layout
    """
    if tools is None:
        tools = create_tools(device=device)

    # Resolve max critic retries
    effective_max_critic_retries = max_critic_retries if max_critic_retries is not None else MAX_CRITIC_RETRIES

    def _is_known_ablation_key(key: str) -> bool:
        for task_prompts in ABLATION_PROMPTS_BY_TASK.values():
            if key in task_prompts:
                return True
        return False

    # Detect ablation mode (single-tool or specific multi-tool combinations)
    ablation_mode = False
    ablation_key = None
    
    if enabled_tools is not None:
        # Check for single-tool ablation
        if len(enabled_tools) == 1:
            ablation_key = enabled_tools[0]
            if _is_known_ablation_key(ablation_key):
                ablation_mode = True
                print(f"[Ablation Mode] Single tool detected: {ablation_key}")
        # Check for multi-tool combination ablation
        else:
            # Try comma-separated key (sorted for consistency)
            sorted_tools = sorted(enabled_tools)
            combo_key = ",".join(sorted_tools)
            if _is_known_ablation_key(combo_key):
                ablation_mode = True
                ablation_key = combo_key
                print(f"[Ablation Mode] Tool combination detected: {ablation_key}")
            else:
                # Try underscore-separated key
                combo_key_underscore = "_".join(sorted_tools)
                if _is_known_ablation_key(combo_key_underscore):
                    ablation_mode = True
                    ablation_key = combo_key_underscore
                    print(f"[Ablation Mode] Tool combination detected: {ablation_key}")
    
    # Resolve model alias to actual API model name
    resolved_model_name = resolve_model_name(model_name)
    print(f"[BenchmarkAgent] Using LLM: {resolved_model_name}")
    
    # Initialize LLM with low temperature
    llm_kwargs = {
        "model": resolved_model_name,
        "temperature": 0.1,
    }
    
    if api_key:
        llm_kwargs["api_key"] = api_key
    elif os.getenv("OPENAI_API_KEY"):
        llm_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
    
    if base_url:
        llm_kwargs["base_url"] = base_url
    elif os.getenv("OPENAI_BASE_URL"):
        llm_kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
    
    llm = ChatOpenAI(**llm_kwargs)
    llm_with_tools = llm.bind_tools(tools)
    
    # =========================================================================
    # Evidence Summary Generator (for critic_llm_summary option)
    # =========================================================================
    
    def generate_evidence_summary(
        messages: List[AnyMessage], 
        critic_feedback: str
    ) -> Dict[str, Any]:
        """
        Use LLM to generate a structured evidence summary for critic retry.
        
        Args:
            messages: Conversation history containing tool outputs
            critic_feedback: The critic's evaluation and issues
            
        Returns:
            Dict with keys: summary, duration_ms, timestamp
        """
        from datetime import datetime
        start_time = time.time()
        
        # Extract tool outputs from messages
        tool_outputs_text = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "unknown")
                # Include full output for LLM to summarize
                tool_outputs_text.append(f"### {tool_name}\n{msg.content}")
        
        tool_outputs_str = "\n\n".join(tool_outputs_text) if tool_outputs_text else "No tool outputs available."
        
        # Format the prompt
        summary_prompt = EVIDENCE_SUMMARY_PROMPT.format(
            tool_outputs=tool_outputs_str,
            critic_issues=critic_feedback
        )
        
        try:
            # Use the same LLM instance (without tools) for summarization
            response = llm.invoke([HumanMessage(content=summary_prompt)])
            summary_text = response.content.strip()
        except Exception as e:
            # Fallback to simple extraction if LLM call fails
            summary_text = f"[Summary generation failed: {e}]\n\nRaw tool outputs:\n{tool_outputs_str[:500]}..."
        
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            "summary": summary_text,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
        }
    
    def chatbot(state: BenchmarkAgentState) -> Dict[str, Any]:
        """Chatbot with task-specific system prompt."""
        messages = state["messages"]
        task_type = state.get("task_type", "diagnosis")
        
        # Select appropriate system prompt based on (task_type, ablation_key) combination
        system_prompt = None
        
        if ablation_mode and ablation_key:
            # First, try task-specific ablation prompts
            task_ablation_prompts = ABLATION_PROMPTS_BY_TASK.get(task_type, {})
            if ablation_key in task_ablation_prompts:
                system_prompt = task_ablation_prompts[ablation_key]
        
        # If not in ablation mode or no ablation prompt found, use normal task prompt
        if system_prompt is None:
            if task_type == "diagnosis":
                if prompt_variant == "v2":
                    system_prompt = DIAGNOSIS_SYSTEM_PROMPT_V2
                elif enabled_tools and "dermogpt_vqa" in enabled_tools and "text_rag" in enabled_tools and "ontology" in enabled_tools:
                    system_prompt = DIAGNOSIS_SYSTEM_PROMPT_DERMOGPT_FULL
                elif enabled_tools and "dermogpt_vqa" in enabled_tools and "text_rag" in enabled_tools:
                    system_prompt = DIAGNOSIS_SYSTEM_PROMPT_DERMOGPT_FULL
                else:
                    system_prompt = TASK_PROMPTS.get(task_type, DIAGNOSIS_SYSTEM_PROMPT)
            elif task_type == "captioning":
                if enable_critic:
                    system_prompt = CAPTIONING_SYSTEM_PROMPT_WITH_CRITIC
                elif enabled_tools and "ontology" in enabled_tools:
                    system_prompt = CAPTIONING_SYSTEM_PROMPT_WITH_ONTOLOGY
                else:
                    system_prompt = TASK_PROMPTS.get(task_type, DIAGNOSIS_SYSTEM_PROMPT)
            else:
                system_prompt = TASK_PROMPTS.get(task_type, DIAGNOSIS_SYSTEM_PROMPT)

        # Dynamically inject options for diagnosis task only
        # This moves options from user message to system prompt, reducing token usage
        # System prompt is replaced each LLM call (not accumulated in history)
        dataset = state.get("dataset")
        options = state.get("options")
        narrowed = state.get("narrowed_candidates")
        retry_count = state.get("critic_retry_count", 0)

        if task_type == "diagnosis" and narrowed and retry_count > 0 and options:
            # ===== Narrowed re-diagnosis: swap options with shortlist =====
            logger.info(
                "[NarrowedDx] Chatbot: injecting narrowed options "
                "(%d→%d) for retry %d",
                len(options), len(narrowed), retry_count,
            )
            options_section = f"""

## Focused Re-diagnosis (Narrowed from {len(options)} to {len(narrowed)} candidates)
Round 1 tools identified these as the most likely candidates:
{chr(10).join(f'- {opt}' for opt in narrowed)}

## Round 2 Tool Instructions
- **panderm_classifier**: Re-classify with candidate_diseases="{','.join(narrowed)}" (REQUIRED)
- **dermogpt_vqa**: Re-analyze with candidate_diseases="{','.join(narrowed)}" for focused CoT reasoning
- **text_rag_retrieval**: Look up differentiating criteria between candidates
- **ontology_query**: Check disease category relationships
- **DO NOT re-call**: make_concept_annotator, rag_retrieval
- Your FINAL_ANSWER MUST be one of the original {len(options)} diagnostic options (not limited to the shortlist).
"""
            system_prompt = system_prompt + options_section
        elif task_type == "diagnosis" and options:
            # ===== V2-specific options section =====
            if prompt_variant == "v2":
                options_section = f"""

## Diagnostic Options (choose ONE from this list ONLY)
{chr(10).join(f'- {opt}' for opt in options)}

## Tool Instructions
- **panderm_classifier**: Pass candidate_diseases="{','.join(options[:20])}..." (full list provided above)
- **dermogpt_vqa**: MUST pass candidate_diseases + reasoning prompt for CoT diagnosis
- **rag_retrieval**: Use IMAGE MODE ONLY with image_path
- **ontology_query**: Query disease taxonomy for disambiguation
- Your FINAL_ANSWER MUST be one of the exact options listed above.
"""
            # ===== V1 original logic (kept as-is) =====
            else:
                options_section = f"""

## Diagnostic Options (choose ONE from this list ONLY)
{chr(10).join(f'- {opt}' for opt in options)}

## Tool Instructions
- **panderm_classifier**: Use dataset="{dataset}" (options loaded automatically from config)
- **rag_retrieval**: Use IMAGE MODE ONLY with image_path
- **make_concept_annotator**: Extract dermoscopic features
- Your FINAL_ANSWER MUST be one of the exact options listed above.
"""
            system_prompt = system_prompt + options_section
        
        # Ensure system prompt is present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        elif isinstance(messages[0], SystemMessage):
            # Replace with task-specific prompt
            messages[0] = SystemMessage(content=system_prompt)
        
        # LLM call with profiling
        start_time = time.perf_counter()
        response = with_rate_limit_retry(llm_with_tools.invoke, messages)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Record LLM call timing and token usage if profiler is enabled
        if profiler.is_enabled():
            # Try to extract token usage from response metadata (OpenAI API returns this)
            usage = {}
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
            elif hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata or {}
            
            profiler.record_llm_call(
                duration_ms=duration_ms,
                input_tokens=usage.get('prompt_tokens') or usage.get('input_tokens'),
                output_tokens=usage.get('completion_tokens') or usage.get('output_tokens'),
                model=resolved_model_name,
            )
        
        return {"messages": [response]}
    
    def should_continue(state: BenchmarkAgentState) -> Literal["tools", "__end__"]:
        """Route to tools or end. Checks max tool calls limit."""
        last_message = state["messages"][-1]
        
        # Check if there are tool calls to make
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # Check max tool calls limit
            current_count = state.get("tool_call_count", 0)
            num_new_calls = len(last_message.tool_calls)
            
            if current_count + num_new_calls > max_tool_calls:
                print(f"[WARNING] Max tool calls limit ({max_tool_calls}) reached. "
                      f"Current: {current_count}, Requested: {num_new_calls}. Stopping agent.")
                return "__end__"
            
            return "tools"
        return "__end__"
    
    # Create tool execution node based on execution_mode
    if execution_mode == "sequential":
        # Sequential execution with optional memory management
        print(f"[BenchmarkAgent] Using SEQUENTIAL tool execution (memory_management={memory_management})")
        
        if memory_management:
            memory_manager = ToolMemoryManager(max_loaded=max_loaded_models, verbose=True)
            print(f"[BenchmarkAgent] LRU Memory Manager enabled (max_loaded={max_loaded_models})")
        else:
            memory_manager = None
        
        # Sequential node already handles call counting
        tools_with_counter = create_sequential_tool_node(tools, memory_manager=memory_manager)
    else:
        # Original parallel execution (default, for reproducibility)
        tool_node = ToolNode(tools)
        
        def tools_with_counter(state: BenchmarkAgentState) -> Dict[str, Any]:
            """
            Execute tools in parallel with caching to prevent duplicate calls.
            
            Features:
            1. Cache tool results by (tool_name, args) - return cached result for duplicates
            2. Track text_rag queries for conflict investigation checking
            3. Increment tool call counter
            """
            last_message = state["messages"][-1]
            if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                return {}
            
            cache = state.get("tool_call_cache", {})
            text_rag_queries = list(state.get("text_rag_queries", []))
            
            cached_results = []
            new_tool_calls = []

            # Hard block: fixed-input tools during narrowed re-diagnosis
            # (Task 1 diagnosis only — gated by three conditions)
            _narrowed = state.get("narrowed_candidates")
            _retry_ct = state.get("critic_retry_count", 0)
            _task_type = state.get("task_type", "diagnosis")
            BLOCKED_ON_NARROWED_RETRY = {
                "make_concept_annotator", "rag_retrieval",
            }
            
            # Separate cached and new tool calls
            for tc in last_message.tool_calls:
                # --- Block fixed-input tools during narrowed re-diagnosis ---
                if (_task_type == "diagnosis"
                        and _narrowed is not None
                        and _retry_ct > 0
                        and tc["name"] in BLOCKED_ON_NARROWED_RETRY):
                    r1_cache_key = _make_cache_key(tc["name"], tc["args"])
                    r1_cached = cache.get(r1_cache_key, "")
                    if r1_cached:
                        blocked_msg = (
                            f"[Blocked] {tc['name']} uses fixed image input "
                            f"— output is identical to Round 1. "
                            f"Use panderm_classifier or dermogpt_vqa with "
                            f"narrowed candidate_diseases instead.\n"
                            f"Round 1 result: {r1_cached[:500]}..."
                        )
                    else:
                        blocked_msg = (
                            f"[Blocked] {tc['name']} cannot be re-called "
                            f"during focused re-diagnosis. Use "
                            f"panderm_classifier or dermogpt_vqa with "
                            f"narrowed candidate_diseases instead."
                        )
                    logger.info(
                        "[NarrowedDx] Blocked tool %s (retry=%d, narrowed=%s)",
                        tc["name"], _retry_ct, _narrowed,
                    )
                    cached_results.append(ToolMessage(
                        content=blocked_msg,
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    ))
                    continue

                cache_key = _make_cache_key(tc["name"], tc["args"])
                
                if cache_key in cache:
                    # Return cached result
                    cached_results.append(ToolMessage(
                        content=f"[Cached - same input as before] {cache[cache_key]}",
                        tool_call_id=tc["id"],
                        name=tc["name"]
                    ))
                else:
                    new_tool_calls.append(tc)
                
                # Track text_rag queries regardless of cache hit
                if tc["name"] == "text_rag_retrieval":
                    query = tc["args"].get("query", "")
                    if query and query not in text_rag_queries:
                        text_rag_queries.append(query)
            
            # Execute only new (non-cached) tool calls
            new_results = []
            if new_tool_calls:
                # Create a modified message with only new tool calls
                modified_message = AIMessage(
                    content=last_message.content,
                    tool_calls=new_tool_calls
                )
                modified_state = {**state, "messages": state["messages"][:-1] + [modified_message]}
                
                # Execute tools
                result = tool_node.invoke(modified_state)
                
                # Extract new results and update cache
                for msg in result.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        new_results.append(msg)
                        # Find the original tool call to get args for cache key
                        for tc in new_tool_calls:
                            if tc["id"] == msg.tool_call_id:
                                cache_key = _make_cache_key(tc["name"], tc["args"])
                                cache[cache_key] = msg.content
                                break
            
            # Combine cached and new results
            all_results = cached_results + new_results
            
            # Update counter
            current_count = state.get("tool_call_count", 0)
            num_calls = len(last_message.tool_calls)
            
            return {
                "messages": all_results,
                "tool_call_count": current_count + num_calls,
                "tool_call_cache": cache,
                "text_rag_queries": text_rag_queries,
            }
    
    # =========================================================================
    # Critic Node - Evaluates tool outputs and decides whether to retry
    # =========================================================================

    def critic_node(state: BenchmarkAgentState) -> Dict[str, Any]:
        """
        Evaluate tool outputs and decide whether to retry.
        
        Directional reviewer design:
        - Identifies issues (low confidence, conflicts, errors, missing tools)
        - Provides evidence summary + available follow-up options
        - Does NOT prescribe exact tool calls or queries
        - LLM autonomously decides which tools to call on retry
        
        Retry-aware logic:
        - retry_count == 0: Detect issues, provide directional guidance
        - retry_count >= 1: Same approach, but pass through if all 6 tools exhausted
        """
        messages = state.get("messages", [])
        tools_called = state.get("tools_called", [])
        retry_count = state.get("critic_retry_count", 0)
        task_type = state.get("task_type", "diagnosis")

        # Critic activation logic
        if enable_critic is None:
            critic_enabled = False
        else:
            critic_enabled = enable_critic

        if not critic_enabled or retry_count >= effective_max_critic_retries:
            return {"last_critic_feedback": None}

        # Collect tool outputs from ToolMessages
        tool_outputs: Dict[str, str] = {}
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", None)
                if tool_name:
                    tool_outputs[tool_name] = msg.content
                    if tool_name not in tools_called:
                        tools_called = tools_called + [tool_name]

        # =====================================================================
        # Graceful exit: all tools for this task have been called
        # =====================================================================
        task_all_tools = ALL_TOOLS_BY_TASK.get(
            task_type, ALL_TOOLS_BY_TASK["diagnosis"])
        uncalled_tools = task_all_tools - set(tools_called)

        if retry_count >= 1 and not uncalled_tools:
            # All tools exhausted on second+ retry → pass through
            return {"last_critic_feedback": None, "tools_called": tools_called}

        # =====================================================================
        # Detect issues (shared: evidence summary & error detection)
        # =====================================================================
        issues: List[str] = []
        evidence_summary: List[str] = []
        tool_errors: List[str] = []

        for tool_name, output in tool_outputs.items():
            if _check_tool_returned_error(tool_outputs, tool_name):
                tool_errors.append(tool_name)
                evidence_summary.append(f"- **{tool_name}**: ERROR (tool failed)")
            else:
                short_output = output[:200].replace("\n", " ").strip()
                if len(output) > 200:
                    short_output += "..."
                evidence_summary.append(f"- **{tool_name}**: {short_output}")

        # =====================================================================
        # Task-specific issue detection
        # =====================================================================
        if task_type == "concept":
            # =================================================================
            # Task 2: Concept Annotation checks
            # =================================================================

            # --- C1: MAKE–DermoGPT concept disagreement ---
            make_output = tool_outputs.get("make_concept_annotator", "")
            dermogpt_output = tool_outputs.get("dermogpt_vqa", "")
            if (make_output and not _check_tool_returned_error(tool_outputs, "make_concept_annotator")
                    and dermogpt_output and not _check_tool_returned_error(tool_outputs, "dermogpt_vqa")):
                all_concepts = ToolOutputParser.parse_make_all_concepts(make_output)
                detected_concepts = [(c, s) for c, s in all_concepts if s > 0.5]
                contradictions: List[str] = []
                for concept, score in detected_concepts:
                    if ToolOutputParser.check_dermogpt_concept_contradiction(
                            dermogpt_output, concept):
                        contradictions.append(
                            f"{concept} (MAKE={score:.2f}, DermoGPT denies)")
                if contradictions:
                    issues.append(
                        f"MAKE–DermoGPT disagreement on concepts: "
                        f"{', '.join(contradictions)}. "
                        "MAKE detected these as present but DermoGPT explicitly "
                        "states they are absent. Verify visually."
                    )

            # --- C2: Borderline MAKE concepts ---
            if make_output and not _check_tool_returned_error(
                    tool_outputs, "make_concept_annotator"):
                borderline = ToolOutputParser.parse_make_borderline_concepts(
                    make_output,
                    low=CONFIDENCE_THRESHOLDS["make_borderline_low"],
                    high=CONFIDENCE_THRESHOLDS["make_borderline_high"],
                )
                if borderline and "dermogpt_vqa" not in tool_outputs:
                    concepts_str = ", ".join(
                        f"{c} ({s:.2f})" for c, s in borderline)
                    issues.append(
                        f"Borderline MAKE concepts near decision threshold: "
                        f"{concepts_str}. "
                        "Visual verification with DermoGPT could resolve ambiguity."
                    )

            # --- C3: Tool coverage ---
            required = REQUIRED_TOOLS_BY_TYPE.get("concept", [])
            missing_required = [
                t for t in required if t not in tools_called]
            if missing_required:
                issues.append(
                    f"Essential tools not yet called: "
                    f"{', '.join(missing_required)}. "
                    "Both MAKE and DermoGPT are needed for reliable concept "
                    "cross-validation."
                )

        elif task_type == "captioning":
            # =================================================================
            # Task 3: Captioning checks
            # =================================================================

            # --- P1: Diagnosis conflict (reuse diagnosis logic) ---
            INVALID_DIAGNOSES = {
                "no definitive diagnosis", "no diagnosis", "unknown", "n/a",
                "none", "uncertain", "unspecified", "not specified",
                "no label", "other",
            }
            diagnoses_by_tool: Dict[str, str] = {}
            if "panderm_classifier" in tool_outputs:
                diagnosis = ToolOutputParser.extract_diagnosis_from_tool(
                    "panderm_classifier",
                    tool_outputs["panderm_classifier"])
                if diagnosis and diagnosis.lower() not in INVALID_DIAGNOSES:
                    diagnoses_by_tool["panderm_classifier"] = diagnosis
            if "rag_retrieval" in tool_outputs:
                all_rag = ToolOutputParser.extract_all_diagnoses_from_rag(
                    tool_outputs["rag_retrieval"])
                for diag in all_rag:
                    if diag.lower() not in INVALID_DIAGNOSES:
                        diagnoses_by_tool["rag_retrieval"] = diag
                        break
            if len(diagnoses_by_tool) >= 2:
                unique_diagnoses = set(diagnoses_by_tool.values())
                if len(unique_diagnoses) > 1:
                    text_rag_queries = state.get("text_rag_queries", [])
                    diagnoses_list = list(unique_diagnoses)
                    if not _check_conflict_investigated(
                            text_rag_queries, diagnoses_list):
                        conflict_desc = ", ".join(
                            f"{k}: {v}" for k, v in diagnoses_by_tool.items())
                        issues.append(
                            f"Diagnosis conflict: {conflict_desc}. "
                            "A wrong diagnostic anchor leads to incorrect "
                            "clinical framing in the caption."
                        )

            # --- P2: Confidence checks (same as diagnosis) ---
            if ("panderm_classifier" in tool_outputs
                    and not _check_tool_returned_error(
                        tool_outputs, "panderm_classifier")):
                confidence = ToolOutputParser.parse_panderm_confidence(
                    tool_outputs["panderm_classifier"])
                if (confidence is not None
                        and confidence < CONFIDENCE_THRESHOLDS["panderm"]):
                    issues.append(
                        f"PanDerm confidence is low "
                        f"({confidence*100:.1f}% < "
                        f"{CONFIDENCE_THRESHOLDS['panderm']*100:.0f}%). "
                        "Low diagnostic confidence may lead to a vague caption."
                    )
            if ("rag_retrieval" in tool_outputs
                    and not _check_tool_returned_error(
                        tool_outputs, "rag_retrieval")):
                similarity = ToolOutputParser.parse_rag_similarity(
                    tool_outputs["rag_retrieval"])
                if (similarity is not None
                        and similarity < CONFIDENCE_THRESHOLDS["rag"]):
                    issues.append(
                        f"RAG similarity is low "
                        f"({similarity:.2f} < "
                        f"{CONFIDENCE_THRESHOLDS['rag']:.2f}). "
                        "No highly similar cases found."
                    )

            # --- P3: Tool coverage (ALL 6 tools required) ---
            CAPTIONING_REQUIRED_TOOLS = {
                "dermogpt_vqa", "make_concept_annotator", "panderm_classifier",
                "rag_retrieval", "text_rag_retrieval", "ontology_query",
            }
            missing_tools = CAPTIONING_REQUIRED_TOOLS - set(tools_called)
            if missing_tools:
                issues.append(
                    f"Required tools not yet called: "
                    f"{', '.join(sorted(missing_tools))}. "
                    "ALL tools must be called for comprehensive captioning "
                    "with diagnosis and treatment context."
                )

            # --- P4: Text RAG dual-topic check (diagnosis + treatment) ---
            text_rag_queries = state.get("text_rag_queries", [])
            if "text_rag_retrieval" in tools_called and text_rag_queries:
                DIAGNOSIS_KEYWORDS = {
                    "clinical", "criteria", "diagnostic", "features",
                    "diagnosis", "symptom", "presentation", "description",
                }
                TREATMENT_KEYWORDS = {
                    "treatment", "therapy", "management", "medication",
                    "drug", "surgical", "intervention", "prognosis",
                }
                has_diagnosis_query = any(
                    any(kw in q.lower() for kw in DIAGNOSIS_KEYWORDS)
                    for q in text_rag_queries
                )
                has_treatment_query = any(
                    any(kw in q.lower() for kw in TREATMENT_KEYWORDS)
                    for q in text_rag_queries
                )
                if not has_diagnosis_query and not has_treatment_query:
                    issues.append(
                        "Text RAG was called but queries lack both "
                        "diagnosis and treatment topics. "
                        "Both diagnostic criteria AND treatment information "
                        "are required for captioning."
                    )
                elif not has_diagnosis_query:
                    issues.append(
                        "Text RAG only queried for treatment. "
                        "A query for diagnostic criteria/clinical features "
                        "is also required."
                    )
                elif not has_treatment_query:
                    issues.append(
                        "Text RAG only queried for diagnosis. "
                        "A query for treatment information is also required "
                        "for the caption."
                    )

        else:
            # =================================================================
            # Diagnosis / VQA checks (existing logic, unchanged)
            # =================================================================

            # --- Check 1: Confidence thresholds ---
            if ("panderm_classifier" in tool_outputs
                    and not _check_tool_returned_error(
                        tool_outputs, "panderm_classifier")):
                confidence = ToolOutputParser.parse_panderm_confidence(
                    tool_outputs["panderm_classifier"])
                if (confidence is not None
                        and confidence < CONFIDENCE_THRESHOLDS["panderm"]):
                    issues.append(
                        f"PanDerm confidence is low "
                        f"({confidence*100:.1f}% < "
                        f"{CONFIDENCE_THRESHOLDS['panderm']*100:.0f}%). "
                        "The classification is uncertain."
                    )

            if ("rag_retrieval" in tool_outputs
                    and not _check_tool_returned_error(
                        tool_outputs, "rag_retrieval")):
                similarity = ToolOutputParser.parse_rag_similarity(
                    tool_outputs["rag_retrieval"])
                if (similarity is not None
                        and similarity < CONFIDENCE_THRESHOLDS["rag"]):
                    issues.append(
                        f"RAG similarity is low "
                        f"({similarity:.2f} < "
                        f"{CONFIDENCE_THRESHOLDS['rag']:.2f}). "
                        "No highly similar cases found in the knowledge base."
                    )

            # --- Check 2: Tool conflicts ---
            INVALID_DIAGNOSES = {
                "no definitive diagnosis", "no diagnosis", "unknown", "n/a",
                "none", "uncertain", "unspecified", "not specified",
                "no label", "other",
            }

            diagnoses_by_tool: Dict[str, str] = {}
            if "panderm_classifier" in tool_outputs:
                diagnosis = ToolOutputParser.extract_diagnosis_from_tool(
                    "panderm_classifier",
                    tool_outputs["panderm_classifier"])
                if diagnosis and diagnosis.lower() not in INVALID_DIAGNOSES:
                    diagnoses_by_tool["panderm_classifier"] = diagnosis

            if "rag_retrieval" in tool_outputs:
                all_rag_diagnoses = (
                    ToolOutputParser.extract_all_diagnoses_from_rag(
                        tool_outputs["rag_retrieval"]))
                for diag in all_rag_diagnoses:
                    if diag.lower() not in INVALID_DIAGNOSES:
                        diagnoses_by_tool["rag_retrieval"] = diag
                        break

            if len(diagnoses_by_tool) >= 2:
                unique_diagnoses = set(diagnoses_by_tool.values())
                if len(unique_diagnoses) > 1:
                    text_rag_queries = state.get("text_rag_queries", [])
                    diagnoses_list = list(unique_diagnoses)
                    conflict_investigated = _check_conflict_investigated(
                        text_rag_queries, diagnoses_list)

                    if not conflict_investigated:
                        conflict_desc = ", ".join(
                            f"{k}: {v}"
                            for k, v in diagnoses_by_tool.items())
                        issues.append(
                            f"Tool conflict: {conflict_desc}. "
                            "Different tools suggest different diagnoses."
                        )

        # --- Shared: Tool errors (all task types) ---
        if tool_errors:
            issues.append(
                f"Tool errors detected: {', '.join(tool_errors)}. "
                "These tools failed and their evidence is missing."
            )

        # =====================================================================
        # Narrowed Re-diagnosis: Build shortlist (Task 1 diagnosis only)
        # =====================================================================
        narrowed_diseases: Optional[List[str]] = None
        if task_type == "diagnosis" and state.get("options"):
            options = state["options"]
            candidate_pool: Dict[str, str] = {}  # lowered name -> original option

            # Source 1: PanDerm — ALL predictions with non-zero confidence
            if ("panderm_classifier" in tool_outputs
                    and not _check_tool_returned_error(
                        tool_outputs, "panderm_classifier")):
                panderm_preds = ToolOutputParser.extract_all_panderm_predictions(
                    tool_outputs["panderm_classifier"])
                for disease, _score in panderm_preds:
                    candidate_pool[disease.lower()] = disease

            # Source 2: RAG — top-2 disease labels (fuzzy match to options)
            if ("rag_retrieval" in tool_outputs
                    and not _check_tool_returned_error(
                        tool_outputs, "rag_retrieval")):
                rag_diagnoses = ToolOutputParser.extract_all_diagnoses_from_rag(
                    tool_outputs["rag_retrieval"])
                for diag in rag_diagnoses[:2]:
                    matched = _fuzzy_match_option(diag, options)
                    if matched and matched.lower() not in candidate_pool:
                        candidate_pool[matched.lower()] = matched

            # Source 3: DermoGPT / Qwen — extracted diagnosis (fuzzy match)
            for vqa_tool in ["dermogpt_vqa"]:
                if (vqa_tool in tool_outputs
                        and not _check_tool_returned_error(
                            tool_outputs, vqa_tool)):
                    diag = ToolOutputParser.extract_dermogpt_diagnosis(
                        tool_outputs[vqa_tool])
                    if diag:
                        matched = _fuzzy_match_option(diag, options)
                        if matched and matched.lower() not in candidate_pool:
                            candidate_pool[matched.lower()] = matched

            if len(candidate_pool) >= 2:
                narrowed_diseases = list(candidate_pool.values())
                logger.info(
                    "[NarrowedDx] Shortlist built: %d candidates from %d options → %s",
                    len(narrowed_diseases),
                    len(options),
                    narrowed_diseases,
                )
            else:
                logger.debug(
                    "[NarrowedDx] Shortlist too small (%d candidates), skipping narrowing",
                    len(candidate_pool),
                )

        # =====================================================================
        # Decision: Retry or Pass
        # =====================================================================
        if not issues:
            return {"last_critic_feedback": None, "tools_called": tools_called}

        # Build directional feedback (NOT prescriptive)
        feedback_parts = ["## Critic Review\n"]

        # Evidence summary
        if evidence_summary:
            feedback_parts.append("### Current Evidence\n")
            for line in evidence_summary:
                feedback_parts.append(f"{line}\n")
            feedback_parts.append("\n")

        # Issues
        feedback_parts.append("### Issues Detected\n")
        for i, issue in enumerate(issues, 1):
            feedback_parts.append(f"{i}. {issue}\n")

        # Available follow-up tools (directional, not prescriptive)
        if uncalled_tools:
            feedback_parts.append("\n### Available Follow-up Tools (not yet called)\n")
            tool_hints = {
                "text_rag_retrieval": "Look up diagnostic criteria, differentiating features between diseases, or clinical guidelines",
                "ontology_query": "Check disease hierarchy, category relationships, or find related conditions",
                "dermogpt_vqa": "Describe lesion morphology, verify concepts, generate clinical description",
                "rag_retrieval": "Find visually similar cases from the knowledge base",
                "panderm_classifier": "Run zero-shot classification with candidate diseases",
                "make_concept_annotator": "Extract dermoscopic concepts and features",
            }
            for tool in sorted(uncalled_tools):
                hint = tool_hints.get(tool, "")
                feedback_parts.append(f"- **{tool}**: {hint}\n")

        feedback_parts.append(
            f"\nUse your clinical judgment to decide which additional tools or "
            f"queries would best address the issues above.\n"
        )
        feedback_parts.append(
            f"\nTools already called: "
            f"{', '.join(tools_called) if tools_called else 'none'}"
        )
        feedback_parts.append(f"\nRetry {retry_count + 1}/{effective_max_critic_retries}")

        feedback = "".join(feedback_parts)

        result: Dict[str, Any] = {
            "last_critic_feedback": feedback,
            "critic_retry_count": retry_count + 1,
            "tools_called": tools_called,
        }
        if narrowed_diseases:
            result["narrowed_candidates"] = narrowed_diseases
        return result

    def should_continue_after_critic(state: BenchmarkAgentState) -> Literal["chatbot", "__end__"]:
        """Route from critic: retry (back to chatbot) or end."""
        feedback = state.get("last_critic_feedback")
        if feedback:
            return "chatbot"
        return "__end__"

    def inject_critic_feedback(state: BenchmarkAgentState) -> Dict[str, Any]:
        """
        Inject critic feedback for retry, with optional LLM summary and image re-injection.
        
        Behavior controlled by:
        - critic_llm_summary: If True, use LLM to generate structured evidence summary
        - critic_image_reinject: If True, re-inject image as HumanMessage (first retry only)
        
        Retry-specific instructions:
        - Retry 1: "Address issues, use available tools to gather additional evidence"
        - Retry 2: "Final attempt, synthesize existing evidence if tools are unavailable"
        """
        feedback = state.get("last_critic_feedback")
        if not feedback:
            return {}
        
        current_image = state.get("current_image")
        messages = state.get("messages", [])
        existing_summaries = state.get("evidence_summaries", [])
        retry_count = state.get("critic_retry_count", 0)  # Already incremented by critic
        
        # Track new evidence summary if generated
        new_summary_record = None
        
        # === Retry-specific instruction text ===
        narrowed = state.get("narrowed_candidates")
        task_type = state.get("task_type", "diagnosis")

        if task_type == "diagnosis" and narrowed and retry_count <= 1:
            # Diagnosis narrowed re-diagnosis: prescriptive instructions
            logger.info(
                "[NarrowedDx] Injecting focused re-diagnosis feedback "
                "(retry=%d, %d candidates: %s)",
                retry_count, len(narrowed), narrowed,
            )
            narrowed_str = ", ".join(narrowed)
            retry_instruction = (
                "**Instructions (Focused Re-diagnosis):**\n"
                f"Round 1 evidence suggests these candidates: "
                f"**{narrowed_str}**\n\n"
                "**Allowed tools for this round:**\n"
                f'1. `panderm_classifier` with candidate_diseases='
                f'"{narrowed_str}" '
                "— re-classify among the narrowed shortlist\n"
                f'2. `dermogpt_vqa` with candidate_diseases='
                f'"{narrowed_str}" '
                "— get focused morphological reasoning on the shortlist\n"
                "3. `text_rag_retrieval` — look up differentiating criteria "
                "between candidates\n"
                "4. `ontology_query` — check disease hierarchy for "
                "disambiguation\n\n"
                "**DO NOT call** make_concept_annotator, rag_retrieval, or "
                "(image-only outputs already available from "
                "Round 1).\n\n"
                "Synthesize Round 1 evidence + Round 2 results to provide "
                "a definitive FINAL_ANSWER."
            )
        elif task_type == "diagnosis" and retry_count > 1:
            # Diagnosis final attempt
            retry_instruction = (
                "**Instructions (Final Attempt):**\n"
                "1. This is your last chance to refine the diagnosis\n"
                "2. Synthesize ALL available evidence from previous rounds\n"
                "3. Provide a definitive FINAL_ANSWER"
            )
        elif retry_count <= 1:
            # Generic retry (non-diagnosis tasks or diagnosis without shortlist)
            retry_instruction = (
                "**Instructions:**\n"
                "1. Review the critic feedback and your existing tool outputs\n"
                "2. Use the available follow-up tools to gather additional "
                "evidence that addresses the identified issues\n"
                "3. Provide a revised FINAL_ANSWER based on all evidence"
            )
        else:
            # Generic final attempt (non-diagnosis tasks)
            retry_instruction = (
                "**Instructions (final attempt):**\n"
                "1. This is your last chance to refine the diagnosis\n"
                "2. If recommended tools fail or are unavailable, rely on "
                "successful tool outputs and your visual analysis\n"
                "3. Synthesize ALL available evidence to make your best "
                "clinical judgment\n"
                "4. Provide a definitive FINAL_ANSWER"
            )
        
        # === Build SystemMessage content ===
        if critic_llm_summary:
            # Generate LLM-based evidence summary (returns Dict)
            summary_result = generate_evidence_summary(messages, feedback)
            evidence_summary_text = summary_result["summary"]
            new_summary_record = summary_result  # Save for state update
            
            system_content = f"""{feedback}

---

{evidence_summary_text}

---

{retry_instruction}
"""
        else:
            system_content = f"""{feedback}

{retry_instruction}
"""
        
        result_messages = [SystemMessage(content=system_content)]
        
        # === Optionally add HumanMessage with image (only on first critic retry) ===
        image_was_reinjected = False
        already_reinjected = state.get("image_reinject_count", 0) > 0
        if critic_image_reinject and current_image and Path(current_image).exists() and not already_reinjected:
            with open(current_image, "rb") as f:
                b64_image = base64.b64encode(f.read()).decode("utf-8")
            suffix = Path(current_image).suffix.lower()
            media_types = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg", 
                ".png": "image/png", ".gif": "image/gif"
            }
            media_type = media_types.get(suffix, "image/jpeg")
            
            human_content = [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64_image}"}},
                {"type": "text", "text": "Re-analyze this image based on the critic feedback above."}
            ]
            result_messages.append(HumanMessage(content=human_content))
            image_was_reinjected = True
        
        # Build return state update
        result = {
            "messages": result_messages,
            "last_critic_feedback": None,
        }
        
        # Append new summary to evidence_summaries list
        if new_summary_record:
            result["evidence_summaries"] = existing_summaries + [new_summary_record]
        
        # Track image reinject count
        if image_was_reinjected:
            result["image_reinject_count"] = state.get("image_reinject_count", 0) + 1
        
        return result

    # =========================================================================
    # Analyze Reinjected Image Node (forces text analysis before tool calls)
    # =========================================================================
    
    def analyze_reinjected_image(state: BenchmarkAgentState) -> Dict[str, Any]:
        """
        Analyze the reinjected image WITHOUT tools.
        Forces LLM to generate text description/analysis before proceeding to tool calls.
        
        This node is only reached when critic_image_reinject=True and image was reinjected.
        """
        messages = state["messages"]
        
        # Add explicit instruction for image analysis
        analysis_instruction = """Based on the critic feedback and the image provided above, please:

1. **Describe what you observe**: Color, shape, texture, borders, distribution of the lesion(s)
2. **Identify key clinical features**: Dermoscopic patterns, morphological characteristics relevant to diagnosis
3. **Preliminary assessment**: Your initial impression before calling diagnostic tools

Be specific and reference visual evidence from the image. After this analysis, you will use the appropriate tools to validate your observations."""
        
        analysis_messages = messages + [HumanMessage(content=analysis_instruction)]
        
        # Call LLM WITHOUT tools - forces text output
        start_time = time.perf_counter()
        response = with_rate_limit_retry(llm.invoke, analysis_messages)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Record profiling if enabled
        if profiler.is_enabled():
            usage = {}
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
            elif hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata or {}
            
            profiler.record_llm_call(
                duration_ms=duration_ms,
                input_tokens=usage.get('prompt_tokens') or usage.get('input_tokens'),
                output_tokens=usage.get('completion_tokens') or usage.get('output_tokens'),
            )
        
        return {"messages": [response]}
    
    def should_analyze_image(state: BenchmarkAgentState) -> Literal["analyze_image", "chatbot"]:
        """
        Route from inject_feedback: go to image analysis if image was reinjected,
        otherwise go directly to chatbot.
        """
        # Check if the last message is a HumanMessage with image (indicating image reinject)
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, HumanMessage):
                content = last_msg.content
                if isinstance(content, list):
                    has_image = any(
                        isinstance(item, dict) and item.get("type") == "image_url"
                        for item in content
                    )
                    if has_image:
                        return "analyze_image"
        return "chatbot"

    # Build graph with critic node
    workflow = StateGraph(BenchmarkAgentState)
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", tools_with_counter)
    workflow.add_node("critic", critic_node)
    workflow.add_node("inject_feedback", inject_critic_feedback)
    workflow.add_node("analyze_reinjected_image", analyze_reinjected_image)

    workflow.set_entry_point("chatbot")

    # chatbot → tools or END (if no tool calls)
    workflow.add_conditional_edges(
        "chatbot",
        should_continue,
        {"tools": "tools", "__end__": "critic"}  # Changed: go to critic instead of END
    )

    # tools → chatbot (continue reasoning with tool results)
    workflow.add_edge("tools", "chatbot")

    # critic → inject_feedback (retry) or END
    workflow.add_conditional_edges(
        "critic",
        should_continue_after_critic,
        {"chatbot": "inject_feedback", "__end__": END}
    )

    # inject_feedback → analyze_image (if image reinjected) or chatbot (if not)
    workflow.add_conditional_edges(
        "inject_feedback",
        should_analyze_image,
        {"analyze_image": "analyze_reinjected_image", "chatbot": "chatbot"}
    )
    
    # analyze_reinjected_image → chatbot (continue with tool calls)
    workflow.add_edge("analyze_reinjected_image", "chatbot")

    return workflow.compile(checkpointer=MemorySaver())


# =============================================================================
# Benchmark Runner with Tracing
# =============================================================================

class BenchmarkRunner:
    """Run agent on benchmark datasets with tracing support."""
    
    def __init__(
        self, 
        agent: StateGraph, 
        tools: List,
        trace_logger: Optional[TraceLogger] = None,
        recursion_limit: int = 30,
        max_tool_calls: int = 20,
        include_image_in_message: bool = False,
        model_name: str = "gpt-4o",
    ):
        self.agent = agent
        self.tools = tools
        self.trace_logger = trace_logger or TraceLogger("./traces")
        self.recursion_limit = recursion_limit
        self.max_tool_calls = max_tool_calls
        self.include_image_in_message = include_image_in_message
        self.model_name = model_name
    
    def run_diagnosis(
        self,
        image_path: str,
        dataset: str,
        ground_truth: str = None,
    ) -> Dict[str, Any]:
        """Run diagnosis task with tracing."""
        config = DATASET_CONFIGS[dataset]
        options = config["options"]
        
        # Generate anonymous image ID and register mapping (prevents filename leakage)
        image_id = generate_image_id(image_path)
        register_image_path(image_id, image_path)
        
        # Simplified user message - options are injected into system prompt by chatbot node
        # This significantly reduces token usage, especially for datasets with many classes (e.g., SNU_500 with 134)
        user_msg = f"""## Task
Diagnose the skin lesion in this image.

## Image
{image_id}

## Instructions
Use panderm_classifier with dataset="{dataset}", cross-validate with other tools (rag_retrieval, make_concept_annotator), 
then provide FINAL_ANSWER with one of the diagnostic options."""
        
        # Create trace
        trace = self.trace_logger.new_trace(
            task_type="diagnosis",
            dataset=dataset,
            image_path=image_path,
            user_query=user_msg,
            model=self.model_name,
        )
        
        # Pass dataset and options to _run for chatbot to inject into system prompt
        result = self._run(
            task_type="diagnosis",
            user_message=user_msg,
            image_path=image_path,
            trace=trace,
            dataset=dataset,
            options=options,
        )
        
        prediction = AnswerParser.parse_diagnosis(result["response"], options)
        
        # Finalize trace
        trace.finalize(result["response"], parsed_answer=prediction)
        trace.ground_truth = ground_truth
        trace.correct = prediction == ground_truth if ground_truth else None
        
        # Save trace
        self.trace_logger.save_trace(trace)
        
        return {
            "image_path": image_path,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": trace.correct,
            "raw_response": result["response"],
            "trace": trace,
        }
    
    def run_concept(
        self,
        image_path: str,
        dataset: str,
        ground_truth: List[str] = None,
    ) -> Dict[str, Any]:
        """Run concept annotation task with tracing."""
        config = DATASET_CONFIGS[dataset]
        concepts = config["concepts"]
        target_concepts_str = ", ".join(concepts)
        
        # Generate anonymous image ID and register mapping (prevents filename leakage)
        image_id = generate_image_id(image_path)
        register_image_path(image_id, image_path)
        
        user_msg = f"""## Task
Identify ALL present dermatological concepts in this image.

## Image
{image_id}

## Valid Concept List (choose from these ONLY)
{target_concepts_str}

## Tool Instructions

1. **make_concept_annotator**: You MUST pass this exact target_concepts parameter:
   target_concepts="{target_concepts_str}"
   Use image_path="{image_id}"

2. **dermogpt_vqa**: Pass target_concepts="{target_concepts_str}" to focus evaluation on these specific concepts.
   Use image_path="{image_id}"

3. **rag_retrieval**: Use IMAGE MODE ONLY with image_path="{image_id}". DO NOT use text_query.

## Instructions
Use ALL available tools with the target_concepts parameter, cross-validate findings, then provide FINAL_ANSWER with concepts from the list above."""
        
        trace = self.trace_logger.new_trace(
            task_type="concept",
            dataset=dataset,
            image_path=image_path,
            user_query=user_msg,
            model=self.model_name,
        )
        
        result = self._run(
            task_type="concept",
            user_message=user_msg,
            image_path=image_path,
            trace=trace,
        )
        
        prediction = AnswerParser.parse_concepts(result["response"], concepts)
        
        trace.finalize(result["response"], parsed_answer=prediction)
        trace.ground_truth = ground_truth
        self.trace_logger.save_trace(trace)
        
        return {
            "image_path": image_path,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "raw_response": result["response"],
            "trace": trace,
        }
    
    def run_captioning(
        self,
        image_path: str,
        ground_truth: str = None,
    ) -> Dict[str, Any]:
        """Run captioning task with tracing."""
        # Generate anonymous image ID and register mapping (prevents filename leakage)
        image_id = generate_image_id(image_path)
        register_image_path(image_id, image_path)
        
        user_msg = f"""## Task
Generate a detailed clinical description of the skin lesion.

## Image
{image_id}

## Tool Instructions
Use image_path="{image_id}" when calling any tools.

## Instructions
Describe: morphology, color, surface, borders, and any special features.
End with FINAL_ANSWER: [your complete caption]"""
        
        trace = self.trace_logger.new_trace(
            task_type="captioning",
            dataset="skin_cap",
            image_path=image_path,
            user_query=user_msg,
            model=self.model_name,
        )
        
        result = self._run(
            task_type="captioning",
            user_message=user_msg,
            image_path=image_path,
            trace=trace,
        )
        
        prediction = AnswerParser.parse_caption(result["response"])
        
        trace.finalize(result["response"], parsed_answer=prediction)
        trace.ground_truth = ground_truth
        self.trace_logger.save_trace(trace)
        
        return {
            "image_path": image_path,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "raw_response": result["response"],
            "trace": trace,
        }
    
    def _run(
        self,
        task_type: str,
        user_message: str,
        image_path: str,
        trace: Optional[AgentTrace] = None,
        include_image: Optional[bool] = None,
        dataset: Optional[str] = None,
        options: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Internal run method.
        
        Args:
            task_type: Type of task (diagnosis, concept, etc.)
            user_message: User message content
            image_path: Path to the image file
            trace: Optional trace logger
            include_image: Override for including image in message (None uses instance setting)
            dataset: Dataset name for diagnosis task (used by chatbot to inject options)
            options: List of valid options for diagnosis task (used by chatbot to inject options)
        """
        import base64
        from pathlib import Path
        
        # Determine whether to include image in message
        should_include_image = include_image if include_image is not None else self.include_image_in_message
        
        if should_include_image and image_path and Path(image_path).exists():
            # Create multimodal message with base64 encoded image
            with open(image_path, "rb") as f:
                b64_image = base64.b64encode(f.read()).decode("utf-8")
            
            # Determine media type
            suffix = Path(image_path).suffix.lower()
            media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", 
                          ".png": "image/png", ".gif": "image/gif"}
            media_type = media_types.get(suffix, "image/jpeg")
            
            content = [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64_image}"}},
                {"type": "text", "text": user_message}
            ]
            messages = [HumanMessage(content=content)]
        else:
            # Standard text-only message
            messages = [HumanMessage(content=user_message)]
        
        input_state = {
            "messages": messages,
            "current_image": image_path,
            "task_type": task_type,
            "dataset": dataset,  # For chatbot to inject options into system prompt
            "options": options,  # For chatbot to inject options into system prompt
            "tool_call_count": 0,  # Initialize tool call counter
            # Critic node state initialization
            "critic_retry_count": 0,
            "tools_called": [],
            "last_critic_feedback": None,
            "evidence_summaries": [],  # LLM summary responses (when critic_llm_summary=True)
            "image_reinject_count": 0,  # Image reinject counter (when critic_image_reinject=True)
            # Tool call caching (prevent duplicate calls with same args)
            "tool_call_cache": {},  # {cache_key -> result}
            # Text RAG query tracking (for conflict investigation check)
            "text_rag_queries": [],  # List of all text_rag queries made
        }

        config = {
            "configurable": {"thread_id": f"benchmark_{hash(image_path)}"},
            "recursion_limit": self.recursion_limit,
            "tags": [task_type, dataset or "", self.model_name],
            "metadata": {
                "task_type": task_type,
                "dataset": dataset or "",
                "model": self.model_name,
                "image_path": image_path,
                "enabled_tools": ",".join(t.name for t in self.tools),
            },
            "run_name": f"{task_type}_{dataset or 'default'}",
        }
        result = with_rate_limit_retry(self.agent.invoke, input_state, config=config)

        # Extract tool calls and results from messages for tracing
        if trace:
            messages = result["messages"]

            # Build a map of tool_call_id -> ToolMessage content
            tool_results_map: Dict[str, str] = {}
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    tool_results_map[msg.tool_call_id] = msg.content

            # Now extract tool calls with their results
            step_count = 0
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    step_count += 1
                    step = AgentStep(
                        step_number=step_count,
                        llm_input_messages=[],
                        llm_output=msg.content or "",
                    )
                    for tc in msg.tool_calls:
                        # Get tool result from ToolMessage
                        tool_call_id = tc.get("id", "")
                        tool_result = tool_results_map.get(tool_call_id, "[no result captured]")
                        # Get timing from registry (pop to avoid reusing)
                        tool_duration = tool_timing_registry.pop_latest(tc["name"])

                        step.tool_calls.append(ToolCall(
                            tool_name=tc["name"],
                            arguments=tc["args"],
                            result=tool_result,
                            duration_ms=tool_duration,
                        ))
                    trace.add_step(step)

            # Capture critic retry information
            trace.critic_retries = result.get("critic_retry_count", 0)
            
            # Capture evidence summaries (LLM-generated summaries during critic retry)
            trace.evidence_summaries = result.get("evidence_summaries", [])
            
            # ================================================================
            # Identify and record critic retry steps with full pipeline info
            # ================================================================
            critic_retry_count = 0
            for i, msg in enumerate(messages):
                if isinstance(msg, SystemMessage):
                    content = msg.content
                    if content and "Critic Evaluation" in content:
                        # Found a critic retry event
                        critic_retry_count += 1
                        trace.critic_feedback_history.append(content)
                        
                        # Check if next message is image reinject (HumanMessage with image)
                        image_reinjected = False
                        if i + 1 < len(messages):
                            next_msg = messages[i + 1]
                            if isinstance(next_msg, HumanMessage):
                                next_content = next_msg.content
                                if isinstance(next_content, list):
                                    image_reinjected = any(
                                        isinstance(item, dict) and item.get("type") == "image_url"
                                        for item in next_content
                                    )
                        
                        # Find LLM responses after retry
                        # When image_reinjected=True:
                        #   - First AIMessage (with text, no tool calls) = image_analysis
                        #   - Second AIMessage (with tool calls) = llm_response_after_retry
                        # When image_reinjected=False:
                        #   - First AIMessage = llm_response_after_retry
                        image_analysis = ""
                        llm_response = ""
                        ai_message_count = 0
                        
                        for j in range(i + 1, len(messages)):
                            if isinstance(messages[j], AIMessage):
                                ai_msg = messages[j]
                                text_content = ai_msg.content or ""
                                
                                # Check if this AIMessage has tool calls
                                has_tool_calls = hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls
                                tool_calls_info = ""
                                if has_tool_calls:
                                    tool_names = [tc.get('name', 'unknown') for tc in ai_msg.tool_calls]
                                    tool_calls_info = f"[Tool calls: {', '.join(tool_names)}]"
                                
                                ai_message_count += 1
                                
                                if image_reinjected:
                                    # First AIMessage: image analysis (should have text, no tool calls)
                                    if ai_message_count == 1:
                                        if text_content and not has_tool_calls:
                                            image_analysis = text_content
                                        elif text_content:
                                            # Has both text and tool calls - use text as analysis
                                            image_analysis = text_content
                                            llm_response = tool_calls_info
                                            break
                                        elif tool_calls_info:
                                            # No text, only tool calls (shouldn't happen with new node)
                                            llm_response = tool_calls_info
                                            break
                                    # Second AIMessage: tool calls response
                                    elif ai_message_count == 2:
                                        if text_content:
                                            llm_response = text_content
                                        elif tool_calls_info:
                                            llm_response = tool_calls_info
                                        break
                                else:
                                    # No image reinject: first AIMessage is the response
                                    if text_content:
                                        llm_response = text_content
                                    elif tool_calls_info:
                                        llm_response = tool_calls_info
                                    break
                        
                        # Get corresponding evidence summary
                        summary = None
                        summary_timestamp = None
                        if critic_retry_count <= len(trace.evidence_summaries):
                            summary = trace.evidence_summaries[critic_retry_count - 1]
                            # Use the summary's timestamp for accurate chronological ordering
                            summary_timestamp = summary.get("timestamp")
                        
                        # Create CriticRetryStep with accurate timestamp
                        trace.critic_retry_steps.append(CriticRetryStep(
                            retry_number=critic_retry_count,
                            critic_feedback=content,
                            evidence_summary=summary,
                            image_reinjected=image_reinjected,
                            image_analysis=image_analysis[:2000] if image_analysis else "",  # Truncate
                            llm_response_after_retry=llm_response[:1000] if llm_response else "",  # Truncate
                            timestamp=summary_timestamp or datetime.now().isoformat(),
                        ))
            
            # Update critic_options with accurate values
            trace.critic_options = {
                "llm_summary": len(trace.evidence_summaries) > 0,
                "image_reinject": result.get("image_reinject_count", 0) > 0,
            }

        return {
            "response": result["messages"][-1].content,
            "messages": result["messages"],
            "critic_retries": result.get("critic_retry_count", 0),
        }

