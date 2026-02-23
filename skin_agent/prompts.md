# DermAgent Prompt Templates & Routing Logic

This document summarizes the system prompts and the routing logic used to select them within `benchmark_agent.py`.

---

## 1. Base Tool Description

The `BASE_TOOL_DESCRIPTION` is shared across many system prompts to ensure the LLM understands the capabilities and requirements of each tool.

```markdown
## Available Tools

1. **panderm_classifier**: Zero-shot skin disease classification
   - IMPORTANT: When calling this tool, you MUST pass the exact candidate_diseases 
     list provided in the task options (comma-separated string)
   - Returns: top-k predictions with confidence scores among the specified candidates
   - Best for: Initial diagnosis hypothesis

2. **make_concept_annotator**: Dermoscopic concept extraction
   - Returns: Detected features (pigment_network, dots_globules, etc.)
   - Best for: Feature-level analysis

3. **qwen_vqa**: Vision-language model for description and QA
   - Returns: Free-form text
   - Best for: Detailed description, answering specific questions

4. **rag_retrieval**: Similar case retrieval from knowledge base
   - Use IMAGE MODE ONLY: Provide 'image_path' to find visually similar cases
   - DO NOT use text_query parameter (text mode is disabled for stability)
   - Returns: Similar cases with diagnosis labels and clinical descriptions
   - Best for: Evidence-based reasoning, finding precedents
```

---

## 2. Task-Specific System Prompts

These prompts are used in standard benchmarking mode (all tools enabled) to guide the agent through a multi-step reasoning process.

### Diagnosis Task (`DIAGNOSIS_SYSTEM_PROMPT`)
- **Task Type**: Single-class classification
- **Focus**: Step-by-step cross-validation between classification, features, and RAG.
- **Required Process**: 
    1. **PanDerm**: Initial Classification with candidate list.
    2. **MAKE**: Feature extraction (pigment network, etc.).
    3. **Qwen VQA**: Visual description of the lesion.
    4. **RAG**: Similar case comparison (Image Mode).
    5. **Cross-Validation**: Synthesize all evidence for the final decision.

### Concept Annotation Task (`CONCEPT_SYSTEM_PROMPT`)
- **Task Type**: Multi-label classification
- **Focus**: Identifying all dermatological concepts from a provided list.
- **Strategy**: Detect with MAKE, verify with VLM description, and reference RAG cases to filter false positives and find missing features.

### Captioning Task (`CAPTIONING_SYSTEM_PROMPT`)
- **Task Type**: Text generation
- **Focus**: Generating detailed clinical descriptions using all 5 tools.
- **Available Tools**: DermoGPT VQA, MAKE, PanDerm, RAG (Image), Text RAG (Guidelines)
- **Multi-Stage Process**:
    1. **Stage 1 - Visual Analysis**: DermoGPT for morphological description + MAKE for concept checklist.
    2. **Stage 2 - Diagnostic Context**: PanDerm for hypothesis + RAG for similar case references.
    3. **Stage 3 - Terminology Enhancement**: Text RAG for guideline-based clinical vocabulary.
    4. **Stage 4 - Caption Synthesis**: Combine all evidence into structured description.
- **Output Structure**: Lesion Type → Color → Surface → Borders → Clinical Context.
- **Quality Requirements**: 50-150 words, professional terminology, mention all detected concepts.

### VQA Task (`VQA_SYSTEM_PROMPT`)
- **Task Type**: Question answering
- **Focus**: Answering specific questions about skin images.
- **Routing Logic**: Uses different tool combinations based on question type (e.g., Diagnosis vs. Description vs. Comparison).

---

## 3. Ablation Study Prompts

Ablation prompts are designed to restrict the agent to a specific tool or combination, forcing it to reason using limited information to measure tool efficacy.

### 3.1 Diagnosis Ablation Prompts (`DIAGNOSIS_ABLATION_PROMPTS`)

These prompts guide single-class classification when only specific tools are available.

| Tool | Prompt Key | Core Logic |
| :--- | :--- | :--- |
| **PanDerm** | `panderm` | Call PanDerm once, select highest confidence. |
| **MAKE** | `make` | Call MAKE once, map features to diagnosis via medical knowledge. |
| **Qwen VQA** | `qwen_vqa` | Call VQA once with a diagnostic question. |
| **RAG** | `rag` | Call RAG once (Image Mode), use majority voting. |
| **DermoGPT** | `dermogpt_vqa` | Call DermoGPT once, parse XML for `<final_diagnosis>`. |
| **PanDerm + RAG** | `panderm,rag` | Cross-validate zero-shot classification with RAG retrieval. |

### 3.2 Concept Annotation Ablation Prompts (`CONCEPT_ABLATION_PROMPTS`)

These prompts guide **multi-label classification** when only specific tools are available. Key differences from diagnosis ablation:

- Emphasizes that **multiple concepts can be present** simultaneously.
- Instructs the model to map detected features to the **"Valid Concept List"**.
- Enforces output format: `FINAL_ANSWER: [concept1, concept2, ...]` or `FINAL_ANSWER: none`.

| Tool | Prompt Key | Core Logic |
| :--- | :--- | :--- |
| **MAKE** | `make` | Call MAKE once, map detected features to concept list. |
| **Qwen VQA** | `qwen_vqa` | Call VQA once to describe features, then map to concept list. |
| **RAG** | `rag` | Call RAG once (Image Mode), use weighted voting from similar cases. |
| **PanDerm** | `panderm` | Use PanDerm for concept similarity scoring. |
| **DermoGPT** | `dermogpt_vqa` | Call DermoGPT once, map response to concept list. |

### 3.3 Captioning Ablation Prompts (`CAPTIONING_ABLATION_PROMPTS`)

These prompts guide **text generation** (clinical description) when only specific tools are available.

| Tool | Prompt Key | Core Logic |
| :--- | :--- | :--- |
| **DermoGPT VQA** | `dermogpt_vqa` | Call DermoGPT once with descriptive query, generate caption from response. |
| **Qwen VQA** | `qwen_vqa` | Call Qwen VQA once with descriptive query, structure as clinical caption. |
| **MAKE** | `make` | Extract concepts, convert to narrative clinical description. |
| **RAG** | `rag` | Find similar cases (Image Mode), adapt their descriptions. |
| **Text RAG** | `text_rag` | Direct visual analysis + guideline lookup for terminology. |
| **DermoGPT + MAKE** | `dermogpt_vqa,make` | VLM description + concept verification for comprehensive caption. |

---

## 4. Prompt Registry Structure

The prompts are organized in a hierarchical dictionary for task-aware selection:

```python
# Task-specific ablation prompt registries
DIAGNOSIS_ABLATION_PROMPTS = {
    "panderm": ABLATION_PANDERM_PROMPT,
    "make": ABLATION_MAKE_PROMPT,
    "rag": ABLATION_RAG_PROMPT,
    "text_rag": ABLATION_TEXT_RAG_DIAGNOSIS_PROMPT,
    "panderm,rag": ABLATION_PANDERM_RAG_PROMPT,
    # ...
}

CONCEPT_ABLATION_PROMPTS = {
    "make": CONCEPT_ABLATION_MAKE_PROMPT,
    "qwen_vqa": CONCEPT_ABLATION_QWEN_VQA_PROMPT,
    "dermogpt_vqa": CONCEPT_ABLATION_DERMOGPT_VQA_PROMPT,
    # ...
}

CAPTIONING_ABLATION_PROMPTS = {
    "dermogpt_vqa": CAPTIONING_ABLATION_DERMOGPT_PROMPT,
    "qwen_vqa": CAPTIONING_ABLATION_QWEN_VQA_PROMPT,
    "make": CAPTIONING_ABLATION_MAKE_PROMPT,
    "rag": CAPTIONING_ABLATION_RAG_PROMPT,
    "text_rag": CAPTIONING_ABLATION_TEXT_RAG_PROMPT,
    "dermogpt_vqa,make": CAPTIONING_ABLATION_DERMOGPT_MAKE_PROMPT,
}

# Master registry
ABLATION_PROMPTS_BY_TASK = {
    "diagnosis": DIAGNOSIS_ABLATION_PROMPTS,
    "concept": CONCEPT_ABLATION_PROMPTS,
    "captioning": CAPTIONING_ABLATION_PROMPTS,
    "vqa": DIAGNOSIS_ABLATION_PROMPTS,  # Fallback
}

# Legacy (backward compatible)
ABLATION_PROMPTS = DIAGNOSIS_ABLATION_PROMPTS
```

---

## 5. Prompt Selection & Routing Logic

The selection logic is handled within `create_benchmark_agent` and the `chatbot` node.

### Ablation Mode Detection (at agent creation)

```python
if enabled_tools is not None:
    if len(enabled_tools) == 1:
        ablation_key = enabled_tools[0]  # e.g., "make"
        ablation_mode = True
    else:
        # Try sorted comma-separated key
        combo_key = ",".join(sorted(enabled_tools))  # e.g., "panderm,rag"
        if combo_key in ABLATION_PROMPTS:
            ablation_mode = True
            ablation_key = combo_key
```

### Runtime Prompt Selection (in `chatbot` function)

The new routing logic considers both `task_type` and `ablation_key`:

```python
def chatbot(state: BenchmarkAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type", "diagnosis")
    
    system_prompt = None
    
    if ablation_mode and ablation_key:
        # 1. Try task-specific ablation prompts first
        task_ablation_prompts = ABLATION_PROMPTS_BY_TASK.get(task_type, {})
        if ablation_key in task_ablation_prompts:
            system_prompt = task_ablation_prompts[ablation_key]
        # 2. Fallback to legacy ABLATION_PROMPTS
        elif ablation_key in ABLATION_PROMPTS:
            system_prompt = ABLATION_PROMPTS[ablation_key]
    
    # 3. If not ablation mode, use normal task prompt
    if system_prompt is None:
        system_prompt = TASK_PROMPTS.get(task_type, DIAGNOSIS_SYSTEM_PROMPT)
```

### Selection Priority

| Priority | Condition | Prompt Source |
| :---: | :--- | :--- |
| 1 | Ablation mode + task-specific ablation exists | `ABLATION_PROMPTS_BY_TASK[task_type][ablation_key]` |
| 2 | Ablation mode + legacy ablation exists | `ABLATION_PROMPTS[ablation_key]` |
| 3 | Normal mode | `TASK_PROMPTS[task_type]` |

### Result Enforcement

All benchmark prompts require the agent to conclude with a structured `FINAL_ANSWER:` tag:

- **Diagnosis**: `FINAL_ANSWER: [disease_name]`
- **Concept**: `FINAL_ANSWER: [concept1, concept2, ...]` or `FINAL_ANSWER: none`
- **Captioning**: `FINAL_ANSWER: [clinical description]`
- **VQA**: `FINAL_ANSWER: [answer]`

---

## 6. Adding New Task-Type Ablation Prompts

To add ablation prompts for a new task type:

1. Create individual prompts: `NEWTASK_ABLATION_TOOL_PROMPT`, etc.
2. Create registry: `NEWTASK_ABLATION_PROMPTS = {"tool": ..., "tool2": ...}`
3. Register in master: `ABLATION_PROMPTS_BY_TASK["newtask"] = NEWTASK_ABLATION_PROMPTS`

**Example (implemented for Captioning):**
- Created 6 prompts: DermoGPT, Qwen VQA, MAKE, RAG, Text RAG, DermoGPT+MAKE
- Registered as `CAPTIONING_ABLATION_PROMPTS`
- Added to `ABLATION_PROMPTS_BY_TASK["captioning"]`

---

**Last Updated**: 2026-01-24  
**Version**: 1.2 (Added Task 3 Captioning ablation prompts with 5-tool support)
