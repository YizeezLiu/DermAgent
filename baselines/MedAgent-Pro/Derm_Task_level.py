"""
Task-level planning for dermatology benchmarks.

Generates plan.json for each Task (1-3) using:
  - RAG from authoritative dermatology URLs (Option B)
  - Planner LLM with domain-specific prompt context
"""

import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()

from RAG import RAG_Module
from Planner import Planner

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ============================================================================
# RAG: Dermatology URLs (Option B — keep original RAG_Module, swap URLs)
# ============================================================================
DERM_RAG_URLS = [
    "https://dermnetnz.org/topics/melanoma-pathology",
    "https://dermnetnz.org/topics/dermoscopy-of-melanoma",
    "https://dermnetnz.org/topics/basal-cell-carcinoma-pathology",
    "https://dermnetnz.org/topics/squamous-cell-carcinoma-of-the-skin-pathology",
    "https://dermnetnz.org/topics/seborrhoeic-keratosis-pathology",
    "https://dermnetnz.org/topics/actinic-keratosis-pathology",
    "https://dermnetnz.org/topics/naevus-pathology",
    "https://dermnetnz.org/topics/dermoscopy-of-pigmented-lesions",
    "https://www.aad.org/public/diseases/skin-cancer",
]

# ============================================================================
# Per-task configuration
# ============================================================================
TASK_CONFIGS = {
    1: {
        "data_root": "Dermatology/task1_diagnosis",
        "rag_prompt": (
            "What are the key dermoscopic features for differentiating "
            "melanoma, nevus, basal cell carcinoma, actinic keratosis, "
            "seborrheic keratosis, and squamous cell carcinoma?"
        ),
        "domain_hint": (
            "Important domain context for dermatology:\n"
            "- Skin lesion classification uses dermoscopic structures: pigment network, "
            "dots/globules, streaks, blue-white veil, regression structures, vascular patterns.\n"
            "- Use zero-shot classification (tool 2) for initial differential diagnosis.\n"
            "- Use concept annotation (tool 3) to extract evidence-based dermoscopic features.\n"
            "- Use retrieval (tool 4) to find similar cases from dermatology databases.\n"
            "- Use VLM (tool 1) for qualitative assessment integrating ALL prior evidence.\n"
            "- The final qualitative step should determine the most likely diagnosis "
            "from the candidate list.\n"
        ),
    },
    2: {
        "data_root": "Dermatology/task2_concept",
        "rag_prompt": (
            "What are the dermoscopic concepts used in skin lesion analysis? "
            "Describe pigment network, blue-white veil, vascular structures, "
            "dots and globules, streaks, and regression structures."
        ),
        "domain_hint": (
            "Important domain context:\n"
            "- This is a concept annotation task: determine which dermoscopic "
            "concepts are present.\n"
            "- Use concept annotation (tool 2) to extract all concepts with scores.\n"
            "- For each concept, use VLM (tool 1) to verify or refute its presence.\n"
            "- List EACH concept as a SEPARATE qualitative step.\n"
            "- Use retrieval (tool 3) to find similar cases for cross-reference.\n"
        ),
    },
    3: {
        "data_root": "Dermatology/task3_captioning",
        "rag_prompt": (
            "How to write a comprehensive clinical description of a skin lesion? "
            "What features should be included in a dermatology image caption?"
        ),
        "domain_hint": (
            "Important domain context:\n"
            "- This is a captioning task: produce a comprehensive clinical description.\n"
            "- Use classification (tool 2) to identify likely disease category.\n"
            "- Use concept annotation (tool 3) for structured features.\n"
            "- Use retrieval (tool 4) for reference descriptions from similar cases.\n"
            "- The FINAL step must be a qualitative VLM step that synthesises ALL "
            "evidence into a single coherent clinical description paragraph.\n"
            "- The final step output_type should be 'final indicator'.\n"
        ),
    },
}


def build_planner_prompt(input_desc: str, disease_goal: str, domain_hint: str) -> str:
    return (
        "Plan a step-by-step, executable workflow using ONLY the available tools.\n"
        f"Input: {input_desc}\n"
        f"Goal: {disease_goal}\n\n"
        f"{domain_hint}\n"
        "Output format (STRICT): an array of objects with fields "
        "[id, tool, action_type, action, input_type, output_type, output_path].\n"
        "- id starts from 1 and increases by 1\n"
        "- tool is an ARRAY of integers (tool ids from the toolset)\n"
        "- action_type is a STRING: 'qualitative' or 'quantitative'\n"
        "- input_type is an ARRAY of integers; use 0 for raw/original inputs, "
        "or a prior step's id if the input is that step's output\n"
        "- The field output_type MUST be EXACTLY one of: "
        "'intermediate result' or 'final indicator'\n"
        "- For any non-image output, set output_path EXACTLY to 'diagnosis.json'\n"
        "- Use a VLM tool to observe potential qualitative indicators; "
        "list EACH indicator as a SEPARATE step.\n"
        "- Qualitative observation/judgement steps MUST set output_type='final indicator'\n"
        "- Quantitative tool steps MUST set output_type='intermediate result' "
        "and be followed by a qualitative VLM judgement step.\n"
        "- Steps must follow strict logical order with no forward references.\n"
        "Return ONLY the JSON array."
    )


def main():
    parser = argparse.ArgumentParser(description="Generate dermatology plans")
    parser.add_argument(
        "--tasks",
        type=str,
        default="1,2,3",
        help="Comma-separated task ids to plan (default: 1,2,3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="chatgpt-4o-latest",
        help="Planner LLM model name",
    )
    args = parser.parse_args()

    task_ids = [int(t.strip()) for t in args.tasks.split(",")]

    rag = RAG_Module(openai_api_key=OPENAI_API_KEY, url_list=DERM_RAG_URLS)
    planner = Planner(api_key=OPENAI_API_KEY)

    for tid in task_ids:
        cfg = TASK_CONFIGS[tid]
        data_root = cfg["data_root"]

        print(f"\n{'='*60}")
        print(f"Task {tid}: Planning  ({data_root})")
        print(f"{'='*60}")

        # RAG retrieval
        print(f"[RAG] Querying: {cfg['rag_prompt'][:80]}...")
        rag_result = rag.query(cfg["rag_prompt"])
        print(f"[RAG] Retrieved {len(rag_result)} chars of context")

        # Load task + toolset
        with open(os.path.join(data_root, "task.json"), "r") as f:
            task = json.load(f)
        with open(os.path.join(data_root, "toolset.json"), "r") as f:
            toolset = json.load(f)

        input_desc = str(task.get("input", "")).strip()
        disease_goal = str(task.get("disease", "")).strip()

        prompt = build_planner_prompt(input_desc, disease_goal, cfg["domain_hint"])

        # Run planner
        print(f"[Planner] Generating plan with {args.model}...")
        plan = planner.plan(
            data_root,
            prompt,
            rag_result,
            filename="plan.json",
            toolset=toolset,
            model=args.model,
        )
        print(f"[Planner] Generated {len(plan)} steps → {data_root}/plan.json")
        for step in plan:
            print(
                f"  Step {step['id']}: [{step['action_type']}] {step['action']}"
            )

    print("\nAll plans generated. Review them before running Derm_Case_level.py.")


if __name__ == "__main__":
    main()
