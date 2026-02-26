#!/usr/bin/env python3
"""
Benchmark script for Skin Agent on SNU 500-sample dataset.

Loads samples from MAKE_SNU_500.csv and runs diagnosis using
the BenchmarkRunner with full tracing support.

SNU dataset contains 134 skin disease classes.

Usage:
    python scripts/run_task1_snu_500_agent.py
    python scripts/run_task1_snu_500_agent.py --max-samples 10
    python scripts/run_task1_snu_500_agent.py --device cuda --model-name gpt-4o
    
Resume mode (retry failed/incomplete samples):
    python scripts/run_task1_snu_500_agent.py --resume results/agent/task1_diagnosis/SNU_500/20260126_113919
    python scripts/run_task1_snu_500_agent.py --resume results/agent/task1_diagnosis/SNU_500/20260126_113919 --retry-errors
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from skin_agent.benchmark_agent import (
    BenchmarkRunner,
    create_benchmark_agent,
    create_tools,
)
from skin_agent.tracing import TraceLogger
from skin_agent.resume import ResumeManager

# =============================================================================
# Constants
# =============================================================================

# SNU class names (134 classes, already full names in diag column)
SNU_CLASSES = [
    "abnormal", "abscess", "acanthosis nigricans", "acne", "actinic keratosis",
    "acute generalized exanthematous pustulosis", "alopecia areata", "amyloidosis",
    "androgenic alopecia", "angiofibroma", "angiokeratoma", "angular cheilitis",
    "atopic dermatitis", "basal cell carcinoma", "becker nevus", "blue nevus",
    "bowen disease", "bullous disease", "cafe au lait macule", "cellulitis",
    "condyloma", "confluent and reticulated papillomatosis", "congenital nevus",
    "contact dermatitis", "dermatofibroma", "drug eruption", "eczema herpeticum",
    "epidermal cyst", "epidermal nevus", "erythema ab igne", "erythema annulare centrifugum",
    "erythema multiforme", "erythema nodosum", "exfoliative dermatitis", "fifth disease",
    "folliculitis", "freckle", "furuncle", "granuloma annulare", "guttate psoriasis",
    "hand eczema", "hemangioma", "herpes simplex", "herpes zoster", "hypertrophic scar",
    "idiopathic guttate hypomelanosis", "impetigo", "inflammed cyst", "ingrowing nail",
    "insect bite", "juvenile xanthogranuloma", "keloid", "keratoacanthoma", "keratoderma",
    "keratosis pilaris", "lentigo", "lichen amyloidosis", "lichen nitidus", "lichen planus",
    "lichen simplex chronicus", "lichen striatus", "livedo reticularis", "lupus erythematosus",
    "lymphangioma", "malignant melanoma", "melanocytic nevus", "melanonychia", "melasma",
    "milia", "molluscum contagiosum", "morphea", "mucocele", "mucosal melanotic macule",
    "mucous cyst", "nail dystrophy", "neurofibroma", "neurofibromatosis", "nevus depigmentosus",
    "nevus spilus", "nummular eczema", "onycholysis", "onychomycosis", "organoid nevus",
    "ota nevus", "palmoplantar pustulosis", "paronychia", "perioral dermatitis",
    "pigmented progressive purpuric dermatosis", "pityriasis alba", "pityriasis lichenoides chronica",
    "pityriasis lichenoides et varioliformis acuta", "pityriasis rosea", "poikiloderma",
    "pompholyx", "porokeratosis", "poroma", "port wine stain", "prurigo nodularis",
    "prurigo pigmentosa", "psoriasis", "pustular psoriasis", "pyoderma gangrenosum",
    "pyogenic granuloma", "riehl melanosis", "rosacea", "scabies", "sebaceous hyperplasia",
    "seborrheic dermatitis", "seborrheic keratosis", "skin tag", "squamous cell carcinoma",
    "staphylococcal scalded skin syndrome", "steatocystoma multiplex", "striae distensae",
    "subungual hematoma", "syringoma", "telangiectasia", "tinea corporis", "tinea cruris",
    "tinea faciale", "tinea pedis", "tinea versicolor", "urticaria", "urticaria pigmentosa",
    "urticarial vasculitis", "varicella", "vasculitis", "venous lake", "verruca plana",
    "viral exanthem", "vitiligo", "wart", "xanthelasma", "xerotic eczema",
]

# Default paths
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "SNU" / "MAKE_SNU_500.csv"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "SNU" / "images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "agent" / "task1_diagnosis" / "SNU_500"

# Default tool configuration
DEFAULT_ENABLED_TOOLS = ["panderm", "make", "qwen_vqa", "rag"]
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MODEL_NAME = "gpt-4o"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Skin Agent benchmark on SNU 500-sample dataset (134 classes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda/cpu). Default: cuda"
    )
    parser.add_argument(
        "--model-name", type=str, default=DEFAULT_MODEL_NAME,
        help=f"LLM model name. Default: {DEFAULT_MODEL_NAME}"
    )
    parser.add_argument(
        "--enabled-tools", type=str, default=",".join(DEFAULT_ENABLED_TOOLS),
        help=f"Comma-separated list of tools. Default: {','.join(DEFAULT_ENABLED_TOOLS)}"
    )
    parser.add_argument(
        "--qwen-model-id", type=str, default=DEFAULT_QWEN_MODEL,
        help=f"Qwen model ID. Default: {DEFAULT_QWEN_MODEL}"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}"
    )
    parser.add_argument(
        "--csv-path", type=str, default=str(DEFAULT_CSV_PATH),
        help=f"Path to benchmark CSV. Default: {DEFAULT_CSV_PATH}"
    )
    parser.add_argument(
        "--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT),
        help=f"Path to dataset root directory. Default: {DEFAULT_DATASET_ROOT}"
    )
    parser.add_argument(
        "--use-flash-attn", action="store_true",
        help="Enable Flash Attention 2 for Qwen (requires flash-attn package)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from existing result directory (timestamp folder path). "
             "Will complete pending samples and optionally retry errors."
    )
    parser.add_argument(
        "--retry-errors", action="store_true",
        help="When resuming, also retry samples that had errors (default: only incomplete)"
    )
    parser.add_argument(
        "--enable-lru", action="store_true",
        help="Enable LRU-based GPU memory management (unload least recently used tools)"
    )
    parser.add_argument(
        "--max-loaded-models", type=int, default=2,
        help="Max models to keep loaded in GPU when LRU enabled (default: 2)"
    )
    # Critic options (generalizable for all tasks)
    parser.add_argument(
        "--enable-critic", action="store_true",
        help="Enable critic node (retry on low confidence/tool conflicts)"
    )
    parser.add_argument(
        "--critic-llm-summary", action="store_true",
        default=False,
        help="Enable LLM-based evidence summary for critic retries"
    )
    parser.add_argument(
        "--critic-image-reinject", action="store_true",
        default=False,
        help="Re-inject image as HumanMessage when critic triggers retry"
    )
    parser.add_argument(
        "--max-critic-retries", type=int, default=None,
        help="Maximum number of critic retry attempts per sample (default: 2)"
    )
    parser.add_argument(
        "--prompt", type=str, default="default",
        choices=["default", "v2"],
        help="Prompt variant to use. 'v2' uses phased workflow with JSON output."
    )

    return parser.parse_args()


def load_benchmark_samples(csv_path: str, max_samples: int = None) -> pd.DataFrame:
    """Load benchmark samples from CSV."""
    df = pd.read_csv(csv_path)
    
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
    
    return df


def load_results_from_traces(traces_dir: Path, dataset_root: Path) -> List[Dict]:
    """Load existing results from trace files."""
    results = []
    
    for trace_file in traces_dir.glob("*.json"):
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            image_path = data.get("image_path", "")
            image_filename = Path(image_path).name if image_path else ""
            
            # Extract filename from image_path (convert back to CSV format)
            filename = f"data/SNU/images/{image_filename}"
            
            results.append({
                "filename": filename,
                "image_filename": image_filename,
                "image_path": image_path,
                "diag": data.get("ground_truth", ""),
                "ground_truth": data.get("ground_truth", ""),
                "prediction": data.get("parsed_answer"),
                "correct": data.get("correct", False),
                "raw_response": data.get("final_response", ""),
                "trace_id": data.get("trace_id", ""),
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse trace file {trace_file}: {e}")
            continue
    
    return results


def main():
    args = parse_args()
    
    # Parse enabled tools
    enabled_tools = [t.strip() for t in args.enabled_tools.split(",") if t.strip()]
    
    print("=" * 60)
    print("Skin Agent Benchmark - SNU 500 Samples (134 Classes)")
    print("With Full Tracing Support")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Model: {args.model_name}")
    print(f"  Enabled Tools: {enabled_tools}")
    print(f"  Qwen Model: {args.qwen_model_id}")
    print(f"  Flash Attention: {args.use_flash_attn}")
    print(f"  Max Samples: {args.max_samples or 'all'}")
    print(f"  CSV Path: {args.csv_path}")
    print(f"  Dataset Root: {args.dataset_root}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Number of Classes: {len(SNU_CLASSES)}")
    print(f"  Resume: {args.resume or 'No'}")
    print(f"  Retry Errors: {args.retry_errors}")
    print(f"  Enable Critic: {args.enable_critic}")
    print(f"  Critic LLM Summary: {args.critic_llm_summary}")
    print(f"  Critic Image Reinject: {args.critic_image_reinject}")
    print(f"  Max Critic Retries: {args.max_critic_retries or 'default (2)'}")
    print(f"  Prompt Variant: {args.prompt}")
    
    # Load benchmark samples
    print("\nLoading benchmark samples...")
    df_full = load_benchmark_samples(args.csv_path, args.max_samples)
    print(f"  Loaded {len(df_full)} samples")
    
    dataset_root = Path(args.dataset_root)
    
    # Add image_path column for resume matching
    df_full["image_path"] = df_full["filename"].apply(
        lambda x: str(dataset_root / "images" / Path(x).name)
    )
    
    # Handle resume mode
    resume_manager = None
    existing_results = []
    failed_paths: Set[str] = set()
    
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            print(f"ERROR: Resume directory not found: {resume_dir}")
            return 1
        
        resume_manager = ResumeManager(resume_dir)
        resume_manager.print_status_report()
        
        # Get samples to run
        df, failed_paths = resume_manager.get_samples_to_run(
            df_full,
            image_path_col="image_path",
            retry_errors=args.retry_errors,
        )
        
        print(f"\nResume mode:")
        print(f"  Pending samples: {len(df)}")
        print(f"  Failed samples to retry: {len(failed_paths) if args.retry_errors else 0}")
        
        if len(df) == 0:
            print("\nNo samples to run. All samples completed successfully!")
            return 0
        
        # Remove traces for failed samples that will be retried
        if args.retry_errors and failed_paths:
            print(f"\nRemoving {len(failed_paths)} failed traces for retry...")
            resume_manager.remove_traces_for_paths(failed_paths)
        
        # Load existing successful results
        existing_results = load_results_from_traces(resume_manager.traces_dir, dataset_root)
        # Filter out failed ones that will be retried
        if args.retry_errors:
            existing_results = [r for r in existing_results if r.get("image_path") not in failed_paths]
        print(f"  Existing successful results: {len(existing_results)}")
        
        # Use existing timestamp directory
        timestamp_dir = resume_dir
        traces_dir = timestamp_dir / "traces"
        timestamp = resume_manager.get_original_timestamp()
    else:
        df = df_full
        # Create output directory structure
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_dir = output_dir / timestamp
        timestamp_dir.mkdir(exist_ok=True)
        traces_dir = timestamp_dir / "traces"
        traces_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {timestamp_dir}")
    
    # Initialize trace logger
    trace_logger = TraceLogger(log_dir=str(traces_dir))
    
    # Create tools and agent
    print("\nInitializing tools and agent...")
    
    try:
        tools = create_tools(
            device=args.device,
            enabled_tools=enabled_tools,
            qwen_model_id=args.qwen_model_id,
            use_flash_attn=args.use_flash_attn,
        )
        agent = create_benchmark_agent(
            tools=tools,
            model_name=args.model_name,
            device=args.device,
            enabled_tools=enabled_tools,  # Pass for ablation mode detection
            execution_mode="sequential",
            memory_management=args.enable_lru,
            max_loaded_models=args.max_loaded_models,
            enable_critic=args.enable_critic if args.enable_critic else None,
            critic_llm_summary=args.critic_llm_summary,
            critic_image_reinject=args.critic_image_reinject,
            max_critic_retries=args.max_critic_retries,
            prompt_variant=args.prompt,
        )
        print("Agent initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Determine if image should be included in message (for text_rag ablation)
    # When using text_rag, GPT-4o needs to see the image directly
    include_image_in_message = "text_rag" in enabled_tools and len(enabled_tools) == 1
    if include_image_in_message:
        print("  [text_rag ablation] Enabling direct image input to GPT-4o")
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        agent=agent,
        tools=tools,
        trace_logger=trace_logger,
        include_image_in_message=include_image_in_message,
        model_name=args.model_name,
    )
    
    # Prepare output file
    output_file = timestamp_dir / f"test_results_{timestamp}.txt"
    dataset_root = Path(args.dataset_root)
    
    results = []
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        # SNU CSV format: filename contains relative path like "data/SNU/images/xxx.jpg"
        filename = row["filename"]
        diag = row["diag"]  # Already full class name
        ground_truth = diag  # Use directly as ground truth
        
        # Extract just the image filename from the path
        image_filename = Path(filename).name
        image_path = dataset_root / "images" / image_filename
        
        print(f"\n{'=' * 60}")
        print(f"Sample {i}/{len(df)}: {image_filename}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Image Path: {image_path}")
        print("=" * 60)
        
        if not image_path.exists():
            result = {
                "filename": filename,
                "image_filename": image_filename,
                "diag": diag,
                "ground_truth": ground_truth,
                "prediction": None,
                "correct": False,
                "error": f"Image not found at {image_path}",
            }
            print(f"ERROR: {result['error']}")
            results.append(result)
            continue
        
        try:
            print("\nRunning diagnosis...")
            
            # Run diagnosis with tracing
            result = runner.run_diagnosis(
                image_path=str(image_path),
                dataset="SNU_500",
                ground_truth=ground_truth,
            )
            
            # Print summary
            print(f"\n--- Results ---")
            print(f"Prediction: {result['prediction']}")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Correct: {result['correct']}")
            
            # Print trace summary
            if result.get("trace"):
                trace_logger.print_trace_summary(result["trace"])
            
            results.append({
                "filename": filename,
                "image_filename": image_filename,
                "diag": diag,
                "ground_truth": ground_truth,
                "prediction": result["prediction"],
                "correct": result["correct"],
                "raw_response": result["raw_response"],
                "trace_id": result["trace"].trace_id if result.get("trace") else None,
            })
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            results.append({
                "filename": filename,
                "image_filename": image_filename,
                "diag": diag,
                "ground_truth": ground_truth,
                "prediction": None,
                "correct": False,
                "error": error_msg,
            })
    
    # Merge with existing results if in resume mode
    if existing_results:
        print(f"\nMerging {len(results)} new results with {len(existing_results)} existing results...")
        all_results = existing_results + results
    else:
        all_results = results
    
    # Calculate accuracy
    correct_count = sum(1 for r in all_results if r.get("correct") is True)
    total = len(all_results)
    accuracy = correct_count / total if total > 0 else 0
    error_count = sum(1 for r in all_results if r.get("error"))
    
    # Print final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY" + (" (After Resume)" if args.resume else ""))
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Correct: {correct_count}")
    print(f"Errors: {error_count}")
    print(f"Accuracy: {accuracy:.1%}")
    
    # Per-class accuracy
    class_stats = {}
    for r in all_results:
        diag = r.get("diag", "unknown")
        if diag not in class_stats:
            class_stats[diag] = {"correct": 0, "total": 0}
        class_stats[diag]["total"] += 1
        if r.get("correct"):
            class_stats[diag]["correct"] += 1
    
    print(f"\nPer-class accuracy ({len(class_stats)} classes):")
    for diag, stats in sorted(class_stats.items()):
        class_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {diag}: {stats['correct']}/{stats['total']} ({class_acc:.1%})")
    
    # Save results to text file
    output_file = timestamp_dir / f"test_results_{timestamp}.txt"
    print(f"\n{'=' * 60}")
    print(f"Saving results to {output_file}")
    print("=" * 60)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Skin Agent Benchmark Test Results")
        if args.resume:
            f.write(" (Resumed)\n")
            f.write(f"Original Timestamp: {timestamp}\n")
            f.write(f"Regenerated: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
        else:
            f.write("\n")
            f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: SNU_500 (134 classes)\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Enabled Tools: {enabled_tools}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write(f"Total: {total}\n")
        f.write(f"Correct: {correct_count}\n")
        f.write(f"Errors: {error_count}\n")
        f.write(f"Accuracy: {accuracy:.1%}\n\n")
        
        f.write(f"Per-class accuracy ({len(class_stats)} classes):\n")
        for diag, stats in sorted(class_stats.items()):
            class_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            f.write(f"  {diag}: {stats['correct']}/{stats['total']} ({class_acc:.1%})\n")
        
        f.write("\n" + "=" * 60 + "\n\n")
        
        for r in all_results:
            f.write(f"Image: {r.get('image_filename', Path(r.get('image_path', '')).name)}\n")
            f.write(f"Ground Truth: {r['ground_truth']}\n")
            f.write(f"Prediction: {r.get('prediction', 'N/A')}\n")
            f.write(f"Correct: {r.get('correct', False)}\n")
            if r.get("trace_id"):
                f.write(f"Trace ID: {r['trace_id']}\n")
            if r.get("error"):
                f.write(f"Error: {r['error']}\n")
            f.write("-" * 40 + "\n")
            if r.get("raw_response"):
                f.write(f"Agent Response:\n{r['raw_response']}\n")
            f.write("\n" + "=" * 60 + "\n\n")
    
    # Save predictions to CSV
    predictions_csv = timestamp_dir / f"predictions_{timestamp}.csv"
    pred_df = pd.DataFrame([
        {
            "filename": r.get("filename", ""),
            "image_filename": r.get("image_filename", Path(r.get("image_path", "")).name),
            "diag": r.get("diag", ""),
            "ground_truth": r["ground_truth"],
            "prediction": r.get("prediction", ""),
            "correct": r.get("correct", False),
            "trace_id": r.get("trace_id", ""),
            "error": r.get("error", ""),
        }
        for r in all_results
    ])
    pred_df.to_csv(predictions_csv, index=False)
    print(f"Predictions saved to: {predictions_csv}")
    
    # Reload all traces for summary (including existing ones in resume mode)
    if args.resume:
        # Reload the trace logger with all traces from disk
        trace_logger_final = TraceLogger(log_dir=str(traces_dir))
        resume_manager.load_existing_traces_to_logger(trace_logger_final)
        # Also add newly created traces
        for t in trace_logger.traces:
            trace_logger_final.traces.append(t)
    else:
        trace_logger_final = trace_logger
    
    # Save traces summary
    traces_summary_path = timestamp_dir / f"traces_summary_{timestamp}.json"
    original_log_dir = trace_logger_final.log_dir
    trace_logger_final.log_dir = timestamp_dir
    trace_logger_final.save_all(f"traces_summary_{timestamp}.json")
    trace_logger_final.log_dir = original_log_dir
    print(f"Traces saved to: {traces_summary_path}")
    
    # Print trace statistics
    stats = trace_logger_final.get_summary_stats()
    print(f"\nTrace Statistics:")
    print(f"  Total traces: {stats['total']}")
    print(f"  Avg duration: {stats.get('avg_duration_ms', 0):.0f}ms")
    print(f"  Avg tool calls: {stats.get('avg_tool_calls', 0):.1f}")
    
    if args.resume:
        print(f"\nResume completed! Results saved to: {timestamp_dir}")
    else:
        print(f"\nBenchmark completed! Results saved to: {timestamp_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
