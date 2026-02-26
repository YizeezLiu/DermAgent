#!/usr/bin/env python3
"""
Benchmark script for Skin Agent on Captioning Task (Task 3).

Uses skin_cap dataset with 100 samples.
Computes BLEU-1, BLEU-4, and ROUGE-L metrics for evaluation.

Usage:
    python scripts/run_task3_captioning_agent.py
    python scripts/run_task3_captioning_agent.py --max-samples 10
    python scripts/run_task3_captioning_agent.py --enabled-tools dermogpt_vqa
    
Resume mode (retry failed/incomplete samples):
    python scripts/run_task3_captioning_agent.py --resume results/agent/task3_captioning/skincap_100/20260126_083212
    python scripts/run_task3_captioning_agent.py --resume results/agent/task3_captioning/skincap_100/20260126_083212 --retry-errors
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set

import numpy as np
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
from benchmark.metrics import compute_captioning_metrics

# =============================================================================
# Dataset Configuration
# =============================================================================

TASK3_DATASET_CONFIG = {
    "csv_path": PROJECT_ROOT / "data" / "skin_cap" / "skin_cap_meta_100.csv",
    "image_dir": PROJECT_ROOT / "datasets" / "skin_cap" / "images",
    "filename_col": "filename",
    "caption_col": "caption_zh_polish_en",
}

# Default tools for Task 3 Captioning (all 6 tools)
DEFAULT_ENABLED_TOOLS = ["dermogpt_vqa", "make", "panderm", "rag", "text_rag", "ontology"]
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MODEL_NAME = "gpt-4o"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Skin Agent benchmark on Captioning Task (Task 3)",
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
        "--output-dir", type=str, default=None,
        help="Output directory. Default: results/agent/task3_captioning/skincap_100"
    )
    parser.add_argument(
        "--use-flash-attn", action="store_true",
        help="Enable Flash Attention 2 for Qwen (requires flash-attn package)"
    )
    parser.add_argument(
        "--include-image-in-message", action="store_true",
        help="Include base64-encoded image in the message to GPT-4o"
    )
    parser.add_argument(
        "--rag-vqa-enhancement", action="store_true",
        help="Enable VQA enhancement for RAG: use DermoGPT to describe retrieved images instead of low-quality database captions"
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
    # Prompt variant
    parser.add_argument(
        "--prompt", type=str, default="default",
        choices=["default", "cot-staged", "cot-flat"],
        help="Prompt variant: 'default' (original), 'cot-staged' (CoT + multi-stage), 'cot-flat' (CoT + flat sequential)"
    )
    
    return parser.parse_args()


def load_results_from_traces(traces_dir: Path, image_dir: Path) -> List[Dict]:
    """Load existing results from trace files."""
    results = []
    
    for trace_file in traces_dir.glob("*.json"):
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            image_path = data.get("image_path", "")
            # Reconstruct filename from image_path
            if image_path:
                rel_path = Path(image_path).relative_to(image_dir) if image_path.startswith(str(image_dir)) else Path(image_path).name
                filename = f"data/skin_cap/{rel_path}" if not str(rel_path).startswith("data/") else str(rel_path)
            else:
                filename = ""
            
            results.append({
                "filename": filename,
                "image_path": image_path,
                "ground_truth": data.get("ground_truth", ""),
                "prediction": data.get("parsed_answer", "") or "",
                "raw_response": data.get("final_response", ""),
                "trace_id": data.get("trace_id", ""),
            })
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Failed to parse trace file {trace_file}: {e}")
            continue
    
    return results


def load_benchmark_samples(csv_path: str, max_samples: int = None) -> pd.DataFrame:
    """Load benchmark samples from CSV."""
    df = pd.read_csv(csv_path)
    
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
    
    return df


def main():
    args = parse_args()
    
    # Get dataset config
    csv_path = TASK3_DATASET_CONFIG["csv_path"]
    image_dir = TASK3_DATASET_CONFIG["image_dir"]
    filename_col = TASK3_DATASET_CONFIG["filename_col"]
    caption_col = TASK3_DATASET_CONFIG["caption_col"]
    
    # Parse enabled tools
    enabled_tools = [t.strip() for t in args.enabled_tools.split(",") if t.strip()]
    
    # Set default output directory
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "agent" / "task3_captioning" / "skincap_100"
    else:
        output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print(f"Skin Agent Benchmark - Task 3 Captioning")
    print(f"Dataset: skin_cap (100 samples)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Model: {args.model_name}")
    print(f"  Enabled Tools: {enabled_tools}")
    print(f"  Qwen Model: {args.qwen_model_id}")
    print(f"  Flash Attention: {args.use_flash_attn}")
    print(f"  Include Image in Message: {args.include_image_in_message}")
    print(f"  RAG VQA Enhancement: {args.rag_vqa_enhancement}")
    print(f"  Max Samples: {args.max_samples or 'all'}")
    print(f"  CSV Path: {csv_path}")
    print(f"  Image Dir: {image_dir}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Resume: {args.resume or 'No'}")
    print(f"  Retry Errors: {args.retry_errors}")
    print(f"  Enable Critic: {args.enable_critic}")
    print(f"  Critic LLM Summary: {args.critic_llm_summary}")
    print(f"  Critic Image Reinject: {args.critic_image_reinject}")
    print(f"  Max Critic Retries: {args.max_critic_retries or 'default (2)'}")
    
    # Load benchmark samples
    print("\nLoading benchmark samples...")
    df_full = load_benchmark_samples(str(csv_path), args.max_samples)
    print(f"  Loaded {len(df_full)} samples")
    
    # Helper to resolve image path
    def resolve_image_path(filename: str) -> str:
        if filename.startswith("data/skin_cap/"):
            relative_path = filename.replace("data/skin_cap/", "")
            return str(image_dir / relative_path)
        else:
            return str(image_dir / "images" / filename)
    
    # Add image_path column for resume matching
    df_full["image_path"] = df_full[filename_col].apply(resolve_image_path)
    
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
        existing_results = load_results_from_traces(resume_manager.traces_dir, image_dir)
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
            rag_vqa_enhancement=args.rag_vqa_enhancement,
        )
        agent = create_benchmark_agent(
            tools=tools,
            model_name=args.model_name,
            device=args.device,
            enabled_tools=enabled_tools,  # Pass for ablation mode detection
            execution_mode="sequential",
            max_tool_calls=20,
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
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        agent=agent,
        tools=tools,
        trace_logger=trace_logger,
        include_image_in_message=args.include_image_in_message,
        model_name=args.model_name,
    )
    
    # Prepare for evaluation
    results = []
    all_ground_truth = []
    all_predictions = []
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        # Parse filename - may contain relative path like "data/skin_cap/images/1490.png"
        filename = row[filename_col]
        ground_truth = row[caption_col]
        
        # Resolve image path
        if filename.startswith("data/skin_cap/"):
            # Remove the prefix since image_dir already points to skin_cap folder
            relative_path = filename.replace("data/skin_cap/", "")
            image_path = image_dir / relative_path
        else:
            image_path = image_dir / "images" / filename
        
        print(f"\n{'=' * 60}")
        print(f"Sample {i}/{len(df)}: {filename}")
        print(f"Ground Truth: {ground_truth[:100]}..." if len(ground_truth) > 100 else f"Ground Truth: {ground_truth}")
        print(f"Image Path: {image_path}")
        print("=" * 60)
        
        if not image_path.exists():
            result = {
                "filename": filename,
                "ground_truth": ground_truth,
                "prediction": "",
                "error": f"Image not found at {image_path}",
            }
            print(f"ERROR: {result['error']}")
            results.append(result)
            all_ground_truth.append(ground_truth)
            all_predictions.append("")
            continue
        
        try:
            print("\nRunning captioning...")
            
            # Run captioning with tracing
            result = runner.run_captioning(
                image_path=str(image_path),
                ground_truth=ground_truth,
            )
            
            prediction = result.get("prediction", "")
            if prediction is None:
                prediction = ""
            
            # Print summary
            print(f"\n--- Results ---")
            print(f"Prediction: {prediction[:150]}..." if len(prediction) > 150 else f"Prediction: {prediction}")
            
            # Print trace summary
            if result.get("trace"):
                trace_logger.print_trace_summary(result["trace"])
            
            results.append({
                "filename": filename,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "raw_response": result.get("raw_response", ""),
                "trace_id": result["trace"].trace_id if result.get("trace") else None,
            })
            
            all_ground_truth.append(ground_truth)
            all_predictions.append(prediction)
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            results.append({
                "filename": filename,
                "ground_truth": ground_truth,
                "prediction": "",
                "error": error_msg,
            })
            all_ground_truth.append(ground_truth)
            all_predictions.append("")
    
    # Merge with existing results if in resume mode
    if existing_results:
        print(f"\nMerging {len(results)} new results with {len(existing_results)} existing results...")
        all_results = existing_results + results
    else:
        all_results = results
    
    # Rebuild all_ground_truth and all_predictions from all_results
    all_ground_truth_final = [r.get("ground_truth", "") for r in all_results]
    all_predictions_final = [r.get("prediction", "") or "" for r in all_results]
    
    # Compute captioning metrics
    print("\n" + "=" * 60)
    print("Computing Captioning Metrics...")
    print("=" * 60)
    
    metrics = compute_captioning_metrics(all_ground_truth_final, all_predictions_final)
    
    print(f"\nCAPTIONING METRICS")
    print(f"  BLEU-1:  {metrics['bleu1']:.4f}")
    print(f"  BLEU-4:  {metrics['bleu4']:.4f}")
    print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
    
    # Save results to text file
    output_file = timestamp_dir / f"test_results_{timestamp}.txt"
    print(f"\n{'=' * 60}")
    print(f"Saving results to {output_file}")
    print("=" * 60)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Skin Agent Benchmark Test Results - Task 3 Captioning")
        if args.resume:
            f.write(" (Resumed)\n")
            f.write(f"Original Timestamp: {timestamp}\n")
            f.write(f"Regenerated: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
        else:
            f.write("\n")
            f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: skin_cap (100 samples)\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Enabled Tools: {enabled_tools}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("SUMMARY METRICS\n")
        f.write(f"BLEU-1:  {metrics['bleu1']:.4f}\n")
        f.write(f"BLEU-4:  {metrics['bleu4']:.4f}\n")
        f.write(f"ROUGE-L: {metrics['rougeL']:.4f}\n")
        f.write("\n" + "=" * 60 + "\n\n")
        
        for r in all_results:
            f.write(f"Filename: {r.get('filename', Path(r.get('image_path', '')).name)}\n")
            f.write(f"Ground Truth: {r['ground_truth']}\n")
            f.write(f"Prediction: {r.get('prediction', '')}\n")
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
    pred_records = []
    for r in all_results:
        record = {
            "filename": r.get("filename", Path(r.get("image_path", "")).name),
            "ground_truth": r["ground_truth"],
            "prediction": r.get("prediction", ""),
            "trace_id": r.get("trace_id", ""),
            "error": r.get("error", ""),
        }
        pred_records.append(record)
    pd.DataFrame(pred_records).to_csv(predictions_csv, index=False)
    print(f"Predictions saved to: {predictions_csv}")
    
    # Save metrics to JSON
    metrics_file = timestamp_dir / f"metrics_{timestamp}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
    
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
