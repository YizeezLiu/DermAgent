"""
Resume Manager for Agent Benchmark Tests

Provides functionality to:
1. Scan existing trace files and identify completed/failed samples
2. Match against original CSV to find pending samples
3. Remove failed traces for retry
4. Regenerate summary files after completion

Usage:
    from skin_agent.resume import ResumeManager
    
    manager = ResumeManager(timestamp_dir)
    pending_df, failed_paths = manager.get_samples_to_run(original_df, retry_errors=True)
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .tracing import TraceLogger, AgentTrace


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TraceStatus:
    """Status of a single trace file."""
    trace_id: str
    image_path: str
    task_type: str
    has_error: bool
    error_details: List[str]
    parsed_answer: Any
    correct: Optional[bool]
    trace_file: Path


# =============================================================================
# Error Detection Patterns
# =============================================================================

ERROR_PATTERNS = [
    r"error",
    r"timed?\s*out",
    r"timeout",
    r"failed",
    r"exception",
    r"cuda.*out.*memory",
    r"oom",
    r"rate.*limit",
    r"429",  # HTTP rate limit error
    r"500",  # Internal server error
    r"502",  # Bad gateway
    r"503",  # Service unavailable
]

ERROR_REGEX = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)


def detect_error_in_text(text: str) -> bool:
    """Check if text contains error indicators."""
    if not text:
        return False
    return bool(ERROR_REGEX.search(text))


def analyze_trace_for_errors(trace_data: Dict) -> Tuple[bool, List[str]]:
    """
    Analyze a trace dictionary for errors.
    
    Returns:
        (has_error, error_details): Whether errors were found and list of error descriptions
    """
    errors = []
    
    # Check parsed_answer
    if trace_data.get("parsed_answer") is None:
        # Only flag as error if there's also an error indicator elsewhere
        pass
    
    # Check steps and tool calls
    for step in trace_data.get("steps", []):
        for tc in step.get("tool_calls", []):
            # Check explicit error field
            if tc.get("error"):
                errors.append(f"Tool {tc.get('tool_name', 'unknown')}: {tc['error']}")
            
            # Check result for error patterns
            result = tc.get("result", "")
            if isinstance(result, str) and detect_error_in_text(result):
                # Extract relevant part of error message
                if "error" in result.lower() or "timeout" in result.lower():
                    # Truncate long error messages
                    error_snippet = result[:200] + "..." if len(result) > 200 else result
                    errors.append(f"Tool {tc.get('tool_name', 'unknown')} result: {error_snippet}")
    
    # Check final response for critical errors
    final_response = trace_data.get("final_response", "")
    if "FINAL_ANSWER" not in final_response and trace_data.get("parsed_answer") is None:
        # No valid answer was generated
        errors.append("No valid FINAL_ANSWER found in response")
    
    return len(errors) > 0, errors


# =============================================================================
# Resume Manager
# =============================================================================

class ResumeManager:
    """
    Manages resume functionality for agent benchmark tests.
    
    Args:
        timestamp_dir: Path to the timestamp directory containing traces/ subdirectory
    """
    
    def __init__(self, timestamp_dir: str | Path):
        self.timestamp_dir = Path(timestamp_dir)
        self.traces_dir = self.timestamp_dir / "traces"
        
        if not self.timestamp_dir.exists():
            raise FileNotFoundError(
                f"Resume directory not found: {self.timestamp_dir}\n"
                f"Please provide a valid timestamp directory from a previous run.\n"
                f"Example: results/agent/task1_diagnosis/SNU_500/20260126_113919"
            )
        
        if not self.traces_dir.exists():
            # Create traces directory if it doesn't exist (e.g., run was interrupted before any traces)
            print(f"Warning: traces/ directory not found in {self.timestamp_dir}")
            print(f"Creating empty traces directory. All samples will be run.")
            self.traces_dir.mkdir(parents=True, exist_ok=True)
        
        self._trace_cache: Dict[str, TraceStatus] = {}
    
    def scan_existing_traces(self) -> Dict[str, TraceStatus]:
        """
        Scan all trace files in the traces directory.
        
        Returns:
            Dict mapping image_path to TraceStatus
        """
        if self._trace_cache:
            return self._trace_cache
        
        traces = {}
        
        for trace_file in self.traces_dir.glob("*.json"):
            try:
                with open(trace_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                image_path = data.get("image_path", "")
                if not image_path:
                    continue
                
                has_error, error_details = analyze_trace_for_errors(data)
                
                status = TraceStatus(
                    trace_id=data.get("trace_id", ""),
                    image_path=image_path,
                    task_type=data.get("task_type", ""),
                    has_error=has_error,
                    error_details=error_details,
                    parsed_answer=data.get("parsed_answer"),
                    correct=data.get("correct"),
                    trace_file=trace_file,
                )
                
                traces[image_path] = status
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse trace file {trace_file}: {e}")
                continue
        
        self._trace_cache = traces
        return traces
    
    def get_completed_paths(self) -> Set[str]:
        """Get set of image paths that completed successfully (no errors)."""
        traces = self.scan_existing_traces()
        return {path for path, status in traces.items() if not status.has_error}
    
    def get_failed_paths(self) -> Set[str]:
        """Get set of image paths that had errors."""
        traces = self.scan_existing_traces()
        return {path for path, status in traces.items() if status.has_error}
    
    def get_all_traced_paths(self) -> Set[str]:
        """Get all image paths that have trace files (regardless of status)."""
        traces = self.scan_existing_traces()
        return set(traces.keys())
    
    def remove_traces_for_paths(self, image_paths: Set[str]) -> int:
        """
        Remove trace files for the given image paths.
        
        Args:
            image_paths: Set of image paths whose traces should be removed
            
        Returns:
            Number of trace files removed
        """
        traces = self.scan_existing_traces()
        removed = 0
        
        for path in image_paths:
            if path in traces:
                trace_file = traces[path].trace_file
                if trace_file.exists():
                    trace_file.unlink()
                    removed += 1
                    print(f"  Removed: {trace_file.name}")
        
        # Clear cache after removal
        self._trace_cache = {}
        
        return removed
    
    def get_samples_to_run(
        self,
        original_df: pd.DataFrame,
        image_path_col: str = "image_path",
        retry_errors: bool = True,
    ) -> Tuple[pd.DataFrame, Set[str]]:
        """
        Determine which samples need to be run.
        
        Args:
            original_df: Original DataFrame with all samples
            image_path_col: Column name containing image paths
            retry_errors: Whether to retry samples that had errors
            
        Returns:
            (samples_to_run, failed_paths): DataFrame of samples to run, set of failed paths
        """
        all_traced = self.get_all_traced_paths()
        failed_paths = self.get_failed_paths()
        
        # Get all image paths from original DataFrame
        all_sample_paths = set(original_df[image_path_col].astype(str))
        
        # Find paths that haven't been traced at all
        pending_paths = all_sample_paths - all_traced
        
        # Determine which paths to run
        paths_to_run = pending_paths.copy()
        if retry_errors:
            paths_to_run |= failed_paths
        
        # Filter DataFrame to only include paths to run
        samples_to_run = original_df[
            original_df[image_path_col].astype(str).isin(paths_to_run)
        ].copy()
        
        return samples_to_run, failed_paths
    
    def print_status_report(self):
        """Print a summary of the current trace status."""
        traces = self.scan_existing_traces()
        
        total = len(traces)
        failed = sum(1 for t in traces.values() if t.has_error)
        success = total - failed
        
        print(f"\n{'='*60}")
        print("Resume Status Report")
        print(f"{'='*60}")
        print(f"Directory: {self.timestamp_dir}")
        print(f"Total traces: {total}")
        print(f"  Successful: {success}")
        print(f"  With errors: {failed}")
        
        if failed > 0:
            print(f"\nFailed samples:")
            for path, status in traces.items():
                if status.has_error:
                    print(f"  - {Path(path).name}")
                    for err in status.error_details[:2]:  # Show first 2 errors
                        print(f"      {err[:80]}...")
        
        print(f"{'='*60}\n")
    
    def get_original_timestamp(self) -> str:
        """Extract the original timestamp from the directory name."""
        return self.timestamp_dir.name
    
    def load_existing_traces_to_logger(self, trace_logger: TraceLogger) -> int:
        """
        Load all existing (non-error) traces into a TraceLogger.
        
        This is useful for regenerating the summary file.
        
        Args:
            trace_logger: TraceLogger instance to populate
            
        Returns:
            Number of traces loaded
        """
        traces = self.scan_existing_traces()
        loaded = 0
        
        for path, status in traces.items():
            if status.has_error:
                continue
            
            try:
                with open(status.trace_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Create AgentTrace from data
                trace = AgentTrace(
                    trace_id=data.get("trace_id", ""),
                    task_type=data.get("task_type", ""),
                    dataset=data.get("dataset", ""),
                    image_path=data.get("image_path", ""),
                    user_query=data.get("user_query", ""),
                    model=data.get("model", "gpt-4o"),
                    temperature=data.get("temperature", 0.1),
                )
                trace.final_response = data.get("final_response", "")
                trace.parsed_answer = data.get("parsed_answer")
                trace.ground_truth = data.get("ground_truth")
                trace.correct = data.get("correct")
                trace.start_time = data.get("start_time", "")
                trace.end_time = data.get("end_time", "")
                trace.total_duration_ms = data.get("total_duration_ms", 0)
                trace.total_tool_calls = data.get("total_tool_calls", 0)
                trace.critic_retries = data.get("critic_retries", 0)
                trace.critic_feedback_history = data.get("critic_feedback_history", [])
                
                trace_logger.traces.append(trace)
                loaded += 1
                
            except Exception as e:
                print(f"Warning: Failed to load trace {status.trace_file}: {e}")
                continue
        
        return loaded


def regenerate_outputs(
    timestamp_dir: Path,
    trace_logger: TraceLogger,
    results: List[Dict[str, Any]],
    task_type: str,
    dataset: str,
    model_name: str,
    enabled_tools: List[str],
    original_timestamp: str,
    class_names: Optional[List[str]] = None,
    concepts: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Regenerate all output files (predictions CSV, test results, traces summary).
    
    Args:
        timestamp_dir: Output directory
        trace_logger: TraceLogger with all traces
        results: List of result dictionaries
        task_type: Type of task (diagnosis, concept, captioning, vqa)
        dataset: Dataset name
        model_name: Model name used
        enabled_tools: List of enabled tools
        original_timestamp: Original timestamp string
        class_names: Class names for classification tasks
        concepts: Concept names for concept tasks
        
    Returns:
        Dict of output file paths
    """
    output_paths = {}
    timestamp = original_timestamp
    
    # Save predictions CSV
    if task_type == "diagnosis":
        pred_df = pd.DataFrame([
            {
                "filename": r.get("filename", ""),
                "image_filename": r.get("image_filename", ""),
                "diag": r.get("diag", ""),
                "ground_truth": r["ground_truth"],
                "prediction": r.get("prediction", ""),
                "correct": r.get("correct", False),
                "trace_id": r.get("trace_id", ""),
                "error": r.get("error", ""),
            }
            for r in results
        ])
    elif task_type == "concept":
        pred_df = pd.DataFrame([
            {
                "image_id": r.get("image_id", ""),
                "image_path": r.get("image_path", ""),
                "ground_truth": ", ".join(r.get("ground_truth", [])) if isinstance(r.get("ground_truth"), list) else r.get("ground_truth", ""),
                "prediction": ", ".join(r.get("prediction", [])) if isinstance(r.get("prediction"), list) else r.get("prediction", ""),
                "trace_id": r.get("trace_id", ""),
                "error": r.get("error", ""),
            }
            for r in results
        ])
    elif task_type == "captioning":
        pred_df = pd.DataFrame([
            {
                "image_id": r.get("image_id", ""),
                "image_path": r.get("image_path", ""),
                "ground_truth": r.get("ground_truth", ""),
                "prediction": r.get("prediction", ""),
                "trace_id": r.get("trace_id", ""),
                "error": r.get("error", ""),
            }
            for r in results
        ])
    elif task_type == "vqa":
        pred_df = pd.DataFrame([
            {
                "image_path": r.get("image_path", ""),
                "question": r.get("question", ""),
                "question_type": r.get("question_type", ""),
                "ground_truth": r.get("ground_truth", ""),
                "prediction": r.get("prediction", ""),
                "trace_id": r.get("trace_id", ""),
                "error": r.get("error", ""),
            }
            for r in results
        ])
    else:
        pred_df = pd.DataFrame(results)
    
    predictions_csv = timestamp_dir / f"predictions_{timestamp}.csv"
    pred_df.to_csv(predictions_csv, index=False)
    output_paths["predictions"] = predictions_csv
    
    # Save test results text file
    output_file = timestamp_dir / f"test_results_{timestamp}.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Skin Agent Benchmark Test Results (Resumed)\n")
        f.write(f"Original Timestamp: {timestamp}\n")
        f.write(f"Regenerated: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Enabled Tools: {enabled_tools}\n")
        f.write("=" * 60 + "\n\n")
        
        # Calculate stats based on task type
        if task_type == "diagnosis":
            correct_count = sum(1 for r in results if r.get("correct") is True)
            total = len(results)
            accuracy = correct_count / total if total > 0 else 0
            error_count = sum(1 for r in results if r.get("error"))
            
            f.write("SUMMARY\n")
            f.write(f"Total: {total}\n")
            f.write(f"Correct: {correct_count}\n")
            f.write(f"Errors: {error_count}\n")
            f.write(f"Accuracy: {accuracy:.1%}\n\n")
        
        elif task_type == "concept":
            error_count = sum(1 for r in results if r.get("error"))
            f.write("SUMMARY\n")
            f.write(f"Total: {len(results)}\n")
            f.write(f"Errors: {error_count}\n\n")
        
        elif task_type == "vqa":
            error_count = sum(1 for r in results if r.get("error"))
            f.write("SUMMARY\n")
            f.write(f"Total: {len(results)}\n")
            f.write(f"Errors: {error_count}\n\n")
        
        f.write("=" * 60 + "\n\n")
        
        for r in results:
            if task_type == "diagnosis":
                f.write(f"Image: {r.get('image_filename', r.get('image_path', 'N/A'))}\n")
                f.write(f"Ground Truth: {r.get('ground_truth', 'N/A')}\n")
                f.write(f"Prediction: {r.get('prediction', 'N/A')}\n")
                f.write(f"Correct: {r.get('correct', False)}\n")
            elif task_type == "concept":
                f.write(f"Image: {r.get('image_id', r.get('image_path', 'N/A'))}\n")
                gt = r.get('ground_truth', [])
                pred = r.get('prediction', [])
                f.write(f"Ground Truth: {', '.join(gt) if isinstance(gt, list) else gt}\n")
                f.write(f"Prediction: {', '.join(pred) if isinstance(pred, list) else pred}\n")
            elif task_type == "vqa":
                f.write(f"Image: {r.get('image_path', 'N/A')}\n")
                f.write(f"Question: {r.get('question', 'N/A')}\n")
                f.write(f"Ground Truth: {r.get('ground_truth', 'N/A')}\n")
                f.write(f"Prediction: {r.get('prediction', 'N/A')}\n")
            else:
                f.write(f"Image: {r.get('image_path', 'N/A')}\n")
                f.write(f"Ground Truth: {r.get('ground_truth', 'N/A')}\n")
                f.write(f"Prediction: {r.get('prediction', 'N/A')}\n")
            
            if r.get("trace_id"):
                f.write(f"Trace ID: {r['trace_id']}\n")
            if r.get("error"):
                f.write(f"Error: {r['error']}\n")
            f.write("-" * 40 + "\n")
            if r.get("raw_response"):
                f.write(f"Agent Response:\n{r['raw_response']}\n")
            f.write("\n" + "=" * 60 + "\n\n")
    
    output_paths["test_results"] = output_file
    
    # Save traces summary
    traces_summary_path = timestamp_dir / f"traces_summary_{timestamp}.json"
    original_log_dir = trace_logger.log_dir
    trace_logger.log_dir = timestamp_dir
    trace_logger.save_all(f"traces_summary_{timestamp}.json")
    trace_logger.log_dir = original_log_dir
    output_paths["traces_summary"] = traces_summary_path
    
    return output_paths
