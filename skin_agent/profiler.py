"""
Profiler module for comprehensive performance tracking.

Automatically enabled when TEST_MODE=true environment variable is set.
Records model load times, LLM inference times, token usage, GPU memory, and tool breakdowns.
"""

import os
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LLMCallRecord:
    """Record of a single LLM call."""
    call_index: int
    duration_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ToolBreakdown:
    """Breakdown of tool execution phases."""
    tool_name: str
    total_ms: float
    phases: Dict[str, float] = field(default_factory=dict)


@dataclass
class GPUSnapshot:
    """Snapshot of GPU memory usage."""
    timestamp: str
    event: str
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float = 0.0


# =============================================================================
# Profiler Singleton
# =============================================================================

class Profiler:
    """
    Global Profiler for performance tracking (thread-safe singleton).
    
    Automatically enabled when TEST_MODE=true environment variable is set.
    Collects:
    - Model load times
    - LLM call durations and token usage
    - Tool execution breakdowns
    - GPU memory snapshots
    
    Usage:
        from skin_agent.profiler import profiler
        
        if profiler.is_enabled():
            start = time.perf_counter()
            # ... do work ...
            profiler.record_model_load("tool_name", (time.perf_counter() - start) * 1000)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance
    
    def _init(self):
        """Initialize profiler state."""
        self._enabled = False
        self._model_load_times: Dict[str, float] = {}
        self._llm_calls: List[LLMCallRecord] = []
        self._tool_breakdowns: Dict[str, ToolBreakdown] = {}
        self._gpu_snapshots: List[GPUSnapshot] = []
        self._llm_call_counter = 0
        self._data_lock = threading.Lock()
    
    def enable(self):
        """Enable profiling."""
        self._enabled = True
    
    def disable(self):
        """Disable profiling."""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self._enabled
    
    # -------------------------------------------------------------------------
    # Model Load Time Recording
    # -------------------------------------------------------------------------
    
    def record_model_load(self, tool_name: str, duration_ms: float):
        """Record model loading time for a tool."""
        if not self._enabled:
            return
        with self._data_lock:
            self._model_load_times[tool_name] = duration_ms
            print(f"    📊 [Profiler] {tool_name} loaded in {duration_ms:.0f}ms")
    
    def get_model_load_times(self) -> Dict[str, float]:
        """Get all recorded model load times."""
        with self._data_lock:
            return dict(self._model_load_times)
    
    # -------------------------------------------------------------------------
    # LLM Call Recording
    # -------------------------------------------------------------------------
    
    def record_llm_call(
        self,
        duration_ms: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        model: str = "",
    ):
        """Record an LLM call with timing and token usage."""
        if not self._enabled:
            return
        with self._data_lock:
            self._llm_call_counter += 1
            record = LLMCallRecord(
                call_index=self._llm_call_counter,
                duration_ms=duration_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens or ((input_tokens or 0) + (output_tokens or 0)),
                model=model,
            )
            self._llm_calls.append(record)
            
            # Log to console
            tokens_info = ""
            if input_tokens is not None or output_tokens is not None:
                tokens_info = f" (in:{input_tokens or '?'}, out:{output_tokens or '?'})"
            print(f"    📊 [Profiler] LLM call #{self._llm_call_counter}: {duration_ms:.0f}ms{tokens_info}")
    
    def get_llm_calls(self) -> List[Dict]:
        """Get all LLM call records as dicts."""
        with self._data_lock:
            return [asdict(r) for r in self._llm_calls]
    
    def get_llm_summary(self) -> Dict[str, Any]:
        """Get summary statistics for LLM calls."""
        with self._data_lock:
            if not self._llm_calls:
                return {}
            
            total_duration = sum(r.duration_ms for r in self._llm_calls)
            total_input = sum(r.input_tokens or 0 for r in self._llm_calls)
            total_output = sum(r.output_tokens or 0 for r in self._llm_calls)
            
            return {
                "total_calls": len(self._llm_calls),
                "total_duration_ms": total_duration,
                "avg_duration_ms": total_duration / len(self._llm_calls),
                "total_input_tokens": total_input if any(r.input_tokens for r in self._llm_calls) else None,
                "total_output_tokens": total_output if any(r.output_tokens for r in self._llm_calls) else None,
            }
    
    # -------------------------------------------------------------------------
    # Tool Breakdown Recording
    # -------------------------------------------------------------------------
    
    def record_tool_phase(self, tool_name: str, phase_name: str, duration_ms: float):
        """Record a phase of tool execution."""
        if not self._enabled:
            return
        with self._data_lock:
            if tool_name not in self._tool_breakdowns:
                self._tool_breakdowns[tool_name] = ToolBreakdown(
                    tool_name=tool_name,
                    total_ms=0.0,
                    phases={},
                )
            self._tool_breakdowns[tool_name].phases[phase_name] = duration_ms
            self._tool_breakdowns[tool_name].total_ms += duration_ms
    
    def record_tool_total(self, tool_name: str, total_ms: float):
        """Record total execution time for a tool (overwrites phase sum)."""
        if not self._enabled:
            return
        with self._data_lock:
            if tool_name not in self._tool_breakdowns:
                self._tool_breakdowns[tool_name] = ToolBreakdown(
                    tool_name=tool_name,
                    total_ms=total_ms,
                    phases={},
                )
            else:
                self._tool_breakdowns[tool_name].total_ms = total_ms
    
    def get_tool_breakdowns(self) -> Dict[str, Dict]:
        """Get all tool breakdowns as dicts."""
        with self._data_lock:
            return {
                name: asdict(breakdown)
                for name, breakdown in self._tool_breakdowns.items()
            }
    
    # -------------------------------------------------------------------------
    # GPU Memory Tracking
    # -------------------------------------------------------------------------
    
    def snapshot_gpu(self, event: str = "snapshot"):
        """Take a snapshot of current GPU memory usage."""
        if not self._enabled:
            return
        if not HAS_TORCH or not torch.cuda.is_available():
            return
        
        with self._data_lock:
            snapshot = GPUSnapshot(
                timestamp=datetime.now().isoformat(),
                event=event,
                allocated_mb=torch.cuda.memory_allocated() / (1024 ** 2),
                reserved_mb=torch.cuda.memory_reserved() / (1024 ** 2),
                max_allocated_mb=torch.cuda.max_memory_allocated() / (1024 ** 2),
            )
            self._gpu_snapshots.append(snapshot)
    
    def get_gpu_memory(self) -> Dict[str, Any]:
        """Get GPU memory statistics."""
        with self._data_lock:
            if not self._gpu_snapshots:
                return {}
            
            peak_allocated = max(s.max_allocated_mb for s in self._gpu_snapshots)
            peak_reserved = max(s.reserved_mb for s in self._gpu_snapshots)
            
            return {
                "peak_allocated_mb": peak_allocated,
                "peak_reserved_mb": peak_reserved,
                "snapshots": [asdict(s) for s in self._gpu_snapshots],
            }
    
    # -------------------------------------------------------------------------
    # Aggregate Methods
    # -------------------------------------------------------------------------
    
    def get_profiling_data(self) -> Dict[str, Any]:
        """
        Get complete profiling data for writing to trace.
        
        Returns:
            Dict containing all profiling metrics
        """
        if not self._enabled:
            return {}
        
        with self._data_lock:
            model_load_times = dict(self._model_load_times)
            total_load_time = sum(model_load_times.values()) if model_load_times else 0
            
            data = {
                "enabled": True,
                "model_load_times_ms": model_load_times,
                "total_load_time_ms": total_load_time,
            }
            
            # LLM calls
            if self._llm_calls:
                data["llm_calls"] = [asdict(r) for r in self._llm_calls]
                data["llm_summary"] = self.get_llm_summary()
            
            # Tool breakdowns
            if self._tool_breakdowns:
                data["tool_breakdowns"] = {
                    name: asdict(breakdown)
                    for name, breakdown in self._tool_breakdowns.items()
                }
            
            # GPU memory
            if self._gpu_snapshots:
                data["gpu_memory"] = self.get_gpu_memory()
            
            return data
    
    def reset(self):
        """Reset all profiling data (call at start of each trace)."""
        with self._data_lock:
            self._model_load_times.clear()
            self._llm_calls.clear()
            self._tool_breakdowns.clear()
            self._gpu_snapshots.clear()
            self._llm_call_counter = 0
    
    def reset_per_sample(self):
        """
        Reset per-sample data (LLM calls, tool breakdowns, GPU snapshots).
        Keeps model load times which are typically done once per run.
        """
        with self._data_lock:
            self._llm_calls.clear()
            self._tool_breakdowns.clear()
            self._gpu_snapshots.clear()
            self._llm_call_counter = 0


# =============================================================================
# Global Singleton Instance
# =============================================================================

profiler = Profiler()


def init_profiler_from_env():
    """
    Initialize profiler based on environment variables.
    
    Enables profiling when TEST_MODE=true.
    """
    test_mode = os.getenv("TEST_MODE", "").lower()
    if test_mode == "true":
        profiler.enable()
        print("[Profiler] Enabled (TEST_MODE=true)")
        return True
    return False
