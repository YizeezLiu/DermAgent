"""
Base tool definitions for skin agent.

Provides common utilities and base classes for dermatology tools.
"""

import gc
import os
import time
import threading
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

import torch
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .base import BaseSkinTool

# Import profiler (lazy to avoid circular imports)
def _get_profiler():
    """Get profiler instance (lazy import to avoid circular imports)."""
    try:
        from ..profiler import profiler
        return profiler
    except ImportError:
        return None


# =============================================================================
# Tool Timing Registry (Thread-safe)
# =============================================================================

class ToolTimingRegistry:
    """
    Global registry for tool execution timing.

    Thread-safe singleton that records tool execution times.
    Used by benchmark_agent to populate duration_ms in traces.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._timings: List[Dict] = []
                    cls._instance._timing_lock = threading.Lock()
        return cls._instance

    def record(self, tool_name: str, duration_ms: float):
        """Record a tool execution timing."""
        with self._timing_lock:
            self._timings.append({
                "tool_name": tool_name,
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            })

    def get_latest(self, tool_name: str) -> Optional[float]:
        """Get the most recent timing for a tool (non-destructive)."""
        with self._timing_lock:
            for t in reversed(self._timings):
                if t["tool_name"] == tool_name:
                    return t["duration_ms"]
        return None

    def pop_latest(self, tool_name: str) -> Optional[float]:
        """Get and remove the most recent timing for a tool."""
        with self._timing_lock:
            for i in range(len(self._timings) - 1, -1, -1):
                if self._timings[i]["tool_name"] == tool_name:
                    return self._timings.pop(i)["duration_ms"]
        return None

    def clear(self):
        """Clear all recorded timings."""
        with self._timing_lock:
            self._timings.clear()

    def get_all(self) -> List[Dict]:
        """Get all recorded timings (copy)."""
        with self._timing_lock:
            return list(self._timings)

    def get_summary(self) -> Dict[str, Dict]:
        """Get summary statistics by tool name."""
        with self._timing_lock:
            from collections import defaultdict
            stats = defaultdict(list)
            for t in self._timings:
                stats[t["tool_name"]].append(t["duration_ms"])

            summary = {}
            for name, times in stats.items():
                summary[name] = {
                    "count": len(times),
                    "total_ms": sum(times),
                    "avg_ms": sum(times) / len(times) if times else 0,
                    "min_ms": min(times) if times else 0,
                    "max_ms": max(times) if times else 0,
                }
            return summary


# Global singleton instance
tool_timing_registry = ToolTimingRegistry()


# =============================================================================
# Tool Memory Manager (LRU-based GPU Memory Management)
# =============================================================================

class ToolMemoryManager:
    """
    LRU-based GPU memory manager for tool models.
    
    Manages model lifecycle with LRU eviction to prevent GPU memory exhaustion
    when running multiple tools sequentially. Keeps track of loaded tools and
    automatically unloads least recently used tools when capacity is exceeded.
    
    Args:
        max_loaded: Maximum number of models to keep loaded in GPU memory.
                   Set to 0 for unlimited (no eviction).
        verbose: Print loading/unloading messages.
    
    Example:
        manager = ToolMemoryManager(max_loaded=2)
        manager.ensure_loaded(tool1)  # Loads tool1
        manager.ensure_loaded(tool2)  # Loads tool2
        manager.ensure_loaded(tool3)  # Evicts tool1 (LRU), loads tool3
    """
    
    def __init__(self, max_loaded: int = 2, verbose: bool = True):
        self.max_loaded = max_loaded
        self.verbose = verbose
        # OrderedDict maintains insertion order; we move to end on access (LRU)
        self.loaded_tools: OrderedDict[str, "BaseSkinTool"] = OrderedDict()
        self._lock = threading.Lock()
    
    def ensure_loaded(self, tool: "BaseSkinTool") -> None:
        """
        Ensure the tool is loaded, evicting LRU if at capacity.
        
        Args:
            tool: The tool to ensure is loaded.
        """
        if tool is None:
            return
        
        with self._lock:
            tool_name = tool.name
            
            # If already loaded, move to end (most recently used)
            if tool_name in self.loaded_tools:
                self.loaded_tools.move_to_end(tool_name)
                return
            
            # Evict LRU tools if at capacity
            while self.max_loaded > 0 and len(self.loaded_tools) >= self.max_loaded:
                self._evict_lru()
            
            # Load the tool (guard for tools not inheriting BaseSkinTool)
            is_init = getattr(tool, "is_initialized", None)
            if is_init is None:
                is_init = getattr(tool, "_initialized", True)
            if not is_init:
                if self.verbose:
                    print(f"[MemoryManager] Loading {tool_name}...")
                if hasattr(tool, "preload"):
                    tool.preload()
                elif hasattr(tool, "_ensure_initialized"):
                    tool._ensure_initialized()
                if self.verbose:
                    print(f"[MemoryManager] {tool_name} loaded.")
            
            # Register as loaded
            self.loaded_tools[tool_name] = tool
    
    def _evict_lru(self) -> None:
        """Unload the least recently used tool."""
        if not self.loaded_tools:
            return
        
        # Get the first item (least recently used)
        lru_name, lru_tool = next(iter(self.loaded_tools.items()))
        
        if self.verbose:
            print(f"[MemoryManager] Evicting LRU tool: {lru_name}")
        
        # Unload the tool
        if hasattr(lru_tool, 'unload'):
            lru_tool.unload()
        
        # Remove from registry
        del self.loaded_tools[lru_name]
    
    def unload_all(self) -> None:
        """Unload all currently loaded tools."""
        with self._lock:
            for tool_name, tool in list(self.loaded_tools.items()):
                if self.verbose:
                    print(f"[MemoryManager] Unloading {tool_name}...")
                if hasattr(tool, 'unload'):
                    tool.unload()
            self.loaded_tools.clear()
    
    def get_loaded_tools(self) -> List[str]:
        """Get list of currently loaded tool names (in LRU order)."""
        with self._lock:
            return list(self.loaded_tools.keys())
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        available, free_gb = check_gpu_memory()
        return {
            "loaded_tools": self.get_loaded_tools(),
            "num_loaded": len(self.loaded_tools),
            "max_loaded": self.max_loaded,
            "gpu_available": available,
            "gpu_free_gb": free_gb,
        }


class ImageInput(BaseModel):
    """Input schema for image-based tools."""
    
    image_path: str = Field(
        description="Path to the skin/dermoscopic image file (JPEG, PNG, etc.)"
    )


class ImageQueryInput(BaseModel):
    """Input schema for image + text query tools."""
    
    image_path: str = Field(
        description="Path to the skin/dermoscopic image file"
    )
    query: str = Field(
        description="Natural language question about the image"
    )
    target_concepts: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated list of target concepts for concept annotation tasks. "
            "When provided, the query will be enhanced to specifically evaluate these concepts. "
            "Example: 'pigment_network, pigmentation, vascular_structures'"
        )
    )


class TextQueryInput(BaseModel):
    """Input schema for text-only query tools (e.g., RAG retrieval)."""
    
    query: str = Field(
        description="Text query for retrieval or search"
    )
    top_k: int = Field(
        default=5,
        description="Number of results to return"
    )
    search_mode: str = Field(
        default="hybrid",
        description=(
            "Search mode: 'vector' (semantic search only), 'keyword' (full-text search only), "
            "or 'hybrid' (both combined with RRF fusion, default). "
            "Use 'hybrid' for best results combining semantic understanding and keyword matching."
        )
    )


class RAGQueryInput(BaseModel):
    """Input schema for multimodal RAG retrieval (image or text query)."""
    
    image_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the skin image for image-based retrieval. "
            "If provided, finds visually similar cases in the knowledge base."
        )
    )
    text_query: Optional[str] = Field(
        default=None,
        description=(
            "Text description of morphological features for text-based retrieval. "
            "E.g., 'irregular borders with dark pigmentation and asymmetric shape'"
        )
    )
    top_k: int = Field(
        default=5,
        description="Number of similar cases to return"
    )


class DermoGPTDiagnosisInput(BaseModel):
    """Input schema for DermoGPT diagnosis task.
    
    Supports three modes:
    - Diagnosis mode: When candidate_diseases is provided, uses structured prompt template
    - Concept mode: When target_concepts is provided, evaluates presence of specific concepts
    - VQA mode: When only query is provided, uses general VQA behavior
    """
    
    image_path: str = Field(
        description="Path to the skin/dermoscopic image file"
    )
    candidate_diseases: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated list of candidate disease names for diagnosis. "
            "When provided, uses structured diagnosis prompt template with XML output. "
            "Example: 'melanocytic nevi, melanoma, basal cell carcinoma'"
        )
    )
    query: Optional[str] = Field(
        default=None,
        description=(
            "Natural language question for general VQA mode. "
            "Used when candidate_diseases is not provided."
        )
    )
    target_concepts: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated list of target concepts for concept annotation tasks. "
            "When provided, evaluates the presence of these specific concepts. "
            "Example: 'pigment_network, pigmentation, vascular_structures'"
        )
    )


def check_gpu_memory() -> tuple[bool, float]:
    """
    Check GPU availability and free memory.
    
    Returns:
        Tuple of (is_available, free_memory_gb)
    """
    if not torch.cuda.is_available():
        return False, 0.0
    
    try:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        free = gpu_memory - allocated
        return True, free / 1e9
    except Exception:
        return False, 0.0


def validate_image_path(path: str) -> Path:
    """
    Validate and normalize image path.
    
    Args:
        path: Path to the image file
        
    Returns:
        Resolved Path object
        
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If format is unsupported
    """
    resolved = Path(path).resolve()
    
    if not resolved.exists():
        raise FileNotFoundError(f"Image not found: {resolved}")
    
    valid_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if resolved.suffix.lower() not in valid_suffixes:
        raise ValueError(f"Unsupported image format: {resolved.suffix}")
    
    return resolved


class BaseSkinTool(BaseTool):
    """
    Base class for dermatology tools.
    
    Provides common functionality for GPU management, error handling,
    and model loading patterns.
    """
    
    device: str = "cuda"
    model: Any = None
    _initialized: bool = False
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.device = device if torch.cuda.is_available() else "cpu"
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Lazy initialization of model resources."""
        if not self._initialized:
            profiler = _get_profiler()
            
            # Record load time if profiler is enabled
            if profiler is not None and profiler.is_enabled():
                start = time.perf_counter()
                self._load_model()
                load_time_ms = (time.perf_counter() - start) * 1000
                profiler.record_model_load(self.name, load_time_ms)
                profiler.snapshot_gpu(f"after_load_{self.name}")
            else:
                self._load_model()
            
            self._initialized = True
    
    def preload(self) -> "BaseSkinTool":
        """
        Eagerly load model resources (for sequential/serial loading).
        
        Call this method explicitly to load the model immediately instead of
        waiting for the first tool invocation. Returns self for chaining.
        
        Example:
            tool = PanDermTool(device="cuda").preload()
        """
        self._ensure_initialized()
        return self
    
    @property
    def is_initialized(self) -> bool:
        """Check if the tool has been initialized."""
        return self._initialized
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load model weights. Subclasses must implement."""
        pass
    
    def unload(self) -> None:
        """
        Unload model from GPU and free memory.
        
        Subclasses should override this to clean up additional resources
        (e.g., processors, tokenizers) but should call super().unload() first.
        
        After unload(), the tool can be reloaded by calling preload() or
        by invoking the tool (lazy loading).
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        self._initialized = False
        
        # Clear GPU cache and garbage collect
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _check_resources(self, min_memory_gb: float = 2.0) -> None:
        """
        Check if sufficient GPU resources are available.
        
        Raises:
            RuntimeError: If insufficient memory or GPU unavailable
        """
        if self.device == "cuda":
            available, free_gb = check_gpu_memory()
            if not available:
                raise RuntimeError("CUDA requested but not available")
            if free_gb < min_memory_gb:
                raise RuntimeError(
                    f"Insufficient GPU memory. Need {min_memory_gb}GB, have {free_gb:.2f}GB"
                )
    
    def _handle_error(self, error: Exception, context: str = "") -> str:
        """
        Format error message for tool output.

        Args:
            error: The exception that occurred
            context: Additional context about the operation

        Returns:
            Formatted error string
        """
        error_type = type(error).__name__
        msg = f"Error in {self.name}"
        if context:
            msg += f" ({context})"
        msg += f": [{error_type}] {str(error)}"
        return msg

    def invoke(self, input, config=None, **kwargs):
        """
        Override invoke to add execution timing.

        Records tool execution time in the global ToolTimingRegistry,
        which can be retrieved by benchmark_agent for tracing.
        """
        start_time = time.perf_counter()
        try:
            result = super().invoke(input, config, **kwargs)
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            tool_timing_registry.record(self.name, duration_ms)
            # Print timing for immediate visibility in logs
            print(f"    ⏱ [{self.name}] {duration_ms:.0f}ms")
        return result


def preload_tools_sequential(tools: list, verbose: bool = True) -> list:
    """
    Sequentially preload all tools to avoid concurrent loading issues.
    
    Args:
        tools: List of BaseSkinTool instances
        verbose: Print loading progress
        
    Returns:
        The same list of tools, now initialized
    """
    for i, tool in enumerate(tools):
        if hasattr(tool, 'preload') and not tool.is_initialized:
            if verbose:
                print(f"\n[Preload {i+1}/{len(tools)}] Loading {tool.name}...")
            try:
                tool.preload()
                if verbose:
                    print(f"[Preload {i+1}/{len(tools)}] {tool.name} loaded successfully.")
            except Exception as e:
                if verbose:
                    print(f"[Preload {i+1}/{len(tools)}] {tool.name} failed: {e}")
            
            # Clear GPU cache and garbage collect between tool loads
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    return tools

