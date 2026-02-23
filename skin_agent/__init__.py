"""
Skin Agent - Multi-modal AI Agent for Dermatological Diagnosis.

Built on LangChain/LangGraph framework with integrated dermatology tools.

Modules:
- agent: Interactive agent for general use
- benchmark_agent: Optimized agent for benchmark evaluation
- tracing: Tool call logging and debugging
- tools: Dermatology tool implementations
"""

# Interactive Agent (optional — not required for benchmark runs)
try:
    from .agent import create_agent, run_agent, SkinAgentState
except ImportError:
    create_agent = run_agent = SkinAgentState = None

# Benchmark Agent
from .benchmark_agent import (
    create_benchmark_agent,
    BenchmarkRunner,
    BenchmarkAgentState,
    AnswerParser,
    TASK_PROMPTS,
)

# Dataset Configurations (separated to avoid circular imports)
from .configs import DATASET_CONFIGS

# Tracing
from .tracing import (
    TraceLogger,
    AgentTrace,
    AgentStep,
    ToolCall,
    TracingCallback,
)

# Profiler
from .profiler import (
    profiler,
    init_profiler_from_env,
    Profiler,
    LLMCallRecord,
    ToolBreakdown,
    GPUSnapshot,
)

# Tools
from .tools import (
    PanDermTool,
    MAKETool,
    Qwen3VLTool,
    RAGTool,
)

__all__ = [
    # Interactive Agent
    "create_agent",
    "run_agent",
    "SkinAgentState",
    # Benchmark Agent
    "create_benchmark_agent",
    "BenchmarkRunner",
    "BenchmarkAgentState",
    "AnswerParser",
    "TASK_PROMPTS",
    "DATASET_CONFIGS",
    # Tracing
    "TraceLogger",
    "AgentTrace",
    "AgentStep",
    "ToolCall",
    "TracingCallback",
    # Profiler
    "profiler",
    "init_profiler_from_env",
    "Profiler",
    "LLMCallRecord",
    "ToolBreakdown",
    "GPUSnapshot",
    # Tools
    "PanDermTool",
    "MAKETool",
    "Qwen3VLTool",
    "RAGTool",
]
