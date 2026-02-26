"""
Skin Agent - Multi-modal AI Agent for Dermatological Diagnosis.

Built on LangChain/LangGraph framework with integrated dermatology tools.

Modules:
- benchmark_agent: Optimized agent for benchmark evaluation
- tracing: Tool call logging and debugging
- tools: Dermatology tool implementations
"""

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
    DermoGPTTool,
    RAGTool,
    TextRAGTool,
    OntologyTool,
)

__all__ = [
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
    "DermoGPTTool",
    "RAGTool",
    "TextRAGTool",
    "OntologyTool",
]
