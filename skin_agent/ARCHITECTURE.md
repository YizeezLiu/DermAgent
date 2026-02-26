# Skin Agent Architecture

**Framework:** LangChain + LangGraph

---

## Overview

**Skin Agent** is a multi-modal AI agent for dermatological diagnosis and analysis built on the LangChain/LangGraph framework. It orchestrates multiple specialized dermatology tools using GPT-4o as the reasoning backbone.

### Key Features

- **Multi-tool orchestration**: Intelligently combines 7 specialized dermatology tools
- **Task-specific optimization**: Benchmark agent with tailored prompts per task
- **Critic module**: Deterministic evidence verification with self-correction
- **Comprehensive tracing**: Detailed logging of reasoning process and tool calls
- **RAG-enhanced**: Retrieval-augmented generation from 413K+ dermatology cases and clinical guidelines

### Architecture Philosophy

```
User Query → LLM Reasoning → Tool Selection → Tool Execution → Result Synthesis → Response
     ↑                                                                              ↓
     └──────────────────────── Critic Feedback Loop ────────────────────────────────┘
```

The agent follows a **plan-execute-reflect** paradigm:
1. **Plan**: LLM analyzes the query and decides which tools to call
2. **Execute**: Tools process images/queries and return structured results
3. **Reflect**: LLM synthesizes tool outputs and decides next steps
4. **Critic**: Deterministic evaluator checks evidence chains and triggers retries if needed

---

## Module Structure

```
skin_agent/
├── benchmark_agent.py       # Benchmark-optimized agent + Critic + AnswerParser
├── configs.py               # Dataset task configurations
├── tracing.py               # Tool call logging system
├── profiler.py              # Performance profiling
├── resume.py                # Checkpoint/resume for long runs
├── prompts.md               # System prompt documentation
├── __init__.py              # Module exports
├── tools/
│   ├── __init__.py          # Tool exports
│   ├── base.py              # Base tool classes + input schemas
│   ├── skin_tools.py        # All 7 tool implementations
│   ├── executor.py          # Tool execution logic
│   └── derm_knowledge_tree/ # Disease ontology JSONs
└── utils/
    ├── retry.py             # Rate-limit retry logic
    └── image_utils.py       # Image path handling
```

---

## LangGraph Workflow

### Graph Structure

```
┌───────┐
│ START │
└───┬───┘
    │
    ▼
┌────────────┐      Tool calls?     ┌────────┐
│  Chatbot   │────────Yes──────────▶│ Tools  │
│   Node     │                      │  Node  │
└────┬───────┘                      └───┬────┘
     │                                  │
     │ No tool calls                    │
     │                                  │ Always
     ▼                                  │
┌────────┐                             │
│  END   │◀────────────────────────────┘
└────────┘
```

**Node Functions:**

1. **Chatbot Node**: Receives messages and state, calls LLM with tool bindings, decides whether to use tools or respond directly
2. **Tools Node**: Executes requested tools, returns results as ToolMessage objects, routes back to chatbot for synthesis
3. **Conditional Routing**: `should_continue()` checks if LLM requested tool calls, routes to "tools" or "__end__"

### Agent State (`BenchmarkAgentState`)

```python
class BenchmarkAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
```

Messages auto-accumulate via the `operator.add` annotation. Thread-based isolation via `thread_id` in config.

### LLM Configuration

- **Model**: GPT-4o (configurable)
- **Temperature**: 0.1 for benchmark evaluation
- **Tool Binding**: Tools bound to LLM via `.bind_tools()`

---

## Tool System

### Base Class

All tools inherit from `BaseSkinTool`:

```python
class BaseSkinTool(BaseTool):
    device: str = "cuda"
    _initialized: bool = False

    def _load_model(self) -> None:
        """Load model weights (lazy, on first use)."""
        raise NotImplementedError

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self._load_model()
            self._initialized = True

    def _run(self, **kwargs) -> str:
        raise NotImplementedError
```

### Tools

| Tool | Model | Purpose | VRAM |
|------|-------|---------|------|
| PanDerm Classifier | `redlessone/DermLIP_PanDerm-base-w-PubMed-256` | Zero-shot disease classification via CLIP similarity | ~2GB |
| MAKE Annotator | `xieji-x/MAKE` (OpenCLIP) | Dermoscopic concept extraction | ~2GB |
| DermoGPT VQA | DermoGPT-RL | Dermatology-specialized visual QA | ~16GB |
| Qwen3-VL VQA | `Qwen/Qwen3-VL-8B-Instruct` (bfloat16) | General visual question answering | ~16GB |
| Image RAG | DermLIP + Qdrant | Case retrieval from 413K diagnosed cases | ~2GB (shared) |
| Text RAG | Qwen3-Embedding + Qdrant | Guideline retrieval (DermNet, Mayo Clinic) | — |
| Ontology | Knowledge Graph (JSON) | Disease hierarchy and taxonomy queries | — |

**Total GPU**: ~20-22GB for all tools loaded simultaneously. Tools use lazy loading (load on first call).

---

## RAG Integration

### Qdrant Vector Database

**Storage Path:** `./qdrant_storage`

**Image RAG Collection (`derm1m`):**
- Dual vectors: `image_embedding` (512-dim) and `caption_embedding` (512-dim)
- Encoder: DermLIP (shared with PanDerm)
- Distance: Cosine similarity
- Build script: `scripts/build_qdrant_db.py`

**Text RAG Collection (`derm_rag`):**
- Encoder: Qwen3-Embedding-8B
- Reranker: Qwen3-Reranker-0.6B
- Sources: DermNet and Mayo Clinic clinical guidelines
- Build script: `scripts/build_qdrant_rag.py`

---

## Tracing & Debugging

### Trace Data Model

```python
@dataclass
class AgentTrace:
    trace_id: str
    task_type: str
    dataset: str
    image_path: str
    user_query: str
    steps: List[AgentStep]       # Each step = LLM call + tool calls
    final_response: str
    parsed_answer: Any
    ground_truth: Any
    correct: Optional[bool]
    critic_retries: int
    critic_retry_steps: List[CriticRetryStep]
    profiling: Optional[Dict]    # Auto-populated timing data
```

### TraceLogger

```python
logger = TraceLogger(log_dir="./traces")
trace = logger.new_trace(task_type="diagnosis", image_path="...", user_query="...")
# ... agent execution ...
trace.finalize(final_response="...", parsed_answer="melanoma")
logger.save_trace(trace, detailed=True)
```

### LangChain Callback Integration

`TracingCallback` automatically captures `on_llm_start`, `on_llm_end`, `on_tool_start`, `on_tool_end`, and `on_tool_error` events.

---

## Benchmark System

### Supported Tasks

| Task | Datasets | Primary Tools |
|------|----------|---------------|
| Diagnosis (classification) | HAM10000, SNU | PanDerm, DermoGPT, RAG, Ontology |
| Concept Annotation | Derm7pt, SkinCon | MAKE |
| Image Captioning | SkinCAP | DermoGPT, MAKE, PanDerm, RAG |

### Answer Parsing

`AnswerParser` extracts structured answers from LLM responses using:
1. `FINAL_ANSWER:` format (primary)
2. V2 JSON block format (fallback)
3. Fuzzy matching to valid options

### GPU Resource Management

- Models load lazily on first call
- Tools share GPU when possible (PanDerm and RAG share DermLIP encoder)
- bfloat16 precision for large VLMs

---

**Maintainer:** DermAgent Team
