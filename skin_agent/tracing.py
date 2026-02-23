"""
工具调用日志系统
记录完整的 Agent 推理流程，便于调试和分析
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ToolCall:
    """单次工具调用记录"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class AgentStep:
    """Agent 单步记录（一次 LLM 调用 + 可能的工具调用）"""
    step_number: int
    llm_input_messages: List[Dict]
    llm_output: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CriticRetryStep:
    """Critic retry 事件记录 - 记录完整的 critic 重试流程"""
    retry_number: int  # 第几次 retry (1, 2, ...)
    critic_feedback: str  # 原始 critic 反馈（不含 LLM summary）
    evidence_summary: Optional[Dict[str, Any]] = None  # LLM summary (if enabled)
    image_reinjected: bool = False  # 是否重新注入了图像
    image_analysis: str = ""  # LLM 对图像的分析描述 (when image_reinjected=True)
    llm_response_after_retry: str = ""  # LLM 对 retry 的响应 (tool calls or final answer)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentTrace:
    """完整的 Agent 执行记录"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: str = ""
    dataset: str = ""
    image_path: str = ""
    user_query: str = ""

    # 执行过程
    steps: List[AgentStep] = field(default_factory=list)

    # 最终结果
    final_response: str = ""
    parsed_answer: Any = None
    ground_truth: Any = None
    correct: Optional[bool] = None

    # 元信息
    model: str = "gpt-4o"
    temperature: float = 0.1
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""
    total_duration_ms: float = 0
    total_tool_calls: int = 0
    total_tokens: Optional[int] = None

    # Critic node tracking
    critic_retries: int = 0
    critic_feedback_history: List[str] = field(default_factory=list)
    
    # Evidence summary tracking (when critic_llm_summary=True)
    # Format: [{"summary": str, "duration_ms": float, "timestamp": str}, ...]
    evidence_summaries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Critic retry steps (完整流程记录，包含 summary + image reinject + LLM response)
    critic_retry_steps: List["CriticRetryStep"] = field(default_factory=list)
    
    # Critic retry options used
    # Format: {"llm_summary": bool, "image_reinject": bool}
    critic_options: Dict[str, Any] = field(default_factory=dict)
    
    # Profiling data (auto-populated when TEST_MODE=true)
    profiling: Optional[Dict[str, Any]] = None
    
    def add_step(self, step: AgentStep):
        """添加一个执行步骤"""
        self.steps.append(step)
        self.total_tool_calls += len(step.tool_calls)
    
    def finalize(self, final_response: str, parsed_answer: Any = None):
        """完成 trace 记录"""
        self.final_response = final_response
        self.parsed_answer = parsed_answer
        self.end_time = datetime.now().isoformat()
        
        # 计算总时长
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        self.total_duration_ms = (end - start).total_seconds() * 1000
        
        # Auto-attach profiling data if profiler is enabled
        try:
            from .profiler import profiler
            if profiler.is_enabled():
                self.profiling = profiler.get_profiling_data()
                profiler.reset_per_sample()  # Reset per-sample data for next trace
        except ImportError:
            pass
    
    def to_dict(self) -> Dict:
        """转为可序列化的字典（摘要版）"""
        return {
            "trace_id": self.trace_id,
            "task_type": self.task_type,
            "dataset": self.dataset,
            "image_path": self.image_path,
            "user_query": self.user_query[:200] + "..." if len(self.user_query) > 200 else self.user_query,
            "steps": [
                {
                    "step_number": s.step_number,
                    "llm_output_preview": s.llm_output[:300] + "..." if len(s.llm_output) > 300 else s.llm_output,
                    "tool_calls": [asdict(tc) for tc in s.tool_calls],
                    "timestamp": s.timestamp,
                }
                for s in self.steps
            ],
            "final_response_preview": self.final_response[:500] + "..." if len(self.final_response) > 500 else self.final_response,
            "parsed_answer": self.parsed_answer,
            "ground_truth": self.ground_truth,
            "correct": self.correct,
            "model": self.model,
            "temperature": self.temperature,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "total_tool_calls": self.total_tool_calls,
            "critic_retries": self.critic_retries,
            "critic_feedback_history": self.critic_feedback_history,
            "evidence_summaries": self.evidence_summaries,
            "critic_retry_steps": [asdict(s) for s in self.critic_retry_steps],
            "critic_options": self.critic_options,
            "profiling": self.profiling,
        }
    
    def to_detailed_dict(self) -> Dict:
        """完整详细版本（用于保存）"""
        return {
            "trace_id": self.trace_id,
            "task_type": self.task_type,
            "dataset": self.dataset,
            "image_path": self.image_path,
            "user_query": self.user_query,
            "steps": [
                {
                    "step_number": s.step_number,
                    "llm_input_messages": s.llm_input_messages,
                    "llm_output": s.llm_output,
                    "tool_calls": [asdict(tc) for tc in s.tool_calls],
                    "timestamp": s.timestamp,
                }
                for s in self.steps
            ],
            "final_response": self.final_response,
            "parsed_answer": self.parsed_answer,
            "ground_truth": self.ground_truth,
            "correct": self.correct,
            "model": self.model,
            "temperature": self.temperature,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "total_tool_calls": self.total_tool_calls,
            "critic_retries": self.critic_retries,
            "critic_feedback_history": self.critic_feedback_history,
            "evidence_summaries": self.evidence_summaries,
            "critic_retry_steps": [asdict(s) for s in self.critic_retry_steps],
            "critic_options": self.critic_options,
            "profiling": self.profiling,
        }


# =============================================================================
# Trace Logger
# =============================================================================

class TraceLogger:
    """日志管理器"""
    
    def __init__(self, log_dir: str = "./traces"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.traces: List[AgentTrace] = []
    
    def new_trace(
        self,
        task_type: str,
        image_path: str,
        user_query: str,
        dataset: str = "",
        model: str = "gpt-4o",
    ) -> AgentTrace:
        """创建新的 trace"""
        trace = AgentTrace(
            task_type=task_type,
            dataset=dataset,
            image_path=image_path,
            user_query=user_query,
            model=model,
        )
        self.traces.append(trace)
        return trace
    
    def save_trace(self, trace: AgentTrace, detailed: bool = True) -> Path:
        """保存单个 trace 到文件"""
        filename = f"{trace.trace_id}_{trace.task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename
        
        data = trace.to_detailed_dict() if detailed else trace.to_dict()
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_all(self, filename: str = None) -> Path:
        """保存所有 traces 到一个文件"""
        if filename is None:
            filename = f"traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        
        data = {
            "total_traces": len(self.traces),
            "traces": [t.to_dict() for t in self.traces],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def print_trace_summary(self, trace: AgentTrace):
        """打印 trace 摘要到控制台"""
        print(f"\n{'='*60}")
        print(f"Trace ID: {trace.trace_id}")
        print(f"Task: {trace.task_type} | Dataset: {trace.dataset}")
        print(f"Image: {trace.image_path}")
        print(f"{'='*60}")
        
        for step in trace.steps:
            print(f"\n--- Step {step.step_number} ---")
            if step.tool_calls:
                for tc in step.tool_calls:
                    status = "✓" if not tc.error else "✗"
                    print(f"  {status} {tc.tool_name}")
                    if tc.error:
                        print(f"    Error: {tc.error}")
                    else:
                        result_preview = str(tc.result)[:100]
                        print(f"    Result: {result_preview}...")
            else:
                print("  [No tool calls - generating response]")
        
        print(f"\n--- Final Answer ---")
        print(f"Parsed: {trace.parsed_answer}")
        print(f"Ground Truth: {trace.ground_truth}")
        print(f"Correct: {trace.correct}")
        print(f"Duration: {trace.total_duration_ms:.0f}ms | Tool Calls: {trace.total_tool_calls}")
        print(f"{'='*60}\n")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """获取所有 traces 的统计摘要"""
        if not self.traces:
            return {"total": 0}
        
        correct_count = sum(1 for t in self.traces if t.correct is True)
        incorrect_count = sum(1 for t in self.traces if t.correct is False)
        
        return {
            "total": len(self.traces),
            "correct": correct_count,
            "incorrect": incorrect_count,
            "accuracy": correct_count / len(self.traces) if self.traces else 0,
            "avg_duration_ms": sum(t.total_duration_ms for t in self.traces) / len(self.traces),
            "avg_tool_calls": sum(t.total_tool_calls for t in self.traces) / len(self.traces),
        }


# =============================================================================
# LangChain Callback Handler
# =============================================================================

class TracingCallback(BaseCallbackHandler):
    """LangChain 回调，自动记录工具调用"""
    
    def __init__(self, trace: AgentTrace):
        self.trace = trace
        self.current_step: Optional[AgentStep] = None
        self.step_count = 0
        self.tool_start_time: Optional[float] = None
        self.current_tool_name: Optional[str] = None
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 开始调用"""
        self.step_count += 1
        self.current_step = AgentStep(
            step_number=self.step_count,
            llm_input_messages=[{"prompt_preview": p[:200]} for p in prompts],
            llm_output="",
        )
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        """LLM 调用结束"""
        if self.current_step:
            if response.generations:
                self.current_step.llm_output = response.generations[0][0].text
            self.trace.add_step(self.current_step)
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """工具开始调用"""
        self.tool_start_time = time.time()
        self.current_tool_name = serialized.get("name", "unknown")
        
        # 添加工具调用记录（结果待填充）
        if self.current_step:
            self.current_step.tool_calls.append(ToolCall(
                tool_name=self.current_tool_name,
                arguments={"input": input_str[:200]},
                result=None,
            ))
    
    def on_tool_end(self, output, **kwargs):
        """工具调用结束"""
        duration = (time.time() - self.tool_start_time) * 1000 if self.tool_start_time else None
        
        if self.current_step and self.current_step.tool_calls:
            # 更新最后一个 tool call 的结果
            self.current_step.tool_calls[-1].result = str(output)[:500]
            self.current_step.tool_calls[-1].duration_ms = duration
    
    def on_tool_error(self, error, **kwargs):
        """工具调用出错"""
        if self.current_step and self.current_step.tool_calls:
            self.current_step.tool_calls[-1].error = str(error)
