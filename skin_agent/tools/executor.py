"""
Tool execution utilities for skin agent.

Provides helper functions for:
- Image path normalization
- Tool output formatting
- Batch tool execution
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool


def normalize_image_path(
    path: str, 
    base_dir: Optional[str] = None
) -> str:
    """
    Normalize and resolve image path.
    
    Args:
        path: Image path (relative or absolute)
        base_dir: Base directory for relative paths
        
    Returns:
        Absolute path as string
    """
    p = Path(path)
    
    if not p.is_absolute() and base_dir:
        p = Path(base_dir) / p
    
    return str(p.resolve())


def format_tool_output(
    tool_name: str,
    result: str,
    success: bool = True
) -> Dict[str, Any]:
    """
    Format tool output for agent consumption.
    
    Args:
        tool_name: Name of the executed tool
        result: Tool execution result
        success: Whether execution was successful
        
    Returns:
        Formatted output dictionary
    """
    return {
        "tool": tool_name,
        "success": success,
        "result": result,
    }


def create_tool_message(
    tool_call_id: str,
    content: str,
    tool_name: str,
) -> ToolMessage:
    """
    Create a ToolMessage for LangGraph.
    
    Args:
        tool_call_id: ID of the tool call
        content: Tool execution result
        tool_name: Name of the tool
        
    Returns:
        ToolMessage instance
    """
    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        name=tool_name,
    )


class ToolExecutor:
    """
    Executes tools and handles results.
    
    Manages a collection of tools and provides unified execution interface.
    """
    
    def __init__(self, tools: List[BaseTool]):
        """
        Initialize executor with tools.
        
        Args:
            tools: List of LangChain tools
        """
        self.tools = {tool.name: tool for tool in tools}
    
    def execute(
        self, 
        tool_name: str, 
        tool_input: Dict[str, Any]
    ) -> str:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Tool input parameters
            
        Returns:
            Tool execution result as string
        """
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {list(self.tools.keys())}"
        
        tool = self.tools[tool_name]
        
        try:
            result = tool.invoke(tool_input)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {type(e).__name__}: {str(e)}"
    
    def execute_tool_calls(
        self, 
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolMessage]:
        """
        Execute multiple tool calls and return messages.
        
        Args:
            tool_calls: List of tool call dictionaries with 'name', 'args', 'id'
            
        Returns:
            List of ToolMessage results
        """
        messages = []
        
        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_args = call.get("args", {})
            tool_id = call.get("id", "")
            
            result = self.execute(tool_name, tool_args)
            
            messages.append(
                create_tool_message(
                    tool_call_id=tool_id,
                    content=result,
                    tool_name=tool_name,
                )
            )
        
        return messages
    
    def get_tool_descriptions(self) -> str:
        """
        Get formatted descriptions of all available tools.
        
        Returns:
            Formatted string of tool descriptions
        """
        descriptions = []
        
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        
        return "\n".join(descriptions)


