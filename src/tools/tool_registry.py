"""Tool registry for agent tools"""

import asyncio
import inspect
from typing import Dict, List, Any, Optional, Callable, Set, Union

from ..utils.config import Config

class Tool:
    """
    Represents a tool that agents can use
    
    Attributes:
        name: Name of the tool
        description: Description of what the tool does
        function: The function to execute
        required_capabilities: Capabilities required to use this tool
        parameters: Parameter descriptions
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        required_capabilities: Optional[Set[str]] = None,
        parameters: Optional[Dict[str, str]] = None
    ):
        self.name = name
        self.description = description
        self.function = function
        self.required_capabilities = required_capabilities or set()
        
        # Extract parameter info from function signature if not provided
        if parameters is None:
            self.parameters = {}
            sig = inspect.signature(function)
            for param_name, param in sig.parameters.items():
                if param_name != 'self':
                    param_desc = f"{param_name}"
                    if param.annotation != inspect.Parameter.empty:
                        param_desc += f" ({param.annotation.__name__})"
                    if param.default != inspect.Parameter.empty:
                        param_desc += f" (default: {param.default})"
                    self.parameters[param_name] = param_desc
        else:
            self.parameters = parameters
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with the given arguments"""
        try:
            result = self.function(**kwargs)
            
            # Handle both regular functions and coroutines
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as e:
            return f"Error executing tool {self.name}: {str(e)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "required_capabilities": list(self.required_capabilities),
            "parameters": self.parameters
        }


class ToolRegistry:
    """
    Registry for tools that agents can use
    
    Attributes:
        config: System configuration
        tools: Dictionary of registered tools
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.tools: Dict[str, Tool] = {}
    
    async def initialize(self):
        """Initialize the tool registry with default tools"""
        # Register default tools
        await self._register_default_tools()
    
    async def _register_default_tools(self):
        """Register default tools"""
        # Calculator tool
        self.register_tool(
            name="calculator",
            description="Perform mathematical calculations",
            function=self._calculator,
            required_capabilities={"basic_reasoning"}
        )
        
        # Current time tool
        self.register_tool(
            name="get_current_time",
            description="Get the current date and time",
            function=self._get_current_time,
            required_capabilities={"basic_reasoning"}
        )
        
        # Text analysis tool
        self.register_tool(
            name="analyze_text",
            description="Analyze text for sentiment, entities, and key phrases",
            function=self._analyze_text,
            required_capabilities={"language_understanding"}
        )
    
    def register_tool(self, name: str, description: str, function: Callable, required_capabilities: Optional[Set[str]] = None, parameters: Optional[Dict[str, str]] = None) -> Tool:
        """
        Register a new tool
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            function: The function to execute
            required_capabilities: Capabilities required to use this tool
            parameters: Parameter descriptions
            
        Returns:
            The registered tool
        """
        tool = Tool(
            name=name,
            description=description,
            function=function,
            required_capabilities=required_capabilities,
            parameters=parameters
        )
        
        self.tools[name] = tool
        return tool
    
    def unregister_tool(self, name: str):
        """
        Unregister a tool
        
        Args:
            name: Name of the tool to unregister
        """
        if name in self.tools:
            del self.tools[name]
    
    async def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name
        
        Args:
            name: Name of the tool to get
            
        Returns:
            The tool if found, None otherwise
        """
        return self.tools.get(name)
    
    async def get_available_tools(self, required_capabilities: Optional[Set[str]] = None) -> Set[str]:
        """
        Get names of available tools, optionally filtered by required capabilities
        
        Args:
            required_capabilities: Capabilities to filter by
            
        Returns:
            Set of tool names
        """
        if required_capabilities is None:
            return set(self.tools.keys())
        
        return {
            name for name, tool in self.tools.items()
            if not tool.required_capabilities or tool.required_capabilities.issubset(required_capabilities)
        }
    
    async def execute_tool(self, name: str, **kwargs) -> Any:
        """
        Execute a tool
        
        Args:
            name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
        """
        tool = await self.get_tool(name)
        if tool:
            return await tool.execute(**kwargs)
        
        return f"Error: Tool '{name}' not found"
    
    # Default tool implementations
    
    def _calculator(self, expression: str) -> Union[float, str]:
        """
        Evaluate a mathematical expression
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of the calculation
        """
        try:
            # Use safer eval with limited globals
            result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "max": max, "min": min, "sum": sum})
            return result
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"
    
    def _get_current_time(self) -> str:
        """
        Get the current date and time
        
        Returns:
            Current date and time as a string
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for sentiment, entities, and key phrases
        This is a placeholder - in a real implementation, it would use NLP libraries
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        # Simple placeholder implementation
        word_count = len(text.split())
        char_count = len(text)
        
        # Very basic sentiment analysis
        positive_words = ["good", "great", "excellent", "happy", "positive", "wonderful", "best", "love"]
        negative_words = ["bad", "terrible", "awful", "sad", "negative", "worst", "hate", "poor"]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        sentiment = "neutral"
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        
        return {
            "word_count": word_count,
            "character_count": char_count,
            "sentiment": sentiment,
            "positive_words": positive_count,
            "negative_words": negative_count
        }