"""
Base Tool Class

This module provides the abstract base class for all tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Abstract base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool"""
        pass
    
    def __call__(self, **kwargs) -> Any:
        """Allow tool to be called directly"""
        return self.run(**kwargs)


class SearchTool(BaseTool):
    """Tool for searching information"""
    
    def __init__(self):
        super().__init__(
            name="search",
            description="Search for information on the internet. Input should be a search query."
        )
    
    def run(self, query: str) -> str:
        """Simulate search (placeholder implementation)"""
        return f"Search results for '{query}': [Placeholder - integrate with real search API]"


class CalculatorTool(BaseTool):
    """Tool for mathematical calculations"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Input should be a mathematical expression."
        )
    
    def run(self, expression: str) -> str:
        """Evaluate mathematical expression"""
        try:
            result = eval(expression, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


class PythonREPLTool(BaseTool):
    """Tool for executing Python code"""
    
    def __init__(self):
        super().__init__(
            name="python_repl",
            description="Execute Python code. Input should be valid Python code."
        )
    
    def run(self, code: str) -> str:
        """Execute Python code safely"""
        try:
            # Create restricted namespace
            namespace = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                }
            }
            exec(code, namespace)
            return "Code executed successfully"
        except Exception as e:
            return f"Error: {str(e)}"
