"""
Code Analyzer - Basic static analysis utilities for code review.
Provides simple pattern matching and code metrics.
"""

import re
from typing import Dict, List, Any


class CodeAnalyzer:
    """
    Simple code analyzer for basic static analysis.
    Used by code generation and review agents.
    """
    
    def __init__(self):
        """Initialize the code analyzer."""
        pass
    
    def analyze_python(self, code: str) -> Dict[str, Any]:
        """
        Analyze Python code for basic metrics and patterns.
        
        Args:
            code: Python source code to analyze
            
        Returns:
            Dictionary with analysis results
        """
        lines = code.split('\n')
        
        return {
            "lines": len(lines),
            "functions": len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE)),
            "classes": len(re.findall(r'^\s*class\s+\w+', code, re.MULTILINE)),
            "imports": len(re.findall(r'^\s*(?:from|import)\s+', code, re.MULTILINE)),
            "comments": len(re.findall(r'^\s*#', code, re.MULTILINE)),
            "docstrings": len(re.findall(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', code)),
        }
    
    def extract_imports(self, code: str, language: str = "python") -> List[str]:
        """
        Extract import statements from code.
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            List of imported modules
        """
        if language == "python":
            imports = re.findall(r'^\s*(?:from|import)\s+(\S+)', code, re.MULTILINE)
            return list(set(imports))
        return []
    
    def check_complexity(self, code: str) -> Dict[str, Any]:
        """
        Basic complexity metrics.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Complexity metrics
        """
        lines = [line for line in code.split('\n') if line.strip()]
        
        # Count control flow statements
        control_flow = len(re.findall(
            r'\b(if|else|elif|for|while|try|except|with)\b',
            code
        ))
        
        return {
            "total_lines": len(lines),
            "control_flow_statements": control_flow,
            "estimated_complexity": control_flow + len(lines) // 10
        }
