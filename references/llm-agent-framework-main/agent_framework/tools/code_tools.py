"""
Code Execution Tools for Agent Framework

Tools for executing and analyzing code.
"""

from typing import Dict, Any, Optional, List
from agent_framework.tools.base_tool import BaseTool
import sys
import io
import traceback
import ast


class PythonExecutorTool(BaseTool):
    """Safe Python code execution tool"""
    
    def __init__(self, timeout: int = 30):
        super().__init__(
            name="python_executor",
            description="Execute Python code safely. Input should be valid Python code."
        )
        self.timeout = timeout
        self.allowed_modules = {
            'math', 'random', 'datetime', 'json', 're',
            'collections', 'itertools', 'functools',
            'statistics', 'decimal', 'fractions'
        }
    
    def _validate_code(self, code: str) -> tuple[bool, str]:
        """Validate code for safety"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check for dangerous operations
        dangerous_nodes = []
        
        for node in ast.walk(tree):
            # Check for imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.allowed_modules:
                        dangerous_nodes.append(f"Import of '{alias.name}' not allowed")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in self.allowed_modules:
                    dangerous_nodes.append(f"Import from '{node.module}' not allowed")
            
            # Check for exec/eval
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', 'compile', '__import__']:
                        dangerous_nodes.append(f"'{node.func.id}' is not allowed")
            
            # Check for file operations
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['open', 'file']:
                        dangerous_nodes.append("File operations not allowed")
        
        if dangerous_nodes:
            return False, "Security violations: " + "; ".join(dangerous_nodes)
        
        return True, ""
    
    def run(self, code: str) -> str:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution output or error message
        """
        # Validate code
        is_valid, error = self._validate_code(code)
        if not is_valid:
            return f"❌ Validation Error: {error}"
        
        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        result = None
        
        try:
            # Create restricted namespace
            namespace = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'abs': abs,
                    'round': round,
                    'sorted': sorted,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'isinstance': isinstance,
                    'type': type,
                    'True': True,
                    'False': False,
                    'None': None,
                }
            }
            
            # Import allowed modules
            import math
            import random
            import datetime
            import json
            import re
            import collections
            import itertools
            import functools
            
            namespace['math'] = math
            namespace['random'] = random
            namespace['datetime'] = datetime
            namespace['json'] = json
            namespace['re'] = re
            namespace['collections'] = collections
            namespace['itertools'] = itertools
            namespace['functools'] = functools
            
            # Execute code
            exec(code, namespace)
            
            # Get output
            stdout_output = sys.stdout.getvalue()
            stderr_output = sys.stderr.getvalue()
            
            output_parts = []
            if stdout_output:
                output_parts.append(f"Output:\n{stdout_output}")
            if stderr_output:
                output_parts.append(f"Errors:\n{stderr_output}")
            
            if output_parts:
                result = "\n".join(output_parts)
            else:
                result = "✅ Code executed successfully (no output)"
        
        except Exception as e:
            result = f"❌ Execution Error:\n{traceback.format_exc()}"
        
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return result


class CodeAnalyzerTool(BaseTool):
    """Tool for analyzing code structure and quality"""
    
    def __init__(self):
        super().__init__(
            name="code_analyzer",
            description="Analyze Python code for structure, complexity, and potential issues."
        )
    
    def run(self, code: str) -> str:
        """
        Analyze Python code.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Analysis report
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"❌ Syntax Error: {e}"
        
        analysis = {
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'complexity': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis['functions'].append({
                    'name': node.name,
                    'args': len(node.args.args),
                    'lines': node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                })
                analysis['complexity'] += 1
            
            elif isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                analysis['classes'].append({
                    'name': node.name,
                    'methods': methods
                })
                analysis['complexity'] += 2
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis['imports'].append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                analysis['imports'].append(f"from {node.module}")
            
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                analysis['complexity'] += 1
        
        # Format report
        report = ["📊 Code Analysis Report", "=" * 40]
        
        report.append(f"\n📦 Imports ({len(analysis['imports'])}):")
        for imp in analysis['imports'][:10]:
            report.append(f"  - {imp}")
        
        report.append(f"\n🔧 Functions ({len(analysis['functions'])}):")
        for func in analysis['functions']:
            report.append(f"  - {func['name']}({func['args']} args, {func['lines']} lines)")
        
        report.append(f"\n🏗️ Classes ({len(analysis['classes'])}):")
        for cls in analysis['classes']:
            report.append(f"  - {cls['name']}: {len(cls['methods'])} methods")
        
        report.append(f"\n📈 Complexity Score: {analysis['complexity']}")
        
        if analysis['complexity'] > 20:
            report.append("⚠️ High complexity - consider refactoring")
        elif analysis['complexity'] > 10:
            report.append("ℹ️ Moderate complexity")
        else:
            report.append("✅ Low complexity")
        
        return "\n".join(report)


class CodeFormatterTool(BaseTool):
    """Tool for formatting Python code"""
    
    def __init__(self):
        super().__init__(
            name="code_formatter",
            description="Format Python code according to PEP 8 style guidelines."
        )
    
    def run(self, code: str) -> str:
        """
        Format Python code.
        
        Args:
            code: Python code to format
            
        Returns:
            Formatted code or error message
        """
        try:
            import black
            
            formatted = black.format_str(code, mode=black.Mode())
            return f"✅ Formatted Code:\n\n```python\n{formatted}\n```"
        
        except ImportError:
            return "❌ black not installed. Run: pip install black"
        except Exception as e:
            return f"❌ Formatting Error: {str(e)}"
