"""
Code Analyzer - Analyzes Python code structure using AST.
"""

import ast
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ParameterInfo:
    """Information about a function parameter."""
    name: str
    type_hint: Optional[str] = None
    default_value: Optional[str] = None


@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str
    parameters: List[ParameterInfo] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    line_number: int = 0
    is_async: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "parameters": [
                {"name": p.name, "type": p.type_hint, "default": p.default_value}
                for p in self.parameters
            ],
            "return_type": self.return_type,
            "docstring": self.docstring,
            "line_number": self.line_number,
            "is_async": self.is_async
        }


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    methods: List[FunctionInfo] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    line_number: int = 0


class CodeAnalyzer:
    """
    Analyzes Python code structure using AST parsing.
    """
    
    def analyze_code(self, source_code: str) -> Dict[str, Any]:
        """
        Analyze Python source code.
        
        Args:
            source_code: Python source code to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return {
                "error": f"Syntax error: {e}",
                "functions": [],
                "classes": []
            }
        
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                functions.append(self._analyze_function(node))
            elif isinstance(node, ast.ClassDef):
                classes.append(self._analyze_class(node))
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": self._extract_imports(tree)
        }
    
    def _analyze_function(self, node) -> FunctionInfo:
        """Analyze a function definition."""
        parameters = []
        for arg in node.args.args:
            param = ParameterInfo(
                name=arg.arg,
                type_hint=ast.unparse(arg.annotation) if arg.annotation else None
            )
            parameters.append(param)
        
        return FunctionInfo(
            name=node.name,
            parameters=parameters,
            return_type=ast.unparse(node.returns) if node.returns else None,
            docstring=ast.get_docstring(node),
            line_number=node.lineno,
            is_async=isinstance(node, ast.AsyncFunctionDef)
        )
    
    def _analyze_class(self, node: ast.ClassDef) -> ClassInfo:
        """Analyze a class definition."""
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._analyze_function(item))
        
        bases = [ast.unparse(base) for base in node.bases]
        
        return ClassInfo(
            name=node.name,
            methods=methods,
            bases=bases,
            docstring=ast.get_docstring(node),
            line_number=node.lineno
        )
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports
