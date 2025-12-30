"""
AST Parser - Python AST parsing and analysis utilities.
Provides code structure analysis and validation.
"""

import ast
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from src.utils.structured_logging import get_logger

logger = get_logger("ast_parser")


@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str
    lineno: int
    end_lineno: int
    args: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    docstring: Optional[str] = None
    returns: Optional[str] = None


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    lineno: int
    end_lineno: int
    methods: List[FunctionInfo] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class CodeStructure:
    """Represents the structure of a Python file."""
    file_path: str
    imports: List[str] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    global_vars: List[str] = field(default_factory=list)
    
    def get_function(self, name: str) -> Optional[FunctionInfo]:
        """Get function by name."""
        for func in self.functions:
            if func.name == name:
                return func
        
        # Check class methods
        for cls in self.classes:
            for method in cls.methods:
                if method.name == name:
                    return method
        
        return None
    
    def get_class(self, name: str) -> Optional[ClassInfo]:
        """Get class by name."""
        for cls in self.classes:
            if cls.name == name:
                return cls
        return None


class ASTParser:
    """
    Parser for Python AST operations.
    Provides analysis and validation of Python code.
    """
    
    def parse_source(self, source: str, file_path: str = "<string>") -> CodeStructure:
        """
        Parse Python source code into structured format.
        
        Args:
            source: Python source code
            file_path: Path to the file (for reference)
            
        Returns:
            CodeStructure with parsed information
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.error(f"Syntax error parsing {file_path}: {e}")
            return CodeStructure(file_path=file_path)
        
        structure = CodeStructure(file_path=file_path)
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    structure.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    structure.imports.append(node.module)
        
        # Extract top-level items
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._extract_function_info(node)
                structure.functions.append(func_info)
            
            elif isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node)
                structure.classes.append(class_info)
            
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        structure.global_vars.append(target.id)
        
        return structure
    
    def _extract_function_info(self, node: ast.FunctionDef) -> FunctionInfo:
        """Extract information from a function definition."""
        args = [arg.arg for arg in node.args.args]
        
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                decorators.append(dec.func.id)
        
        docstring = ast.get_docstring(node)
        
        returns = None
        if node.returns:
            try:
                returns = ast.unparse(node.returns)
            except:
                returns = None
        
        return FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            args=args,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            docstring=docstring,
            returns=returns
        )
    
    def _extract_class_info(self, node: ast.ClassDef) -> ClassInfo:
        """Extract information from a class definition."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
        
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
        
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function_info(item))
        
        docstring = ast.get_docstring(node)
        
        return ClassInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            methods=methods,
            bases=bases,
            decorators=decorators,
            docstring=docstring
        )
    
    def validate_syntax(self, source: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax.
        
        Args:
            source: Python source code
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(source)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def get_line_range(self, source: str, start_line: int, end_line: int) -> str:
        """
        Extract a range of lines from source code.
        
        Args:
            source: Source code
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (inclusive)
            
        Returns:
            Extracted code as string
        """
        lines = source.splitlines(keepends=True)
        # Convert to 0-indexed
        return ''.join(lines[start_line - 1:end_line])
    
    def find_imports(self, source: str) -> List[str]:
        """
        Find all imports in source code.
        
        Args:
            source: Python source code
            
        Returns:
            List of imported module names
        """
        imports = []
        
        try:
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except SyntaxError:
            pass
        
        return imports
    
    def find_function_calls(self, source: str, function_name: str) -> List[int]:
        """
        Find all calls to a specific function.
        
        Args:
            source: Python source code
            function_name: Name of function to find
            
        Returns:
            List of line numbers where function is called
        """
        call_lines = []
        
        try:
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == function_name:
                        call_lines.append(node.lineno)
        except SyntaxError:
            pass
        
        return call_lines
