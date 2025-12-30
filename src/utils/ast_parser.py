"""
AST Parser - Utilities for parsing and analyzing Python code.
Used by the evolution engine to understand code structure.
"""

import ast
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.structured_logging import get_logger

logger = get_logger("ast_parser")


@dataclass
class FunctionInfo:
    """Information about a function or method."""
    name: str
    lineno: int
    end_lineno: int
    args: List[str]
    decorators: List[str]
    docstring: Optional[str]
    is_async: bool = False
    is_method: bool = False
    class_name: Optional[str] = None
    calls: List[str] = field(default_factory=list)
    
    @property
    def full_name(self) -> str:
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    lineno: int
    end_lineno: int
    bases: List[str]
    decorators: List[str]
    docstring: Optional[str]
    methods: List[FunctionInfo] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str
    names: List[str]  # Specific names imported, or ['*'] for star import
    alias: Optional[str] = None
    lineno: int = 0
    is_from_import: bool = False


@dataclass
class CodeStructure:
    """Complete structure of a Python file."""
    file_path: str
    imports: List[ImportInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    global_variables: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def all_functions(self) -> List[FunctionInfo]:
        """Get all functions including methods."""
        all_funcs = list(self.functions)
        for cls in self.classes:
            all_funcs.extend(cls.methods)
        return all_funcs
    
    def get_function(self, name: str) -> Optional[FunctionInfo]:
        """Find a function by name."""
        for func in self.all_functions:
            if func.name == name or func.full_name == name:
                return func
        return None
    
    def get_class(self, name: str) -> Optional[ClassInfo]:
        """Find a class by name."""
        for cls in self.classes:
            if cls.name == name:
                return cls
        return None


class ASTParser:
    """
    Parses Python source code into structured information.
    """
    
    def parse_file(self, file_path: str) -> CodeStructure:
        """
        Parse a Python file and extract its structure.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            CodeStructure with all extracted information
        """
        structure = CodeStructure(file_path=file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            return self.parse_source(source, file_path)
            
        except FileNotFoundError:
            structure.errors.append(f"File not found: {file_path}")
        except Exception as e:
            structure.errors.append(f"Parse error: {str(e)}")
        
        return structure
    
    def parse_source(self, source: str, file_path: str = "<string>") -> CodeStructure:
        """
        Parse Python source code string.
        
        Args:
            source: Python source code
            file_path: Optional file path for error messages
            
        Returns:
            CodeStructure with all extracted information
        """
        structure = CodeStructure(file_path=file_path)
        
        try:
            tree = ast.parse(source)
            
            # Extract imports
            structure.imports = self._extract_imports(tree)
            
            # Extract classes and functions
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._extract_class(node)
                    structure.classes.append(class_info)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info = self._extract_function(node)
                    structure.functions.append(func_info)
                elif isinstance(node, ast.Assign):
                    # Global variable assignments
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            structure.global_variables.append(target.id)
            
        except SyntaxError as e:
            structure.errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            structure.errors.append(f"Parse error: {str(e)}")
        
        return structure
    
    def _extract_imports(self, tree: ast.AST) -> List[ImportInfo]:
        """Extract all imports from an AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[alias.name],
                        alias=alias.asname,
                        lineno=node.lineno,
                        is_from_import=False
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append(ImportInfo(
                    module=module,
                    names=names,
                    lineno=node.lineno,
                    is_from_import=True
                ))
        
        return imports
    
    def _extract_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extract class information from a ClassDef node."""
        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base))
        
        # Get decorators
        decorators = self._extract_decorators(node)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        class_info = ClassInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            bases=bases,
            decorators=decorators,
            docstring=docstring
        )
        
        # Extract methods and attributes
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._extract_function(item, class_name=node.name)
                class_info.methods.append(method)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info.attributes.append(target.id)
        
        return class_info
    
    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        class_name: str = None
    ) -> FunctionInfo:
        """Extract function information from a FunctionDef node."""
        # Get arguments
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        
        # Get decorators
        decorators = self._extract_decorators(node)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Get function calls within the body
        calls = self._extract_calls(node)
        
        return FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            args=args,
            decorators=decorators,
            docstring=docstring,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=class_name is not None,
            class_name=class_name,
            calls=calls
        )
    
    def _extract_decorators(self, node) -> List[str]:
        """Extract decorator names from a node."""
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(ast.unparse(dec.func))
            elif isinstance(dec, ast.Attribute):
                decorators.append(ast.unparse(dec))
        return decorators
    
    def _extract_calls(self, node: ast.AST) -> List[str]:
        """Extract function/method calls from a node."""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return list(set(calls))  # Deduplicate
    
    def get_line_range(self, source: str, start: int, end: int) -> str:
        """
        Extract lines from source code.
        
        Args:
            source: Full source code
            start: Start line (1-indexed)
            end: End line (1-indexed, inclusive)
            
        Returns:
            Extracted lines as string
        """
        lines = source.split('\n')
        return '\n'.join(lines[start-1:end])
    
    def validate_syntax(self, source: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(source)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
