"""
Code Transformer - AST-based code transformations.
Provides safe, structural code modifications.
"""

import ast
import astor  # pip install astor
from typing import Optional, List, Callable, Any
from dataclasses import dataclass

from src.utils.structured_logging import get_logger

logger = get_logger("code_transformer")


@dataclass
class TransformResult:
    """Result of a code transformation."""
    success: bool
    code: str = ""
    changes_made: List[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.changes_made is None:
            self.changes_made = []


class AddImportTransformer(ast.NodeTransformer):
    """AST transformer that adds imports."""
    
    def __init__(self, imports_to_add: List[str]):
        self.imports_to_add = imports_to_add
        self.existing_imports = set()
    
    def visit_Import(self, node):
        for alias in node.names:
            self.existing_imports.add(alias.name)
        return node
    
    def visit_ImportFrom(self, node):
        if node.module:
            self.existing_imports.add(node.module)
        return node


class AddDecoratorTransformer(ast.NodeTransformer):
    """AST transformer that adds decorators to functions."""
    
    def __init__(self, function_name: str, decorator: str):
        self.function_name = function_name
        self.decorator = decorator
        self.modified = False
    
    def visit_FunctionDef(self, node):
        if node.name == self.function_name:
            # Add decorator
            decorator_node = ast.Name(id=self.decorator, ctx=ast.Load())
            node.decorator_list.insert(0, decorator_node)
            self.modified = True
        return node
    
    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)


class TypeHintAdder(ast.NodeTransformer):
    """AST transformer that adds type hints to functions."""
    
    def __init__(self, type_hints: dict):
        """
        Args:
            type_hints: Dict mapping arg names to type strings
        """
        self.type_hints = type_hints
        self.modified = False
    
    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            if arg.arg in self.type_hints and not arg.annotation:
                arg.annotation = ast.parse(self.type_hints[arg.arg], mode='eval').body
                self.modified = True
        return node


class CodeTransformer:
    """
    High-level code transformation utilities.
    Uses AST for safe, structural modifications.
    """
    
    def add_import(self, source: str, import_statement: str) -> TransformResult:
        """
        Add an import statement to source code.
        
        Args:
            source: Original source code
            import_statement: Import to add (e.g., "from typing import Optional")
            
        Returns:
            Transformation result
        """
        try:
            tree = ast.parse(source)
            
            # Check if import already exists
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if import_statement.endswith(alias.name):
                            return TransformResult(
                                success=True,
                                code=source,
                                changes_made=["Import already exists"]
                            )
            
            # Parse the import statement
            import_node = ast.parse(import_statement).body[0]
            
            # Find the right position (after existing imports)
            insert_pos = 0
            for i, node in enumerate(tree.body):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    insert_pos = i + 1
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    # Skip docstrings
                    insert_pos = i + 1
                else:
                    break
            
            tree.body.insert(insert_pos, import_node)
            
            new_code = astor.to_source(tree)
            
            return TransformResult(
                success=True,
                code=new_code,
                changes_made=[f"Added import: {import_statement}"]
            )
            
        except Exception as e:
            return TransformResult(
                success=False,
                error=str(e)
            )
    
    def add_decorator(
        self,
        source: str,
        function_name: str,
        decorator: str
    ) -> TransformResult:
        """
        Add a decorator to a function.
        
        Args:
            source: Original source code
            function_name: Name of function to decorate
            decorator: Decorator to add (without @)
            
        Returns:
            Transformation result
        """
        try:
            tree = ast.parse(source)
            
            transformer = AddDecoratorTransformer(function_name, decorator)
            new_tree = transformer.visit(tree)
            
            if not transformer.modified:
                return TransformResult(
                    success=False,
                    error=f"Function '{function_name}' not found"
                )
            
            new_code = astor.to_source(new_tree)
            
            return TransformResult(
                success=True,
                code=new_code,
                changes_made=[f"Added @{decorator} to {function_name}"]
            )
            
        except Exception as e:
            return TransformResult(
                success=False,
                error=str(e)
            )
    
    def rename_function(
        self,
        source: str,
        old_name: str,
        new_name: str
    ) -> TransformResult:
        """
        Rename a function and update all calls.
        
        Args:
            source: Original source code
            old_name: Current function name
            new_name: New function name
            
        Returns:
            Transformation result
        """
        try:
            tree = ast.parse(source)
            changes = []
            
            class RenameTransformer(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    if node.name == old_name:
                        node.name = new_name
                        changes.append(f"Renamed function definition: {old_name} -> {new_name}")
                    self.generic_visit(node)
                    return node
                
                def visit_AsyncFunctionDef(self, node):
                    return self.visit_FunctionDef(node)
                
                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name) and node.func.id == old_name:
                        node.func.id = new_name
                        changes.append(f"Updated call: {old_name}() -> {new_name}()")
                    self.generic_visit(node)
                    return node
            
            transformer = RenameTransformer()
            new_tree = transformer.visit(tree)
            
            if not changes:
                return TransformResult(
                    success=False,
                    error=f"Function '{old_name}' not found"
                )
            
            new_code = astor.to_source(new_tree)
            
            return TransformResult(
                success=True,
                code=new_code,
                changes_made=changes
            )
            
        except Exception as e:
            return TransformResult(
                success=False,
                error=str(e)
            )
    
    def extract_function(
        self,
        source: str,
        function_name: str,
        start_line: int,
        end_line: int,
        new_function_name: str
    ) -> TransformResult:
        """
        Extract lines from a function into a new function.
        
        Args:
            source: Original source code
            function_name: Function to extract from
            start_line: Start line of code to extract (relative to function)
            end_line: End line of code to extract
            new_function_name: Name for the new function
            
        Returns:
            Transformation result with extracted function
        """
        # This is a complex refactoring - simplified version
        try:
            lines = source.split('\n')
            
            # Find the function
            tree = ast.parse(source)
            target_func = None
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == function_name:
                        target_func = node
                        break
            
            if not target_func:
                return TransformResult(
                    success=False,
                    error=f"Function '{function_name}' not found"
                )
            
            # For now, return a placeholder - full implementation is complex
            return TransformResult(
                success=False,
                error="Extract function not fully implemented - use code generator instead"
            )
            
        except Exception as e:
            return TransformResult(
                success=False,
                error=str(e)
            )
    
    def add_error_handling(
        self,
        source: str,
        function_name: str,
        exception_type: str = "Exception"
    ) -> TransformResult:
        """
        Wrap function body in try-except.
        
        Args:
            source: Original source code
            function_name: Function to add error handling to
            exception_type: Type of exception to catch
            
        Returns:
            Transformation result
        """
        try:
            tree = ast.parse(source)
            changes = []
            
            class ErrorHandlingTransformer(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    if node.name == function_name:
                        # Check if already has try-except
                        if any(isinstance(stmt, ast.Try) for stmt in node.body):
                            return node
                        
                        # Create try-except wrapper
                        try_node = ast.Try(
                            body=node.body,
                            handlers=[
                                ast.ExceptHandler(
                                    type=ast.Name(id=exception_type, ctx=ast.Load()),
                                    name='e',
                                    body=[
                                        ast.Raise(
                                            exc=ast.Call(
                                                func=ast.Name(id='RuntimeError', ctx=ast.Load()),
                                                args=[
                                                    ast.JoinedStr(values=[
                                                        ast.Constant(value=f'{function_name} failed: '),
                                                        ast.FormattedValue(
                                                            value=ast.Name(id='e', ctx=ast.Load()),
                                                            conversion=-1
                                                        )
                                                    ])
                                                ],
                                                keywords=[]
                                            ),
                                            cause=ast.Name(id='e', ctx=ast.Load())
                                        )
                                    ]
                                )
                            ],
                            orelse=[],
                            finalbody=[]
                        )
                        
                        node.body = [try_node]
                        changes.append(f"Added try-except to {function_name}")
                    
                    return node
            
            transformer = ErrorHandlingTransformer()
            new_tree = transformer.visit(tree)
            
            if not changes:
                return TransformResult(
                    success=False,
                    error=f"Function '{function_name}' not found or already has error handling"
                )
            
            new_code = astor.to_source(new_tree)
            
            return TransformResult(
                success=True,
                code=new_code,
                changes_made=changes
            )
            
        except Exception as e:
            return TransformResult(
                success=False,
                error=str(e)
            )
