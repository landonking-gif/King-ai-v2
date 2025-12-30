"""
Code Quality Analyzer - Metrics and quality checks for Python code.
"""

import ast
import re
from typing import Dict, List, Any
from dataclasses import dataclass, field

from src.utils.ast_parser import ASTParser, CodeStructure
from src.utils.structured_logging import get_logger

logger = get_logger("code_quality")


@dataclass
class QualityMetrics:
    """Quality metrics for a piece of code."""
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    function_count: int = 0
    class_count: int = 0
    avg_function_length: float = 0.0
    max_function_length: int = 0
    docstring_coverage: float = 0.0
    type_hint_coverage: float = 0.0
    issues: List[str] = field(default_factory=list)
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Penalize high complexity
        if self.cyclomatic_complexity > 10:
            score -= min(20, (self.cyclomatic_complexity - 10) * 2)
        
        # Penalize long functions
        if self.max_function_length > 50:
            score -= min(15, (self.max_function_length - 50) * 0.3)
        
        # Reward docstrings
        score += self.docstring_coverage * 10
        
        # Reward type hints
        score += self.type_hint_coverage * 10
        
        # Penalize issues
        score -= len(self.issues) * 2
        
        return max(0, min(100, score))


class CodeQualityAnalyzer:
    """
    Analyzes code quality and generates metrics.
    """
    
    def __init__(self):
        self.parser = ASTParser()
    
    def analyze_file(self, file_path: str) -> QualityMetrics:
        """Analyze a Python file for quality metrics."""
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        return self.analyze_source(source, file_path)
    
    def analyze_source(self, source: str, file_path: str = "<string>") -> QualityMetrics:
        """Analyze source code for quality metrics."""
        metrics = QualityMetrics()
        
        # Basic metrics
        lines = source.split('\n')
        metrics.lines_of_code = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        
        # Parse structure
        structure = self.parser.parse_source(source, file_path)
        
        if structure.errors:
            metrics.issues.extend(structure.errors)
            return metrics
        
        metrics.function_count = len(structure.all_functions)
        metrics.class_count = len(structure.classes)
        
        # Function length analysis
        if structure.all_functions:
            lengths = [f.end_lineno - f.lineno + 1 for f in structure.all_functions]
            metrics.avg_function_length = sum(lengths) / len(lengths)
            metrics.max_function_length = max(lengths)
        
        # Cyclomatic complexity
        try:
            tree = ast.parse(source)
            metrics.cyclomatic_complexity = self._calculate_complexity(tree)
        except (SyntaxError, ValueError) as e:
            logger.debug(f"Could not calculate complexity for {file_path}: {e}")
        
        # Docstring coverage
        with_docs = sum(1 for f in structure.all_functions if f.docstring)
        if structure.all_functions:
            metrics.docstring_coverage = with_docs / len(structure.all_functions)
        
        # Type hint coverage
        metrics.type_hint_coverage = self._analyze_type_hints(source)
        
        # Run quality checks
        metrics.issues.extend(self._check_quality(source, structure))
        
        return metrics
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Branches add complexity
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # and/or add complexity
                complexity += len(node.values) - 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
                if node.ifs:
                    complexity += len(node.ifs)
        
        return complexity
    
    def _analyze_type_hints(self, source: str) -> float:
        """Analyze type hint coverage."""
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return 0.0
        
        total_params = 0
        typed_params = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    total_params += 1
                    if arg.annotation:
                        typed_params += 1
                
                # Return type
                total_params += 1
                if node.returns:
                    typed_params += 1
        
        return typed_params / total_params if total_params > 0 else 1.0
    
    def _check_quality(self, source: str, structure: CodeStructure) -> List[str]:
        """Run quality checks and return issues."""
        issues = []
        
        # Check for long lines
        for i, line in enumerate(source.split('\n'), 1):
            if len(line) > 120:
                issues.append(f"Line {i} exceeds 120 characters")
        
        # Check for missing docstrings on public functions
        for func in structure.all_functions:
            if not func.name.startswith('_') and not func.docstring:
                issues.append(f"Public function '{func.full_name}' missing docstring")
        
        # Check for too many arguments
        for func in structure.all_functions:
            if len(func.args) > 7:
                issues.append(f"Function '{func.full_name}' has too many arguments ({len(func.args)})")
        
        return issues[:10]  # Limit to 10 issues
