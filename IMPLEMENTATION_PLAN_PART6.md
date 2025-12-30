# King AI v2 - Implementation Plan Part 6
## Evolution Engine - Code Analysis & AST Tools

**Target Timeline:** Week 5
**Objective:** Build AST-based code analysis tools for safe self-modification and impact assessment.

---

## Overview of All Parts

| Part | Title | Status |
|------|-------|--------|
| 1 | Infrastructure Layer & Core System Hardening | ✅ Complete |
| 2 | Master AI Brain - Core Enhancements | ✅ Complete |
| 3 | Master AI Brain - Context & Memory System | ✅ Complete |
| 4 | Master AI Brain - Planning & ReAct Implementation | ✅ Complete |
| 5 | Evolution Engine - Core Models & Proposal System | ✅ Complete |
| **6** | **Evolution Engine - Code Analysis & AST Tools** | 🔄 Current |
| 7 | Evolution Engine - Code Patching & Generation | ⏳ Pending |
| 8 | Evolution Engine - Git Integration & Rollback | ⏳ Pending |
| 9 | Evolution Engine - Sandbox Testing | ⏳ Pending |
| 10 | Sub-Agent: Research (Web/API) | ⏳ Pending |
| 11 | Sub-Agent: Code Generator | ⏳ Pending |
| 12 | Sub-Agent: Content (Blog/SEO) | ⏳ Pending |
| 13 | Sub-Agent: Commerce - Shopify | ⏳ Pending |
| 14 | Sub-Agent: Commerce - Suppliers | ⏳ Pending |
| 15 | Sub-Agent: Finance - Stripe | ⏳ Pending |
| 16 | Sub-Agent: Finance - Plaid/Banking | ⏳ Pending |
| 17 | Sub-Agent: Analytics | ⏳ Pending |
| 18 | Sub-Agent: Legal | ⏳ Pending |
| 19 | Business: Lifecycle Engine | ⏳ Pending |
| 20 | Business: Playbook System | ⏳ Pending |
| 21 | Business: Portfolio Management | ⏳ Pending |
| 22 | Dashboard: React Components | ⏳ Pending |
| 23 | Dashboard: Approval Workflows | ⏳ Pending |
| 24 | Dashboard: WebSocket & Monitoring | ⏳ Pending |

---

## Part 6 Scope

This part focuses on:
1. Python AST parsing and analysis utilities
2. Code structure extraction (classes, functions, imports)
3. Dependency graph building
4. Impact analysis for proposed changes
5. Safe modification boundary detection

---

## Task 6.1: Create AST Parser Utility

**File:** `src/utils/ast_parser.py` (CREATE NEW FILE)

```python
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
```

---

## Task 6.2: Create Dependency Analyzer

**File:** `src/utils/dependency_analyzer.py` (CREATE NEW FILE)

```python
"""
Dependency Analyzer - Analyzes code dependencies and relationships.
Builds dependency graphs for impact analysis.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import os

from src.utils.ast_parser import ASTParser, CodeStructure, ImportInfo
from src.utils.structured_logging import get_logger

logger = get_logger("dependency_analyzer")


@dataclass
class Dependency:
    """A dependency relationship between code elements."""
    source: str       # The file/function that depends
    target: str       # The file/function being depended on
    dep_type: str     # "import", "call", "inherit", "reference"
    lineno: int = 0
    
    def __hash__(self):
        return hash((self.source, self.target, self.dep_type))


@dataclass
class DependencyGraph:
    """Graph of dependencies in the codebase."""
    nodes: Set[str] = field(default_factory=set)
    edges: List[Dependency] = field(default_factory=list)
    
    # Quick lookups
    _outgoing: Dict[str, List[Dependency]] = field(default_factory=dict)
    _incoming: Dict[str, List[Dependency]] = field(default_factory=dict)
    
    def add_dependency(self, dep: Dependency):
        """Add a dependency to the graph."""
        self.nodes.add(dep.source)
        self.nodes.add(dep.target)
        self.edges.append(dep)
        
        # Update lookups
        if dep.source not in self._outgoing:
            self._outgoing[dep.source] = []
        self._outgoing[dep.source].append(dep)
        
        if dep.target not in self._incoming:
            self._incoming[dep.target] = []
        self._incoming[dep.target].append(dep)
    
    def get_dependents(self, target: str) -> List[str]:
        """Get all nodes that depend on target."""
        deps = self._incoming.get(target, [])
        return [d.source for d in deps]
    
    def get_dependencies(self, source: str) -> List[str]:
        """Get all nodes that source depends on."""
        deps = self._outgoing.get(source, [])
        return [d.target for d in deps]
    
    def get_transitive_dependents(self, target: str) -> Set[str]:
        """Get all nodes that directly or indirectly depend on target."""
        result = set()
        to_visit = [target]
        
        while to_visit:
            current = to_visit.pop()
            dependents = self.get_dependents(current)
            for dep in dependents:
                if dep not in result:
                    result.add(dep)
                    to_visit.append(dep)
        
        return result


@dataclass
class ImpactAnalysis:
    """Result of impact analysis for a proposed change."""
    changed_file: str
    changed_functions: List[str]
    affected_files: List[str]
    affected_functions: List[str]
    risk_score: float  # 0.0 to 1.0
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_high_impact(self) -> bool:
        return self.risk_score > 0.7 or len(self.affected_files) > 5


class DependencyAnalyzer:
    """
    Analyzes dependencies across the codebase.
    """
    
    def __init__(self, project_root: str):
        """
        Initialize the analyzer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.parser = ASTParser()
        self._file_cache: Dict[str, CodeStructure] = {}
        self._graph: Optional[DependencyGraph] = None
    
    def build_dependency_graph(self, include_patterns: List[str] = None) -> DependencyGraph:
        """
        Build a dependency graph for the entire project.
        
        Args:
            include_patterns: Glob patterns to include (default: ['src/**/*.py'])
            
        Returns:
            Complete dependency graph
        """
        if include_patterns is None:
            include_patterns = ['src/**/*.py']
        
        graph = DependencyGraph()
        
        # Find all Python files
        python_files = []
        for pattern in include_patterns:
            python_files.extend(self.project_root.glob(pattern))
        
        logger.info(f"Analyzing {len(python_files)} Python files")
        
        # Parse each file and extract dependencies
        for file_path in python_files:
            rel_path = str(file_path.relative_to(self.project_root))
            
            structure = self.parser.parse_file(str(file_path))
            self._file_cache[rel_path] = structure
            
            # Add import dependencies
            for imp in structure.imports:
                target = self._resolve_import(imp, rel_path)
                if target:
                    graph.add_dependency(Dependency(
                        source=rel_path,
                        target=target,
                        dep_type="import",
                        lineno=imp.lineno
                    ))
            
            # Add inheritance dependencies
            for cls in structure.classes:
                for base in cls.bases:
                    graph.add_dependency(Dependency(
                        source=f"{rel_path}:{cls.name}",
                        target=base,
                        dep_type="inherit",
                        lineno=cls.lineno
                    ))
            
            # Add function call dependencies
            for func in structure.all_functions:
                for call in func.calls:
                    graph.add_dependency(Dependency(
                        source=f"{rel_path}:{func.full_name}",
                        target=call,
                        dep_type="call",
                        lineno=func.lineno
                    ))
        
        self._graph = graph
        logger.info(f"Built graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        return graph
    
    def _resolve_import(self, imp: ImportInfo, from_file: str) -> Optional[str]:
        """Resolve an import to a file path."""
        if imp.is_from_import:
            # Convert module path to file path
            parts = imp.module.split('.')
            possible_paths = [
                '/'.join(parts) + '.py',
                '/'.join(parts) + '/__init__.py'
            ]
            
            for path in possible_paths:
                if (self.project_root / path).exists():
                    return path
        
        return imp.module  # Return module name as-is if not found
    
    def analyze_impact(
        self,
        file_path: str,
        changed_lines: Tuple[int, int] = None,
        changed_functions: List[str] = None
    ) -> ImpactAnalysis:
        """
        Analyze the impact of changes to a file.
        
        Args:
            file_path: Path to the changed file (relative to project root)
            changed_lines: Tuple of (start, end) line numbers
            changed_functions: List of function names being changed
            
        Returns:
            Impact analysis result
        """
        if self._graph is None:
            self.build_dependency_graph()
        
        # If we have line numbers, find affected functions
        if changed_lines and not changed_functions:
            structure = self._file_cache.get(file_path)
            if structure:
                changed_functions = self._find_functions_in_range(
                    structure, changed_lines[0], changed_lines[1]
                )
        
        changed_functions = changed_functions or []
        
        # Find affected files
        affected_files = set()
        affected_functions = set()
        
        # Files that import this file
        file_dependents = self._graph.get_transitive_dependents(file_path)
        affected_files.update(f.split(':')[0] for f in file_dependents)
        
        # Functions that call changed functions
        for func_name in changed_functions:
            full_name = f"{file_path}:{func_name}"
            func_dependents = self._graph.get_transitive_dependents(full_name)
            affected_functions.update(func_dependents)
            affected_files.update(f.split(':')[0] for f in func_dependents if ':' in f)
        
        # Calculate risk score
        risk_score = self._calculate_risk(
            file_path, changed_functions, affected_files, affected_functions
        )
        
        # Generate warnings
        warnings = self._generate_warnings(
            file_path, changed_functions, affected_files
        )
        
        return ImpactAnalysis(
            changed_file=file_path,
            changed_functions=changed_functions,
            affected_files=list(affected_files),
            affected_functions=list(affected_functions),
            risk_score=risk_score,
            warnings=warnings
        )
    
    def _find_functions_in_range(
        self,
        structure: CodeStructure,
        start: int,
        end: int
    ) -> List[str]:
        """Find functions that overlap with a line range."""
        functions = []
        
        for func in structure.all_functions:
            if (func.lineno <= end and func.end_lineno >= start):
                functions.append(func.full_name)
        
        return functions
    
    def _calculate_risk(
        self,
        file_path: str,
        changed_functions: List[str],
        affected_files: Set[str],
        affected_functions: Set[str]
    ) -> float:
        """Calculate risk score for changes."""
        risk = 0.0
        
        # Base risk from file type
        if 'brain.py' in file_path or 'evolution' in file_path:
            risk += 0.3
        if 'database' in file_path:
            risk += 0.2
        if 'api' in file_path:
            risk += 0.1
        
        # Risk from scope of impact
        risk += min(0.3, len(affected_files) * 0.05)
        risk += min(0.2, len(affected_functions) * 0.02)
        
        # Risk from number of changes
        risk += min(0.2, len(changed_functions) * 0.05)
        
        return min(1.0, risk)
    
    def _generate_warnings(
        self,
        file_path: str,
        changed_functions: List[str],
        affected_files: Set[str]
    ) -> List[str]:
        """Generate warnings about the changes."""
        warnings = []
        
        if len(affected_files) > 10:
            warnings.append(f"High impact: {len(affected_files)} files affected")
        
        if 'brain.py' in file_path:
            warnings.append("CRITICAL: Changes to Master AI brain require extra review")
        
        if any('__init__' in f for f in changed_functions):
            warnings.append("Constructor changes may break dependent code")
        
        if 'api/' in file_path:
            warnings.append("API changes may affect external consumers")
        
        return warnings
    
    def get_safe_modification_boundaries(self, file_path: str) -> Dict[str, Tuple[int, int]]:
        """
        Get safe line boundaries for modifications.
        Returns ranges that can be modified with minimal impact.
        
        Returns:
            Dict mapping function names to (start, end) line tuples
        """
        structure = self._file_cache.get(file_path)
        if not structure:
            structure = self.parser.parse_file(str(self.project_root / file_path))
        
        boundaries = {}
        
        for func in structure.all_functions:
            # Check if this function has dependents
            full_name = f"{file_path}:{func.full_name}"
            dependents = self._graph.get_dependents(full_name) if self._graph else []
            
            if len(dependents) == 0:
                # Safe to modify - no dependents
                boundaries[func.full_name] = (func.lineno, func.end_lineno)
        
        return boundaries
```

---

## Task 6.3: Create Code Quality Analyzer

**File:** `src/utils/code_quality.py` (CREATE NEW FILE)

```python
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
        except:
            pass
        
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
        except:
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
```

---

## Testing Requirements

**File:** `tests/test_ast_parser.py` (CREATE NEW FILE)

```python
"""Tests for AST parser and analysis tools."""

import pytest
from src.utils.ast_parser import ASTParser, CodeStructure
from src.utils.dependency_analyzer import DependencyAnalyzer, DependencyGraph
from src.utils.code_quality import CodeQualityAnalyzer


class TestASTParser:
    """Tests for AST parser."""
    
    @pytest.fixture
    def parser(self):
        return ASTParser()
    
    def test_parse_simple_function(self, parser):
        source = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"
'''
        structure = parser.parse_source(source)
        
        assert len(structure.functions) == 1
        func = structure.functions[0]
        assert func.name == "hello"
        assert func.args == ["name"]
        assert func.docstring == "Say hello."
    
    def test_parse_class(self, parser):
        source = '''
class MyClass(BaseClass):
    """A test class."""
    
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
'''
        structure = parser.parse_source(source)
        
        assert len(structure.classes) == 1
        cls = structure.classes[0]
        assert cls.name == "MyClass"
        assert "BaseClass" in cls.bases
        assert len(cls.methods) == 2
    
    def test_parse_imports(self, parser):
        source = '''
import os
from typing import List, Dict
from src.utils import helper
'''
        structure = parser.parse_source(source)
        
        assert len(structure.imports) == 3
    
    def test_syntax_validation(self, parser):
        valid, error = parser.validate_syntax("def foo(): pass")
        assert valid
        assert error is None
        
        valid, error = parser.validate_syntax("def foo(")
        assert not valid
        assert error is not None


class TestCodeQuality:
    """Tests for code quality analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        return CodeQualityAnalyzer()
    
    def test_quality_metrics(self, analyzer):
        source = '''
def documented_function():
    """This function has a docstring."""
    pass

def undocumented():
    pass
'''
        metrics = analyzer.analyze_source(source)
        
        assert metrics.function_count == 2
        assert metrics.docstring_coverage == 0.5
        assert metrics.quality_score > 0
```

---

## Acceptance Criteria

- [ ] `src/utils/ast_parser.py` - Parse Python files into structured info
- [ ] `src/utils/dependency_analyzer.py` - Build dependency graphs
- [ ] `src/utils/code_quality.py` - Calculate quality metrics
- [ ] `tests/test_ast_parser.py` - All tests passing
- [ ] Can parse any Python file in the project
- [ ] Can identify impact of changes to any file

---

## File Summary

| Action | File Path |
|--------|-----------|
| CREATE | `src/utils/ast_parser.py` |
| CREATE | `src/utils/dependency_analyzer.py` |
| CREATE | `src/utils/code_quality.py` |
| CREATE | `tests/test_ast_parser.py` |

---

*End of Part 6*
