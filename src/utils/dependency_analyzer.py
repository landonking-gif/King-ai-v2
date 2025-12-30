"""
Dependency Analyzer - Analyzes code dependencies and relationships.
Builds dependency graphs for impact analysis.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import os

from src.utils.ast_parser import ASTParser, CodeStructure, ImportInfo
import structlog

logger = structlog.get_logger("dependency_analyzer")


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
            warnings.append("CRITICAL: Changes to master AI brain require extra review")
        
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
