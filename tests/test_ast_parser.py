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
