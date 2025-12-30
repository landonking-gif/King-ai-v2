"""Tests for code generator agent."""

import pytest
from unittest.mock import AsyncMock

from src.agents.code_generator import (
    CodeGeneratorAgent,
    GenerationRequest,
    GenerationMode,
    Language
)
from src.agents.code_reviewer import CodeReviewerAgent, IssueSeverity
from src.agents.code_templates import TemplateLibrary, TemplateType


class TestTemplateLibrary:
    """Tests for template library."""
    
    def test_get_python_class_template(self):
        """Test getting Python class template."""
        template = TemplateLibrary.get(
            Language.PYTHON,
            TemplateType.CLASS
        )
        
        assert template is not None
        assert template.language == Language.PYTHON
        assert "class" in template.template.lower()
    
    def test_get_python_function_template(self):
        """Test getting Python function template."""
        template = TemplateLibrary.get(
            Language.PYTHON,
            TemplateType.FUNCTION
        )
        
        assert template is not None
        assert "def" in template.template
    
    def test_list_templates_by_language(self):
        """Test listing templates by language."""
        templates = TemplateLibrary.list_templates(language=Language.PYTHON)
        
        assert len(templates) > 0
        assert all(t.language == Language.PYTHON for t in templates)
    
    def test_template_render(self):
        """Test template rendering."""
        template = TemplateLibrary.get(
            Language.PYTHON,
            TemplateType.FUNCTION
        )
        
        result = template.render(
            function_name="test_func",
            parameters="x: int",
            return_type="int"
        )
        
        assert "test_func" in result
        assert "x: int" in result


class TestCodeGeneratorAgent:
    """Tests for code generator."""
    
    @pytest.fixture
    def mock_llm_agent(self):
        """Create a mocked code generator agent."""
        agent = CodeGeneratorAgent()
        agent._ask_llm = AsyncMock(return_value="""
```python
def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b
```
""")
        return agent
    
    def test_extract_code_from_response(self):
        """Test code extraction from LLM response."""
        agent = CodeGeneratorAgent()
        response = """Here's the code:
```python
def hello():
    return "Hello"
```
That's all."""
        
        code = agent._extract_code(response, Language.PYTHON)
        
        assert "def hello" in code
        assert "return" in code
    
    @pytest.mark.asyncio
    async def test_generate_function(self, mock_llm_agent):
        """Test function generation."""
        result = await mock_llm_agent.create_function(
            name="add",
            description="adds two numbers",
            parameters=[
                {"name": "a", "type": "int"},
                {"name": "b", "type": "int"}
            ],
            return_type="int"
        )
        
        assert result.code
        assert result.language == Language.PYTHON
    
    @pytest.mark.asyncio
    async def test_generate_with_mode(self, mock_llm_agent):
        """Test generation with different modes."""
        request = GenerationRequest(
            description="A simple calculator",
            language=Language.PYTHON,
            mode=GenerationMode.CREATE
        )
        
        result = await mock_llm_agent.generate(request)
        
        assert result.code
    
    @pytest.mark.asyncio
    async def test_execute_task_new_format(self, mock_llm_agent):
        """Test execute with new task format."""
        task = {
            "input": {
                "description": "Create a hello world function",
                "language": "python",
                "mode": "create"
            }
        }
        
        result = await mock_llm_agent.execute(task)
        
        assert result["success"] is True
        assert "output" in result
    
    @pytest.mark.asyncio
    async def test_execute_task_legacy_format(self, mock_llm_agent):
        """Test execute with legacy task format."""
        task = {
            "description": "Create a hello world function",
            "language": "python",
            "mode": "create"
        }
        
        result = await mock_llm_agent.execute(task)
        
        assert result["success"] is True
        assert "output" in result


class TestCodeReviewerAgent:
    """Tests for code reviewer."""
    
    @pytest.fixture
    def mock_reviewer_agent(self):
        """Create a mocked code reviewer agent."""
        agent = CodeReviewerAgent()
        agent._ask_llm = AsyncMock(return_value="""
ISSUE:
Line: 5
Category: security
Severity: critical
Message: Hardcoded password detected
Suggestion: Use environment variables
---

SUMMARY: Code has security issues
SCORE: 40
RECOMMENDATIONS:
- Remove hardcoded credentials
- Add input validation
""")
        return agent
    
    def test_pattern_check_eval(self):
        """Test detection of eval usage."""
        reviewer = CodeReviewerAgent()
        code = """
result = eval(user_input)
"""
        issues = reviewer._check_patterns(code, Language.PYTHON)
        
        assert len(issues) > 0
        assert issues[0].severity == IssueSeverity.CRITICAL
    
    def test_pattern_check_password(self):
        """Test detection of hardcoded password."""
        reviewer = CodeReviewerAgent()
        code = """
password = "secret123"
"""
        issues = reviewer._check_patterns(code, Language.PYTHON)
        
        assert len(issues) > 0
        assert any("password" in i.message.lower() for i in issues)
    
    @pytest.mark.asyncio
    async def test_full_review(self, mock_reviewer_agent):
        """Test full code review."""
        code = """
def process(data):
    result = eval(data)
    password = "test123"
    return result
"""
        review = await mock_reviewer_agent.review(code, Language.PYTHON)
        
        assert review.issues
        assert review.overall_score <= 100
        assert review.summary
    
    @pytest.mark.asyncio
    async def test_execute_task(self, mock_reviewer_agent):
        """Test execute with task."""
        task = {
            "input": {
                "code": "def test(): pass",
                "language": "python"
            }
        }
        
        result = await mock_reviewer_agent.execute(task)
        
        assert result["success"] is True
        assert "output" in result
