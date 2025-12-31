"""
Test Generator - Generates tests for evolution proposals.
Uses LLM to create test cases for new or modified code.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.utils.ollama_client import OllamaClient
from src.utils.code_analyzer import CodeAnalyzer
from src.utils.ast_parser import FunctionInfo, ClassInfo
from src.utils.logging import get_logger

logger = get_logger("test_generator")


class TestType(str, Enum):
    """Types of tests to generate."""
    UNIT = "unit"
    INTEGRATION = "integration"
    EDGE_CASE = "edge_case"
    ERROR_HANDLING = "error_handling"


@dataclass
class GeneratedTest:
    """A generated test case."""
    name: str
    test_type: TestType
    code: str
    description: str
    target_function: str
    confidence: float


class TestGenerator:
    """
    Generates test cases for code changes.
    Uses LLM for intelligent test generation.
    """
    
    # Configuration constants
    MAX_CODE_LENGTH = 3000  # Maximum characters of source code to send to LLM
    MAX_CONTEXT_FUNCTIONS = 5  # Maximum number of functions to include in context
    
    TEST_GENERATION_PROMPT = """Generate pytest test cases for the following Python code.

Code to test:
```python
{code}
```

Context about the code:
{context}

Requirements:
1. Generate comprehensive test cases covering:
   - Normal operation (happy path)
   - Edge cases (empty inputs, boundaries)
   - Error conditions (invalid inputs, exceptions)
2. Use pytest fixtures where appropriate
3. Include docstrings explaining each test
4. Use meaningful test names (test_<function>_<scenario>)
5. Use assert statements with clear messages

Generate {num_tests} test cases.

Output the tests as valid Python code:
```python
"""
    
    def __init__(self, llm_client: OllamaClient, code_analyzer: CodeAnalyzer):
        """
        Initialize test generator.
        
        Args:
            llm_client: LLM client for generation
            code_analyzer: Code analyzer for understanding code
        """
        self.llm = llm_client
        self.analyzer = code_analyzer
    
    async def generate_tests(
        self,
        source_code: str,
        file_path: str,
        num_tests: int = 5,
        test_types: List[TestType] = None
    ) -> List[GeneratedTest]:
        """
        Generate tests for source code.
        
        Args:
            source_code: Python source code to test
            file_path: Path of the source file
            num_tests: Number of tests to generate
            test_types: Types of tests to generate
            
        Returns:
            List of generated tests
        """
        if test_types is None:
            test_types = list(TestType)
        
        # Analyze the code
        analysis = self.analyzer.analyze_code(source_code)
        
        # Build context
        context = self._build_context(analysis)
        
        # Generate tests via LLM
        prompt = self.TEST_GENERATION_PROMPT.format(
            code=source_code[:self.MAX_CODE_LENGTH],  # Limit code size
            context=context,
            num_tests=num_tests
        )
        
        response = await self.llm.generate(prompt)
        
        # Parse generated tests
        tests = self._parse_generated_tests(response, analysis)
        
        logger.info(
            f"Generated {len(tests)} tests",
            file=file_path,
            functions=len(analysis.get("functions", []))
        )
        
        return tests
    
    def _build_context(self, analysis: Dict[str, Any]) -> str:
        """Build context string from code analysis."""
        parts = []
        
        functions = analysis.get("functions", [])
        if functions:
            parts.append(f"Functions: {', '.join(f.name for f in functions)}")
            for func in functions[:self.MAX_CONTEXT_FUNCTIONS]:
                params = ", ".join(f"{p.name}: {p.type_hint or 'Any'}" for p in func.parameters)
                parts.append(f"  - {func.name}({params}) -> {func.return_type or 'None'}")
        
        classes = analysis.get("classes", [])
        if classes:
            parts.append(f"Classes: {', '.join(c.name for c in classes)}")
        
        return "\n".join(parts)
    
    def _parse_generated_tests(
        self,
        llm_response: str,
        analysis: Dict[str, Any]
    ) -> List[GeneratedTest]:
        """Parse LLM response into test objects."""
        tests = []
        
        # Extract code block
        code_start = llm_response.find("```python")
        code_end = llm_response.rfind("```")
        
        if code_start != -1 and code_end > code_start:
            test_code = llm_response[code_start + 9:code_end].strip()
        else:
            test_code = llm_response
        
        # Parse individual test functions
        import ast
        try:
            tree = ast.parse(test_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    # Extract test info
                    test_name = node.name
                    docstring = ast.get_docstring(node) or ""
                    
                    # Determine test type from name
                    test_type = TestType.UNIT
                    if "error" in test_name.lower() or "exception" in test_name.lower():
                        test_type = TestType.ERROR_HANDLING
                    elif "edge" in test_name.lower() or "empty" in test_name.lower():
                        test_type = TestType.EDGE_CASE
                    
                    # Get the function code
                    func_code = ast.unparse(node)
                    
                    # Try to identify target function
                    target = self._identify_target_function(test_name, analysis)
                    
                    tests.append(GeneratedTest(
                        name=test_name,
                        test_type=test_type,
                        code=func_code,
                        description=docstring,
                        target_function=target,
                        confidence=0.8
                    ))
                    
        except SyntaxError as e:
            logger.warning(f"Failed to parse generated tests: {e}")
        
        return tests
    
    def _identify_target_function(
        self,
        test_name: str,
        analysis: Dict[str, Any]
    ) -> str:
        """Identify which function a test is targeting."""
        functions = analysis.get("functions", [])
        
        # Remove test_ prefix and common suffixes
        name_parts = test_name.replace("test_", "").split("_")
        
        for func in functions:
            if func.name.lower() in test_name.lower():
                return func.name
        
        return "unknown"
    
    async def generate_tests_for_function(
        self,
        function: FunctionInfo,
        source_context: str = ""
    ) -> List[GeneratedTest]:
        """Generate tests specifically for one function."""
        # Build a proper code representation
        params = ", ".join(f"{p.name}: {p.type_hint or 'Any'}" for p in function.parameters)
        func_signature = f"def {function.name}({params}) -> {function.return_type or 'None'}"
        
        prompt = f"""Generate pytest test cases for this function:

```python
{func_signature}
    \"\"\"
    {function.docstring or 'No documentation'}
    \"\"\"
    pass
```

{f'Context: {source_context}' if source_context else ''}

Generate 3-5 comprehensive tests covering:
1. Normal operation with typical inputs
2. Edge cases (empty, None, boundary values)
3. Error handling (invalid inputs)

Output valid pytest code:
```python
"""
        
        response = await self.llm.generate(prompt)
        
        # Simplified parsing for single function
        tests = self._parse_generated_tests(
            response,
            {"functions": [function]}
        )
        
        return tests
    
    def generate_test_file_skeleton(
        self,
        source_file: str,
        functions: List[FunctionInfo]
    ) -> str:
        """Generate a test file skeleton."""
        module_name = source_file.replace("/", ".").replace(".py", "")
        
        imports = [
            "import pytest",
            f"from {module_name} import *",
            "",
            "",
        ]
        
        test_stubs = []
        for func in functions:
            test_stubs.append(f'''
def test_{func.name}_basic():
    """Test {func.name} with basic inputs."""
    # TODO: Implement test
    pass


def test_{func.name}_edge_cases():
    """Test {func.name} with edge cases."""
    # TODO: Implement test
    pass
''')
        
        return "\n".join(imports) + "\n".join(test_stubs)
