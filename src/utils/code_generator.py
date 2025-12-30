"""
Code Generator - LLM-based code generation for system improvements.
Generates safe, validated code modifications.
"""

import json
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from src.utils.llm_router import LLMRouter, TaskContext
from src.utils.ast_parser import ASTParser, CodeStructure
from src.utils.code_patcher import CodePatch, CodePatcher
from src.utils.structured_logging import get_logger
from src.utils.retry import with_retry, LLM_RETRY_CONFIG

logger = get_logger("code_generator")


@dataclass
class GenerationRequest:
    """Request for code generation."""
    goal: str
    file_path: str
    context: str = ""
    constraints: List[str] = None
    existing_code: Optional[str] = None
    function_name: Optional[str] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []


@dataclass
class GenerationResult:
    """Result of code generation."""
    success: bool
    code: Optional[str] = None
    explanation: str = ""
    patch: Optional[CodePatch] = None
    validation_errors: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []


CODE_GENERATION_PROMPT = '''You are an expert Python developer. Generate code based on the following request.

GOAL: {goal}

FILE: {file_path}

EXISTING CODE:
```python
{existing_code}
```

CONTEXT: {context}

CONSTRAINTS:
{constraints}

REQUIREMENTS:
1. Write clean, well-documented Python code
2. Include type hints for all functions
3. Include docstrings for all public functions
4. Follow PEP 8 style guidelines
5. Handle errors appropriately
6. Do NOT use dangerous functions (eval, exec, os.system)

Respond with JSON:
{{
    "code": "your generated code here",
    "explanation": "brief explanation of what the code does",
    "imports_needed": ["list", "of", "imports"]
}}
'''


FUNCTION_IMPROVEMENT_PROMPT = '''You are an expert Python developer. Improve this function.

FUNCTION TO IMPROVE:
```python
{function_code}
```

IMPROVEMENT GOAL: {goal}

CONTEXT: {context}

REQUIREMENTS:
1. Keep the same function signature (name and parameters)
2. Improve based on the goal while maintaining functionality
3. Add proper type hints and docstring
4. Handle edge cases
5. Follow best practices

Respond with JSON:
{{
    "improved_code": "the improved function code",
    "changes_made": ["list of changes"],
    "explanation": "why these changes improve the code"
}}
'''


class CodeGenerator:
    """
    Generates code using LLM with validation and safety checks.
    """
    
    def __init__(self, llm_router: LLMRouter, project_root: str):
        """
        Initialize the generator.
        
        Args:
            llm_router: LLM router for inference
            project_root: Root directory of the project
        """
        self.llm = llm_router
        self.project_root = project_root
        self.parser = ASTParser()
        self.patcher = CodePatcher(project_root)
    
    @with_retry(LLM_RETRY_CONFIG)
    async def generate_code(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate code based on a request.
        
        Args:
            request: Generation request details
            
        Returns:
            Generation result with code and patch
        """
        logger.info(f"Generating code for: {request.goal[:50]}...")
        
        # Read existing code if not provided
        if request.existing_code is None:
            try:
                full_path = f"{self.project_root}/{request.file_path}"
                with open(full_path, 'r') as f:
                    request.existing_code = f.read()
            except FileNotFoundError:
                request.existing_code = "# New file"
        
        # Build prompt
        prompt = CODE_GENERATION_PROMPT.format(
            goal=request.goal,
            file_path=request.file_path,
            existing_code=request.existing_code[:3000],
            context=request.context[:1000],
            constraints='\n'.join(f"- {c}" for c in request.constraints)
        )
        
        # Generate with LLM
        llm_context = TaskContext(
            task_type="code_generation",
            risk_level="medium",
            requires_accuracy=True,
            token_estimate=2000,
            priority="high"
        )
        
        response = await self.llm.complete(prompt, context=llm_context)
        
        # Parse response
        try:
            data = self._parse_json_response(response)
            generated_code = data.get("code", "")
            explanation = data.get("explanation", "")
        except Exception as e:
            return GenerationResult(
                success=False,
                explanation=f"Failed to parse LLM response: {e}"
            )
        
        # Validate the generated code
        is_valid, errors = self._validate_code(generated_code, request.file_path)
        
        if not is_valid:
            return GenerationResult(
                success=False,
                code=generated_code,
                explanation=explanation,
                validation_errors=errors
            )
        
        # Create a patch
        patch = self.patcher.create_patch(
            file_path=request.file_path,
            new_content=generated_code,
            description=request.goal
        )
        
        return GenerationResult(
            success=True,
            code=generated_code,
            explanation=explanation,
            patch=patch
        )
    
    @with_retry(LLM_RETRY_CONFIG)
    async def improve_function(
        self,
        file_path: str,
        function_name: str,
        goal: str,
        context: str = ""
    ) -> GenerationResult:
        """
        Improve a specific function.
        
        Args:
            file_path: Path to the file
            function_name: Name of function to improve
            goal: Improvement goal
            context: Additional context
            
        Returns:
            Generation result
        """
        logger.info(f"Improving function: {function_name}")
        
        # Read and parse the file
        full_path = f"{self.project_root}/{file_path}"
        with open(full_path, 'r') as f:
            source = f.read()
        
        structure = self.parser.parse_source(source, file_path)
        func = structure.get_function(function_name)
        
        if not func:
            return GenerationResult(
                success=False,
                explanation=f"Function '{function_name}' not found"
            )
        
        # Extract function code
        function_code = self.parser.get_line_range(
            source,
            func.lineno,
            func.end_lineno
        )
        
        # Generate improvement
        prompt = FUNCTION_IMPROVEMENT_PROMPT.format(
            function_code=function_code,
            goal=goal,
            context=context
        )
        
        llm_context = TaskContext(
            task_type="code_generation",
            risk_level="medium",
            requires_accuracy=True,
            token_estimate=1500,
            priority="high"
        )
        
        response = await self.llm.complete(prompt, context=llm_context)
        
        try:
            data = self._parse_json_response(response)
            improved_code = data.get("improved_code", "")
            changes = data.get("changes_made", [])
            explanation = data.get("explanation", "")
        except Exception as e:
            return GenerationResult(
                success=False,
                explanation=f"Failed to parse response: {e}"
            )
        
        # Validate improved code
        is_valid, errors = self._validate_function(improved_code)
        
        if not is_valid:
            return GenerationResult(
                success=False,
                code=improved_code,
                explanation=explanation,
                validation_errors=errors
            )
        
        # Create patch for function replacement
        try:
            patch = self.patcher.create_patch_from_function(
                file_path=file_path,
                function_name=function_name,
                new_function_code=improved_code
            )
        except Exception as e:
            return GenerationResult(
                success=False,
                code=improved_code,
                explanation=f"Failed to create patch: {e}"
            )
        
        return GenerationResult(
            success=True,
            code=improved_code,
            explanation=f"{explanation}\n\nChanges: {', '.join(changes)}",
            patch=patch
        )
    
    async def add_method_to_class(
        self,
        file_path: str,
        class_name: str,
        method_signature: str,
        method_description: str
    ) -> GenerationResult:
        """
        Add a new method to an existing class.
        
        Args:
            file_path: Path to the file
            class_name: Name of the class
            method_signature: Signature like "async def process(self, data: str) -> bool"
            method_description: What the method should do
            
        Returns:
            Generation result
        """
        request = GenerationRequest(
            goal=f"Add method to class {class_name}: {method_signature}\nDescription: {method_description}",
            file_path=file_path,
            constraints=[
                f"Add the method inside the {class_name} class",
                "Follow existing class patterns",
                "Include proper type hints and docstring"
            ]
        )
        
        return await self.generate_code(request)
    
    def _validate_code(self, code: str, file_path: str) -> Tuple[bool, List[str]]:
        """Validate generated code."""
        errors = []
        
        if not code.strip():
            errors.append("Empty code generated")
            return False, errors
        
        # Syntax check for Python files
        if file_path.endswith('.py'):
            is_valid, error = self.parser.validate_syntax(code)
            if not is_valid:
                errors.append(f"Syntax error: {error}")
        
        # Check for dangerous patterns
        dangerous = [
            ('eval(', "eval() is dangerous"),
            ('exec(', "exec() is dangerous"),
            ('os.system(', "os.system() is dangerous"),
            ('__import__(', "dynamic import is dangerous"),
        ]
        
        for pattern, message in dangerous:
            if pattern in code:
                errors.append(message)
        
        return len(errors) == 0, errors
    
    def _validate_function(self, code: str) -> Tuple[bool, List[str]]:
        """Validate a function code snippet."""
        errors = []
        
        if not code.strip():
            errors.append("Empty function code")
            return False, errors
        
        # Check it's a valid function definition
        if not re.match(r'^\s*(async\s+)?def\s+\w+', code):
            errors.append("Code does not appear to be a function definition")
        
        # Validate syntax
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
        
        return len(errors) == 0, errors
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to extract JSON from response
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        
        # Clean up common issues
        response = response.strip()
        
        return json.loads(response)
