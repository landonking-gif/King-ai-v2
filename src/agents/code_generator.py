"""
Code Generator Agent - Intelligent code generation and modification.
Uses LLM for context-aware code generation with template support.
"""

import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from src.agents.base import SubAgent
from src.agents.code_templates import (
    TemplateLibrary,
    CodeTemplate,
    Language,
    TemplateType
)
from src.utils.code_analyzer import CodeAnalyzer
from src.utils.metrics import TASKS_EXECUTED


class GenerationMode(str, Enum):
    """Code generation modes."""
    CREATE = "create"  # Create new code
    MODIFY = "modify"  # Modify existing code
    OPTIMIZE = "optimize"  # Optimize for performance
    REFACTOR = "refactor"  # Refactor for clarity
    DOCUMENT = "document"  # Add documentation
    TEST = "test"  # Generate tests


@dataclass
class GenerationRequest:
    """Request for code generation."""
    description: str
    language: Language
    mode: GenerationMode = GenerationMode.CREATE
    existing_code: Optional[str] = None
    template_type: Optional[TemplateType] = None
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)


@dataclass
class GeneratedCode:
    """Generated code result."""
    code: str
    language: Language
    file_path: Optional[str] = None
    explanation: str = ""
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "language": self.language.value,
            "file_path": self.file_path,
            "explanation": self.explanation,
            "imports": self.imports,
            "dependencies": self.dependencies,
            "warnings": self.warnings
        }


class CodeGeneratorAgent(SubAgent):
    """
    Agent for intelligent code generation.
    Combines LLM capabilities with structured templates.
    """
    
    name = "code_generator"
    description = "Generates and modifies source code for business applications."
    
    # Prompts for different generation modes
    PROMPTS = {
        GenerationMode.CREATE: """Generate {language} code based on the following requirements:

Requirements:
{description}

{context}

Constraints:
{constraints}

Generate clean, well-documented code following best practices.
Include type hints and docstrings.
Output ONLY the code without explanations.

```{language}
""",
        GenerationMode.MODIFY: """Modify the following {language} code according to the requirements:

Current Code:
```{language}
{existing_code}
```

Modification Required:
{description}

{context}

Constraints:
{constraints}

Output the complete modified code.

```{language}
""",
        GenerationMode.OPTIMIZE: """Optimize the following {language} code for better performance:

Current Code:
```{language}
{existing_code}
```

Optimization Goals:
{description}

Focus on:
- Time complexity
- Memory usage
- Readability
- Best practices

Output the optimized code with comments explaining changes.

```{language}
""",
        GenerationMode.REFACTOR: """Refactor the following {language} code for better maintainability:

Current Code:
```{language}
{existing_code}
```

Refactoring Goals:
{description}

Focus on:
- Code clarity
- Single responsibility
- DRY principles
- Meaningful naming

Output the refactored code.

```{language}
""",
        GenerationMode.DOCUMENT: """Add comprehensive documentation to the following {language} code:

Code:
```{language}
{existing_code}
```

Add:
- Module docstring
- Class/function docstrings
- Inline comments for complex logic
- Type hints if missing

Output the fully documented code.

```{language}
""",
        GenerationMode.TEST: """Generate comprehensive tests for the following {language} code:

Code to Test:
```{language}
{existing_code}
```

Test Requirements:
{description}

Generate:
- Unit tests for all functions/methods
- Edge case tests
- Error handling tests
- Use pytest conventions

```{language}
"""
    }
    
    def __init__(self):
        """Initialize code generator agent."""
        super().__init__()
        self.analyzer = CodeAnalyzer()
        self.template_library = TemplateLibrary
    
    async def execute(self, task: dict) -> dict:
        """Execute a code generation task."""
        try:
            # Support both new and legacy task formats
            if "input" in task:
                # New format
                input_data = task.get("input", {})
                request = GenerationRequest(
                    description=input_data.get("description", task.get("description", "")),
                    language=Language(input_data.get("language", "python")),
                    mode=GenerationMode(input_data.get("mode", "create")),
                    existing_code=input_data.get("existing_code"),
                    template_type=TemplateType(input_data["template_type"]) if input_data.get("template_type") else None,
                    context=input_data.get("context", {}),
                    constraints=input_data.get("constraints", [])
                )
            else:
                # Legacy format
                request = GenerationRequest(
                    description=task.get("description", "Generate code"),
                    language=Language(task.get("language", "python")),
                    mode=GenerationMode(task.get("mode", "create")),
                    existing_code=task.get("existing_code"),
                    template_type=TemplateType(task["template_type"]) if task.get("template_type") else None,
                    context=task.get("context", {}),
                    constraints=task.get("constraints", [])
                )
            
            result = await self.generate(request)
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            
            return {
                "success": True,
                "output": result.to_dict(),
                "metadata": {"type": "code_gen", "language": result.language.value}
            }
            
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {
                "success": False,
                "error": str(e),
                "metadata": {"type": "code_gen"}
            }
    
    async def generate(self, request: GenerationRequest) -> GeneratedCode:
        """
        Generate code based on request.
        
        Args:
            request: Generation request
            
        Returns:
            Generated code result
        """
        # Check for applicable template
        if request.template_type and request.mode == GenerationMode.CREATE:
            template = self.template_library.get(
                request.language,
                request.template_type
            )
            if template:
                return await self._generate_from_template(request, template)
        
        # Generate using LLM
        return await self._generate_with_llm(request)
    
    async def _generate_from_template(
        self,
        request: GenerationRequest,
        template: CodeTemplate
    ) -> GeneratedCode:
        """Generate code from template with LLM filling in details."""
        # Use LLM to generate template variables
        prompt = f"""Fill in the template variables for a {template.template_type.value}.

Description: {request.description}

Template variables needed: {', '.join(template.variables)}

For each variable, provide an appropriate value based on the description.
Format as:
variable_name: value

Variables:"""

        response = await self._ask_llm(prompt)
        
        # Parse variables
        variables = self._parse_template_variables(response, template.variables)
        
        # Render template
        code = template.render(**variables)
        
        return GeneratedCode(
            code=code,
            language=request.language,
            explanation=f"Generated from {template.name} template",
            dependencies=template.dependencies
        )
    
    def _parse_template_variables(
        self,
        response: str,
        expected_vars: List[str]
    ) -> Dict[str, str]:
        """Parse template variables from LLM response."""
        variables = {}
        
        for line in response.split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip().lower().replace(' ', '_')
                value = parts[1].strip()
                
                # Match to expected variables
                for var in expected_vars:
                    if var.lower() in key or key in var.lower():
                        variables[var] = value
                        break
        
        # Fill in missing with placeholders
        for var in expected_vars:
            if var not in variables:
                variables[var] = f"# TODO: {var}"
        
        return variables
    
    async def _generate_with_llm(
        self,
        request: GenerationRequest
    ) -> GeneratedCode:
        """Generate code using LLM."""
        # Build prompt
        prompt_template = self.PROMPTS.get(request.mode, self.PROMPTS[GenerationMode.CREATE])
        
        context_str = ""
        if request.context:
            context_str = "Context:\n" + "\n".join(
                f"- {k}: {v}" for k, v in request.context.items()
            )
        
        constraints_str = "\n".join(f"- {c}" for c in request.constraints) if request.constraints else "None"
        
        prompt = prompt_template.format(
            language=request.language.value,
            description=request.description,
            existing_code=request.existing_code or "",
            context=context_str,
            constraints=constraints_str
        )
        
        # Generate
        response = await self._ask_llm(prompt)
        
        # Extract code from response
        code = self._extract_code(response, request.language)
        
        # Analyze generated code
        analysis = self._analyze_generated_code(code, request.language)
        
        return GeneratedCode(
            code=code,
            language=request.language,
            explanation=self._extract_explanation(response),
            imports=analysis.get("imports", []),
            dependencies=analysis.get("dependencies", []),
            warnings=analysis.get("warnings", [])
        )
    
    def _extract_code(self, response: str, language: Language) -> str:
        """Extract code block from LLM response."""
        # Try to find code block
        patterns = [
            rf"```{language.value}\n(.*?)```",
            r"```\n(.*?)```",
            rf"```{language.value}(.*?)```",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no code block, return cleaned response
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code = not in_code
                continue
            if in_code or (not line.strip().startswith('#') and '=' in line or 'def ' in line or 'class ' in line):
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else response
    
    def _extract_explanation(self, response: str) -> str:
        """Extract explanation from response."""
        # Look for text before or after code block
        parts = response.split('```')
        
        explanations = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Not code
                text = part.strip()
                if text and len(text) > 20:
                    explanations.append(text)
        
        return ' '.join(explanations)[:500]
    
    def _analyze_generated_code(
        self,
        code: str,
        language: Language
    ) -> Dict[str, Any]:
        """Analyze generated code for issues."""
        result = {
            "imports": [],
            "dependencies": [],
            "warnings": []
        }
        
        if language == Language.PYTHON:
            # Extract imports
            import_lines = re.findall(r'^(?:from|import)\s+(\S+)', code, re.MULTILINE)
            result["imports"] = list(set(import_lines))
            
            # Check for potential issues
            if 'eval(' in code or 'exec(' in code:
                result["warnings"].append("Code contains eval/exec - security risk")
            
            if 'import *' in code:
                result["warnings"].append("Wildcard imports detected - consider explicit imports")
            
            # Identify external dependencies
            stdlib = {'os', 'sys', 'json', 'datetime', 'typing', 're', 'asyncio', 'pathlib'}
            for imp in result["imports"]:
                base_module = imp.split('.')[0]
                if base_module not in stdlib:
                    result["dependencies"].append(base_module)
        
        return result
    
    async def create_function(
        self,
        name: str,
        description: str,
        parameters: List[Dict[str, str]],
        return_type: str,
        language: Language = Language.PYTHON
    ) -> GeneratedCode:
        """
        Generate a function with specified signature.
        
        Args:
            name: Function name
            description: What the function does
            parameters: List of {"name": str, "type": str} dicts
            return_type: Return type
            language: Programming language
            
        Returns:
            Generated function code
        """
        params_str = ", ".join(
            f"{p['name']}: {p.get('type', 'Any')}" for p in parameters
        )
        
        request = GenerationRequest(
            description=f"""Create a function named '{name}' that {description}.

Signature: def {name}({params_str}) -> {return_type}

Parameters:
{chr(10).join(f"- {p['name']}: {p.get('description', p.get('type', 'Any'))}" for p in parameters)}

Returns: {return_type}""",
            language=language,
            mode=GenerationMode.CREATE
        )
        
        return await self.generate(request)
    
    async def create_class(
        self,
        name: str,
        description: str,
        attributes: List[Dict[str, str]],
        methods: List[str],
        language: Language = Language.PYTHON
    ) -> GeneratedCode:
        """
        Generate a class with specified structure.
        
        Args:
            name: Class name
            description: What the class does
            attributes: List of {"name": str, "type": str} dicts
            methods: List of method descriptions
            language: Programming language
            
        Returns:
            Generated class code
        """
        attrs_str = "\n".join(
            f"- {a['name']}: {a.get('type', 'Any')}" for a in attributes
        )
        methods_str = "\n".join(f"- {m}" for m in methods)
        
        request = GenerationRequest(
            description=f"""Create a class named '{name}' that {description}.

Attributes:
{attrs_str}

Methods to implement:
{methods_str}""",
            language=language,
            mode=GenerationMode.CREATE,
            template_type=TemplateType.CLASS
        )
        
        return await self.generate(request)
    
    async def add_documentation(
        self,
        code: str,
        language: Language = Language.PYTHON
    ) -> GeneratedCode:
        """Add documentation to existing code."""
        request = GenerationRequest(
            description="Add comprehensive documentation",
            language=language,
            mode=GenerationMode.DOCUMENT,
            existing_code=code
        )
        
        return await self.generate(request)
    
    async def generate_tests(
        self,
        code: str,
        language: Language = Language.PYTHON
    ) -> GeneratedCode:
        """Generate tests for code."""
        request = GenerationRequest(
            description="Generate comprehensive test coverage",
            language=language,
            mode=GenerationMode.TEST,
            existing_code=code
        )
        
        return await self.generate(request)
    
    async def optimize_code(
        self,
        code: str,
        optimization_goals: List[str],
        language: Language = Language.PYTHON
    ) -> GeneratedCode:
        """Optimize existing code."""
        request = GenerationRequest(
            description=f"Optimize for: {', '.join(optimization_goals)}",
            language=language,
            mode=GenerationMode.OPTIMIZE,
            existing_code=code,
            constraints=optimization_goals
        )
        
        return await self.generate(request)
