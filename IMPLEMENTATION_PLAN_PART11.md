# King AI v2 - Implementation Plan Part 11
## Sub-Agent: Code Generator

**Target Timeline:** Week 8
**Objective:** Implement an intelligent code generation agent capable of creating, modifying, and optimizing code across multiple languages.

---

## Overview of All Parts

| Part | Title | Status |
|------|-------|--------|
| 1 | Infrastructure Layer & Core System Hardening | ✅ Complete |
| 2 | Master AI Brain - Core Enhancements | ✅ Complete |
| 3 | Master AI Brain - Context & Memory System | ✅ Complete |
| 4 | Master AI Brain - Planning & ReAct Implementation | ✅ Complete |
| 5 | Evolution Engine - Core Models & Proposal System | ✅ Complete |
| 6 | Evolution Engine - Code Analysis & AST Tools | ✅ Complete |
| 7 | Evolution Engine - Code Patching & Generation | ✅ Complete |
| 8 | Evolution Engine - Git Integration & Rollback | ✅ Complete |
| 9 | Evolution Engine - Sandbox Testing | ✅ Complete |
| 10 | Sub-Agent: Research (Web/API) | ✅ Complete |
| **11** | **Sub-Agent: Code Generator** | 🔄 Current |
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

## Part 11 Scope

This part focuses on:
1. Multi-language code generation
2. Code templates and scaffolding
3. Code optimization and refactoring
4. Documentation generation
5. Test generation for code
6. Code review and suggestions

---

## Task 11.1: Create Code Templates System

**File:** `src/agents/code_templates.py` (CREATE NEW FILE)

```python
"""
Code Templates System - Reusable code templates and scaffolding.
Provides structured templates for common code patterns.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from string import Template
import re


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    YAML = "yaml"
    JSON = "json"


class TemplateType(str, Enum):
    """Types of code templates."""
    CLASS = "class"
    FUNCTION = "function"
    API_ENDPOINT = "api_endpoint"
    DATABASE_MODEL = "database_model"
    TEST_FILE = "test_file"
    REACT_COMPONENT = "react_component"
    DOCKERFILE = "dockerfile"
    CONFIG_FILE = "config_file"


@dataclass
class CodeTemplate:
    """A code template definition."""
    name: str
    language: Language
    template_type: TemplateType
    template: str
    description: str
    variables: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def render(self, **kwargs) -> str:
        """Render template with variables."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"${{{key}}}", str(value))
            result = result.replace(f"${key}", str(value))
        return result


class TemplateLibrary:
    """
    Library of code templates for various languages and patterns.
    """
    
    TEMPLATES: Dict[str, CodeTemplate] = {}
    
    @classmethod
    def register(cls, template: CodeTemplate):
        """Register a template."""
        key = f"{template.language.value}_{template.template_type.value}_{template.name}"
        cls.TEMPLATES[key] = template
    
    @classmethod
    def get(
        cls,
        language: Language,
        template_type: TemplateType,
        name: str = "default"
    ) -> Optional[CodeTemplate]:
        """Get a template by language, type, and name."""
        key = f"{language.value}_{template_type.value}_{name}"
        return cls.TEMPLATES.get(key)
    
    @classmethod
    def list_templates(
        cls,
        language: Language = None,
        template_type: TemplateType = None
    ) -> List[CodeTemplate]:
        """List available templates."""
        templates = list(cls.TEMPLATES.values())
        
        if language:
            templates = [t for t in templates if t.language == language]
        if template_type:
            templates = [t for t in templates if t.template_type == template_type]
        
        return templates


# Register Python Templates
TemplateLibrary.register(CodeTemplate(
    name="default",
    language=Language.PYTHON,
    template_type=TemplateType.CLASS,
    description="Standard Python class with docstring",
    template='''"""
${description}
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class ${class_name}:
    """
    ${class_description}
    
    Attributes:
        ${attributes_doc}
    """
    
    ${attributes}
    
    def __init__(self, ${init_params}):
        """Initialize ${class_name}."""
        ${init_body}
    
    ${methods}
''',
    variables=["class_name", "description", "class_description", "attributes", "methods"]
))

TemplateLibrary.register(CodeTemplate(
    name="default",
    language=Language.PYTHON,
    template_type=TemplateType.FUNCTION,
    description="Standard Python function with type hints",
    template='''def ${function_name}(${parameters}) -> ${return_type}:
    """
    ${docstring}
    
    Args:
        ${args_doc}
    
    Returns:
        ${returns_doc}
    
    Raises:
        ${raises_doc}
    """
    ${body}
''',
    variables=["function_name", "parameters", "return_type", "docstring", "body"]
))

TemplateLibrary.register(CodeTemplate(
    name="default",
    language=Language.PYTHON,
    template_type=TemplateType.API_ENDPOINT,
    description="FastAPI endpoint with validation",
    template='''from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/${route_prefix}", tags=["${tag}"])


class ${request_model}(BaseModel):
    """Request model for ${endpoint_name}."""
    ${request_fields}


class ${response_model}(BaseModel):
    """Response model for ${endpoint_name}."""
    ${response_fields}


@router.${method}("/${path}")
async def ${endpoint_name}(
    request: ${request_model}
) -> ${response_model}:
    """
    ${endpoint_description}
    """
    try:
        ${implementation}
        return ${response_model}(${response_params})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
''',
    variables=["route_prefix", "tag", "request_model", "response_model", "endpoint_name", "method", "path"],
    dependencies=["fastapi", "pydantic"]
))

TemplateLibrary.register(CodeTemplate(
    name="default",
    language=Language.PYTHON,
    template_type=TemplateType.DATABASE_MODEL,
    description="SQLAlchemy async model",
    template='''"""
${model_name} database model.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from src.database.connection import Base


class ${model_name}(Base):
    """
    ${model_description}
    """
    
    __tablename__ = "${table_name}"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ${columns}
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    ${relationships}
    
    def __repr__(self):
        return f"<${model_name}(id={self.id})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            "id": str(self.id),
            ${to_dict_fields}
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
''',
    variables=["model_name", "model_description", "table_name", "columns", "relationships"],
    dependencies=["sqlalchemy"]
))

TemplateLibrary.register(CodeTemplate(
    name="default",
    language=Language.PYTHON,
    template_type=TemplateType.TEST_FILE,
    description="Pytest test file with fixtures",
    template='''"""
Tests for ${module_name}.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

${imports}


class Test${class_name}:
    """Tests for ${class_name}."""
    
    @pytest.fixture
    def ${fixture_name}(self):
        """Create test fixture."""
        ${fixture_body}
    
    ${test_methods}


class Test${class_name}Integration:
    """Integration tests for ${class_name}."""
    
    @pytest.mark.asyncio
    async def test_${integration_test_name}(self):
        """Test ${integration_description}."""
        ${integration_body}
''',
    variables=["module_name", "class_name", "fixture_name", "test_methods"],
    dependencies=["pytest", "pytest-asyncio"]
))

# Register JavaScript/TypeScript Templates
TemplateLibrary.register(CodeTemplate(
    name="default",
    language=Language.TYPESCRIPT,
    template_type=TemplateType.REACT_COMPONENT,
    description="React functional component with TypeScript",
    template='''import React, { useState, useEffect } from 'react';
${imports}

interface ${component_name}Props {
    ${props_interface}
}

export const ${component_name}: React.FC<${component_name}Props> = ({
    ${destructured_props}
}) => {
    ${state_declarations}
    
    useEffect(() => {
        ${effect_body}
    }, [${effect_deps}]);
    
    ${handlers}
    
    return (
        <div className="${css_class}">
            ${jsx_content}
        </div>
    );
};

export default ${component_name};
''',
    variables=["component_name", "props_interface", "jsx_content"]
))

TemplateLibrary.register(CodeTemplate(
    name="default",
    language=Language.PYTHON,
    template_type=TemplateType.DOCKERFILE,
    description="Multi-stage Python Dockerfile",
    template='''# Build stage
FROM python:${python_version}-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Production stage
FROM python:${python_version}-slim

WORKDIR /app

# Create non-root user
RUN useradd --create-home appuser

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
${env_vars}

EXPOSE ${port}

CMD ["${command}"]
''',
    variables=["python_version", "port", "command", "env_vars"]
))
```

---

## Task 11.2: Create Code Generator Agent

**File:** `src/agents/code_generator.py` (REPLACE EXISTING FILE)

```python
"""
Code Generator Agent - Intelligent code generation and modification.
Uses LLM for context-aware code generation with template support.
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.agents.base import BaseAgent, AgentCapability, AgentResult
from src.agents.code_templates import (
    TemplateLibrary,
    CodeTemplate,
    Language,
    TemplateType
)
from src.utils.ollama_client import OllamaClient
from src.utils.code_analyzer import CodeAnalyzer
from src.utils.structured_logging import get_logger

logger = get_logger("code_generator")


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


class CodeGeneratorAgent(BaseAgent):
    """
    Agent for intelligent code generation.
    Combines LLM capabilities with structured templates.
    """
    
    CAPABILITIES = [
        AgentCapability.CODE_GENERATION,
        AgentCapability.CODE_ANALYSIS,
        AgentCapability.DOCUMENTATION
    ]
    
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
    
    def __init__(self, llm_client: OllamaClient):
        """
        Initialize code generator agent.
        
        Args:
            llm_client: LLM client for generation
        """
        super().__init__("code_generator", llm_client)
        self.analyzer = CodeAnalyzer()
        self.template_library = TemplateLibrary
    
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Execute a code generation task."""
        try:
            request = GenerationRequest(
                description=task.get("description", ""),
                language=Language(task.get("language", "python")),
                mode=GenerationMode(task.get("mode", "create")),
                existing_code=task.get("existing_code"),
                template_type=TemplateType(task["template_type"]) if task.get("template_type") else None,
                context=task.get("context", {}),
                constraints=task.get("constraints", [])
            )
            
            result = await self.generate(request)
            
            return AgentResult(
                success=True,
                data=result.to_dict(),
                message=f"Generated {result.language.value} code"
            )
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                message="Code generation failed"
            )
    
    async def generate(self, request: GenerationRequest) -> GeneratedCode:
        """
        Generate code based on request.
        
        Args:
            request: Generation request
            
        Returns:
            Generated code result
        """
        logger.info(
            f"Generating code",
            language=request.language.value,
            mode=request.mode.value
        )
        
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

        response = await self.llm.generate(prompt)
        
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
        response = await self.llm.generate(prompt)
        
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
```

---

## Task 11.3: Create Code Review Agent

**File:** `src/agents/code_reviewer.py` (CREATE NEW FILE)

```python
"""
Code Reviewer Agent - Automated code review and suggestions.
Analyzes code for quality, security, and best practices.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from src.agents.base import BaseAgent, AgentCapability, AgentResult
from src.utils.ollama_client import OllamaClient
from src.utils.code_analyzer import CodeAnalyzer
from src.agents.code_templates import Language
from src.utils.structured_logging import get_logger

logger = get_logger("code_reviewer")


class IssueSeverity(str, Enum):
    """Severity levels for code issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(str, Enum):
    """Categories of code issues."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    CORRECTNESS = "correctness"
    STYLE = "style"
    DOCUMENTATION = "documentation"


@dataclass
class CodeIssue:
    """A code issue found during review."""
    line: Optional[int]
    category: IssueCategory
    severity: IssueSeverity
    message: str
    suggestion: str
    code_snippet: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "line": self.line,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "code_snippet": self.code_snippet
        }


@dataclass
class CodeReview:
    """Complete code review result."""
    file_path: Optional[str]
    language: Language
    issues: List[CodeIssue]
    overall_score: float  # 0-100
    summary: str
    recommendations: List[str]
    
    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)
    
    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.HIGH)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "language": self.language.value,
            "issues": [i.to_dict() for i in self.issues],
            "issue_count": len(self.issues),
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "recommendations": self.recommendations
        }


class CodeReviewerAgent(BaseAgent):
    """
    Agent for automated code review.
    Identifies issues and provides actionable suggestions.
    """
    
    CAPABILITIES = [
        AgentCapability.CODE_ANALYSIS,
        AgentCapability.SECURITY_ANALYSIS
    ]
    
    REVIEW_PROMPT = """Review the following {language} code for issues and improvements.

Code:
```{language}
{code}
```

Analyze for:
1. Security vulnerabilities (injection, exposure, etc.)
2. Performance issues (complexity, memory leaks)
3. Maintainability (complexity, coupling)
4. Correctness (bugs, edge cases)
5. Style (naming, formatting)
6. Documentation (missing docs, unclear code)

For each issue found, provide:
- Line number (if applicable)
- Category (security/performance/maintainability/correctness/style/documentation)
- Severity (critical/high/medium/low/info)
- Description of the issue
- Suggested fix

Format each issue as:
ISSUE:
Line: [number or "N/A"]
Category: [category]
Severity: [severity]
Message: [description]
Suggestion: [how to fix]
---

At the end, provide:
SUMMARY: [overall assessment]
SCORE: [0-100]
RECOMMENDATIONS:
- [recommendation 1]
- [recommendation 2]
"""
    
    # Common security patterns to check
    SECURITY_PATTERNS = {
        Language.PYTHON: [
            (r'eval\(', "Use of eval() - potential code injection", IssueSeverity.CRITICAL),
            (r'exec\(', "Use of exec() - potential code injection", IssueSeverity.CRITICAL),
            (r'pickle\.loads?\(', "Pickle deserialization - potential security risk", IssueSeverity.HIGH),
            (r'subprocess\.(call|run|Popen).*shell\s*=\s*True', "Shell=True in subprocess - command injection risk", IssueSeverity.HIGH),
            (r'password\s*=\s*["\']', "Hardcoded password detected", IssueSeverity.CRITICAL),
            (r'api_key\s*=\s*["\']', "Hardcoded API key detected", IssueSeverity.CRITICAL),
        ],
        Language.JAVASCRIPT: [
            (r'eval\(', "Use of eval() - potential code injection", IssueSeverity.CRITICAL),
            (r'innerHTML\s*=', "innerHTML assignment - potential XSS", IssueSeverity.HIGH),
            (r'document\.write\(', "document.write() - potential XSS", IssueSeverity.MEDIUM),
        ]
    }
    
    def __init__(self, llm_client: OllamaClient):
        """
        Initialize code reviewer.
        
        Args:
            llm_client: LLM client for analysis
        """
        super().__init__("code_reviewer", llm_client)
        self.analyzer = CodeAnalyzer()
    
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Execute a code review task."""
        try:
            code = task.get("code", "")
            language = Language(task.get("language", "python"))
            file_path = task.get("file_path")
            
            review = await self.review(code, language, file_path)
            
            return AgentResult(
                success=True,
                data=review.to_dict(),
                message=f"Review complete: {len(review.issues)} issues found"
            )
            
        except Exception as e:
            logger.error(f"Code review failed: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                message="Code review failed"
            )
    
    async def review(
        self,
        code: str,
        language: Language,
        file_path: str = None
    ) -> CodeReview:
        """
        Perform comprehensive code review.
        
        Args:
            code: Code to review
            language: Programming language
            file_path: Optional file path for context
            
        Returns:
            Complete code review
        """
        logger.info(f"Reviewing code", language=language.value, lines=len(code.split('\n')))
        
        issues: List[CodeIssue] = []
        
        # Static pattern checks
        pattern_issues = self._check_patterns(code, language)
        issues.extend(pattern_issues)
        
        # LLM-based review
        llm_issues, summary, score, recommendations = await self._llm_review(code, language)
        issues.extend(llm_issues)
        
        # Dedup and sort issues
        issues = self._deduplicate_issues(issues)
        issues.sort(key=lambda i: (
            list(IssueSeverity).index(i.severity),
            i.line or 0
        ))
        
        return CodeReview(
            file_path=file_path,
            language=language,
            issues=issues,
            overall_score=score,
            summary=summary,
            recommendations=recommendations
        )
    
    def _check_patterns(
        self,
        code: str,
        language: Language
    ) -> List[CodeIssue]:
        """Check code against known security patterns."""
        import re
        
        issues = []
        patterns = self.SECURITY_PATTERNS.get(language, [])
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, message, severity in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        line=i,
                        category=IssueCategory.SECURITY,
                        severity=severity,
                        message=message,
                        suggestion="Review and address this security concern",
                        code_snippet=line.strip()
                    ))
        
        return issues
    
    async def _llm_review(
        self,
        code: str,
        language: Language
    ) -> tuple[List[CodeIssue], str, float, List[str]]:
        """Perform LLM-based code review."""
        prompt = self.REVIEW_PROMPT.format(
            language=language.value,
            code=code[:8000]  # Limit code size
        )
        
        response = await self.llm.generate(prompt)
        
        # Parse issues
        issues = self._parse_issues(response)
        
        # Parse summary and score
        summary = self._extract_summary(response)
        score = self._extract_score(response)
        recommendations = self._extract_recommendations(response)
        
        return issues, summary, score, recommendations
    
    def _parse_issues(self, response: str) -> List[CodeIssue]:
        """Parse issues from LLM response."""
        issues = []
        
        issue_blocks = response.split('ISSUE:')[1:] if 'ISSUE:' in response else []
        
        for block in issue_blocks:
            if '---' in block:
                block = block.split('---')[0]
            
            issue = self._parse_issue_block(block)
            if issue:
                issues.append(issue)
        
        return issues
    
    def _parse_issue_block(self, block: str) -> Optional[CodeIssue]:
        """Parse a single issue block."""
        try:
            lines_dict = {}
            current_key = None
            
            for line in block.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    lines_dict[key] = value.strip()
                    current_key = key
                elif current_key and line.strip():
                    lines_dict[current_key] += ' ' + line.strip()
            
            # Parse line number
            line_num = None
            if 'line' in lines_dict:
                try:
                    line_num = int(lines_dict['line'])
                except ValueError:
                    pass
            
            # Parse category
            category = IssueCategory.MAINTAINABILITY
            if 'category' in lines_dict:
                cat_str = lines_dict['category'].lower()
                for cat in IssueCategory:
                    if cat.value in cat_str:
                        category = cat
                        break
            
            # Parse severity
            severity = IssueSeverity.MEDIUM
            if 'severity' in lines_dict:
                sev_str = lines_dict['severity'].lower()
                for sev in IssueSeverity:
                    if sev.value in sev_str:
                        severity = sev
                        break
            
            return CodeIssue(
                line=line_num,
                category=category,
                severity=severity,
                message=lines_dict.get('message', 'Issue detected'),
                suggestion=lines_dict.get('suggestion', 'Review this code')
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse issue: {e}")
            return None
    
    def _extract_summary(self, response: str) -> str:
        """Extract summary from response."""
        if 'SUMMARY:' in response:
            parts = response.split('SUMMARY:')[1]
            if 'SCORE:' in parts:
                parts = parts.split('SCORE:')[0]
            return parts.strip()[:500]
        return "Code review completed"
    
    def _extract_score(self, response: str) -> float:
        """Extract score from response."""
        import re
        
        if 'SCORE:' in response:
            score_part = response.split('SCORE:')[1][:20]
            match = re.search(r'(\d+)', score_part)
            if match:
                return min(100, max(0, float(match.group(1))))
        
        return 50.0  # Default score
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract recommendations from response."""
        recommendations = []
        
        if 'RECOMMENDATIONS:' in response:
            rec_part = response.split('RECOMMENDATIONS:')[1]
            for line in rec_part.split('\n'):
                line = line.strip().lstrip('- •*')
                if line and len(line) > 10:
                    recommendations.append(line)
                if len(recommendations) >= 5:
                    break
        
        return recommendations
    
    def _deduplicate_issues(self, issues: List[CodeIssue]) -> List[CodeIssue]:
        """Remove duplicate issues."""
        seen = set()
        unique = []
        
        for issue in issues:
            key = (issue.line, issue.message[:50])
            if key not in seen:
                seen.add(key)
                unique.append(issue)
        
        return unique
```

---

## Task 11.4: Create Code Generation API Routes

**File:** `src/api/routes/codegen.py` (CREATE NEW FILE)

```python
"""
Code Generation API Routes - REST endpoints for code generation.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agents.code_generator import (
    CodeGeneratorAgent,
    GenerationRequest,
    GenerationMode,
    Language
)
from src.agents.code_reviewer import CodeReviewerAgent
from src.agents.code_templates import TemplateLibrary, TemplateType
from src.utils.ollama_client import OllamaClient
from src.utils.structured_logging import get_logger

logger = get_logger("codegen_api")
router = APIRouter(prefix="/codegen", tags=["code-generation"])


class GenerateCodeRequest(BaseModel):
    """Request for code generation."""
    description: str
    language: str = "python"
    mode: str = "create"
    existing_code: Optional[str] = None
    template_type: Optional[str] = None
    constraints: List[str] = []


class GenerateFunctionRequest(BaseModel):
    """Request for function generation."""
    name: str
    description: str
    parameters: List[dict]
    return_type: str
    language: str = "python"


class GenerateClassRequest(BaseModel):
    """Request for class generation."""
    name: str
    description: str
    attributes: List[dict]
    methods: List[str]
    language: str = "python"


class ReviewCodeRequest(BaseModel):
    """Request for code review."""
    code: str
    language: str = "python"
    file_path: Optional[str] = None


class CodeResponse(BaseModel):
    """Response with generated code."""
    code: str
    language: str
    explanation: str
    imports: List[str]
    dependencies: List[str]
    warnings: List[str]


class ReviewResponse(BaseModel):
    """Response with code review."""
    issues: List[dict]
    issue_count: int
    critical_count: int
    overall_score: float
    summary: str
    recommendations: List[str]


# Global instances
_generator: Optional[CodeGeneratorAgent] = None
_reviewer: Optional[CodeReviewerAgent] = None


def get_generator() -> CodeGeneratorAgent:
    """Get or create code generator."""
    global _generator
    if _generator is None:
        llm = OllamaClient()
        _generator = CodeGeneratorAgent(llm)
    return _generator


def get_reviewer() -> CodeReviewerAgent:
    """Get or create code reviewer."""
    global _reviewer
    if _reviewer is None:
        llm = OllamaClient()
        _reviewer = CodeReviewerAgent(llm)
    return _reviewer


@router.post("/generate", response_model=CodeResponse)
async def generate_code(request: GenerateCodeRequest):
    """Generate code based on description."""
    try:
        generator = get_generator()
        
        gen_request = GenerationRequest(
            description=request.description,
            language=Language(request.language),
            mode=GenerationMode(request.mode),
            existing_code=request.existing_code,
            template_type=TemplateType(request.template_type) if request.template_type else None,
            constraints=request.constraints
        )
        
        result = await generator.generate(gen_request)
        
        return CodeResponse(
            code=result.code,
            language=result.language.value,
            explanation=result.explanation,
            imports=result.imports,
            dependencies=result.dependencies,
            warnings=result.warnings
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/function", response_model=CodeResponse)
async def generate_function(request: GenerateFunctionRequest):
    """Generate a function."""
    try:
        generator = get_generator()
        
        result = await generator.create_function(
            name=request.name,
            description=request.description,
            parameters=request.parameters,
            return_type=request.return_type,
            language=Language(request.language)
        )
        
        return CodeResponse(
            code=result.code,
            language=result.language.value,
            explanation=result.explanation,
            imports=result.imports,
            dependencies=result.dependencies,
            warnings=result.warnings
        )
        
    except Exception as e:
        logger.error(f"Function generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/class", response_model=CodeResponse)
async def generate_class(request: GenerateClassRequest):
    """Generate a class."""
    try:
        generator = get_generator()
        
        result = await generator.create_class(
            name=request.name,
            description=request.description,
            attributes=request.attributes,
            methods=request.methods,
            language=Language(request.language)
        )
        
        return CodeResponse(
            code=result.code,
            language=result.language.value,
            explanation=result.explanation,
            imports=result.imports,
            dependencies=result.dependencies,
            warnings=result.warnings
        )
        
    except Exception as e:
        logger.error(f"Class generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/review", response_model=ReviewResponse)
async def review_code(request: ReviewCodeRequest):
    """Review code for issues."""
    try:
        reviewer = get_reviewer()
        
        result = await reviewer.review(
            code=request.code,
            language=Language(request.language),
            file_path=request.file_path
        )
        
        return ReviewResponse(
            issues=[i.to_dict() for i in result.issues],
            issue_count=len(result.issues),
            critical_count=result.critical_count,
            overall_score=result.overall_score,
            summary=result.summary,
            recommendations=result.recommendations
        )
        
    except Exception as e:
        logger.error(f"Review failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def list_templates(
    language: Optional[str] = None,
    template_type: Optional[str] = None
):
    """List available code templates."""
    templates = TemplateLibrary.list_templates(
        language=Language(language) if language else None,
        template_type=TemplateType(template_type) if template_type else None
    )
    
    return {
        "templates": [
            {
                "name": t.name,
                "language": t.language.value,
                "type": t.template_type.value,
                "description": t.description,
                "variables": t.variables
            }
            for t in templates
        ]
    }
```

---

## Testing Requirements

**File:** `tests/test_code_generator.py` (CREATE NEW FILE)

```python
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
    def mock_llm(self):
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="""
```python
def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b
```
""")
        return llm
    
    @pytest.fixture
    def generator(self, mock_llm):
        return CodeGeneratorAgent(mock_llm)
    
    def test_extract_code_from_response(self, generator):
        """Test code extraction from LLM response."""
        response = """Here's the code:
```python
def hello():
    return "Hello"
```
That's all."""
        
        code = generator._extract_code(response, Language.PYTHON)
        
        assert "def hello" in code
        assert "return" in code
    
    @pytest.mark.asyncio
    async def test_generate_function(self, generator):
        """Test function generation."""
        result = await generator.create_function(
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
    async def test_generate_with_mode(self, generator):
        """Test generation with different modes."""
        request = GenerationRequest(
            description="A simple calculator",
            language=Language.PYTHON,
            mode=GenerationMode.CREATE
        )
        
        result = await generator.generate(request)
        
        assert result.code


class TestCodeReviewerAgent:
    """Tests for code reviewer."""
    
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="""
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
        return llm
    
    @pytest.fixture
    def reviewer(self, mock_llm):
        return CodeReviewerAgent(mock_llm)
    
    def test_pattern_check_eval(self, reviewer):
        """Test detection of eval usage."""
        code = """
result = eval(user_input)
"""
        issues = reviewer._check_patterns(code, Language.PYTHON)
        
        assert len(issues) > 0
        assert issues[0].severity == IssueSeverity.CRITICAL
    
    def test_pattern_check_password(self, reviewer):
        """Test detection of hardcoded password."""
        code = """
password = "secret123"
"""
        issues = reviewer._check_patterns(code, Language.PYTHON)
        
        assert len(issues) > 0
        assert any("password" in i.message.lower() for i in issues)
    
    @pytest.mark.asyncio
    async def test_full_review(self, reviewer):
        """Test full code review."""
        code = """
def process(data):
    result = eval(data)
    password = "test123"
    return result
"""
        review = await reviewer.review(code, Language.PYTHON)
        
        assert review.issues
        assert review.overall_score < 100
        assert review.summary
```

---

## Acceptance Criteria

- [ ] `src/agents/code_templates.py` - Template library with multiple languages
- [ ] `src/agents/code_generator.py` - Full code generation agent
- [ ] `src/agents/code_reviewer.py` - Automated code review
- [ ] `src/api/routes/codegen.py` - REST API endpoints
- [ ] `tests/test_code_generator.py` - All tests passing
- [ ] Multiple generation modes working
- [ ] Code review detecting security issues
- [ ] Template rendering working correctly

---

## File Summary

| Action | File Path |
|--------|-----------|
| CREATE | `src/agents/code_templates.py` |
| REPLACE | `src/agents/code_generator.py` |
| CREATE | `src/agents/code_reviewer.py` |
| CREATE | `src/api/routes/codegen.py` |
| CREATE | `tests/test_code_generator.py` |

---

*End of Part 11*
