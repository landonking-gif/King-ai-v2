"""
Prompt Template Engine.
Manages and renders LLM prompts with variables and versioning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Set
from pathlib import Path
from enum import Enum
import re
import hashlib
import json
import yaml

from src.utils.structured_logging import get_logger

logger = get_logger("prompt_engine")


class TemplateFormat(str, Enum):
    """Template format types."""
    PLAIN = "plain"  # Simple {{variable}} substitution
    JINJA = "jinja"  # Jinja2-style templating
    MARKDOWN = "markdown"  # Markdown with variable blocks


class TemplateCategory(str, Enum):
    """Template categories."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    SUMMARIZATION = "summarization"


@dataclass
class TemplateVariable:
    """Variable definition for a template."""
    name: str
    description: str = ""
    required: bool = True
    default: Any = None
    type_hint: str = "str"
    validators: List[Callable[[Any], bool]] = field(default_factory=list)
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against this variable definition."""
        if value is None:
            if self.required and self.default is None:
                return False, f"Required variable '{self.name}' is missing"
            return True, None
        
        for validator in self.validators:
            try:
                if not validator(value):
                    return False, f"Validation failed for '{self.name}'"
            except Exception as e:
                return False, f"Validation error for '{self.name}': {e}"
        
        return True, None


@dataclass
class PromptTemplate:
    """A versioned prompt template."""
    id: str
    name: str
    content: str
    category: TemplateCategory
    format: TemplateFormat = TemplateFormat.PLAIN
    description: str = ""
    version: str = "1.0.0"
    variables: Dict[str, TemplateVariable] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        # Auto-detect variables from content
        if not self.variables:
            self.variables = self._extract_variables()
    
    def _extract_variables(self) -> Dict[str, TemplateVariable]:
        """Extract variables from template content."""
        variables = {}
        
        # Match {{variable}} or {{ variable }}
        pattern = r'\{\{\s*(\w+)\s*\}\}'
        matches = re.findall(pattern, self.content)
        
        for var_name in set(matches):
            variables[var_name] = TemplateVariable(
                name=var_name,
                required=True,
            )
        
        return variables
    
    @property
    def content_hash(self) -> str:
        """Hash of template content for change detection."""
        return hashlib.md5(self.content.encode()).hexdigest()[:8]
    
    def render(self, context: Dict[str, Any]) -> str:
        """Render template with context."""
        result = self.content
        
        # Apply defaults
        full_context = {}
        for var_name, var_def in self.variables.items():
            if var_name in context:
                full_context[var_name] = context[var_name]
            elif var_def.default is not None:
                full_context[var_name] = var_def.default
        
        # Simple substitution
        if self.format == TemplateFormat.PLAIN:
            for key, value in full_context.items():
                pattern = r'\{\{\s*' + re.escape(key) + r'\s*\}\}'
                result = re.sub(pattern, str(value), result)
        
        return result
    
    def validate_context(self, context: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate context against template variables."""
        errors = []
        
        for var_name, var_def in self.variables.items():
            value = context.get(var_name)
            is_valid, error = var_def.validate(value)
            if not is_valid:
                errors.append(error)
        
        return len(errors) == 0, errors


@dataclass
class PromptChain:
    """Chain of prompts for multi-step interactions."""
    id: str
    name: str
    templates: List[str]  # Template IDs in order
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateVersion:
    """Version history entry for a template."""
    version: str
    content: str
    content_hash: str
    created_at: datetime
    change_description: str = ""


class TemplateRegistry:
    """Registry of all templates with versioning."""
    
    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._versions: Dict[str, List[TemplateVersion]] = {}
        self._chains: Dict[str, PromptChain] = {}
    
    def register(self, template: PromptTemplate) -> None:
        """Register a template."""
        self._templates[template.id] = template
        
        # Store version
        if template.id not in self._versions:
            self._versions[template.id] = []
        
        self._versions[template.id].append(TemplateVersion(
            version=template.version,
            content=template.content,
            content_hash=template.content_hash,
            created_at=template.updated_at,
        ))
        
        logger.info(f"Registered template: {template.name} v{template.version}")
    
    def get(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)
    
    def get_version(
        self,
        template_id: str,
        version: str,
    ) -> Optional[TemplateVersion]:
        """Get a specific version of a template."""
        versions = self._versions.get(template_id, [])
        for v in versions:
            if v.version == version:
                return v
        return None
    
    def list_templates(
        self,
        category: Optional[TemplateCategory] = None,
    ) -> List[PromptTemplate]:
        """List all templates, optionally filtered by category."""
        templates = list(self._templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return templates
    
    def register_chain(self, chain: PromptChain) -> None:
        """Register a prompt chain."""
        self._chains[chain.id] = chain
    
    def get_chain(self, chain_id: str) -> Optional[PromptChain]:
        """Get a prompt chain."""
        return self._chains.get(chain_id)


class PromptBuilder:
    """Fluent builder for constructing prompts."""
    
    def __init__(self):
        self._parts: List[str] = []
        self._variables: Dict[str, Any] = {}
    
    def add_system(self, content: str) -> "PromptBuilder":
        """Add system instruction."""
        self._parts.append(f"[SYSTEM]\n{content}\n")
        return self
    
    def add_context(self, context: str, label: str = "Context") -> "PromptBuilder":
        """Add context section."""
        self._parts.append(f"[{label.upper()}]\n{context}\n")
        return self
    
    def add_examples(self, examples: List[Dict[str, str]]) -> "PromptBuilder":
        """Add few-shot examples."""
        self._parts.append("[EXAMPLES]\n")
        for i, ex in enumerate(examples, 1):
            self._parts.append(f"Example {i}:\n")
            self._parts.append(f"Input: {ex.get('input', '')}\n")
            self._parts.append(f"Output: {ex.get('output', '')}\n\n")
        return self
    
    def add_task(self, task: str) -> "PromptBuilder":
        """Add task description."""
        self._parts.append(f"[TASK]\n{task}\n")
        return self
    
    def add_constraints(self, constraints: List[str]) -> "PromptBuilder":
        """Add output constraints."""
        self._parts.append("[CONSTRAINTS]\n")
        for c in constraints:
            self._parts.append(f"- {c}\n")
        return self
    
    def add_output_format(self, format_spec: str) -> "PromptBuilder":
        """Specify output format."""
        self._parts.append(f"[OUTPUT FORMAT]\n{format_spec}\n")
        return self
    
    def with_variable(self, name: str, value: Any) -> "PromptBuilder":
        """Add a variable for substitution."""
        self._variables[name] = value
        return self
    
    def build(self) -> str:
        """Build the final prompt."""
        result = "\n".join(self._parts)
        
        # Substitute variables
        for key, value in self._variables.items():
            pattern = r'\{\{\s*' + re.escape(key) + r'\s*\}\}'
            result = re.sub(pattern, str(value), result)
        
        return result.strip()


class PromptTemplateEngine:
    """
    Manages LLM prompts with templating and versioning.
    
    Features:
    - Template registration and versioning
    - Variable substitution and validation
    - Prompt chains for multi-step workflows
    - Template optimization suggestions
    - Usage analytics
    """
    
    def __init__(self):
        self.registry = TemplateRegistry()
        self._usage_stats: Dict[str, int] = {}
        self._render_times: Dict[str, List[float]] = {}
    
    def register_template(
        self,
        template_id: str,
        name: str,
        content: str,
        category: TemplateCategory,
        description: str = "",
        variables: Optional[Dict[str, Dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> PromptTemplate:
        """Register a new template."""
        var_defs = {}
        if variables:
            for var_name, var_config in variables.items():
                var_defs[var_name] = TemplateVariable(
                    name=var_name,
                    description=var_config.get("description", ""),
                    required=var_config.get("required", True),
                    default=var_config.get("default"),
                    type_hint=var_config.get("type", "str"),
                )
        
        template = PromptTemplate(
            id=template_id,
            name=name,
            content=content,
            category=category,
            description=description,
            variables=var_defs,
            metadata=metadata or {},
        )
        
        self.registry.register(template)
        return template
    
    def render(
        self,
        template_id: str,
        context: Dict[str, Any],
        validate: bool = True,
    ) -> str:
        """Render a template with context."""
        template = self.registry.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        # Validate context
        if validate:
            is_valid, errors = template.validate_context(context)
            if not is_valid:
                raise ValueError(f"Context validation failed: {errors}")
        
        # Track usage
        self._usage_stats[template_id] = self._usage_stats.get(template_id, 0) + 1
        
        return template.render(context)
    
    def render_chain(
        self,
        chain_id: str,
        contexts: List[Dict[str, Any]],
    ) -> List[str]:
        """Render all templates in a chain."""
        chain = self.registry.get_chain(chain_id)
        if not chain:
            raise ValueError(f"Chain not found: {chain_id}")
        
        if len(contexts) != len(chain.templates):
            raise ValueError(
                f"Context count ({len(contexts)}) doesn't match template count ({len(chain.templates)})"
            )
        
        results = []
        for template_id, context in zip(chain.templates, contexts):
            results.append(self.render(template_id, context))
        
        return results
    
    def builder(self) -> PromptBuilder:
        """Get a new prompt builder."""
        return PromptBuilder()
    
    def load_from_yaml(self, path: Path) -> List[str]:
        """Load templates from YAML file."""
        loaded_ids = []
        
        content = yaml.safe_load(path.read_text())
        
        for template_def in content.get("templates", []):
            template = self.register_template(
                template_id=template_def["id"],
                name=template_def["name"],
                content=template_def["content"],
                category=TemplateCategory(template_def.get("category", "user")),
                description=template_def.get("description", ""),
                variables=template_def.get("variables"),
                metadata=template_def.get("metadata"),
            )
            loaded_ids.append(template.id)
        
        return loaded_ids
    
    def export_template(self, template_id: str) -> Dict[str, Any]:
        """Export a template to dictionary format."""
        template = self.registry.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        return {
            "id": template.id,
            "name": template.name,
            "content": template.content,
            "category": template.category.value,
            "description": template.description,
            "version": template.version,
            "variables": {
                name: {
                    "description": var.description,
                    "required": var.required,
                    "default": var.default,
                    "type": var.type_hint,
                }
                for name, var in template.variables.items()
            },
            "metadata": template.metadata,
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get template usage statistics."""
        return {
            "usage_counts": self._usage_stats.copy(),
            "most_used": sorted(
                self._usage_stats.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "total_templates": len(self.registry._templates),
        }
    
    def optimize_template(self, template_id: str) -> List[str]:
        """Suggest optimizations for a template."""
        template = self.registry.get(template_id)
        if not template:
            return []
        
        suggestions = []
        
        # Check length
        if len(template.content) > 4000:
            suggestions.append(
                "Template is quite long. Consider breaking into smaller, focused templates."
            )
        
        # Check for unused variables
        content_vars = set(re.findall(r'\{\{\s*(\w+)\s*\}\}', template.content))
        defined_vars = set(template.variables.keys())
        
        unused = defined_vars - content_vars
        if unused:
            suggestions.append(f"Unused variables defined: {unused}")
        
        undefined = content_vars - defined_vars
        if undefined:
            suggestions.append(f"Variables used but not defined: {undefined}")
        
        # Check for common prompt patterns
        if "json" in template.content.lower() and "```" not in template.content:
            suggestions.append(
                "Template mentions JSON. Consider adding format example with code block."
            )
        
        return suggestions


# Global engine instance
prompt_engine = PromptTemplateEngine()


def get_prompt_engine() -> PromptTemplateEngine:
    """Get the global prompt engine instance."""
    return prompt_engine


# Pre-built templates
def register_default_templates():
    """Register commonly used templates."""
    
    # Analysis template
    prompt_engine.register_template(
        template_id="analysis.business",
        name="Business Analysis",
        content="""Analyze the following business data and provide insights:

{{business_context}}

Focus areas:
{{focus_areas}}

Provide:
1. Key findings
2. Trends identified  
3. Recommendations
4. Risk factors

Format your response as structured JSON.""",
        category=TemplateCategory.ANALYSIS,
        description="Standard business analysis prompt",
        variables={
            "business_context": {"description": "Business data to analyze", "required": True},
            "focus_areas": {"description": "Specific areas to focus on", "default": "revenue, growth, efficiency"},
        },
    )
    
    # Content generation template
    prompt_engine.register_template(
        template_id="content.marketing",
        name="Marketing Content Generator",
        content="""Create marketing content for:

Product: {{product_name}}
Target Audience: {{target_audience}}
Tone: {{tone}}
Type: {{content_type}}

Requirements:
{{requirements}}

Generate engaging content that resonates with the target audience.""",
        category=TemplateCategory.GENERATION,
        description="Marketing content generation",
        variables={
            "product_name": {"description": "Name of product/service", "required": True},
            "target_audience": {"description": "Target audience description", "required": True},
            "tone": {"description": "Content tone", "default": "professional"},
            "content_type": {"description": "Type of content", "default": "social media post"},
            "requirements": {"description": "Additional requirements", "default": "Keep it concise"},
        },
    )
    
    # Code review template
    prompt_engine.register_template(
        template_id="code.review",
        name="Code Review",
        content="""Review the following code for:
- Security vulnerabilities
- Performance issues
- Best practices violations
- Code quality

Language: {{language}}

Code:
```{{language}}
{{code}}
```

Provide specific, actionable feedback with line references.""",
        category=TemplateCategory.ANALYSIS,
        description="Code review and analysis",
        variables={
            "language": {"description": "Programming language", "required": True},
            "code": {"description": "Code to review", "required": True},
        },
    )
    
    # Summarization template
    prompt_engine.register_template(
        template_id="summarize.document",
        name="Document Summarizer",
        content="""Summarize the following document:

{{document}}

Summary requirements:
- Length: {{summary_length}}
- Focus on: {{focus}}
- Include key points and action items""",
        category=TemplateCategory.SUMMARIZATION,
        description="Document summarization",
        variables={
            "document": {"description": "Document to summarize", "required": True},
            "summary_length": {"description": "Desired summary length", "default": "3-5 paragraphs"},
            "focus": {"description": "Areas to focus on", "default": "main points and conclusions"},
        },
    )


# Auto-register default templates
register_default_templates()
