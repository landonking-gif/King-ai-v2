"""
Code Templates System - Reusable code templates and scaffolding.
Provides structured templates for common code patterns.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


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
