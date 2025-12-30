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

router = APIRouter()


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
        _generator = CodeGeneratorAgent()
    return _generator


def get_reviewer() -> CodeReviewerAgent:
    """Get or create code reviewer."""
    global _reviewer
    if _reviewer is None:
        _reviewer = CodeReviewerAgent()
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
