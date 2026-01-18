"""
Code Executor Service - FastAPI application.
Module: code-exec/service/main.py
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse

from .config import settings
from .executor import SandboxedExecutor
from .models import (
    ErrorResponse,
    ExecutionRequest,
    ExecutionResult,
    SafetyFlag,
    SkillListResponse,
    SkillMetadata,
    SkillRegistrationRequest,
    SkillSchemaResponse,
)
from .registry import SkillNotFoundError, SkillRegistry, SkillValidationError

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format,
)
logger = logging.getLogger(__name__)

# Global registry and executor (initialized in lifespan)
registry: Optional[SkillRegistry] = None
executor: Optional[SandboxedExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager for startup and shutdown.

    Args:
        app: FastAPI application instance
    """
    global registry, executor

    # Startup
    logger.info("Starting Code Executor Service")
    logger.info(f"Skills directory: {settings.skills_directory}")

    # Initialize registry
    registry = SkillRegistry(settings.skills_directory)
    await registry.load_all_skills()

    # Initialize executor
    executor = SandboxedExecutor(registry)

    logger.info(f"Loaded {len(registry.skills)} skills")
    logger.info(f"Service ready on {settings.host}:{settings.port}")

    yield

    # Shutdown
    logger.info("Shutting down Code Executor Service")


# Create FastAPI app
app = FastAPI(
    title="Code Executor Service",
    description="Sandboxed execution service for deterministic skills with provenance tracking",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Health"])
async def root() -> dict:
    """Root endpoint - service health check."""
    return {
        "service": settings.service_name,
        "status": "healthy",
        "skills_loaded": len(registry.skills) if registry else 0,
    }


@app.get("/health", tags=["Health"])
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "registry_loaded": registry is not None,
        "executor_ready": executor is not None,
        "skills_count": len(registry.skills) if registry else 0,
    }


@app.post(
    "/skills/execute",
    response_model=ExecutionResult,
    status_code=status.HTTP_200_OK,
    tags=["Execution"],
)
async def execute_skill(request: ExecutionRequest) -> ExecutionResult:
    """
    Execute a skill with input validation and provenance tracking.

    Args:
        request: Execution request with skill name and arguments

    Returns:
        Execution result with provenance record

    Raises:
        HTTPException: If skill not found or execution fails
    """
    if not executor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Executor not initialized",
        )

    try:
        result = await executor.execute(request)

        if not result.success:
            # Return error result with 200 status but success=False
            # This allows client to see provenance even for failures
            return result

        return result

    except SkillNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except SkillValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error during execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}",
        )


@app.get(
    "/skills/list",
    response_model=SkillListResponse,
    tags=["Skills"],
)
async def list_skills(
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    safety_flag: Optional[SafetyFlag] = Query(None, description="Filter by safety flag"),
) -> SkillListResponse:
    """
    List all available skills with optional filtering.

    Args:
        tags: Optional list of tags to filter by
        safety_flag: Optional safety flag to filter by

    Returns:
        List of skill metadata
    """
    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Registry not initialized",
        )

    skills = registry.list_skills(tags=tags, safety_flag=safety_flag)

    return SkillListResponse(
        skills=skills,
        total=len(skills),
    )


@app.get(
    "/skills/{skill_id}/schema",
    response_model=SkillSchemaResponse,
    tags=["Skills"],
)
async def get_skill_schema(skill_id: str) -> SkillSchemaResponse:
    """
    Get input/output JSON schemas for a skill.

    Args:
        skill_id: Skill name/identifier

    Returns:
        Skill schemas

    Raises:
        HTTPException: If skill not found
    """
    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Registry not initialized",
        )

    try:
        skill = registry.get_skill(skill_id)

        return SkillSchemaResponse(
            skill_name=skill.name,
            version=skill.version,
            input_schema=skill.input_schema,
            output_schema=skill.output_schema,
        )

    except SkillNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {skill_id}",
        )


@app.get(
    "/skills/{skill_id}",
    response_model=SkillMetadata,
    tags=["Skills"],
)
async def get_skill_metadata(skill_id: str) -> SkillMetadata:
    """
    Get full metadata for a skill.

    Args:
        skill_id: Skill name/identifier

    Returns:
        Skill metadata

    Raises:
        HTTPException: If skill not found
    """
    if not registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Registry not initialized",
        )

    try:
        return registry.get_skill(skill_id)

    except SkillNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {skill_id}",
        )


@app.post(
    "/skills/register",
    response_model=SkillMetadata,
    status_code=status.HTTP_201_CREATED,
    tags=["Skills"],
)
async def register_skill(request: SkillRegistrationRequest) -> SkillMetadata:
    """
    Register a new skill (for future implementation).

    Args:
        request: Skill registration request

    Returns:
        Registered skill metadata

    Raises:
        HTTPException: Not yet implemented
    """
    # TODO: Implement dynamic skill registration
    # This would involve:
    # 1. Validating the handler code
    # 2. Creating skill directory structure
    # 3. Writing skill.yaml and schema.json
    # 4. Loading the skill into registry

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Dynamic skill registration not yet implemented. "
        "Please add skills to the skills directory and restart the service.",
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for unhandled errors.

    Args:
        request: Request object
        exc: Exception

    Returns:
        JSON error response
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    error_response = ErrorResponse(
        error="internal_server_error",
        message="An unexpected error occurred",
        details={"exception_type": type(exc).__name__},
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
