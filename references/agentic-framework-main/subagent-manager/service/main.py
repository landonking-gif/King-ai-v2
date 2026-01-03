"""
Subagent Manager FastAPI service.

This service manages isolated LLM contexts (subagents) with capability enforcement,
schema validation, and bounded lifetimes.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List

import anyio
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel

from subagent_manager.service.config import config
from subagent_manager.service.lifecycle import SubagentLifecycleManager
from subagent_manager.service.models import (
    SubagentDestroyRequest,
    SubagentExecuteRequest,
    SubagentInfo,
    SubagentListResponse,
    SubagentResponse,
    SubagentSpawnRequest,
)
from subagent_manager.service.validator import SchemaValidator

# Metrics
subagent_spawned = Counter(
    "subagent_spawned_total", "Total number of subagents spawned", ["role"]
)
subagent_executed = Counter(
    "subagent_executed_total", "Total number of task executions", ["status"]
)
execution_duration = Histogram(
    "subagent_execution_duration_seconds", "Task execution duration"
)


# Global lifecycle manager
lifecycle_manager: SubagentLifecycleManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """
    Application lifespan handler.

    Initializes services on startup and cleans up on shutdown.
    """
    global lifecycle_manager

    # Startup
    validator = SchemaValidator(config.schema_registry_path)
    lifecycle_manager = SubagentLifecycleManager(config, validator)

    # Start background tasks (cleanup loop)
    # Note: In production, this would be managed by a task group
    # For now, we'll skip the background task to keep it simple
    # await lifecycle_manager.start()

    yield

    # Shutdown
    # Cleanup all subagents
    if lifecycle_manager:
        for subagent_id in list(lifecycle_manager.subagents.keys()):
            await lifecycle_manager.destroy_subagent(subagent_id)


# Create FastAPI app
app = FastAPI(
    title="Subagent Manager",
    description="Manages isolated LLM contexts with capability enforcement",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    active_subagents: int


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Service health status
    """
    active_count = len(lifecycle_manager.subagents) if lifecycle_manager else 0
    return HealthResponse(
        status="healthy",
        service="subagent-manager",
        version="1.0.0",
        active_subagents=active_count,
    )


@app.post("/subagent/spawn", response_model=SubagentInfo, status_code=status.HTTP_201_CREATED)
async def spawn_subagent(request: SubagentSpawnRequest) -> SubagentInfo:
    """
    Spawn a new subagent instance.

    Args:
        request: Subagent spawn configuration

    Returns:
        Information about the created subagent

    Raises:
        HTTPException: If spawning fails
    """
    if not lifecycle_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lifecycle manager not initialized",
        )

    try:
        info = await lifecycle_manager.spawn_subagent(request)
        subagent_spawned.labels(role=request.role.value).inc()
        return info
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to spawn subagent: {str(e)}"
        )


@app.post("/subagent/execute", response_model=SubagentResponse)
async def execute_task(request: SubagentExecuteRequest) -> SubagentResponse:
    """
    Execute a task with a subagent.

    Args:
        request: Task execution request

    Returns:
        Execution response with validated output

    Raises:
        HTTPException: If execution fails
    """
    if not lifecycle_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lifecycle manager not initialized",
        )

    try:
        with execution_duration.time():
            response = await lifecycle_manager.execute_task(request)

        subagent_executed.labels(status=response.status.value).inc()
        return response

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Execution failed: {str(e)}"
        )


@app.post("/subagent/destroy", status_code=status.HTTP_204_NO_CONTENT)
async def destroy_subagent(request: SubagentDestroyRequest) -> None:
    """
    Destroy a subagent and cleanup its context.

    Args:
        request: Destruction request

    Raises:
        HTTPException: If subagent not found
    """
    if not lifecycle_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lifecycle manager not initialized",
        )

    destroyed = await lifecycle_manager.destroy_subagent(request.subagent_id)
    if not destroyed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Subagent {request.subagent_id} not found",
        )


@app.get("/subagent/{subagent_id}/status", response_model=SubagentInfo)
async def get_subagent_status(subagent_id: str) -> SubagentInfo:
    """
    Get status of a specific subagent.

    Args:
        subagent_id: Subagent identifier

    Returns:
        Subagent information

    Raises:
        HTTPException: If subagent not found
    """
    if not lifecycle_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lifecycle manager not initialized",
        )

    info = await lifecycle_manager.get_subagent_status(subagent_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Subagent {subagent_id} not found",
        )

    return info


@app.get("/subagents", response_model=SubagentListResponse)
async def list_subagents() -> SubagentListResponse:
    """
    List all active subagents.

    Returns:
        List of all subagent instances
    """
    if not lifecycle_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lifecycle manager not initialized",
        )

    subagents = await lifecycle_manager.list_subagents()
    active_count = sum(1 for s in subagents if s.status.value in ("ready", "executing"))

    return SubagentListResponse(
        subagents=subagents,
        total=len(subagents),
        active=active_count,
    )


@app.get("/schemas")
async def list_schemas() -> Dict[str, List[str]]:
    """
    List all available artifact schemas.

    Returns:
        Dictionary with schema names
    """
    if not lifecycle_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lifecycle manager not initialized",
        )

    schema_names = lifecycle_manager.validator.list_schemas()
    return {"schemas": schema_names}


@app.get("/schemas/{schema_name}")
async def get_schema(schema_name: str) -> Dict[str, Any]:
    """
    Get a specific schema definition.

    Args:
        schema_name: Name of the schema

    Returns:
        Schema definition

    Raises:
        HTTPException: If schema not found
    """
    if not lifecycle_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lifecycle manager not initialized",
        )

    schema = lifecycle_manager.validator.get_schema(schema_name)
    if not schema:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Schema {schema_name} not found"
        )

    return schema


@app.get("/metrics")
async def metrics() -> Any:
    """
    Prometheus metrics endpoint.

    Returns:
        Metrics in Prometheus format
    """
    from fastapi.responses import Response

    return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "subagent_manager.service.main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=True,
    )
