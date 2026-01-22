"""
Workflow API routes.

Provides endpoints for workflow management and execution.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Response
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import json
import asyncio
from datetime import datetime

from src.workflows.executor import WorkflowExecutor
from src.workflows.models import WorkflowManifest, WorkflowRun
from src.database.connection import get_db
from src.utils.structured_logging import get_logger
from src.workflows.loader import get_workflow_loader

logger = get_logger("workflow_api")
router = APIRouter()

# Global workflow executor instance
workflow_executor = WorkflowExecutor()

# Active workflow executions (for WebSocket connections)
active_executions: Dict[str, Dict[str, Any]] = {}


class WorkflowCreateRequest(BaseModel):
    """Request to create and execute a workflow."""
    name: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    variables: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    """Response containing workflow execution details."""
    execution_id: str
    status: str
    created_at: datetime


@router.post("/execute", response_model=WorkflowResponse)
async def execute_workflow(
    request: WorkflowCreateRequest,
    background_tasks: BackgroundTasks
) -> WorkflowResponse:
    """
    Execute a workflow from the visual editor.

    Converts the visual workflow definition to a manifest and executes it.
    """
    try:
        # Convert visual workflow to manifest
        manifest = convert_visual_workflow_to_manifest(request)

        # Create workflow run
        execution_id = f"wf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request.name.replace(' ', '_')}"

        # Initialize execution tracking
        active_executions[execution_id] = {
            "status": "running",
            "logs": [],
            "progress": 0,
            "start_time": datetime.utcnow()
        }

        # Execute workflow in background
        background_tasks.add_task(
            execute_workflow_background,
            execution_id,
            manifest
        )

        return WorkflowResponse(
            execution_id=execution_id,
            status="running",
            created_at=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Failed to start workflow execution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute workflow: {str(e)}")


@router.get("/executions/{execution_id}")
async def get_workflow_execution(execution_id: str) -> Dict[str, Any]:
    """Get the status and logs of a workflow execution."""
    if execution_id not in active_executions:
        raise HTTPException(status_code=404, detail="Workflow execution not found")

    return active_executions[execution_id]


@router.websocket("/executions/{execution_id}/ws")
async def workflow_execution_websocket(websocket: WebSocket, execution_id: str):
    """WebSocket endpoint for real-time workflow execution updates."""
    await websocket.accept()

    if execution_id not in active_executions:
        await websocket.send_json({"error": "Workflow execution not found"})
        await websocket.close()
        return

    try:
        # Send initial status
        await websocket.send_json(active_executions[execution_id])

        # Keep connection alive and send updates
        while True:
            if execution_id in active_executions:
                execution_data = active_executions[execution_id]
                await websocket.send_json(execution_data)

                # Check if execution is complete
                if execution_data.get("status") in ["completed", "failed"]:
                    break

            await asyncio.sleep(1)  # Send updates every second

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for execution {execution_id}")
    except Exception as e:
        logger.error(f"WebSocket error for execution {execution_id}: {e}")


@router.get("/templates")
async def get_workflow_templates() -> List[Dict[str, Any]]:
    """Get available workflow templates."""
    templates = [
        {
            "id": "content-generation",
            "name": "Content Generation",
            "description": "Generate blog posts, articles, or marketing content",
            "nodes": [
                {"id": "1", "type": "input", "data": {"label": "Start"}, "position": {"x": 100, "y": 100}},
                {"id": "2", "type": "llm", "data": {"label": "Generate Content", "model": "gpt-4"}, "position": {"x": 300, "y": 100}},
                {"id": "3", "type": "output", "data": {"label": "Save Content"}, "position": {"x": 500, "y": 100}}
            ],
            "edges": [
                {"id": "e1-2", "source": "1", "target": "2"},
                {"id": "e2-3", "source": "2", "target": "3"}
            ]
        },
        {
            "id": "data-analysis",
            "name": "Data Analysis",
            "description": "Analyze datasets and generate insights",
            "nodes": [
                {"id": "1", "type": "input", "data": {"label": "Load Data"}, "position": {"x": 100, "y": 100}},
                {"id": "2", "type": "data", "data": {"label": "Process Data", "operation": "analyze"}, "position": {"x": 300, "y": 100}},
                {"id": "3", "type": "llm", "data": {"label": "Generate Insights", "model": "gpt-4"}, "position": {"x": 500, "y": 100}},
                {"id": "4", "type": "output", "data": {"label": "Save Report"}, "position": {"x": 700, "y": 100}}
            ],
            "edges": [
                {"id": "e1-2", "source": "1", "target": "2"},
                {"id": "e2-3", "source": "2", "target": "3"},
                {"id": "e3-4", "source": "3", "target": "4"}
            ]
        },
        {
            "id": "research-pipeline",
            "name": "Research Pipeline",
            "description": "Research topics and generate comprehensive reports",
            "nodes": [
                {"id": "1", "type": "input", "data": {"label": "Research Topic"}, "position": {"x": 100, "y": 100}},
                {"id": "2", "type": "tool", "data": {"label": "Web Search", "tool": "web-search"}, "position": {"x": 300, "y": 100}},
                {"id": "3", "type": "llm", "data": {"label": "Analyze Results", "model": "gpt-4"}, "position": {"x": 500, "y": 100}},
                {"id": "4", "type": "tool", "data": {"label": "Generate Report", "tool": "document-generator"}, "position": {"x": 700, "y": 100}},
                {"id": "5", "type": "output", "data": {"label": "Save Report"}, "position": {"x": 900, "y": 100}}
            ],
            "edges": [
                {"id": "e1-2", "source": "1", "target": "2"},
                {"id": "e2-3", "source": "2", "target": "3"},
                {"id": "e3-4", "source": "3", "target": "4"},
                {"id": "e4-5", "source": "4", "target": "5"}
            ]
        }
    ]

    return templates


def convert_visual_workflow_to_manifest(request: WorkflowCreateRequest) -> WorkflowManifest:
    """Convert visual workflow definition to WorkflowManifest."""
    from src.workflows.models import WorkflowStep
    
    steps = []
    for node in request.nodes:
        if node["type"] == "llm":
            step = WorkflowStep(
                id=node["id"],
                name=node["data"]["label"],
                description=f"LLM call using {node['data'].get('model', 'gpt-4')}",
                agent="llm",
                inputs={
                    "model": node["data"].get("model", "gpt-4"),
                    "prompt": "Execute this step in the workflow",
                    "temperature": 0.7
                },
                outputs=["result"],
                depends_on=[]
            )
        elif node["type"] == "tool":
            step = WorkflowStep(
                id=node["id"],
                name=node["data"]["label"],
                description=f"Tool call to {node['data'].get('tool', 'web-search')}",
                agent="tool",
                inputs={
                    "tool": node["data"].get("tool", "web-search"),
                    "parameters": {}
                },
                outputs=["result"],
                depends_on=[]
            )
        elif node["type"] == "data":
            step = WorkflowStep(
                id=node["id"],
                name=node["data"]["label"],
                description=f"Data processing operation: {node['data'].get('operation', 'query')}",
                agent="data",
                inputs={
                    "operation": node["data"].get("operation", "query")
                },
                outputs=["result"],
                depends_on=[]
            )
        else:
            # Default step for input/output nodes
            step = WorkflowStep(
                id=node["id"],
                name=node["data"]["label"],
                description="Workflow input/output step",
                agent="generic",
                inputs={},
                outputs=["result"],
                depends_on=[]
            )
        
        steps.append(step)

    # Add dependencies based on edges
    for edge in request.edges:
        source_id = edge["source"]
        target_id = edge["target"]

        # Find target step and add dependency
        for step in steps:
            if step.id == target_id:
                if source_id not in step.depends_on:
                    step.depends_on.append(source_id)
                break

    manifest = WorkflowManifest(
        id=f"wf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request.name.replace(' ', '_')}",
        name=request.name,
        description=f"Workflow created from visual editor: {request.name}",
        version="1.0.0",
        steps=steps,
        variables=[{"name": k, "description": "", "type": "string", "default": v} for k, v in (request.variables or {}).items()],
        created_at=datetime.utcnow()
    )

    return manifest


async def execute_workflow_background(execution_id: str, manifest: WorkflowManifest):
    """Execute workflow in background and update execution status."""
    try:
        logger.info(f"Starting workflow execution {execution_id}: {manifest.name}")

        # Update status to running
        active_executions[execution_id]["status"] = "running"
        active_executions[execution_id]["logs"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": "info",
            "message": f"Starting workflow execution: {manifest.name}"
        })

        # Execute workflow using the executor
        run = await workflow_executor.execute_workflow(manifest)

        # Update execution results
        active_executions[execution_id]["status"] = getattr(run, "status", "completed")
        active_executions[execution_id]["result"] = {
            "run_id": getattr(run, "id", None),
            "completed_at": run.completed_at.isoformat() if getattr(run, "completed_at", None) else None,
            "duration_ms": int((run.completed_at - run.started_at).total_seconds() * 1000) if getattr(run, "completed_at", None) and getattr(run, "started_at", None) else 0,
            "step_results": [
                {
                    "step_id": step_id,
                    "success": (step_state == "completed"),
                    "error": None,
                }
                for step_id, step_state in (getattr(run, "step_states", {}) or {}).items()
            ]
        }

        # Add completion log
        active_executions[execution_id]["logs"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": "info",
            "message": f"Workflow execution completed with status: {active_executions[execution_id]['status']}"
        })

        logger.info(f"Workflow execution {execution_id} completed")

    except Exception as e:
        logger.error(f"Workflow execution {execution_id} failed: {e}")

        # Update status to failed
        active_executions[execution_id]["status"] = "failed"
        active_executions[execution_id]["error"] = str(e)
        active_executions[execution_id]["logs"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": "error",
            "message": f"Workflow execution failed: {str(e)}"
        })