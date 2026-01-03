"""
Orchestrator module for the Agentic Framework.

This module provides the Lead Agent/Orchestrator service responsible for:
- Workflow ingestion and execution
- Task planning and decomposition
- Subagent spawning and coordination
- Policy enforcement
- Artifact validation
- Commit decision making
- Final synthesis

The orchestrator is the central coordination point for multi-agent workflows.
"""

__version__ = "1.0.0"
__author__ = "Agentic Framework Team"

from .service.config import OrchestratorConfig, config
from .service.models import (
    ArtifactType,
    ClaimVerification,
    CodePatch,
    ProvenanceRecord,
    ResearchSnippet,
    SubagentRole,
    SynthesisResult,
    WorkflowManifest,
    WorkflowStatus,
)
from .service.workflow_engine import WorkflowEngine

__all__ = [
    "OrchestratorConfig",
    "config",
    "ArtifactType",
    "ClaimVerification",
    "CodePatch",
    "ProvenanceRecord",
    "ResearchSnippet",
    "SubagentRole",
    "SynthesisResult",
    "WorkflowManifest",
    "WorkflowStatus",
    "WorkflowEngine",
]
