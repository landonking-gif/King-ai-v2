"""
Typed Artifacts System.

Provides structured artifacts with provenance tracking for full traceability
of all outputs in the King AI system.
"""

from src.artifacts.models import (
    ArtifactType,
    SafetyClass,
    Artifact,
    ResearchArtifact,
    CodeArtifact,
    ContentArtifact,
    BusinessPlanArtifact,
    FinanceArtifact,
    LegalArtifact,
    ProvenanceRecord,
)
from src.artifacts.store import ArtifactStore
from src.artifacts.lineage import LineageTracker

__all__ = [
    "ArtifactType",
    "SafetyClass",
    "Artifact",
    "ResearchArtifact",
    "CodeArtifact",
    "ContentArtifact",
    "BusinessPlanArtifact",
    "FinanceArtifact",
    "LegalArtifact",
    "ProvenanceRecord",
    "ArtifactStore",
    "LineageTracker",
]
