"""
Lineage Tracker.

Tracks artifact lineage and provides visualization of provenance chains.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

from src.artifacts.models import Artifact, ProvenanceRecord
from src.artifacts.store import ArtifactStore, get_artifact_store
from src.utils.structured_logging import get_logger

logger = get_logger("lineage_tracker")


@dataclass
class LineageNode:
    """A node in the lineage graph."""
    artifact_id: str
    artifact_type: str
    name: str
    created_by: str
    created_at: datetime
    children: List["LineageNode"] = field(default_factory=list)
    depth: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "name": self.name,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "children": [c.to_dict() for c in self.children],
            "depth": self.depth,
        }


@dataclass
class LineageGraph:
    """Complete lineage graph for an artifact."""
    root_id: str
    nodes: Dict[str, LineageNode] = field(default_factory=dict)
    edges: List[tuple] = field(default_factory=list)  # (parent_id, child_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_id": self.root_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": self.edges,
        }
    
    def to_mermaid(self) -> str:
        """Generate Mermaid diagram syntax for visualization."""
        lines = ["graph TD"]
        
        for node_id, node in self.nodes.items():
            # Escape special characters
            label = node.name.replace('"', '\\"')[:30]
            lines.append(f'    {node_id[:8]}["{label}"]')
        
        for parent_id, child_id in self.edges:
            lines.append(f"    {parent_id[:8]} --> {child_id[:8]}")
        
        return "\n".join(lines)


class LineageTracker:
    """
    Tracks and visualizes artifact lineage.
    
    Provides:
    - Full lineage chain traversal
    - Impact analysis (what depends on this?)
    - Lineage graph generation
    - Compliance reporting
    """
    
    def __init__(self, store: Optional[ArtifactStore] = None):
        """Initialize with artifact store."""
        self.store = store or get_artifact_store()
    
    async def get_ancestors(
        self,
        artifact_id: str,
        max_depth: int = 10
    ) -> List[Artifact]:
        """
        Get all ancestor artifacts in the lineage chain.
        
        Args:
            artifact_id: Starting artifact
            max_depth: Maximum depth to traverse
            
        Returns:
            List of ancestor artifacts (oldest first)
        """
        ancestors = []
        seen: Set[str] = set()
        
        async def traverse(aid: str, depth: int):
            if depth > max_depth or aid in seen:
                return
            seen.add(aid)
            
            artifact = await self.store.get(aid)
            if not artifact:
                return
            
            # Traverse parents first (to get oldest first)
            for parent_id in artifact.parent_artifact_ids:
                await traverse(parent_id, depth + 1)
            
            ancestors.append(artifact)
        
        await traverse(artifact_id, 0)
        return ancestors[:-1]  # Exclude the starting artifact
    
    async def get_descendants(
        self,
        artifact_id: str,
        max_depth: int = 10
    ) -> List[Artifact]:
        """
        Get all descendant artifacts (impact analysis).
        
        Returns:
            List of descendant artifacts
        """
        descendants = []
        seen: Set[str] = set()
        
        async def traverse(aid: str, depth: int):
            if depth > max_depth or aid in seen:
                return
            seen.add(aid)
            
            children = await self.store.get_children(aid)
            for child in children:
                descendants.append(child)
                await traverse(child.id, depth + 1)
        
        await traverse(artifact_id, 0)
        return descendants
    
    async def build_lineage_graph(
        self,
        artifact_id: str,
        include_ancestors: bool = True,
        include_descendants: bool = True,
        max_depth: int = 5
    ) -> LineageGraph:
        """
        Build a complete lineage graph for an artifact.
        
        Args:
            artifact_id: Central artifact
            include_ancestors: Include parent lineage
            include_descendants: Include child artifacts
            max_depth: Maximum depth in each direction
            
        Returns:
            LineageGraph with nodes and edges
        """
        graph = LineageGraph(root_id=artifact_id)
        
        # Add central artifact
        root = await self.store.get(artifact_id)
        if root:
            graph.nodes[artifact_id] = LineageNode(
                artifact_id=root.id,
                artifact_type=root.artifact_type.value,
                name=root.name,
                created_by=root.created_by,
                created_at=root.created_at,
                depth=0,
            )
        
        # Add ancestors
        if include_ancestors:
            ancestors = await self.get_ancestors(artifact_id, max_depth)
            for i, ancestor in enumerate(ancestors):
                graph.nodes[ancestor.id] = LineageNode(
                    artifact_id=ancestor.id,
                    artifact_type=ancestor.artifact_type.value,
                    name=ancestor.name,
                    created_by=ancestor.created_by,
                    created_at=ancestor.created_at,
                    depth=-(len(ancestors) - i),
                )
            
            # Add edges for ancestors
            for ancestor in ancestors:
                for parent_id in ancestor.parent_artifact_ids:
                    if parent_id in graph.nodes:
                        graph.edges.append((parent_id, ancestor.id))
        
        # Add descendants
        if include_descendants:
            descendants = await self.get_descendants(artifact_id, max_depth)
            for descendant in descendants:
                if descendant.id not in graph.nodes:
                    graph.nodes[descendant.id] = LineageNode(
                        artifact_id=descendant.id,
                        artifact_type=descendant.artifact_type.value,
                        name=descendant.name,
                        created_by=descendant.created_by,
                        created_at=descendant.created_at,
                        depth=1,  # Will be updated below
                    )
                
                for parent_id in descendant.parent_artifact_ids:
                    if parent_id in graph.nodes:
                        graph.edges.append((parent_id, descendant.id))
        
        return graph
    
    async def get_provenance_chain(
        self,
        artifact_id: str
    ) -> List[ProvenanceRecord]:
        """
        Get the full provenance chain for an artifact.
        
        Returns:
            List of provenance records in chronological order
        """
        ancestors = await self.get_ancestors(artifact_id)
        artifact = await self.store.get(artifact_id)
        
        if artifact:
            ancestors.append(artifact)
        
        chain = []
        for a in ancestors:
            if a.provenance:
                chain.append(a.provenance)
        
        return chain
    
    async def generate_audit_report(
        self,
        artifact_id: str
    ) -> Dict[str, Any]:
        """
        Generate a compliance audit report for an artifact.
        
        Includes:
        - Full lineage chain
        - All actors involved
        - All tools used
        - Timestamps and durations
        """
        graph = await self.build_lineage_graph(artifact_id)
        chain = await self.get_provenance_chain(artifact_id)
        
        actors = set()
        tools = set()
        models = set()
        total_tokens = 0
        total_duration_ms = 0
        
        for record in chain:
            actors.add(f"{record.actor_type}:{record.actor_id}")
            tools.update(record.tool_ids)
            if record.model_used:
                models.add(record.model_used)
            total_tokens += record.tokens_used
            total_duration_ms += record.duration_ms
        
        artifact = await self.store.get(artifact_id)
        
        return {
            "artifact_id": artifact_id,
            "artifact_name": artifact.name if artifact else "Unknown",
            "generated_at": datetime.utcnow().isoformat(),
            "lineage": {
                "ancestor_count": len([n for n in graph.nodes.values() if n.depth < 0]),
                "descendant_count": len([n for n in graph.nodes.values() if n.depth > 0]),
                "graph": graph.to_dict(),
            },
            "provenance": {
                "record_count": len(chain),
                "actors": list(actors),
                "tools_used": list(tools),
                "models_used": list(models),
                "total_tokens": total_tokens,
                "total_duration_ms": total_duration_ms,
            },
            "mermaid_diagram": graph.to_mermaid(),
        }
    
    async def find_common_ancestor(
        self,
        artifact_ids: List[str]
    ) -> Optional[Artifact]:
        """
        Find the common ancestor of multiple artifacts.
        
        Returns:
            The most recent common ancestor, or None
        """
        if not artifact_ids:
            return None
        
        # Get ancestor sets for each artifact
        ancestor_sets: List[Set[str]] = []
        for aid in artifact_ids:
            ancestors = await self.get_ancestors(aid)
            ancestor_sets.append({a.id for a in ancestors})
        
        # Find intersection
        if not ancestor_sets:
            return None
        
        common = ancestor_sets[0]
        for s in ancestor_sets[1:]:
            common = common.intersection(s)
        
        if not common:
            return None
        
        # Return the most recent common ancestor
        for aid in artifact_ids:
            ancestors = await self.get_ancestors(aid)
            for ancestor in reversed(ancestors):
                if ancestor.id in common:
                    return ancestor
        
        return None


# Global tracker instance
_lineage_tracker: Optional[LineageTracker] = None


def get_lineage_tracker() -> LineageTracker:
    """Get or create the global lineage tracker."""
    global _lineage_tracker
    if _lineage_tracker is None:
        _lineage_tracker = LineageTracker()
    return _lineage_tracker
