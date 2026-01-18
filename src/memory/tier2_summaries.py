"""
Tier 2 Memory - Session Summaries.

Stores LLM-generated summaries of completed sessions/tasks.
These provide condensed context from previous work.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

from src.utils.structured_logging import get_logger

logger = get_logger("tier2_memory")


@dataclass
class SessionSummary:
    """A summary of a completed session or task."""
    
    id: str
    session_id: str
    task_id: Optional[str] = None
    
    # Summary content
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    decisions_made: List[str] = field(default_factory=list)
    follow_up_items: List[str] = field(default_factory=list)
    
    # Metadata
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: datetime = field(default_factory=datetime.utcnow)
    agents_used: List[str] = field(default_factory=list)
    artifacts_created: List[str] = field(default_factory=list)
    
    # Token stats
    total_tokens: int = 0
    summary_tokens: int = 0
    compression_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "decisions_made": self.decisions_made,
            "follow_up_items": self.follow_up_items,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "agents_used": self.agents_used,
            "artifacts_created": self.artifacts_created,
            "total_tokens": self.total_tokens,
            "summary_tokens": self.summary_tokens,
            "compression_ratio": self.compression_ratio,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionSummary":
        data = data.copy()
        for dt_field in ["started_at", "ended_at"]:
            if dt_field in data and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        return cls(**data)
    
    def to_context_string(self) -> str:
        """Convert to a string for inclusion in context."""
        lines = [f"Session Summary ({self.ended_at.strftime('%Y-%m-%d')}):"]
        lines.append(f"  {self.summary}")
        
        if self.key_findings:
            lines.append("  Key findings:")
            for finding in self.key_findings[:3]:
                lines.append(f"    - {finding}")
        
        if self.decisions_made:
            lines.append("  Decisions:")
            for decision in self.decisions_made[:2]:
                lines.append(f"    - {decision}")
        
        return "\n".join(lines)


# LLM prompt for generating summaries
SUMMARY_PROMPT = """Create a concise summary of this task session.

Goal: {goal}
Agents Used: {agents}
Key Messages:
{messages}

Return a JSON object with this exact structure:
{{
    "summary": "One sentence summary of what was accomplished",
    "key_findings": ["Key finding 1", "Key finding 2", "Key finding 3"],
    "decisions_made": ["Decision 1", "Decision 2"],
    "follow_up_items": ["Follow-up item 1"]
}}

Be concise - each finding should be one sentence. Return ONLY valid JSON."""


class Tier2Memory:
    """
    Tier 2 Memory - Session Summaries.
    
    Stores LLM-generated summaries of completed sessions.
    These summaries are created when:
    - A task completes
    - Tier 1 memory reaches capacity
    - User explicitly ends a session
    
    Features:
    - LLM-powered summarization
    - Automatic compression of old context
    - Fast retrieval by project/business
    """
    
    MAX_SUMMARIES_PER_PROJECT = 20
    
    def __init__(
        self,
        llm_client=None,
        redis_client=None,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize Tier 2 memory.
        
        Args:
            llm_client: LLM client for generating summaries
            redis_client: Redis client for persistence
            storage_path: File-based storage path
        """
        self.llm = llm_client
        self.redis = redis_client
        self.storage_path = storage_path or Path("data/memory/tier2")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Cache per project
        self._cache: Dict[str, List[SessionSummary]] = {}
    
    async def create_summary(
        self,
        project_id: str,
        session_id: str,
        goal: str,
        messages: List[Dict[str, str]],
        agents_used: List[str],
        task_id: Optional[str] = None,
        artifacts_created: Optional[List[str]] = None,
    ) -> SessionSummary:
        """
        Create a session summary using LLM.
        
        Args:
            project_id: Project/business identifier
            session_id: Session identifier
            goal: What the session was trying to accomplish
            messages: List of messages to summarize
            agents_used: Agents that were invoked
            task_id: Associated task ID (optional)
            artifacts_created: IDs of artifacts created
            
        Returns:
            Created session summary
        """
        # Format messages for prompt
        message_text = "\n".join([
            f"[{m.get('role', 'unknown').upper()}]: {m.get('content', '')[:200]}"
            for m in messages[-10:]  # Last 10 messages
        ])
        
        # Calculate original token count (estimate)
        original_tokens = sum(len(m.get("content", "").split()) * 1.3 for m in messages)
        
        # Generate summary with LLM
        summary_data = await self._generate_summary(
            goal=goal,
            agents=", ".join(agents_used) if agents_used else "None",
            messages=message_text,
        )
        
        # Create summary object
        summary = SessionSummary(
            id=f"sum_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            session_id=session_id,
            task_id=task_id,
            summary=summary_data.get("summary", f"Completed: {goal}"),
            key_findings=summary_data.get("key_findings", []),
            decisions_made=summary_data.get("decisions_made", []),
            follow_up_items=summary_data.get("follow_up_items", []),
            agents_used=agents_used,
            artifacts_created=artifacts_created or [],
            total_tokens=int(original_tokens),
            summary_tokens=len(summary_data.get("summary", "").split()),
        )
        
        # Calculate compression ratio
        if summary.total_tokens > 0:
            summary.compression_ratio = summary.summary_tokens / summary.total_tokens
        
        # Store summary
        await self._store_summary(project_id, summary)
        
        logger.info(
            "Created session summary",
            project_id=project_id,
            session_id=session_id,
            compression_ratio=f"{summary.compression_ratio:.2%}",
        )
        
        return summary
    
    async def get_summaries(
        self,
        project_id: str,
        limit: Optional[int] = None
    ) -> List[SessionSummary]:
        """
        Get summaries for a project.
        
        Args:
            project_id: Project/business identifier
            limit: Maximum summaries to return
            
        Returns:
            List of summaries, most recent first
        """
        summaries = await self._load_summaries(project_id)
        limit = limit or self.MAX_SUMMARIES_PER_PROJECT
        return summaries[-limit:]
    
    async def get_context_string(
        self,
        project_id: str,
        max_summaries: int = 5
    ) -> str:
        """
        Get summaries as formatted context string.
        
        Args:
            project_id: Project/business identifier
            max_summaries: Maximum summaries to include
            
        Returns:
            Formatted context string
        """
        summaries = await self.get_summaries(project_id, limit=max_summaries)
        
        if not summaries:
            return "No previous session summaries available."
        
        lines = ["Previous session summaries:"]
        for summary in summaries:
            lines.append(summary.to_context_string())
            lines.append("")
        
        return "\n".join(lines)
    
    async def search_summaries(
        self,
        project_id: str,
        query: str,
        limit: int = 5
    ) -> List[SessionSummary]:
        """
        Search summaries by keyword.
        
        Args:
            project_id: Project identifier
            query: Search query
            limit: Maximum results
            
        Returns:
            Matching summaries
        """
        summaries = await self._load_summaries(project_id)
        query_lower = query.lower()
        
        matches = []
        for summary in summaries:
            # Search in summary text and findings
            searchable = " ".join([
                summary.summary,
                " ".join(summary.key_findings),
                " ".join(summary.decisions_made),
            ]).lower()
            
            if query_lower in searchable:
                matches.append(summary)
                if len(matches) >= limit:
                    break
        
        return matches
    
    async def get_follow_up_items(
        self,
        project_id: str
    ) -> List[str]:
        """Get all pending follow-up items from summaries."""
        summaries = await self._load_summaries(project_id)
        
        items = []
        for summary in summaries:
            items.extend(summary.follow_up_items)
        
        return items
    
    async def delete_summary(
        self,
        project_id: str,
        summary_id: str
    ) -> bool:
        """Delete a specific summary."""
        summaries = await self._load_summaries(project_id)
        
        original_count = len(summaries)
        summaries = [s for s in summaries if s.id != summary_id]
        
        if len(summaries) < original_count:
            await self._save_summaries(project_id, summaries)
            return True
        
        return False
    
    # Private methods
    
    async def _generate_summary(
        self,
        goal: str,
        agents: str,
        messages: str
    ) -> Dict[str, Any]:
        """Generate summary using LLM."""
        if not self.llm:
            # Fallback when no LLM available
            return {
                "summary": f"Completed task: {goal[:100]}",
                "key_findings": [],
                "decisions_made": [],
                "follow_up_items": [],
            }
        
        prompt = SUMMARY_PROMPT.format(
            goal=goal,
            agents=agents,
            messages=messages,
        )
        
        try:
            response = await self.llm.complete(prompt)
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")
        
        return {
            "summary": f"Completed: {goal[:100]}",
            "key_findings": [],
            "decisions_made": [],
            "follow_up_items": [],
        }
    
    async def _store_summary(
        self,
        project_id: str,
        summary: SessionSummary
    ) -> None:
        """Store a summary."""
        summaries = await self._load_summaries(project_id)
        summaries.append(summary)
        
        # Trim to max
        if len(summaries) > self.MAX_SUMMARIES_PER_PROJECT:
            summaries = summaries[-self.MAX_SUMMARIES_PER_PROJECT:]
        
        await self._save_summaries(project_id, summaries)
        self._cache[project_id] = summaries
    
    async def _load_summaries(self, project_id: str) -> List[SessionSummary]:
        """Load summaries from cache or storage."""
        if project_id in self._cache:
            return self._cache[project_id]
        
        summaries = []
        
        if self.redis:
            data = await self.redis.get(f"tier2:{project_id}")
            if data:
                summaries = [SessionSummary.from_dict(s) for s in json.loads(data)]
        else:
            file_path = self.storage_path / f"{project_id}.json"
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)
                    summaries = [SessionSummary.from_dict(s) for s in data]
        
        self._cache[project_id] = summaries
        return summaries
    
    async def _save_summaries(
        self,
        project_id: str,
        summaries: List[SessionSummary]
    ) -> None:
        """Persist summaries to storage."""
        data = [s.to_dict() for s in summaries]
        
        if self.redis:
            await self.redis.set(
                f"tier2:{project_id}",
                json.dumps(data),
                ex=86400 * 60  # 60 day expiry
            )
        else:
            file_path = self.storage_path / f"{project_id}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
