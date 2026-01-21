"""
Diary and Reflection system for King AI agents.

Inspired by Claude Diary plugin, adapted for agentic memory learning.
Generates diary entries from agent sessions and performs reflection to update long-term memory.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import uuid

from src.utils.structured_logging import get_logger
from src.memory.tier3_longterm import LongTermMemory, MemoryCategory

logger = get_logger("diary_reflection")


class AgentDiary:
    """Manages diary entries for agent sessions."""

    def __init__(self, project_id: str, memory_dir: Path = None):
        self.project_id = project_id
        self.memory_dir = memory_dir or Path(".king_ai/memory")
        self.diary_dir = self.memory_dir / "diary"
        self.diary_dir.mkdir(parents=True, exist_ok=True)
        self.processed_log = self.memory_dir / "processed.log"

    def generate_diary_entry(self, session_data: Dict[str, Any]) -> str:
        """
        Generate a diary entry from session data.

        Args:
            session_data: Dictionary containing session information including:
                - agent_id: ID of the agent
                - task: Task description
                - actions: List of actions taken
                - decisions: Important decisions made
                - outcomes: Results achieved
                - challenges: Problems encountered
                - solutions: Solutions implemented
                - learnings: Key learnings

        Returns:
            Path to the created diary file
        """
        date = datetime.now().strftime("%Y-%m-%d")
        session_id = str(uuid.uuid4())[:8]
        filename = f"{date}-session-{session_id}.md"
        filepath = self.diary_dir / filename

        content = f"""# Diary Entry: {date} - Session {session_id}

## Agent Information
- Agent ID: {session_data.get('agent_id', 'Unknown')}
- Project: {self.project_id}

## Task Summary
{session_data.get('task', 'No task specified')}

## Work Done
{self._format_actions(session_data.get('actions', []))}

## Design Decisions
{self._format_list(session_data.get('decisions', []))}

## User Preferences
{self._format_list(session_data.get('preferences', []))}

## Code Review Feedback
{self._format_list(session_data.get('feedback', []))}

## Challenges
{self._format_list(session_data.get('challenges', []))}

## Solutions
{self._format_list(session_data.get('solutions', []))}

## Code Patterns
{self._format_list(session_data.get('patterns', []))}

## Learnings
{self._format_list(session_data.get('learnings', []))}
"""

        filepath.write_text(content)
        logger.info(f"Diary entry created: {filepath}")
        return str(filepath)

    def _format_actions(self, actions: List[Dict]) -> str:
        """Format actions list for diary."""
        if not actions:
            return "No actions recorded."

        formatted = []
        for action in actions:
            action_type = action.get('type', 'Unknown')
            description = action.get('description', 'No description')
            result = action.get('result', '')
            formatted.append(f"- **{action_type}**: {description}")
            if result:
                formatted.append(f"  - Result: {result}")

        return "\n".join(formatted)

    def _format_list(self, items: List[str]) -> str:
        """Format list items for diary."""
        if not items:
            return "None recorded."

        return "\n".join(f"- {item}" for item in items)

    def get_unprocessed_entries(self) -> List[Path]:
        """Get list of unprocessed diary entries."""
        processed = set()
        if self.processed_log.exists():
            processed = set(self.processed_log.read_text().splitlines())

        all_entries = list(self.diary_dir.glob("*.md"))
        return [entry for entry in all_entries if entry.name not in processed]

    def mark_processed(self, entries: List[str]):
        """Mark diary entries as processed."""
        current = set()
        if self.processed_log.exists():
            current = set(self.processed_log.read_text().splitlines())

        current.update(entries)
        self.processed_log.write_text("\n".join(sorted(current)))


class ReflectionEngine:
    """Performs reflection on diary entries to extract patterns and update memory."""

    def __init__(self, project_id: str, memory_manager):
        self.project_id = project_id
        self.memory_manager = memory_manager
        self.diary = AgentDiary(project_id)

    async def perform_reflection(self, entries: Optional[List[Path]] = None) -> Dict[str, Any]:
        """
        Analyze diary entries and extract patterns.

        Args:
            entries: Specific entries to analyze, or None for unprocessed

        Returns:
            Dictionary with analysis results
        """
        if entries is None:
            entries = self.diary.get_unprocessed_entries()

        if not entries:
            return {"message": "No new diary entries to reflect on."}

        analysis = {
            "entries_analyzed": len(entries),
            "patterns": {},
            "insights": [],
            "memory_updates": []
        }

        # Read and analyze entries
        all_content = []
        for entry_path in entries:
            content = entry_path.read_text()
            all_content.append(content)

            # Extract patterns from this entry
            entry_patterns = self._analyze_entry(content)
            for category, patterns in entry_patterns.items():
                if category not in analysis["patterns"]:
                    analysis["patterns"][category] = []
                analysis["patterns"][category].extend(patterns)

        # Synthesize insights
        analysis["insights"] = self._synthesize_insights(analysis["patterns"])

        # Create long-term memories
        analysis["memory_updates"] = await self._create_memories(analysis["insights"])

        # Mark entries as processed
        self.diary.mark_processed([entry.name for entry in entries])

        logger.info(f"Reflection completed: {len(entries)} entries analyzed, {len(analysis['memory_updates'])} memories created")
        return analysis

    def _analyze_entry(self, content: str) -> Dict[str, List[str]]:
        """Analyze a single diary entry for patterns."""
        patterns = {
            "preferences": [],
            "decisions": [],
            "challenges": [],
            "solutions": [],
            "patterns": [],
            "learnings": []
        }

        lines = content.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('## '):
                current_section = line[3:].lower().replace(' ', '_')
            elif line.startswith('- ') and current_section in patterns:
                patterns[current_section].append(line[2:])

        return patterns

    def _synthesize_insights(self, patterns: Dict[str, List[str]]) -> List[str]:
        """Synthesize insights from patterns."""
        insights = []

        # Find recurring patterns (appearing in 2+ entries would be better, but for now any)
        for category, items in patterns.items():
            if len(items) > 1:
                insights.append(f"Recurring {category}: {', '.join(items[:3])}")

        # Generate general insights
        if patterns.get("solutions"):
            insights.append("Problem-solving approaches: " + ", ".join(patterns["solutions"][:2]))

        if patterns.get("learnings"):
            insights.append("Key learnings: " + ", ".join(patterns["learnings"][:2]))

        return insights

    async def _create_memories(self, insights: List[str]) -> List[LongTermMemory]:
        """Create long-term memories from insights."""
        memories = []

        for insight in insights:
            memory = LongTermMemory(
                id=str(uuid.uuid4()),
                project_id=self.project_id,
                content=insight,
                category=MemoryCategory.PREFERENCE,  # Default category
                source="reflection",
                importance=0.7,
                tags=["reflection", "agent_learning"]
            )
            memories.append(memory)

            # Add to memory manager
            await self.memory_manager.store_memory(
                project_id=self.project_id,
                content=insight,
                category=MemoryCategory.PREFERENCE,
                source="reflection",
                importance=0.7
            )

        return memories