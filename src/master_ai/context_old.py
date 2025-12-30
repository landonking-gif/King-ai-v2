"""
Context Manager - Builds the full context window for the Master AI.

This module is responsible for:
1. Loading the current state from the database (Active Businesses, Tasks, History).
2. Summarizing and condensing data to fit within the LLM's token limits.
3. Providing "Long-term Memory" via historical conversation retrieval.
"""

from src.database.connection import get_db
from src.database.models import BusinessUnit, Task, ConversationMessage, Log
from config.settings import settings
from sqlalchemy import select

class ContextManager:
    """
    Manages the global state visibility for the MasterAI.
    Acts as a 'Sensory System' for the brain.
    """
    
    # 100k tokens is roughly 300-400k characters.
    # Higher limits allow for more detailed historical context.
    MAX_CONTEXT_TOKENS = 100000 
    
    async def build_context(self) -> str:
        """
        Constructs a comprehensive snapshot of the entire empire.
        
        Returns a formatted string containing:
        - Temporal markers (Current time)
        - Governance settings (Risk profile)
        - Financial health across all business units
        - Operational status (Recent tasks and bottlenecks)
        - Psychological context (Conversation history)
        """
        sections = []
        
        # Add basic system status
        sections.append(f"CURRENT TIME: {self._get_current_time()}")
        sections.append(f"RISK PROFILE: {settings.risk_profile}")
        
        # Gather data from dynamic database tables
        sections.append(await self._get_businesses_summary())
        sections.append(await self._get_recent_tasks())
        sections.append(await self._get_conversation_history())
        sections.append(await self._get_pending_approvals())
        
        full_context = "\n\n".join(sections)
        
        # Safety check: Truncate context if it exceeds model limits to prevent 400 errors
        if len(full_context) > self.MAX_CONTEXT_TOKENS * 4: # 4 chars per token estimate
            full_context = self._truncate_context(full_context)
        
        return full_context
    
    def _get_current_time(self):
        """Returns ISO formatted current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _truncate_context(self, text: str) -> str:
        """Simple truncation logic for keeping context within bounds."""
        return text[:self.MAX_CONTEXT_TOKENS * 4] + "... (TRUNCATED)"

    async def _get_businesses_summary(self) -> str:
        """
        Fetches all active business units and summarizes their financial performance.
        """
        async with get_db() as db:
            # Query all business units, newest first
            result = await db.execute(
                select(BusinessUnit).order_by(BusinessUnit.created_at.desc())
            )
            businesses = result.scalars().all()
            
            if not businesses:
                return "BUSINESSES: None active. Ready to start new ventures."
            
            lines = ["BUSINESSES:"]
            for b in businesses:
                revenue = b.total_revenue or 0.0
                expenses = b.total_expenses or 0.0
                profit = revenue - expenses
                lines.append(
                    f"  - {b.name} ({b.type}): {b.status.value} | "
                    f"Revenue: ${revenue:,.2f} | "
                    f"Profit: ${profit:,.2f}"
                )
            return "\n".join(lines)
    
    async def _get_recent_tasks(self) -> str:
        """
        Retrieves the last 20 tasks to show what the system has been doing.
        """
        async with get_db() as db:
            result = await db.execute(
                select(Task).order_by(Task.created_at.desc()).limit(20)
            )
            tasks = result.scalars().all()
            
            if not tasks:
                return "RECENT TASKS: None"
            
            lines = ["RECENT TASKS:"]
            for t in tasks:
                lines.append(f"  - [{t.status}] {t.name} (agent: {t.agent})")
            return "\n".join(lines)
    
    async def _get_conversation_history(self, limit: int = 10) -> str:
        """
        Provides short-term conversational memory to maintain flow.
        """
        async with get_db() as db:
            result = await db.execute(
                select(ConversationMessage).order_by(ConversationMessage.created_at.desc()).limit(limit)
            )
            messages = result.scalars().all()
            
            if not messages:
                return "CONVERSATION HISTORY: New session started."
            
            lines = ["RECENT CONVERSATION:"]
            # Reverse because DB holds newest-first, but LLM needs chronological order
            for m in reversed(messages):
                lines.append(f"  {m.role.upper()}: {m.content[:200]}...")
            return "\n".join(lines)
    
    async def _get_pending_approvals(self) -> str:
        """
        Flags items that are currently blocked waiting for human input.
        """
        async with get_db() as db:
            result = await db.execute(
                select(Task).where(Task.status == 'pending_approval')
            )
            tasks = result.scalars().all()
            
            if not tasks:
                return "PENDING APPROVALS: None"
            
            lines = ["PENDING APPROVALS:"]
            for t in tasks:
                lines.append(f"  - {t.name}: {t.description}")
            return "\n".join(lines)
