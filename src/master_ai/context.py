"""
Enhanced Context Manager - Builds the full context window for the Master AI.

This module is responsible for:
1. Loading the current state from the database (Active Businesses, Tasks, History).
2. RAG integration for relevant memory retrieval.
3. Intelligent token budgeting and prioritization.
4. Conversation summarization for long histories.
5. Semantic search for related business context.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional

from src.database.connection import get_db
from src.database.models import BusinessUnit, Task, ConversationMessage, Log
from src.database.vector_store import vector_store, MemoryType
from src.utils.token_manager import (
    ContextBudget, ContextSection, estimate_tokens,
    smart_truncate, ConversationSummarizer
)
from src.utils.structured_logging import get_logger
from config.settings import settings
from sqlalchemy import select

logger = get_logger("context_manager")


class ContextManager:
    """
    Manages the global state visibility for the MasterAI.
    Acts as a 'Sensory System' for the brain with RAG capabilities.
    """
    
    # Model context window limits
    CONTEXT_LIMITS = {
        "llama3.1:70b": 128000,
        "llama3.1:8b": 128000,
        "gemini-pro": 32000,
        "default": 100000
    }
    
    def __init__(self, llm_client=None):
        """
        Initialize context manager.
        
        Args:
            llm_client: Optional LLM client for summarization
        """
        self.llm_client = llm_client
        self._summarizer = None
        self._conversation_cache: List[Dict[str, str]] = []
        self._rolling_summary: str = ""
        self._last_summary_turn = 0
        self._last_facts: Dict[str, Any] = {}  # Store extracted facts
        
        # Get context limit for current model
        model = settings.ollama_model
        self.max_context_tokens = self.CONTEXT_LIMITS.get(
            model, self.CONTEXT_LIMITS["default"]
        )
        
        # Reserve 20% for response
        self.available_tokens = int(self.max_context_tokens * 0.8)
        
        logger.info(
            "Context manager initialized",
            model=model,
            max_tokens=self.available_tokens
        )
    
    @property
    def summarizer(self) -> Optional[ConversationSummarizer]:
        """Lazy-load summarizer when LLM client is available."""
        if self._summarizer is None and self.llm_client:
            self._summarizer = ConversationSummarizer(self.llm_client)
        return self._summarizer
    
    async def build_context(
        self,
        query: str = None,
        include_rag: bool = True,
        focus_areas: List[str] = None
    ) -> str:
        """
        Constructs a comprehensive snapshot of the entire empire.
        
        Args:
            query: Optional query for RAG-enhanced context and focus
            include_rag: Whether to include semantic memory search
            focus_areas: Optional list of areas to prioritize (e.g., ['finance', 'businesses'])
        
        Returns:
            Formatted context string optimized for token budget
        """
        logger.debug(
            "Building context",
            query_provided=query is not None,
            focus_areas=focus_areas
        )
        
        # Initialize budget
        budget = ContextBudget(total_tokens=self.available_tokens)
        
        # Section 1: Current State (always include)
        current_state = await self._build_current_state()
        budget.set_content(ContextSection.CURRENT_STATE, current_state)
        
        # Section 2: Business Data
        business_data = await self._get_businesses_summary(query=query)
        budget.set_content(
            ContextSection.BUSINESS_DATA,
            smart_truncate(business_data, budget.get_budget(ContextSection.BUSINESS_DATA).allocated)
        )
        
        # Section 3: RAG - Relevant Memory (if query provided)
        if include_rag and query:
            relevant_memory = await self._get_relevant_memories(
                query,
                max_tokens=budget.get_budget(ContextSection.RELEVANT_MEMORY).allocated
            )
            budget.set_content(ContextSection.RELEVANT_MEMORY, relevant_memory)
        
        # Section 4: Recent Conversation
        conversation = await self._get_conversation_context(
            max_tokens=budget.get_budget(ContextSection.RECENT_CONVERSATION).allocated
        )
        budget.set_content(ContextSection.RECENT_CONVERSATION, conversation)
        
        # Section 5: Task History
        task_history = await self._get_recent_tasks()
        budget.set_content(
            ContextSection.TASK_HISTORY,
            smart_truncate(task_history, budget.get_budget(ContextSection.TASK_HISTORY).allocated)
        )
        
        # Section 6: Pending Approvals
        pending = await self._get_pending_approvals()
        # Add to current state if there's room
        if pending and budget.can_add(estimate_tokens(pending)):
            current = budget.get_budget(ContextSection.CURRENT_STATE).content
            budget.set_content(ContextSection.CURRENT_STATE, current + "\n\n" + pending)
        
        # Build final context
        full_context = budget.build_context()
        
        # Extract and store facts for validation (optional metadata)
        self._last_facts = self.extract_facts_from_context(full_context)
        
        logger.info(
            "Context built",
            total_tokens=budget.get_total_used(),
            sections_used=len([s for s in budget.sections.values() if s.content]),
            facts_extracted=len(self._last_facts["businesses"])
        )
        
        return full_context
    
    async def _build_current_state(self) -> str:
        """Build the current system state section."""
        sections = [
            f"CURRENT TIME: {datetime.now().isoformat()}",
            f"RISK PROFILE: {settings.risk_profile}",
            f"AUTONOMOUS MODE: {'Enabled' if settings.enable_autonomous_mode else 'Disabled'}",
        ]
        
        # Add quick stats
        async with get_db() as db:
            result = await db.execute(select(BusinessUnit))
            businesses = result.scalars().all()
            
            if businesses:
                total_revenue = sum(b.total_revenue or 0 for b in businesses)
                total_expenses = sum(b.total_expenses or 0 for b in businesses)
                active_count = len([b for b in businesses if b.status.value != "sunset"])
                
                sections.append(f"EMPIRE OVERVIEW:")
                sections.append(f"  - Active Businesses: {active_count}")
                sections.append(f"  - Total Revenue: ${total_revenue:,.2f}")
                sections.append(f"  - Total Profit: ${total_revenue - total_expenses:,.2f}")
        
        return "\n".join(sections)

    async def _get_businesses_summary(self, query: str = None) -> str:
        """
        Fetches all active business units and summarizes their financial performance.
        Optionally filters or prioritizes based on query keywords.
        """
        async with get_db() as db:
            result = await db.execute(
                select(BusinessUnit).order_by(BusinessUnit.created_at.desc())
            )
            businesses = result.scalars().all()
            
            if not businesses:
                return "BUSINESSES: None active. Ready to start new ventures."
            
            # Extract query keywords for relevance scoring if provided
            query_keywords = set()
            if query:
                query_lower = query.lower()
                query_keywords = set([
                    word for word in query_lower.split()
                    if len(word) > 3 and word not in {'what', 'when', 'where', 'which', 'have', 'been', 'this', 'that', 'from'}
                ])
            
            lines = ["BUSINESSES:"]
            for b in businesses:
                revenue = b.total_revenue or 0.0
                expenses = b.total_expenses or 0.0
                profit = revenue - expenses
                status = b.status.value if hasattr(b.status, 'value') else str(b.status)
                
                # Calculate relevance score if query provided
                relevance = 0
                if query_keywords:
                    business_text = f"{b.name} {b.type} {b.description or ''}".lower()
                    relevance = sum(1 for kw in query_keywords if kw in business_text)
                
                business_line = (
                    f"  - {b.name} ({b.type}): {status} | "
                    f"Revenue: ${revenue:,.2f} | Profit: ${profit:,.2f}"
                )
                
                # Mark as relevant if query matches
                if relevance > 0:
                    business_line = "[RELEVANT] " + business_line
                
                lines.append(business_line)
                
                # Add KPIs if available
                if b.kpis:
                    kpi_str = ", ".join([f"{k}: {v}" for k, v in list(b.kpis.items())[:3]])
                    lines.append(f"    KPIs: {kpi_str}")
            
            return "\n".join(lines)
    
    def extract_facts_from_context(self, context: str) -> Dict[str, Any]:
        """
        Extract structured facts from context for validation.
        
        Args:
            context: The context string
            
        Returns:
            Dictionary of extractable facts (numbers, metrics, statuses)
        """
        import re
        
        facts = {
            "numbers": set(),
            "businesses": [],
            "metrics": {},
        }
        
        # Extract all numbers with context
        number_patterns = re.findall(r'\$?[\d,]+\.?\d*%?', context)
        facts["numbers"] = set(number_patterns)
        
        # Extract business names (after "- " in BUSINESSES section)
        business_matches = re.findall(r'- ([^(]+) \(', context)
        facts["businesses"] = [b.strip() for b in business_matches]
        
        # Extract key metrics
        metric_patterns = [
            (r'Active Businesses: (\d+)', 'active_businesses'),
            (r'Total Revenue: \$([\d,.]+)', 'total_revenue'),
            (r'Total Profit: \$([\d,.]+)', 'total_profit'),
        ]
        
        for pattern, key in metric_patterns:
            match = re.search(pattern, context)
            if match:
                facts["metrics"][key] = match.group(1)
        
        return facts
    
    async def _get_relevant_memories(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Retrieve relevant context from vector store using RAG.
        """
        if not query:
            return ""
        
        try:
            context = await vector_store.get_relevant_context(
                query=query,
                max_tokens=max_tokens
            )
            
            if context:
                return f"RELEVANT CONTEXT FROM MEMORY:\n{context}"
            return ""
            
        except Exception as e:
            logger.warning("Failed to retrieve memories", error=str(e))
            return ""
    
    async def _get_conversation_context(self, max_tokens: int = 2000) -> str:
        """
        Get conversation history with intelligent summarization.
        """
        async with get_db() as db:
            result = await db.execute(
                select(ConversationMessage)
                .order_by(ConversationMessage.created_at.desc())
                .limit(50)
            )
            messages = result.scalars().all()
            
            if not messages:
                return ""
            
            # Reverse to chronological order
            messages = list(reversed(messages))
            
            # Check if we need summarization
            total_tokens = sum(estimate_tokens(m.content) for m in messages)
            
            if total_tokens <= max_tokens:
                # Can include full history
                lines = ["RECENT CONVERSATION:"]
                for m in messages[-20:]:  # Last 20 messages
                    role = m.role.upper()
                    content = m.content[:500]  # Truncate individual messages
                    lines.append(f"  {role}: {content}")
                return "\n".join(lines)
            
            # Need to summarize older messages
            if self.summarizer:
                # Split into old (to summarize) and recent (to keep full)
                split_point = len(messages) // 2
                old_messages = [{"role": m.role, "content": m.content} for m in messages[:split_point]]
                recent_messages = messages[split_point:]
                
                # Get or create summary of older messages
                if len(old_messages) > self._last_summary_turn + 10:
                    self._rolling_summary = await self.summarizer.summarize_conversation(
                        old_messages,
                        max_tokens=max_tokens // 3
                    )
                    self._last_summary_turn = len(old_messages)
                
                lines = ["CONVERSATION HISTORY:"]
                if self._rolling_summary:
                    lines.append(f"[Earlier summary]: {self._rolling_summary}")
                
                lines.append("\nRecent messages:")
                for m in recent_messages[-10:]:
                    role = m.role.upper()
                    content = m.content[:300]
                    lines.append(f"  {role}: {content}")
                
                return smart_truncate("\n".join(lines), max_tokens)
            
            # Fallback: simple truncation
            lines = ["RECENT CONVERSATION:"]
            for m in messages[-10:]:
                role = m.role.upper()
                content = m.content[:200]
                lines.append(f"  {role}: {content}")
            return "\n".join(lines)
    
    async def _get_recent_tasks(self) -> str:
        """
        Retrieves recent tasks to show what the system has been doing.
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
                status_icon = "✅" if t.status == "completed" else "⏳" if t.status == "pending" else "❌"
                lines.append(f"  {status_icon} {t.name} ({t.agent or 'system'}) - {t.status}")
            
            return "\n".join(lines)
    
    async def _get_pending_approvals(self) -> str:
        """Get pending items requiring human approval."""
        async with get_db() as db:
            result = await db.execute(
                select(Task)
                .where(Task.status == "pending_approval")
                .order_by(Task.created_at.desc())
            )
            pending = result.scalars().all()
            
            if not pending:
                return ""
            
            lines = ["⚠️ PENDING APPROVALS:"]
            for t in pending:
                lines.append(f"  - [{t.id[:8]}] {t.name}: {t.description or 'No description'}")
            
            return "\n".join(lines)
    
    async def add_to_conversation(
        self,
        role: str,
        content: str,
        persist: bool = True
    ):
        """
        Add a message to conversation history.
        
        Args:
            role: 'user', 'assistant', or 'system'
            content: Message content
            persist: Whether to save to database
        """
        # Add to cache
        self._conversation_cache.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Persist to database
        if persist:
            async with get_db() as db:
                from uuid import uuid4
                msg = ConversationMessage(
                    id=str(uuid4()),
                    role=role,
                    content=content
                )
                add_result = db.add(msg)
                try:
                    import inspect
                    if inspect.isawaitable(add_result):
                        await add_result
                except Exception:
                    # If db.add is synchronous (normal SQLAlchemy), ignore.
                    pass
                await db.commit()
        
        # Store significant exchanges in vector memory
        if len(self._conversation_cache) % 10 == 0:
            await self._store_conversation_to_memory()
    
    async def _store_conversation_to_memory(self):
        """Store conversation summary to long-term memory."""
        if not self._conversation_cache:
            return
        
        try:
            # Create summary for storage
            recent = self._conversation_cache[-10:]
            summary_text = "\n".join([
                f"{m['role']}: {m['content'][:200]}"
                for m in recent
            ])
            
            await vector_store.store_memory(
                text=f"Conversation on {datetime.now().isoformat()}:\n{summary_text}",
                memory_type=MemoryType.CONVERSATION,
                metadata={"turn_count": len(self._conversation_cache)}
            )
            
            logger.debug("Stored conversation to long-term memory")
            
        except Exception as e:
            logger.warning("Failed to store conversation", error=str(e))
    
    async def store_decision(
        self,
        decision: str,
        rationale: str,
        outcome: str = None,
        business_id: str = None
    ):
        """Store a significant decision in long-term memory."""
        await vector_store.store_decision(
            decision=decision,
            rationale=rationale,
            outcome=outcome,
            business_id=business_id
        )
    
    async def get_similar_decisions(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve similar past decisions for reference."""
        return await vector_store.search_memories(
            query=query,
            memory_types=[MemoryType.DECISION],
            top_k=top_k
        )
    
    def clear_conversation_cache(self):
        """Clear the in-memory conversation cache."""
        self._conversation_cache = []
        self._rolling_summary = ""
        self._last_summary_turn = 0
