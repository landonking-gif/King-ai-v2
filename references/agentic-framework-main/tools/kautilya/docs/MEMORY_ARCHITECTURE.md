# Kautilya Memory Architecture Design

## Executive Summary

This document outlines a modern memory management system for Kautilya, incorporating the latest research in agentic AI memory (2025). The architecture enables context-aware conversations across sessions, personalized responses, and intelligent memory retrieval.

---

## Current State Analysis

### What Exists Today
- **In-memory ChatHistory**: Simple list of messages, max 50, lost on session end
- **No persistence**: Each CLI invocation starts fresh
- **No cross-session context**: Previous conversations completely forgotten
- **Memory Service exists**: Separate service with Redis/Postgres/Vector DB - but NOT integrated

### Key Gaps
1. No session persistence
2. No user preference learning
3. No semantic retrieval (only recency-based)
4. No memory consolidation (episodic → semantic)
5. No contradiction handling

---

## Proposed Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Kautilya CLI                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Memory Manager                                │    │
│  │  ┌─────────────┬─────────────┬─────────────┬───────────────────┐   │    │
│  │  │   Working   │  Episodic   │  Semantic   │    Procedural     │   │    │
│  │  │   Memory    │   Memory    │   Memory    │     Memory        │   │    │
│  │  ├─────────────┼─────────────┼─────────────┼───────────────────┤   │    │
│  │  │ Current     │ Past        │ User prefs  │ Learned patterns  │   │    │
│  │  │ session     │ sessions    │ Facts       │ Successful tools  │   │    │
│  │  │ context     │ Interactions│ Knowledge   │ Query strategies  │   │    │
│  │  │             │             │ graph       │                   │   │    │
│  │  └──────┬──────┴──────┬──────┴──────┬──────┴─────────┬─────────┘   │    │
│  │         │             │             │                │             │    │
│  │         └─────────────┴─────────────┴────────────────┘             │    │
│  │                              │                                      │    │
│  │                    ┌─────────▼─────────┐                           │    │
│  │                    │  Retrieval Engine │                           │    │
│  │                    │  - Relevance      │                           │    │
│  │                    │  - Recency        │                           │    │
│  │                    │  - Importance     │                           │    │
│  │                    └─────────┬─────────┘                           │    │
│  │                              │                                      │    │
│  └──────────────────────────────┼──────────────────────────────────────┘    │
│                                 │                                           │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     Storage Layer         │
                    ├───────────────────────────┤
                    │ Redis    - Working/Hot    │
                    │ Postgres - Structured     │
                    │ ChromaDB - Vectors        │
                    │ SQLite   - Local fallback │
                    └───────────────────────────┘
```

---

## Memory Types (Cognitive Science Inspired)

### 1. Working Memory
**Purpose**: Current conversation context
**Storage**: In-memory (Python) + Redis (hot cache)
**TTL**: Session duration

```python
@dataclass
class WorkingMemory:
    session_id: str
    messages: List[Message]          # Current conversation
    active_context: Dict[str, Any]   # Current topic, entities mentioned
    pending_tools: List[ToolCall]    # Tools being executed
    attention_focus: List[str]       # Key topics user is focused on
```

**Operations**:
- Add message
- Update attention focus
- Get recent context (last N turns)
- Clear on session end (but archive to episodic)

---

### 2. Episodic Memory
**Purpose**: "What happened" - specific past interactions with full context
**Storage**: Postgres (metadata) + ChromaDB (embeddings)
**Retention**: 90 days default, importance-based archival

```python
@dataclass
class EpisodicMemory:
    episode_id: str
    session_id: str
    timestamp: datetime

    # Context
    user_query: str
    agent_response: str
    tools_used: List[str]
    sources_consulted: List[SourceEntry]

    # Metadata
    topic_tags: List[str]           # Auto-extracted topics
    entities: List[str]             # Named entities mentioned
    sentiment: str                  # positive/neutral/negative
    importance_score: float         # 0.0 - 1.0

    # Embeddings (stored in vector DB)
    query_embedding: List[float]
    response_embedding: List[float]
```

**Operations**:
- Store interaction after each turn
- Retrieve by semantic similarity
- Retrieve by topic/entity
- Retrieve by time range
- Update importance score based on references

---

### 3. Semantic Memory
**Purpose**: "How things work" - generalized knowledge and user preferences
**Storage**: Postgres (structured) + ChromaDB (embeddings)
**Retention**: Permanent, with confidence decay

```python
@dataclass
class SemanticMemory:
    memory_id: str
    memory_type: Literal["fact", "preference", "belief", "knowledge"]

    # Content
    subject: str                    # What this is about
    predicate: str                  # Relationship type
    object: str                     # The value/fact

    # Example: ("user", "prefers", "technical analysis over fundamentals")
    # Example: ("gold_price", "correlates_with", "USD weakness")

    # Metadata
    confidence: float               # 0.0 - 1.0, decays over time
    source_episodes: List[str]      # Episodic memories this was derived from
    created_at: datetime
    last_confirmed: datetime        # When last reinforced
    contradiction_count: int        # Times contradicted

    # For knowledge graph
    related_memories: List[str]     # Links to related semantic memories
```

**Memory Categories**:

| Category | Example | Update Trigger |
|----------|---------|----------------|
| **User Preferences** | "User prefers concise answers" | Explicit feedback or pattern detection |
| **User Expertise** | "User understands Python, learning Rust" | Query complexity analysis |
| **Domain Facts** | "Company X acquired Company Y in 2024" | Extracted from conversations |
| **User Context** | "User works in fintech" | Stated or inferred |
| **Beliefs** | "User is bullish on AI stocks" | Stated opinions with confidence |

**Operations**:
- Extract facts from episodic memories (consolidation)
- Update confidence on reinforcement
- Decay confidence over time
- Handle contradictions (update or create competing beliefs)
- Query by subject/predicate/object

---

### 4. Procedural Memory
**Purpose**: "How to do things" - learned patterns and successful strategies
**Storage**: Postgres (structured)
**Retention**: Permanent, with effectiveness scoring

```python
@dataclass
class ProceduralMemory:
    procedure_id: str
    procedure_type: Literal["tool_pattern", "query_strategy", "response_style"]

    # Pattern
    trigger_pattern: str            # When to apply this procedure
    action_sequence: List[str]      # What to do

    # Example:
    # trigger: "market analysis query"
    # action: ["web_search recent news", "check trusted sources", "synthesize with historical context"]

    # Effectiveness
    success_count: int
    failure_count: int
    effectiveness_score: float      # success / (success + failure)

    # Context
    applicable_domains: List[str]   # finance, tech, general, etc.
    source_episodes: List[str]      # Episodes where this was learned
```

**Operations**:
- Learn new procedure from successful interaction
- Update effectiveness on use
- Retrieve applicable procedures for current query
- Prune low-effectiveness procedures

---

## Memory Manager Implementation

### Core Class Structure

```python
class MemoryManager:
    """
    Central memory management for Kautilya.
    Coordinates all memory types and storage backends.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.user_id = config.user_id or self._get_or_create_user_id()
        self.session_id = self._create_session_id()

        # Memory stores
        self.working = WorkingMemoryStore(redis_client)
        self.episodic = EpisodicMemoryStore(postgres_client, vector_db)
        self.semantic = SemanticMemoryStore(postgres_client, vector_db)
        self.procedural = ProceduralMemoryStore(postgres_client)

        # Background consolidation
        self.consolidator = MemoryConsolidator(self)

    # === Core Operations ===

    async def remember(self, interaction: Interaction) -> None:
        """Store an interaction in appropriate memory stores."""

        # 1. Update working memory
        await self.working.add(interaction)

        # 2. Create episodic memory
        episode = await self._create_episode(interaction)
        await self.episodic.store(episode)

        # 3. Extract and update semantic memories (async background)
        asyncio.create_task(self._extract_semantics(interaction))

        # 4. Update procedural memory if tools were used
        if interaction.tools_used:
            asyncio.create_task(self._update_procedures(interaction))

    async def recall(self, query: str, context: Dict) -> MemoryContext:
        """Retrieve relevant memories for a query."""

        # 1. Get working memory (current session)
        working_context = await self.working.get_recent(limit=10)

        # 2. Search episodic memory (semantic + recency + importance)
        relevant_episodes = await self.episodic.search(
            query=query,
            user_id=self.user_id,
            limit=5,
            weights={"relevance": 0.5, "recency": 0.3, "importance": 0.2}
        )

        # 3. Get relevant semantic memories
        semantic_facts = await self.semantic.query(
            query=query,
            user_id=self.user_id,
            min_confidence=0.6
        )

        # 4. Get applicable procedures
        procedures = await self.procedural.get_applicable(
            query_type=self._classify_query(query),
            domain=context.get("domain")
        )

        # 5. Compose memory context
        return MemoryContext(
            working=working_context,
            episodes=relevant_episodes,
            facts=semantic_facts,
            procedures=procedures,
            user_profile=await self.get_user_profile()
        )

    async def get_user_profile(self) -> UserProfile:
        """Get aggregated user profile from semantic memory."""
        preferences = await self.semantic.query_by_subject(
            subject="user",
            predicate_types=["prefers", "expertise", "interested_in"]
        )
        return UserProfile.from_semantic_memories(preferences)
```

---

## Retrieval Engine

### Multi-Factor Scoring

```python
class RetrievalEngine:
    """
    Scores and ranks memories using multiple factors.
    Inspired by Generative Agents (Stanford) paper.
    """

    def score_memory(
        self,
        memory: Union[EpisodicMemory, SemanticMemory],
        query: str,
        current_time: datetime
    ) -> float:
        """
        Calculate composite score for memory relevance.

        Score = w1 * relevance + w2 * recency + w3 * importance
        """

        # 1. Relevance (semantic similarity)
        relevance = self._cosine_similarity(
            self._embed(query),
            memory.embedding
        )

        # 2. Recency (exponential decay)
        hours_ago = (current_time - memory.timestamp).total_seconds() / 3600
        recency = math.exp(-self.decay_rate * hours_ago)

        # 3. Importance (pre-computed or derived)
        importance = memory.importance_score

        # Weighted combination
        score = (
            self.weights["relevance"] * relevance +
            self.weights["recency"] * recency +
            self.weights["importance"] * importance
        )

        return score

    def retrieve(
        self,
        query: str,
        memory_store: MemoryStore,
        top_k: int = 10
    ) -> List[ScoredMemory]:
        """Retrieve top-k relevant memories."""

        # Get candidate memories (pre-filter by vector similarity)
        candidates = memory_store.vector_search(query, limit=top_k * 3)

        # Score each candidate
        scored = [
            ScoredMemory(memory=m, score=self.score_memory(m, query, now()))
            for m in candidates
        ]

        # Return top-k by composite score
        return sorted(scored, key=lambda x: x.score, reverse=True)[:top_k]
```

---

## Memory Consolidation

### Episodic → Semantic Transformation

```python
class MemoryConsolidator:
    """
    Background process that consolidates episodic memories into semantic knowledge.
    Runs periodically or on-demand.
    """

    async def consolidate(self, user_id: str) -> ConsolidationReport:
        """
        Analyze recent episodic memories and extract semantic knowledge.
        """

        # 1. Get unconsolidated episodes
        episodes = await self.episodic.get_unconsolidated(
            user_id=user_id,
            since=timedelta(days=7)
        )

        # 2. Cluster episodes by topic
        clusters = self._cluster_by_topic(episodes)

        # 3. Extract patterns from each cluster
        for cluster in clusters:
            # Use LLM to extract facts and preferences
            extractions = await self._extract_knowledge(cluster)

            for extraction in extractions:
                # Check for existing semantic memory
                existing = await self.semantic.find_similar(extraction)

                if existing:
                    # Reinforce or contradict
                    if self._is_consistent(existing, extraction):
                        await self.semantic.reinforce(existing.id)
                    else:
                        await self.semantic.add_contradiction(existing.id, extraction)
                else:
                    # Create new semantic memory
                    await self.semantic.create(extraction)

        # 4. Mark episodes as consolidated
        await self.episodic.mark_consolidated([e.id for e in episodes])

        return ConsolidationReport(...)

    async def _extract_knowledge(self, episodes: List[EpisodicMemory]) -> List[SemanticExtraction]:
        """Use LLM to extract semantic knowledge from episode cluster."""

        prompt = f"""
        Analyze these past interactions and extract:
        1. User preferences (communication style, detail level, interests)
        2. User expertise areas
        3. Facts or knowledge mentioned
        4. Recurring topics or concerns

        Interactions:
        {self._format_episodes(episodes)}

        Output as JSON:
        {{
            "preferences": [{{"subject": "user", "predicate": "prefers", "object": "...", "confidence": 0.8}}],
            "expertise": [...],
            "facts": [...],
            "topics": [...]
        }}
        """

        response = await self.llm.complete(prompt)
        return self._parse_extractions(response)
```

---

## User Profile System

### Profile Structure

```python
@dataclass
class UserProfile:
    """Aggregated user profile from semantic memories."""

    user_id: str

    # Communication preferences
    preferred_detail_level: Literal["brief", "moderate", "detailed"]
    preferred_format: Literal["prose", "bullets", "technical"]
    tone_preference: Literal["formal", "casual", "professional"]

    # Expertise
    expertise_areas: Dict[str, float]  # domain -> confidence
    # Example: {"python": 0.9, "finance": 0.7, "rust": 0.3}

    # Interests
    topics_of_interest: List[str]
    recurring_queries: List[str]

    # Context
    inferred_role: Optional[str]       # "developer", "analyst", "manager"
    inferred_industry: Optional[str]   # "fintech", "healthcare", etc.

    # Behavioral
    typical_query_time: Optional[str]  # "morning", "evening"
    session_patterns: Dict[str, Any]   # avg length, tools used, etc.
```

### Profile Update Flow

```
User Interaction
       │
       ▼
┌──────────────┐
│ Extract      │──► Entities, topics, sentiment
│ Signals      │──► Query complexity, tool usage
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Update       │──► Increment expertise if correct
│ Profile      │──► Adjust preferences on feedback
└──────┬───────┘──► Decay old preferences
       │
       ▼
┌──────────────┐
│ Persist      │──► Store in semantic memory
└──────────────┘
```

---

## Storage Schema

### PostgreSQL Tables

```sql
-- User identification
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP,
    metadata JSONB
);

-- Sessions
CREATE TABLE sessions (
    session_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    started_at TIMESTAMP DEFAULT NOW(),
    ended_at TIMESTAMP,
    metadata JSONB
);

-- Episodic memories
CREATE TABLE episodic_memories (
    episode_id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(session_id),
    user_id UUID REFERENCES users(user_id),

    timestamp TIMESTAMP DEFAULT NOW(),
    user_query TEXT NOT NULL,
    agent_response TEXT NOT NULL,

    tools_used TEXT[],
    sources JSONB,

    topic_tags TEXT[],
    entities TEXT[],
    importance_score FLOAT DEFAULT 0.5,

    consolidated BOOLEAN DEFAULT FALSE,
    embedding_id VARCHAR(255),  -- Reference to vector DB

    created_at TIMESTAMP DEFAULT NOW()
);

-- Semantic memories
CREATE TABLE semantic_memories (
    memory_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),

    memory_type VARCHAR(50) NOT NULL,  -- fact, preference, belief, knowledge
    subject VARCHAR(255) NOT NULL,
    predicate VARCHAR(255) NOT NULL,
    object TEXT NOT NULL,

    confidence FLOAT DEFAULT 0.7,
    source_episodes UUID[],

    created_at TIMESTAMP DEFAULT NOW(),
    last_confirmed TIMESTAMP,
    contradiction_count INT DEFAULT 0,

    embedding_id VARCHAR(255)
);

-- Procedural memories
CREATE TABLE procedural_memories (
    procedure_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),

    procedure_type VARCHAR(50) NOT NULL,
    trigger_pattern TEXT NOT NULL,
    action_sequence JSONB NOT NULL,

    success_count INT DEFAULT 0,
    failure_count INT DEFAULT 0,
    effectiveness_score FLOAT DEFAULT 0.5,

    applicable_domains TEXT[],
    source_episodes UUID[],

    created_at TIMESTAMP DEFAULT NOW(),
    last_used TIMESTAMP
);

-- Indexes
CREATE INDEX idx_episodic_user_time ON episodic_memories(user_id, timestamp DESC);
CREATE INDEX idx_episodic_topics ON episodic_memories USING GIN(topic_tags);
CREATE INDEX idx_semantic_user_type ON semantic_memories(user_id, memory_type);
CREATE INDEX idx_semantic_subject ON semantic_memories(subject, predicate);
```

### ChromaDB Collections

```python
# Episodic embeddings
episodic_collection = chroma_client.create_collection(
    name="episodic_memories",
    metadata={"hnsw:space": "cosine"}
)

# Semantic embeddings
semantic_collection = chroma_client.create_collection(
    name="semantic_memories",
    metadata={"hnsw:space": "cosine"}
)
```

---

## Integration with Kautilya

### Modified Chat Flow

```python
# In interactive.py - _handle_chat method

async def _handle_chat(self, user_input: str) -> None:
    # 1. Recall relevant context
    memory_context = await self.memory_manager.recall(
        query=user_input,
        context={"session_id": self.session_id}
    )

    # 2. Build enhanced prompt with memory context
    enhanced_messages = self._build_messages_with_memory(
        user_input=user_input,
        memory_context=memory_context
    )

    # 3. Get LLM response (existing flow)
    response = await self.llm_client.chat(enhanced_messages, ...)

    # 4. Remember this interaction
    await self.memory_manager.remember(Interaction(
        session_id=self.session_id,
        user_query=user_input,
        agent_response=response,
        tools_used=tools_used,
        sources=get_source_tracker().get_sources()
    ))

    # 5. Display response (existing flow)
    ...

def _build_messages_with_memory(
    self,
    user_input: str,
    memory_context: MemoryContext
) -> List[Message]:
    """Inject relevant memories into conversation context."""

    messages = []

    # System message with user profile
    system_msg = self._get_system_prompt()
    if memory_context.user_profile:
        system_msg += f"\n\n## User Profile\n{memory_context.user_profile.to_prompt()}"

    # Add relevant past context
    if memory_context.episodes:
        system_msg += "\n\n## Relevant Past Conversations\n"
        for ep in memory_context.episodes[:3]:
            system_msg += f"- Q: {ep.user_query[:100]}...\n  A: {ep.agent_response[:100]}...\n"

    # Add known facts
    if memory_context.facts:
        system_msg += "\n\n## Known Facts About User\n"
        for fact in memory_context.facts[:5]:
            system_msg += f"- {fact.subject} {fact.predicate} {fact.object}\n"

    messages.append({"role": "system", "content": system_msg})

    # Add working memory (recent conversation)
    messages.extend(memory_context.working)

    # Add current query
    messages.append({"role": "user", "content": user_input})

    return messages
```

---

## CLI Commands

### New Memory Commands

```
/memory status          Show memory statistics
/memory search <query>  Search past conversations
/memory forget <id>     Delete specific memory
/memory profile         Show inferred user profile
/memory export          Export memories to JSON
/memory import <file>   Import memories from JSON
/memory consolidate     Trigger manual consolidation
/memory clear           Clear all memories (with confirmation)
```

---

## Configuration

### Memory Config File (`.kautilya/memory.yaml`)

```yaml
memory:
  enabled: true

  # Storage backends
  storage:
    working:
      backend: redis
      url: ${REDIS_URL:-redis://localhost:6379}
      ttl: 3600  # 1 hour

    episodic:
      backend: postgres
      url: ${POSTGRES_URL}
      retention_days: 90

    semantic:
      backend: postgres
      url: ${POSTGRES_URL}

    vectors:
      backend: chromadb
      path: ~/.kautilya/chroma
      # Or for remote:
      # url: ${CHROMA_URL}

  # Retrieval settings
  retrieval:
    weights:
      relevance: 0.5
      recency: 0.3
      importance: 0.2

    episodic_limit: 5
    semantic_limit: 10
    decay_rate: 0.01  # Per hour

  # Consolidation
  consolidation:
    enabled: true
    interval: daily  # or: hourly, weekly, manual
    min_episodes: 5  # Minimum episodes before consolidation

  # Privacy
  privacy:
    local_only: false  # If true, never send memories to cloud
    anonymize_pii: true
    retention_days: 90
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Create `MemoryManager` class
- [ ] Implement `WorkingMemory` with Redis
- [ ] Add session persistence (SQLite fallback)
- [ ] Integrate into `_handle_chat` flow

### Phase 2: Episodic Memory (Week 3-4)
- [ ] Implement `EpisodicMemoryStore`
- [ ] Add ChromaDB integration for embeddings
- [ ] Implement retrieval with multi-factor scoring
- [ ] Add `/memory search` command

### Phase 3: Semantic Memory (Week 5-6)
- [ ] Implement `SemanticMemoryStore`
- [ ] Build knowledge extraction pipeline
- [ ] Create `MemoryConsolidator`
- [ ] Add user profile system

### Phase 4: Procedural Memory (Week 7-8)
- [ ] Implement `ProceduralMemoryStore`
- [ ] Add pattern learning from successful interactions
- [ ] Integrate procedure suggestions into prompts
- [ ] Add effectiveness tracking

### Phase 5: Polish & Optimization (Week 9-10)
- [ ] Add all `/memory` CLI commands
- [ ] Performance optimization
- [ ] Add memory export/import
- [ ] Documentation and testing

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Cross-session context recall | 80%+ relevant | User survey |
| User preference accuracy | 85%+ | A/B testing |
| Memory retrieval latency | <100ms | p95 latency |
| Storage efficiency | <1MB/100 conversations | Storage audit |
| User satisfaction | 4.5/5 | NPS survey |

---

## References

- [Hindsight: Agentic Memory Framework](https://venturebeat.com/data/with-91-accuracy-open-source-hindsight-agentic-memory-provides-20-20-vision)
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/html/2502.12110v11)
- [Generative Agents: Interactive Simulacra (Stanford)](https://arxiv.org/abs/2304.03442)
- [LangMem: Long-term Memory for LangChain](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
- [Mem0: Memory Layer for AI](https://mem0.ai)
- [ACM Survey on Memory Mechanisms](https://dl.acm.org/doi/10.1145/3748302)
