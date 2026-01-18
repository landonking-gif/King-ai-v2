"""
Memory Service Client Example.

Module: memory-service/examples/client_example.py

Demonstrates how to use the Memory Service API from other services.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
from pydantic import BaseModel


class MemoryClient:
    """Client for interacting with Memory Service."""

    def __init__(self, base_url: str = "http://localhost:8001") -> None:
        """
        Initialize memory client.

        Args:
            base_url: Base URL of memory service
        """
        self.base_url = base_url
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "MemoryClient":
        """Async context manager entry."""
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def commit_artifact(
        self,
        artifact_type: str,
        content: Dict[str, Any],
        created_by: str,
        actor_id: str,
        actor_type: str,
        session_id: Optional[str] = None,
        tool_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        generate_embedding: bool = True,
        store_in_cold: bool = False,
    ) -> Dict[str, Any]:
        """
        Commit an artifact to memory.

        Args:
            artifact_type: Type of artifact
            content: Artifact content
            created_by: Creator ID
            actor_id: Actor ID committing artifact
            actor_type: Actor type (subagent, lead_agent, skill)
            session_id: Optional session ID
            tool_ids: Tools used to create artifact
            tags: Artifact tags
            generate_embedding: Whether to generate embedding
            store_in_cold: Whether to store in cold storage

        Returns:
            Commit response
        """
        if not self.client:
            raise RuntimeError("Client not initialized")

        payload = {
            "artifact": {
                "artifact_type": artifact_type,
                "content": content,
                "created_by": created_by,
                "session_id": session_id,
                "tags": tags or [],
            },
            "actor_id": actor_id,
            "actor_type": actor_type,
            "tool_ids": tool_ids or [],
            "generate_embedding": generate_embedding,
            "store_in_cold": store_in_cold,
        }

        response = await self.client.post("/memory/commit", json=payload)
        response.raise_for_status()
        return response.json()

    async def query_artifacts(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        filter_artifact_type: Optional[str] = None,
        filter_session_id: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Query similar artifacts.

        Args:
            query_text: Text query
            query_embedding: Embedding vector query
            top_k: Number of results
            filter_artifact_type: Filter by type
            filter_session_id: Filter by session
            min_similarity: Minimum similarity threshold

        Returns:
            Query results
        """
        if not self.client:
            raise RuntimeError("Client not initialized")

        payload = {
            "query_text": query_text,
            "query_embedding": query_embedding,
            "top_k": top_k,
            "filter_artifact_type": filter_artifact_type,
            "filter_session_id": filter_session_id,
            "min_similarity": min_similarity,
        }

        response = await self.client.post("/memory/query", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_provenance(self, artifact_id: str) -> Dict[str, Any]:
        """
        Get provenance chain for artifact.

        Args:
            artifact_id: Artifact ID

        Returns:
            Provenance chain
        """
        if not self.client:
            raise RuntimeError("Client not initialized")

        response = await self.client.get(f"/memory/provenance/{artifact_id}")
        response.raise_for_status()
        return response.json()

    async def compact_memory(
        self,
        session_id: str,
        strategy: str = "summarize",
        target_tokens: int = 5000,
        preserve_artifact_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compact memory for session.

        Args:
            session_id: Session ID
            strategy: Compaction strategy (summarize, truncate, none)
            target_tokens: Target token count
            preserve_artifact_ids: Artifacts to preserve

        Returns:
            Compaction results
        """
        if not self.client:
            raise RuntimeError("Client not initialized")

        payload = {
            "session_id": session_id,
            "strategy": strategy,
            "target_tokens": target_tokens,
            "preserve_artifact_ids": preserve_artifact_ids or [],
        }

        response = await self.client.post("/memory/compact", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get session statistics.

        Args:
            session_id: Session ID

        Returns:
            Session stats
        """
        if not self.client:
            raise RuntimeError("Client not initialized")

        response = await self.client.get(f"/memory/session/{session_id}/stats")
        response.raise_for_status()
        return response.json()


async def example_research_workflow() -> None:
    """Example: Research agent workflow with memory."""
    async with MemoryClient() as client:
        session_id = "research-session-001"

        # 1. Commit research findings
        print("1. Committing research findings...")
        research1 = await client.commit_artifact(
            artifact_type="research_snippet",
            content={
                "text": "LLM agents can use tools to interact with external systems",
                "summary": "Tool usage in LLM agents",
                "source": {"url": "https://arxiv.org/example"},
                "confidence": 0.9,
            },
            created_by="research-agent",
            actor_id="agent-research-1",
            actor_type="subagent",
            session_id=session_id,
            tool_ids=["web_search", "arxiv_reader"],
            tags=["llm", "tools", "agents"],
        )
        print(f"   Committed artifact: {research1['artifact_id']}")

        # 2. Commit another finding
        research2 = await client.commit_artifact(
            artifact_type="research_snippet",
            content={
                "text": "Memory systems enable agents to maintain context across interactions",
                "summary": "Agent memory systems",
                "source": {"url": "https://example.com/memory"},
                "confidence": 0.85,
            },
            created_by="research-agent",
            actor_id="agent-research-1",
            actor_type="subagent",
            session_id=session_id,
            tool_ids=["web_search"],
            tags=["memory", "context", "agents"],
        )
        print(f"   Committed artifact: {research2['artifact_id']}")

        # 3. Query related research
        print("\n2. Querying related research...")
        results = await client.query_artifacts(
            query_text="agent memory and tools", top_k=5, min_similarity=0.5
        )
        print(f"   Found {len(results['items'])} similar artifacts")
        for item in results["items"]:
            print(f"   - {item['artifact_id']}: similarity={item['similarity']:.2f}")

        # 4. Check session stats
        print("\n3. Checking session statistics...")
        stats = await client.get_session_stats(session_id)
        print(f"   Artifacts: {stats['artifact_count']}")
        print(f"   Total tokens: {stats['total_tokens']}")
        print(f"   Needs compaction: {stats['needs_compaction']}")

        # 5. Get provenance
        print("\n4. Getting provenance chain...")
        provenance = await client.get_provenance(research1["artifact_id"])
        print(f"   Chain depth: {provenance['depth']}")
        print(f"   Root artifacts: {provenance['root_artifacts']}")


async def example_code_generation_workflow() -> None:
    """Example: Code generation with verification."""
    async with MemoryClient() as client:
        session_id = "codegen-session-001"

        # 1. Store code patch
        print("1. Storing code patch...")
        patch = await client.commit_artifact(
            artifact_type="code_patch",
            content={
                "repo": "agent-framework",
                "base_commit": "abc123",
                "files_changed": ["memory_service/main.py"],
                "patch_summary": "Add new query endpoint",
                "tests": ["test_query_endpoint"],
            },
            created_by="code-agent",
            actor_id="agent-code-1",
            actor_type="subagent",
            session_id=session_id,
            tool_ids=["code_generator", "test_runner"],
            tags=["code", "api"],
            parent_artifact_ids=[],
        )
        print(f"   Committed patch: {patch['artifact_id']}")

        # 2. Store verification result
        verification = await client.commit_artifact(
            artifact_type="claim_verification",
            content={
                "claim_text": "Code patch passes all tests",
                "verdict": "VERIFIED",
                "confidence": 0.95,
                "evidence_refs": ["test_output.log"],
            },
            created_by="verify-agent",
            actor_id="agent-verify-1",
            actor_type="subagent",
            session_id=session_id,
            tool_ids=["test_runner", "static_analyzer"],
            tags=["verification", "tests"],
            parent_artifact_ids=[patch["artifact_id"]],
        )
        print(f"   Committed verification: {verification['artifact_id']}")

        # 3. Get full provenance
        print("\n2. Getting provenance chain...")
        provenance = await client.get_provenance(verification["artifact_id"])
        print(f"   Chain includes {len(provenance['chain'])} entries")
        for entry in provenance["chain"]:
            print(f"   - {entry['actor_type']}: {entry['artifact_id'][:8]}...")


async def example_memory_compaction() -> None:
    """Example: Memory compaction for long-running session."""
    async with MemoryClient() as client:
        session_id = "long-session-001"

        # Simulate creating many artifacts
        print("1. Creating multiple artifacts...")
        artifact_ids = []
        for i in range(10):
            result = await client.commit_artifact(
                artifact_type="generic",
                content={"text": f"Artifact {i}", "data": "x" * 1000},
                created_by="test-agent",
                actor_id="agent-test",
                actor_type="subagent",
                session_id=session_id,
            )
            artifact_ids.append(result["artifact_id"])

        # Check stats
        stats = await client.get_session_stats(session_id)
        print(f"   Created {stats['artifact_count']} artifacts")
        print(f"   Total tokens: {stats['total_tokens']}")

        # Compact if needed
        if stats["needs_compaction"]:
            print("\n2. Compacting memory...")
            result = await client.compact_memory(
                session_id=session_id,
                strategy="truncate",
                target_tokens=5000,
                preserve_artifact_ids=artifact_ids[:2],  # Keep first 2
            )
            print(f"   Tokens before: {result['tokens_before']}")
            print(f"   Tokens after: {result['tokens_after']}")
            print(f"   Artifacts removed: {result['artifacts_removed']}")


async def main() -> None:
    """Run all examples."""
    print("=" * 60)
    print("Memory Service Client Examples")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Example 1: Research Workflow")
    print("=" * 60)
    await example_research_workflow()

    print("\n" + "=" * 60)
    print("Example 2: Code Generation Workflow")
    print("=" * 60)
    await example_code_generation_workflow()

    print("\n" + "=" * 60)
    print("Example 3: Memory Compaction")
    print("=" * 60)
    await example_memory_compaction()


if __name__ == "__main__":
    asyncio.run(main())
