#!/usr/bin/env python3
"""
King AI v3 Component Test Suite

Tests all components of the King AI v3 agentic framework:
- Orchestrator API
- Memory Service
- Diary/Reflect endpoints
- Ralph agent integration
- Agent listing
- Chat functionality
"""

import asyncio
import json
import sys
import os
from typing import Tuple

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    import httpx
except ImportError:
    print("Installing httpx...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
    import httpx


class TestRunner:
    """Runs tests against King AI v3 services."""
    
    def __init__(self, orchestrator_url: str = "http://localhost:8000", memory_url: str = "http://localhost:8002"):
        self.orchestrator_url = orchestrator_url
        self.memory_url = memory_url
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    async def run_all_tests(self) -> bool:
        """Run all tests and return success status."""
        print("=" * 60)
        print("  King AI v3 Component Test Suite")
        print("=" * 60)
        print(f"\nOrchestrator: {self.orchestrator_url}")
        print(f"Memory Service: {self.memory_url}")
        print("\n" + "-" * 60 + "\n")
        
        # Orchestrator tests
        await self.test_orchestrator_health()
        await self.test_orchestrator_agents()
        await self.test_orchestrator_chat()
        await self.test_orchestrator_chat_message()
        
        # Memory service tests
        await self.test_memory_health()
        await self.test_memory_diary_list()
        await self.test_memory_diary_create()
        await self.test_memory_reflect()
        await self.test_memory_commit()
        await self.test_memory_query()
        
        # Integration tests
        await self.test_ralph_agent()
        await self.test_workflow_agents()
        
        # Print summary
        print("\n" + "=" * 60)
        print("  Test Summary")
        print("=" * 60)
        print(f"\n  ✓ Passed: {self.passed}")
        print(f"  ✗ Failed: {self.failed}")
        print(f"  Total: {self.passed + self.failed}")
        
        if self.errors:
            print("\n  Errors:")
            for error in self.errors:
                print(f"    - {error}")
        
        print("\n" + "=" * 60)
        
        return self.failed == 0
    
    async def _test(self, name: str, test_func) -> bool:
        """Run a single test and track results."""
        try:
            print(f"Testing: {name}... ", end="", flush=True)
            success, message = await test_func()
            if success:
                print(f"✓ PASS - {message}")
                self.passed += 1
            else:
                print(f"✗ FAIL - {message}")
                self.failed += 1
                self.errors.append(f"{name}: {message}")
            return success
        except Exception as e:
            print(f"✗ ERROR - {e}")
            self.failed += 1
            self.errors.append(f"{name}: {e}")
            return False
    
    async def test_orchestrator_health(self):
        """Test orchestrator health endpoint."""
        async def test():
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.orchestrator_url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    return True, f"Status: {data.get('status', 'unknown')}"
                return False, f"Status code: {resp.status_code}"
        await self._test("Orchestrator Health", test)
    
    async def test_orchestrator_agents(self):
        """Test orchestrator agents endpoint."""
        async def test():
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.orchestrator_url}/api/agents")
                if resp.status_code == 200:
                    data = resp.json()
                    count = data.get('count', len(data.get('agents', [])))
                    return True, f"Found {count} agents"
                return False, f"Status code: {resp.status_code}"
        await self._test("Orchestrator Agents List", test)
    
    async def test_orchestrator_chat(self):
        """Test orchestrator chat endpoint."""
        async def test():
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.orchestrator_url}/api/chat",
                    json={"message": "hello", "session_id": "test-session"}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    has_response = 'response' in data or 'message' in data
                    return has_response, f"Got response: {str(data)[:100]}..."
                return False, f"Status code: {resp.status_code}, Body: {resp.text[:200]}"
        await self._test("Orchestrator Chat", test)
    
    async def test_orchestrator_chat_message(self):
        """Test orchestrator chat/message endpoint."""
        async def test():
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.orchestrator_url}/api/chat/message",
                    json={"text": "list available agents", "session_id": "test-session"}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return True, f"Got response: {str(data)[:100]}..."
                return False, f"Status code: {resp.status_code}"
        await self._test("Orchestrator Chat Message", test)
    
    async def test_memory_health(self):
        """Test memory service health endpoint."""
        async def test():
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.memory_url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    return True, f"Status: {data.get('status', 'unknown')}"
                return False, f"Status code: {resp.status_code}"
        await self._test("Memory Service Health", test)
    
    async def test_memory_diary_list(self):
        """Test memory service diary list endpoint."""
        async def test():
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.memory_url}/diary")
                if resp.status_code == 200:
                    data = resp.json()
                    count = data.get('count', 0)
                    return True, f"Found {count} diary entries"
                return False, f"Status code: {resp.status_code}"
        await self._test("Memory Diary List", test)
    
    async def test_memory_diary_create(self):
        """Test memory service diary create endpoint."""
        async def test():
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.memory_url}/diary",
                    json={
                        "session_id": "test-session-123",
                        "open_files": ["test.py", "main.py"],
                        "git_diff": "+ Added new test\n- Removed old code"
                    }
                )
                if resp.status_code in [200, 201]:
                    data = resp.json()
                    diary_id = data.get('diary_id', 'unknown')
                    return True, f"Created diary entry: {diary_id[:20]}..."
                return False, f"Status code: {resp.status_code}, Body: {resp.text[:200]}"
        await self._test("Memory Diary Create", test)
    
    async def test_memory_reflect(self):
        """Test memory service reflect endpoint."""
        async def test():
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.memory_url}/reflect",
                    json={"max_entries": 10, "min_unprocessed": 1}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    entries = data.get('entries_processed', 0)
                    patterns = len(data.get('patterns_found', []))
                    return True, f"Processed {entries} entries, found {patterns} patterns"
                return False, f"Status code: {resp.status_code}"
        await self._test("Memory Reflect", test)
    
    async def test_memory_commit(self):
        """Test memory service commit endpoint."""
        async def test():
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.memory_url}/memory/commit",
                    json={
                        "artifact": {
                            "id": "test-artifact-123",
                            "artifact_type": "generic",
                            "content": {"test": "data"},
                            "created_by": "test-runner",
                            "session_id": "test-session",
                            "safety_class": "internal",
                            "tags": ["test"]
                        },
                        "actor_id": "test-runner",
                        "actor_type": "system",
                        "tool_ids": ["test"],
                        "generate_embedding": False,
                        "store_in_cold": False
                    }
                )
                if resp.status_code in [200, 201]:
                    data = resp.json()
                    memory_id = data.get('memory_id', 'unknown')
                    return True, f"Committed: {memory_id[:20]}..."
                return False, f"Status code: {resp.status_code}, Body: {resp.text[:200]}"
        await self._test("Memory Commit", test)
    
    async def test_memory_query(self):
        """Test memory service query endpoint."""
        async def test():
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.memory_url}/memory/query",
                    json={
                        "query_text": "test query",
                        "top_k": 5
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    count = len(data.get('results', []))
                    return True, f"Query returned {count} results"
                return False, f"Status code: {resp.status_code}"
        await self._test("Memory Query", test)
    
    async def test_ralph_agent(self):
        """Test Ralph agent availability."""
        async def test():
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.orchestrator_url}/api/agents")
                if resp.status_code == 200:
                    data = resp.json()
                    agents = data.get('agents', [])
                    ralph_agents = [a for a in agents if 'ralph' in a.get('name', '').lower()]
                    if ralph_agents:
                        return True, f"Ralph agent found: {ralph_agents[0].get('name')}"
                    return False, "Ralph agent not found in agent list"
                return False, f"Status code: {resp.status_code}"
        await self._test("Ralph Agent Available", test)
    
    async def test_workflow_agents(self):
        """Test workflow agents availability."""
        async def test():
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.orchestrator_url}/api/agents")
                if resp.status_code == 200:
                    data = resp.json()
                    agents = data.get('agents', [])
                    workflow_agents = [a for a in agents if 'workflow' in a.get('name', '').lower() or 'research' in a.get('name', '').lower()]
                    return True, f"Found {len(workflow_agents)} workflow/research agents"
                return False, f"Status code: {resp.status_code}"
        await self._test("Workflow Agents Available", test)


async def main():
    """Main entry point."""
    # Get URLs from environment or command line
    orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
    memory_url = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8002")
    
    # Command line override
    if len(sys.argv) > 1:
        orchestrator_url = sys.argv[1]
    if len(sys.argv) > 2:
        memory_url = sys.argv[2]
    
    runner = TestRunner(orchestrator_url, memory_url)
    success = await runner.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
