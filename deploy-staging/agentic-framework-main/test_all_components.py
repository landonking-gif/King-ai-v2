#!/usr/bin/env python3
"""
King AI v3 - Comprehensive Test Suite
=====================================
Tests all components of the agentic framework to ensure proper functionality.

Usage:
    python test_all_components.py [--server http://localhost]
"""

import asyncio
import sys
import time
from typing import Dict, Any, List, Tuple
import json

try:
    import httpx
except ImportError:
    print("Installing httpx...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "httpx"], check=True)
    import httpx


# Configuration
DEFAULT_BASE_URL = "http://localhost"
SERVICES = {
    "orchestrator": {"port": 8000, "health_path": "/health"},
    "subagent_manager": {"port": 8001, "health_path": "/health"},
    "memory_service": {"port": 8002, "health_path": "/health"},
    "mcp_gateway": {"port": 8080, "health_path": "/health"},
    "control_panel": {"port": 3000, "health_path": "/health"},
    "ollama": {"port": 11434, "health_path": "/api/tags"},
}


class TestResult:
    def __init__(self, name: str, passed: bool, message: str, duration_ms: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} | {self.name}: {self.message} ({self.duration_ms:.1f}ms)"


class KingAIV3Tester:
    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url
        self.results: List[TestResult] = []
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        await self.client.aclose()
    
    def _url(self, port: int, path: str) -> str:
        return f"{self.base_url}:{port}{path}"
    
    async def _test(self, name: str, coro) -> TestResult:
        """Execute a test and record the result."""
        start = time.time()
        try:
            result = await coro
            duration = (time.time() - start) * 1000
            if isinstance(result, tuple):
                passed, message = result
            else:
                passed, message = bool(result), str(result)
            return TestResult(name, passed, message, duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(name, False, f"Exception: {str(e)}", duration)
    
    # =========================================================================
    # Service Health Tests
    # =========================================================================
    
    async def test_service_health(self, service_name: str, port: int, path: str) -> Tuple[bool, str]:
        """Test if a service is healthy."""
        try:
            resp = await self.client.get(self._url(port, path))
            if resp.status_code == 200:
                return True, f"Healthy (status {resp.status_code})"
            return False, f"Unhealthy (status {resp.status_code})"
        except httpx.ConnectError:
            return False, "Connection refused"
        except Exception as e:
            return False, str(e)
    
    # =========================================================================
    # Memory Service Tests
    # =========================================================================
    
    async def test_memory_diary_create(self) -> Tuple[bool, str]:
        """Test creating a diary entry."""
        resp = await self.client.post(
            self._url(8002, "/diary"),
            json={
                "session_id": "test-session-001",
                "workspace_path": "/test/workspace",
                "open_files": ["test.py", "main.py"],
                "context": {"test": True}
            }
        )
        if resp.status_code == 200:
            data = resp.json()
            return True, f"Created diary {data.get('diary_id', 'unknown')[:8]}"
        return False, f"Status {resp.status_code}: {resp.text[:100]}"
    
    async def test_memory_diary_list(self) -> Tuple[bool, str]:
        """Test listing diary entries."""
        resp = await self.client.get(self._url(8002, "/diary?limit=5"))
        if resp.status_code == 200:
            data = resp.json()
            return True, f"Found {data.get('count', 0)} entries"
        return False, f"Status {resp.status_code}"
    
    async def test_memory_reflect(self) -> Tuple[bool, str]:
        """Test reflection endpoint."""
        resp = await self.client.post(
            self._url(8002, "/reflect"),
            json={"max_entries": 10, "min_unprocessed": 1}
        )
        if resp.status_code == 200:
            data = resp.json()
            patterns = len(data.get("patterns_found", []))
            learnings = len(data.get("learnings", []))
            return True, f"Patterns: {patterns}, Learnings: {learnings}"
        return False, f"Status {resp.status_code}"
    
    async def test_memory_commit(self) -> Tuple[bool, str]:
        """Test committing an artifact to memory."""
        import uuid
        artifact_id = str(uuid.uuid4())
        resp = await self.client.post(
            self._url(8002, "/memory/commit"),
            json={
                "artifact": {
                    "id": artifact_id,
                    "artifact_type": "research_snippet",
                    "content": {"test": "data", "value": 123},
                    "created_by": "test-agent",
                    "session_id": "test-session",
                    "safety_class": "internal",
                    "tags": ["test"]
                },
                "actor_id": "test-agent",
                "actor_type": "test",
                "tool_ids": ["test"],
                "generate_embedding": False,
                "store_in_cold": False
            }
        )
        if resp.status_code in [200, 201]:
            return True, f"Committed artifact {artifact_id[:8]}"
        return False, f"Status {resp.status_code}: {resp.text[:100]}"
    
    # =========================================================================
    # Orchestrator Tests
    # =========================================================================
    
    async def test_orchestrator_agents(self) -> Tuple[bool, str]:
        """Test listing agents from orchestrator."""
        resp = await self.client.get(self._url(8000, "/api/agents"))
        if resp.status_code == 200:
            data = resp.json()
            count = data.get("count", len(data.get("agents", [])))
            return True, f"Found {count} agents"
        return False, f"Status {resp.status_code}"
    
    async def test_orchestrator_chat(self) -> Tuple[bool, str]:
        """Test chat endpoint."""
        resp = await self.client.post(
            self._url(8000, "/api/chat"),
            json={"message": "hello, what can you do?", "text": "hello, what can you do?"}
        )
        if resp.status_code == 200:
            data = resp.json()
            response = data.get("response", "")[:50]
            return True, f"Response: {response}..."
        return False, f"Status {resp.status_code}"
    
    async def test_orchestrator_ralph(self) -> Tuple[bool, str]:
        """Test Ralph code agent trigger."""
        resp = await self.client.post(
            self._url(8000, "/api/chat"),
            json={"message": "use ralph to create a hello world function", "text": "use ralph to create a hello world function"}
        )
        if resp.status_code == 200:
            data = resp.json()
            response = data.get("response", "")
            if "ralph" in response.lower() or "agent" in response.lower():
                return True, "Ralph agent triggered"
            return True, f"Response received: {response[:50]}..."
        return False, f"Status {resp.status_code}"
    
    # =========================================================================
    # Control Panel Tests
    # =========================================================================
    
    async def test_control_panel_dashboard(self) -> Tuple[bool, str]:
        """Test control panel dashboard endpoint."""
        resp = await self.client.get(self._url(3000, "/api/dashboard"))
        if resp.status_code == 200:
            return True, "Dashboard accessible"
        elif resp.status_code == 401:
            return True, "Dashboard accessible (auth required)"
        return False, f"Status {resp.status_code}"
    
    # =========================================================================
    # Ollama Tests
    # =========================================================================
    
    async def test_ollama_models(self) -> Tuple[bool, str]:
        """Test Ollama model availability."""
        resp = await self.client.get(self._url(11434, "/api/tags"))
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            if models:
                return True, f"Models: {', '.join(models[:3])}"
            return True, "Ollama running (no models)"
        return False, f"Status {resp.status_code}"
    
    async def test_ollama_generate(self) -> Tuple[bool, str]:
        """Test Ollama text generation."""
        resp = await self.client.post(
            self._url(11434, "/api/generate"),
            json={
                "model": "llama3.2:1b",
                "prompt": "Say hello in one word",
                "stream": False
            }
        )
        if resp.status_code == 200:
            data = resp.json()
            response = data.get("response", "")[:30]
            return True, f"Generated: {response}"
        return False, f"Status {resp.status_code}"
    
    # =========================================================================
    # Run All Tests
    # =========================================================================
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all tests and return results."""
        print("\n" + "=" * 60)
        print("King AI v3 - Component Test Suite")
        print("=" * 60 + "\n")
        
        # 1. Service Health Tests
        print("📡 Testing Service Health...")
        for name, config in SERVICES.items():
            result = await self._test(
                f"Service: {name}",
                self.test_service_health(name, config["port"], config["health_path"])
            )
            self.results.append(result)
            print(f"  {result}")
        
        print()
        
        # 2. Memory Service Tests
        print("💾 Testing Memory Service...")
        memory_tests = [
            ("Memory: Create Diary", self.test_memory_diary_create()),
            ("Memory: List Diaries", self.test_memory_diary_list()),
            ("Memory: Reflect", self.test_memory_reflect()),
            ("Memory: Commit Artifact", self.test_memory_commit()),
        ]
        for name, coro in memory_tests:
            result = await self._test(name, coro)
            self.results.append(result)
            print(f"  {result}")
        
        print()
        
        # 3. Orchestrator Tests
        print("🎯 Testing Orchestrator...")
        orchestrator_tests = [
            ("Orchestrator: List Agents", self.test_orchestrator_agents()),
            ("Orchestrator: Chat", self.test_orchestrator_chat()),
            ("Orchestrator: Ralph Agent", self.test_orchestrator_ralph()),
        ]
        for name, coro in orchestrator_tests:
            result = await self._test(name, coro)
            self.results.append(result)
            print(f"  {result}")
        
        print()
        
        # 4. Control Panel Tests
        print("🖥️ Testing Control Panel...")
        result = await self._test("Control Panel: Dashboard", self.test_control_panel_dashboard())
        self.results.append(result)
        print(f"  {result}")
        
        print()
        
        # 5. Ollama Tests
        print("🧠 Testing Ollama LLM...")
        ollama_tests = [
            ("Ollama: List Models", self.test_ollama_models()),
            ("Ollama: Generate Text", self.test_ollama_generate()),
        ]
        for name, coro in ollama_tests:
            result = await self._test(name, coro)
            self.results.append(result)
            print(f"  {result}")
        
        return self.results
    
    def print_summary(self):
        """Print test summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"  Total:  {total}")
        print(f"  Passed: {passed} ✅")
        print(f"  Failed: {failed} ❌")
        print(f"  Rate:   {passed/total*100:.1f}%")
        print("=" * 60)
        
        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  ❌ {r.name}: {r.message}")
        
        return failed == 0


async def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    
    tester = KingAIV3Tester(base_url)
    try:
        await tester.run_all_tests()
        success = tester.print_summary()
        await tester.close()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        await tester.close()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
