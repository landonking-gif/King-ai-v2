"""
Infrastructure component tests.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.utils.vllm_client import VLLMClient, InferenceRequest
from src.utils.llm_router import LLMRouter, ProviderType, TaskContext
from src.utils.monitoring import DatadogMonitor


class TestVLLMClient:
    """Tests for vLLM client."""
    
    @pytest.fixture
    def client(self):
        return VLLMClient(base_url="http://localhost:8080", model="test-model")
    
    @pytest.mark.asyncio
    async def test_complete_success(self, client):
        """Test successful completion."""
        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Hello!"}}]
            }
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response
            
            result = await client.complete("Hello")
            assert result == "Hello!"
    
    @pytest.mark.asyncio
    async def test_batch_complete(self, client):
        """Test batch completion."""
        requests = [
            InferenceRequest(prompt="Test 1", request_id="1"),
            InferenceRequest(prompt="Test 2", request_id="2"),
        ]
        
        with patch.object(client, 'complete', new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "Response"
            
            results = await client.batch_complete(requests)
            assert len(results) == 2
            assert all(r["success"] for r in results)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test health check when server is healthy."""
        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = await client.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        """Test health check when server is down."""
        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            
            result = await client.health_check()
            assert result is False


class TestLLMRouter:
    """Tests for LLM routing logic."""
    
    @pytest.fixture
    def router(self):
        with patch('src.utils.llm_router.settings') as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_model = "llama3.1:8b"
            mock_settings.vllm_url = None
            mock_settings.vllm_model = "meta-llama/Llama-3.1-70B-Instruct"
            mock_settings.anthropic_api_key = None
            mock_settings.claude_model = "claude-3-5-sonnet-20241022"
            mock_settings.gemini_api_key = None
            mock_settings.gemini_api_keys = ""
            return LLMRouter()
    
    @pytest.mark.asyncio
    async def test_fallback_chain(self, router):
        """Test fallback when primary fails."""
        router._provider_health[ProviderType.VLLM] = False
        
        with patch.object(router.ollama, 'complete', new_callable=AsyncMock) as mock_ollama:
            mock_ollama.return_value = "Ollama response"
            
            result = await router.complete("Test")
            assert result == "Ollama response"
    
    def test_circuit_breaker(self, router):
        """Test circuit breaker opens after failures."""
        provider = ProviderType.OLLAMA
        
        for _ in range(router._failure_threshold):
            router._record_failure(provider)
        
        assert router._is_circuit_open(provider)
    
    def test_circuit_breaker_recovery(self, router):
        """Test circuit breaker closes after timeout."""
        import time
        provider = ProviderType.OLLAMA
        
        # Open circuit
        for _ in range(router._failure_threshold):
            router._record_failure(provider)
        
        assert router._is_circuit_open(provider)
        
        # Simulate timeout passage
        router._circuit_open_until[provider] = time.time() - 1
        
        # Circuit should be closed now
        assert not router._is_circuit_open(provider)
    
    @pytest.mark.asyncio
    async def test_routing_decision_high_stakes(self, router):
        """Test that high-stakes tasks route to accurate provider."""
        # Set up Gemini as available
        router.gemini = MagicMock()
        router._provider_health[ProviderType.GEMINI] = True
        router._provider_health[ProviderType.VLLM] = True
        router.vllm = MagicMock()
        
        context = TaskContext(
            task_type="finance",
            risk_level="high",
            requires_accuracy=True,
            token_estimate=1000,
            priority="critical"
        )
        
        decision = await router._route(context)
        assert decision.provider == ProviderType.GEMINI
        assert "accuracy" in decision.reason.lower()
    
    def test_fallback_chain_order(self, router):
        """Test that fallback chain has correct order."""
        chain = router._get_fallback_chain(ProviderType.OLLAMA)
        
        assert chain[0] == ProviderType.OLLAMA
        assert len(chain) == 3
        assert ProviderType.VLLM in chain
        assert ProviderType.GEMINI in chain


class TestDatadogMonitor:
    """Tests for monitoring integration."""
    
    def test_tags_format(self):
        """Test metric tags are properly formatted."""
        monitor = DatadogMonitor()
        tags = monitor._get_tags({"custom": "tag"})
        
        assert any("env:" in t for t in tags)
        assert any("service:" in t for t in tags)
        assert "custom:tag" in tags
    
    def test_increment_disabled(self):
        """Test that metrics don't fail when Datadog is disabled."""
        monitor = DatadogMonitor()
        monitor.enabled = False
        
        # Should not raise an error
        monitor.increment("test.metric", value=1)
    
    def test_timing_context_manager(self):
        """Test timing context manager."""
        import time
        monitor = DatadogMonitor()
        monitor.enabled = False
        
        with monitor.timed("test.operation"):
            time.sleep(0.01)
        
        # Should complete without errors
    
    def test_trace_decorator_async(self):
        """Test trace decorator on async function."""
        monitor = DatadogMonitor()
        monitor.enabled = False
        
        @monitor.trace("test.operation")
        async def async_func():
            return "result"
        
        # Should return the decorated function
        assert asyncio.iscoroutinefunction(async_func)


class TestInferenceRequest:
    """Tests for InferenceRequest dataclass."""
    
    def test_inference_request_defaults(self):
        """Test InferenceRequest with defaults."""
        req = InferenceRequest(prompt="test prompt")
        
        assert req.prompt == "test prompt"
        assert req.max_tokens == 4096
        assert req.temperature == 0.7
        assert req.request_id is None
    
    def test_inference_request_custom_values(self):
        """Test InferenceRequest with custom values."""
        req = InferenceRequest(
            prompt="custom prompt",
            max_tokens=2048,
            temperature=0.5,
            request_id="123"
        )
        
        assert req.prompt == "custom prompt"
        assert req.max_tokens == 2048
        assert req.temperature == 0.5
        assert req.request_id == "123"


# Import asyncio for async tests
import asyncio
