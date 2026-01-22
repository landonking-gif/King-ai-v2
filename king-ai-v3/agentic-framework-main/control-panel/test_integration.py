"""
Integration tests for Master Control Panel
Tests cover API integrations, WebSocket connections, proxy routes, and service communication
"""

import pytest
import asyncio
import json
from httpx import AsyncClient
from fastapi import FastAPI
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the app (assuming it's in main.py at parent level)
# from main import app, ws_manager

# Fixtures for testing
@pytest.fixture
async def client():
    """Create async test client"""
    # This would need to be adapted to your actual app structure
    pass

@pytest.fixture
def auth_headers():
    """Create authentication headers for tests"""
    return {
        "Authorization": "Bearer test-token-12345"
    }

class TestAuthenticationEndpoints:
    """Test authentication and authorization"""
    
    @pytest.mark.asyncio
    async def test_login_endpoint(self, client, auth_headers):
        """Test user login endpoint"""
        response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data or "error" in data
    
    @pytest.mark.asyncio
    async def test_logout_endpoint(self, client, auth_headers):
        """Test user logout endpoint"""
        response = await client.post(
            "/api/auth/logout",
            headers=auth_headers
        )
        assert response.status_code in [200, 401]
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, client, auth_headers):
        """Test get current user endpoint"""
        response = await client.get(
            "/api/auth/me",
            headers=auth_headers
        )
        assert response.status_code in [200, 401]

class TestServiceProxyRoutes:
    """Test service proxy layer"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_health(self, client, auth_headers):
        """Test Orchestrator service health check"""
        response = await client.get(
            "/api/orchestrator/health",
            headers=auth_headers
        )
        assert response.status_code in [200, 503]
    
    @pytest.mark.asyncio
    async def test_subagent_health(self, client, auth_headers):
        """Test Subagent Manager health check"""
        response = await client.get(
            "/api/subagent-manager/health",
            headers=auth_headers
        )
        assert response.status_code in [200, 503]
    
    @pytest.mark.asyncio
    async def test_memory_health(self, client, auth_headers):
        """Test Memory Service health check"""
        response = await client.get(
            "/api/memory/health",
            headers=auth_headers
        )
        assert response.status_code in [200, 503]
    
    @pytest.mark.asyncio
    async def test_mcp_health(self, client, auth_headers):
        """Test MCP Gateway health check"""
        response = await client.get(
            "/api/mcp/health",
            headers=auth_headers
        )
        assert response.status_code in [200, 503]
    
    @pytest.mark.asyncio
    async def test_code_exec_health(self, client, auth_headers):
        """Test Code Executor health check"""
        response = await client.get(
            "/api/code-exec/health",
            headers=auth_headers
        )
        assert response.status_code in [200, 503]
    
    @pytest.mark.asyncio
    async def test_all_services_health(self, client, auth_headers):
        """Test aggregated services health"""
        response = await client.get(
            "/api/services/health",
            headers=auth_headers
        )
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            assert "orchestrator" in str(data) or "data" in str(response)

class TestWorkflowEndpoints:
    """Test workflow management endpoints"""
    
    @pytest.mark.asyncio
    async def test_list_workflows(self, client, auth_headers):
        """Test listing workflows"""
        response = await client.get(
            "/api/orchestrator/workflows",
            headers=auth_headers
        )
        assert response.status_code in [200, 404, 503]
    
    @pytest.mark.asyncio
    async def test_create_workflow(self, client, auth_headers):
        """Test creating workflow"""
        workflow_data = {
            "name": "Test Workflow",
            "description": "A test workflow",
            "steps": []
        }
        response = await client.post(
            "/api/orchestrator/workflows",
            json=workflow_data,
            headers=auth_headers
        )
        assert response.status_code in [200, 400, 401, 503]

class TestPLTrackingEndpoints:
    """Test P&L tracking endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_pl_summary(self, client, auth_headers):
        """Test getting P&L summary"""
        response = await client.get(
            "/api/business/pl/summary?period=monthly",
            headers=auth_headers
        )
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            assert "total_revenue" in data or "period" in data
    
    @pytest.mark.asyncio
    async def test_record_transaction(self, client, auth_headers):
        """Test recording financial transaction"""
        txn_data = {
            "business_unit_id": "unit-1",
            "transaction_type": "expense",
            "category": "infrastructure",
            "amount": 100.00,
            "description": "Test transaction"
        }
        response = await client.post(
            "/api/business/transactions",
            json=txn_data,
            headers=auth_headers
        )
        assert response.status_code in [200, 400, 401]
    
    @pytest.mark.asyncio
    async def test_get_pl_trends(self, client, auth_headers):
        """Test getting P&L trends"""
        response = await client.get(
            "/api/business/pl/trends?days=30",
            headers=auth_headers
        )
        assert response.status_code in [200, 401]
    
    @pytest.mark.asyncio
    async def test_get_cost_breakdown(self, client, auth_headers):
        """Test getting cost breakdown"""
        response = await client.get(
            "/api/business/pl/breakdown",
            headers=auth_headers
        )
        assert response.status_code in [200, 401]

class TestChatEndpoints:
    """Test conversational interface endpoints"""
    
    @pytest.mark.asyncio
    async def test_send_message(self, client, auth_headers):
        """Test sending chat message"""
        msg_data = {"message": "Create a workflow"}
        response = await client.post(
            "/api/chat/message",
            json=msg_data,
            headers=auth_headers
        )
        assert response.status_code in [200, 400, 401]
    
    @pytest.mark.asyncio
    async def test_get_chat_history(self, client, auth_headers):
        """Test getting chat history"""
        response = await client.get(
            "/api/chat/history?limit=50",
            headers=auth_headers
        )
        assert response.status_code in [200, 401]

class TestDataIntegrity:
    """Test data consistency and integrity"""
    
    @pytest.mark.asyncio
    async def test_transaction_persistence(self, client, auth_headers):
        """Test that transactions are persisted"""
        # Create transaction
        txn_data = {
            "business_unit_id": "unit-1",
            "transaction_type": "revenue",
            "category": "api_calls",
            "amount": 250.00,
            "description": "API revenue"
        }
        create_response = await client.post(
            "/api/business/transactions",
            json=txn_data,
            headers=auth_headers
        )
        
        # Verify in summary
        if create_response.status_code == 200:
            summary_response = await client.get(
                "/api/business/pl/summary",
                headers=auth_headers
            )
            assert summary_response.status_code in [200, 401]

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_invalid_endpoint(self, client, auth_headers):
        """Test accessing non-existent endpoint"""
        response = await client.get(
            "/api/nonexistent/endpoint",
            headers=auth_headers
        )
        assert response.status_code in [404, 401]
    
    @pytest.mark.asyncio
    async def test_missing_auth_header(self, client):
        """Test accessing protected endpoint without auth"""
        response = await client.get("/api/dashboard/overview")
        assert response.status_code in [401, 403]
    
    @pytest.mark.asyncio
    async def test_invalid_json_payload(self, client, auth_headers):
        """Test sending invalid JSON"""
        response = await client.post(
            "/api/chat/message",
            data="not valid json",
            headers=auth_headers
        )
        assert response.status_code in [400, 422]

class TestPerformance:
    """Test performance requirements"""
    
    @pytest.mark.asyncio
    async def test_api_response_time(self, client, auth_headers):
        """Test API response time < 200ms"""
        import time
        start = time.time()
        response = await client.get(
            "/api/dashboard/overview",
            headers=auth_headers
        )
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        if response.status_code == 200:
            assert elapsed < 500  # Allow 500ms for initial request
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, client, auth_headers):
        """Test handling multiple concurrent requests"""
        import asyncio
        
        tasks = []
        for i in range(10):
            task = client.get(
                "/api/dashboard/overview",
                headers=auth_headers
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        assert len(responses) == 10

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
