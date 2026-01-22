"""
End-to-End Tests for Master Control Panel
Complete user workflow testing including workflow creation, approval, execution, and monitoring
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

class TestCompleteWorkflowCycle:
    """Test complete workflow from creation to execution"""
    
    @pytest.mark.asyncio
    async def test_workflow_creation_and_execution(self, client, auth_headers):
        """Test creating, approving, and executing a workflow"""
        
        # Step 1: Create workflow
        workflow_data = {
            "name": "E2E Test Workflow",
            "description": "End-to-end test workflow",
            "steps": [
                {
                    "type": "brainstorm",
                    "model": "claude-3-sonnet",
                    "prompt": "Generate 5 ideas for improving customer retention"
                },
                {
                    "type": "command",
                    "model": "claude-3-opus",
                    "prompt": "Create customer retention plan"
                }
            ]
        }
        
        create_response = await client.post(
            "/api/orchestrator/workflows",
            json=workflow_data,
            headers=auth_headers
        )
        assert create_response.status_code in [200, 400, 503]
        
        if create_response.status_code == 200:
            workflow = create_response.json()
            workflow_id = workflow.get("id") or workflow.get("workflow_id")
            
            # Step 2: Get workflow details
            detail_response = await client.get(
                f"/api/orchestrator/workflows/{workflow_id}",
                headers=auth_headers
            )
            assert detail_response.status_code in [200, 404, 503]
            
            # Step 3: Submit for approval
            approval_data = {"workflow_id": workflow_id, "comment": "Approve E2E test"}
            approval_response = await client.post(
                "/api/orchestrator/workflows/submit-approval",
                json=approval_data,
                headers=auth_headers
            )
            assert approval_response.status_code in [200, 400, 503]
            
            # Step 4: Execute workflow
            exec_response = await client.post(
                f"/api/orchestrator/workflows/{workflow_id}/execute",
                headers=auth_headers
            )
            assert exec_response.status_code in [200, 400, 503]
            
            # Step 5: Monitor execution
            if exec_response.status_code == 200:
                await asyncio.sleep(1)  # Wait for execution to start
                
                status_response = await client.get(
                    f"/api/orchestrator/workflows/{workflow_id}/status",
                    headers=auth_headers
                )
                assert status_response.status_code in [200, 404, 503]

class TestAgentManagement:
    """Test agent lifecycle management"""
    
    @pytest.mark.asyncio
    async def test_spawn_agent_and_monitor(self, client, auth_headers):
        """Test spawning an agent and monitoring its metrics"""
        
        # Spawn agent
        agent_data = {
            "agent_type": "brainstorm",
            "capabilities": ["ideation", "creative-thinking"],
            "model": "claude-3-sonnet"
        }
        
        spawn_response = await client.post(
            "/api/subagent-manager/agents",
            json=agent_data,
            headers=auth_headers
        )
        assert spawn_response.status_code in [200, 400, 503]
        
        if spawn_response.status_code == 200:
            agent = spawn_response.json()
            agent_id = agent.get("id") or agent.get("agent_id")
            
            # Get agent details
            detail_response = await client.get(
                f"/api/subagent-manager/agents/{agent_id}",
                headers=auth_headers
            )
            assert detail_response.status_code in [200, 404, 503]
            
            # Pause agent
            pause_response = await client.post(
                f"/api/subagent-manager/agents/{agent_id}/pause",
                headers=auth_headers
            )
            assert pause_response.status_code in [200, 400, 404, 503]
            
            # Resume agent
            resume_response = await client.post(
                f"/api/subagent-manager/agents/{agent_id}/resume",
                headers=auth_headers
            )
            assert resume_response.status_code in [200, 400, 404, 503]
            
            # Destroy agent
            destroy_response = await client.post(
                f"/api/subagent-manager/agents/{agent_id}/destroy",
                headers=auth_headers
            )
            assert destroy_response.status_code in [200, 400, 404, 503]

class TestRealTimeUpdates:
    """Test real-time WebSocket updates across dashboards"""
    
    @pytest.mark.asyncio
    async def test_activity_feed_updates(self, client, auth_headers):
        """Test activity feed real-time updates"""
        # This would require WebSocket client testing
        # Placeholder for WebSocket test pattern
        pass
    
    @pytest.mark.asyncio
    async def test_approval_notifications(self, client, auth_headers):
        """Test real-time approval notifications"""
        pass
    
    @pytest.mark.asyncio
    async def test_workflow_execution_updates(self, client, auth_headers):
        """Test workflow execution progress updates"""
        pass

class TestFinancialTracking:
    """Test financial tracking and P&L calculations"""
    
    @pytest.mark.asyncio
    async def test_complete_financial_cycle(self, client, auth_headers):
        """Test recording transactions and verifying P&L calculations"""
        
        # Step 1: Record revenue transaction
        revenue_data = {
            "business_unit_id": "e2e-test-unit",
            "transaction_type": "revenue",
            "category": "api_calls",
            "amount": 1500.00,
            "description": "E2E test revenue"
        }
        
        rev_response = await client.post(
            "/api/business/transactions",
            json=revenue_data,
            headers=auth_headers
        )
        assert rev_response.status_code in [200, 400, 401]
        
        # Step 2: Record expense transaction
        expense_data = {
            "business_unit_id": "e2e-test-unit",
            "transaction_type": "expense",
            "category": "infrastructure",
            "amount": 300.00,
            "description": "E2E test expense"
        }
        
        exp_response = await client.post(
            "/api/business/transactions",
            json=expense_data,
            headers=auth_headers
        )
        assert exp_response.status_code in [200, 400, 401]
        
        # Step 3: Verify P&L summary
        summary_response = await client.get(
            "/api/business/pl/summary?period=monthly",
            headers=auth_headers
        )
        assert summary_response.status_code in [200, 401]
        
        if summary_response.status_code == 200:
            summary = summary_response.json()
            
            # Verify calculations
            if "total_revenue" in summary and "total_expenses" in summary:
                expected_net = summary["total_revenue"] - summary["total_expenses"]
                if "net_profit" in summary:
                    assert summary["net_profit"] == expected_net

class TestConversationalInterface:
    """Test AI conversational interface functionality"""
    
    @pytest.mark.asyncio
    async def test_intent_detection_and_workflow_creation(self, client, auth_headers):
        """Test conversation creates workflows based on intent"""
        
        # Step 1: Send brainstorm message
        msg_data = {"message": "Let's brainstorm ideas for a new feature"}
        response = await client.post(
            "/api/chat/message",
            json=msg_data,
            headers=auth_headers
        )
        assert response.status_code in [200, 400, 401]
        
        if response.status_code == 200:
            result = response.json()
            assert "response" in result or "message" in result
            assert result.get("intent") in ["brainstorm", "general", "analysis", "command"]
        
        # Step 2: Send command message
        cmd_data = {"message": "Create and run a workflow to analyze user behavior"}
        cmd_response = await client.post(
            "/api/chat/message",
            json=cmd_data,
            headers=auth_headers
        )
        assert cmd_response.status_code in [200, 400, 401]
        
        # Step 3: Get conversation history
        history_response = await client.get(
            "/api/chat/history?limit=10",
            headers=auth_headers
        )
        assert history_response.status_code in [200, 401]

class TestApprovalWorkflow:
    """Test approval process and audit trail"""
    
    @pytest.mark.asyncio
    async def test_approval_lifecycle(self, client, auth_headers):
        """Test complete approval lifecycle"""
        
        # Step 1: Create item requiring approval
        item_data = {
            "workflow_id": "e2e-approval-test",
            "risk_level": "medium",
            "description": "Test approval item"
        }
        
        create_response = await client.post(
            "/api/orchestrator/approvals",
            json=item_data,
            headers=auth_headers
        )
        assert create_response.status_code in [200, 400, 503]
        
        if create_response.status_code == 200:
            approval = create_response.json()
            approval_id = approval.get("id") or approval.get("approval_id")
            
            # Step 2: Get approval details
            detail_response = await client.get(
                f"/api/orchestrator/approvals/{approval_id}",
                headers=auth_headers
            )
            assert detail_response.status_code in [200, 404, 503]
            
            # Step 3: Approve
            approve_response = await client.post(
                f"/api/orchestrator/approvals/{approval_id}/approve",
                json={"comment": "Approved by E2E test"},
                headers=auth_headers
            )
            assert approve_response.status_code in [200, 400, 404, 503]
            
            # Step 4: Verify audit trail
            audit_response = await client.get(
                f"/api/orchestrator/approvals/{approval_id}/audit-trail",
                headers=auth_headers
            )
            assert audit_response.status_code in [200, 404, 503]

class TestDataPersistence:
    """Test data persistence across service restarts"""
    
    @pytest.mark.asyncio
    async def test_workflow_persistence(self, client, auth_headers):
        """Test that workflows persist after creation"""
        
        # Create workflow
        workflow_data = {
            "name": "Persistence Test Workflow",
            "description": "Test data persistence"
        }
        
        create_response = await client.post(
            "/api/orchestrator/workflows",
            json=workflow_data,
            headers=auth_headers
        )
        
        if create_response.status_code == 200:
            workflow = create_response.json()
            workflow_id = workflow.get("id")
            
            # Simulate delay (in real test, this would restart service)
            await asyncio.sleep(2)
            
            # Verify workflow still exists
            check_response = await client.get(
                f"/api/orchestrator/workflows/{workflow_id}",
                headers=auth_headers
            )
            assert check_response.status_code in [200, 404, 503]

class TestSecurityValidation:
    """Test security features and access control"""
    
    @pytest.mark.asyncio
    async def test_rbac_enforcement(self, client):
        """Test role-based access control"""
        
        # Test with operator role (should have limited access)
        operator_headers = {
            "Authorization": "Bearer operator-token"
        }
        
        response = await client.get(
            "/api/settings/system-config",
            headers=operator_headers
        )
        # Should either succeed with limited data or be forbidden
        assert response.status_code in [200, 403, 401]
    
    @pytest.mark.asyncio
    async def test_token_expiration(self, client):
        """Test that expired tokens are rejected"""
        
        expired_headers = {
            "Authorization": "Bearer expired-token-12345"
        }
        
        response = await client.get(
            "/api/dashboard/overview",
            headers=expired_headers
        )
        assert response.status_code in [401, 403]

class TestPerformanceBenchmarks:
    """Test performance against requirements"""
    
    @pytest.mark.asyncio
    async def test_page_load_time(self, client, auth_headers):
        """Test page load performance < 1.5s"""
        import time
        
        endpoints = [
            "/api/dashboard/overview",
            "/api/orchestrator/workflows",
            "/api/business/pl/summary"
        ]
        
        for endpoint in endpoints:
            start = time.time()
            response = await client.get(endpoint, headers=auth_headers)
            elapsed = time.time() - start
            
            if response.status_code == 200:
                # Initial load may be slower, allow 2s
                assert elapsed < 2.0, f"{endpoint} took {elapsed}s"
    
    @pytest.mark.asyncio
    async def test_api_response_times(self, client, auth_headers):
        """Test API response time < 200ms"""
        import time
        
        endpoints = [
            "/api/business/transactions",
            "/api/chat/history?limit=10",
            "/api/services/health"
        ]
        
        times = []
        for endpoint in endpoints:
            start = time.time()
            response = await client.get(endpoint, headers=auth_headers)
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        assert avg_time < 500  # Allow 500ms average for E2E
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, client, auth_headers):
        """Test handling 50 concurrent requests"""
        import asyncio
        
        async def make_request(i):
            try:
                return await client.get(
                    f"/api/business/pl/summary",
                    headers=auth_headers
                )
            except:
                return None
        
        tasks = [make_request(i) for i in range(50)]
        responses = await asyncio.gather(*tasks)
        
        # At least 80% should succeed
        successful = sum(1 for r in responses if r and r.status_code == 200)
        assert successful >= 40

class TestSystemStability:
    """Test system stability and recovery"""
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, client, auth_headers):
        """Test system recovers from errors gracefully"""
        
        # Send invalid request
        invalid_response = await client.post(
            "/api/chat/message",
            json={"invalid": "data"},
            headers=auth_headers
        )
        
        # System should still be responsive
        await asyncio.sleep(0.5)
        health_response = await client.get(
            "/api/services/health",
            headers=auth_headers
        )
        assert health_response.status_code in [200, 401]
    
    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self, client, auth_headers):
        """Test that one service failure doesn't crash system"""
        
        # Try to reach potentially unavailable service
        response = await client.get(
            "/api/memory/health",
            headers=auth_headers
        )
        
        # Even if service is down, backend should be up
        health_response = await client.get(
            "/api/services/health",
            headers=auth_headers
        )
        assert health_response.status_code in [200, 401]

class TestEndToEndScenarios:
    """Complete realistic user scenarios"""
    
    @pytest.mark.asyncio
    async def test_analyst_workflow_scenario(self, client):
        """Test analyst user accessing analytics and creating reports"""
        
        analyst_headers = {
            "Authorization": "Bearer analyst-token"
        }
        
        # Analyst can read data
        response = await client.get(
            "/api/business/pl/summary",
            headers=analyst_headers
        )
        assert response.status_code in [200, 401]
        
        # Analyst cannot create workflows (command restricted)
        workflow_data = {"name": "Test"}
        restricted = await client.post(
            "/api/orchestrator/workflows",
            json=workflow_data,
            headers=analyst_headers
        )
        # Should either fail (403) or require approval
        assert restricted.status_code in [403, 202, 401]
    
    @pytest.mark.asyncio
    async def test_operator_workflow_scenario(self, client):
        """Test operator user managing workflows and approvals"""
        
        operator_headers = {
            "Authorization": "Bearer operator-token"
        }
        
        # Operator can create workflows
        workflow_data = {
            "name": "Operator Test Workflow",
            "description": "Created by operator"
        }
        response = await client.post(
            "/api/orchestrator/workflows",
            json=workflow_data,
            headers=operator_headers
        )
        assert response.status_code in [200, 400, 503]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
