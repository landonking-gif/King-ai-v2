"""Integration tests for Lifecycle API Routes."""
import pytest
from fastapi.testclient import TestClient
from src.api.routes.lifecycle import router, get_engine, _engine
from src.business.lifecycle import EnhancedLifecycleEngine


@pytest.fixture
def reset_engine():
    """Reset the global engine between tests."""
    global _engine
    import src.api.routes.lifecycle as lifecycle_module
    lifecycle_module._engine = None
    yield
    lifecycle_module._engine = None


@pytest.fixture
def client():
    """Create test client for the lifecycle router."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestLifecycleAPI:
    def test_list_stages(self, client, reset_engine):
        """Test listing all lifecycle stages."""
        response = client.get("/lifecycle/stages")
        assert response.status_code == 200
        data = response.json()
        assert "stages" in data
        assert len(data["stages"]) == 8  # All lifecycle stages

    def test_get_stage_info(self, client, reset_engine):
        """Test getting info for a specific stage."""
        response = client.get("/lifecycle/stages/growth")
        assert response.status_code == 200
        data = response.json()
        assert data["stage"] == "growth"
        assert "description" in data
        assert data["typical_duration_days"] == 180

    def test_get_invalid_stage_info(self, client, reset_engine):
        """Test getting info for an invalid stage."""
        response = client.get("/lifecycle/stages/invalid")
        assert response.status_code == 400

    def test_initialize_business(self, client, reset_engine):
        """Test initializing a business lifecycle."""
        response = client.post("/lifecycle/init", json={
            "business_id": "test_biz_1",
            "initial_stage": "ideation"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["business_id"] == "test_biz_1"
        assert data["stage"] == "ideation"
        assert data["milestones"] > 0

    def test_initialize_business_invalid_stage(self, client, reset_engine):
        """Test initializing with invalid stage."""
        response = client.post("/lifecycle/init", json={
            "business_id": "test_biz_2",
            "initial_stage": "invalid"
        })
        assert response.status_code == 400

    def test_get_state(self, client, reset_engine):
        """Test getting business state."""
        # Initialize first
        client.post("/lifecycle/init", json={
            "business_id": "test_biz_3",
            "initial_stage": "ideation"
        })
        
        # Get state
        response = client.get("/lifecycle/state/test_biz_3")
        assert response.status_code == 200
        data = response.json()
        assert data["business_id"] == "test_biz_3"
        assert data["stage"] == "ideation"
        assert "milestones" in data
        assert "health_score" in data

    def test_get_state_not_found(self, client, reset_engine):
        """Test getting state for nonexistent business."""
        response = client.get("/lifecycle/state/nonexistent")
        assert response.status_code == 404

    def test_transition(self, client, reset_engine):
        """Test transitioning to a new stage."""
        # Initialize first
        client.post("/lifecycle/init", json={
            "business_id": "test_biz_4",
            "initial_stage": "ideation"
        })
        
        # Transition with force
        response = client.post("/lifecycle/transition", json={
            "business_id": "test_biz_4",
            "to_stage": "validation",
            "notes": "Test transition",
            "force": True
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_transition_invalid(self, client, reset_engine):
        """Test invalid transition."""
        # Initialize first
        client.post("/lifecycle/init", json={
            "business_id": "test_biz_5",
            "initial_stage": "ideation"
        })
        
        # Try invalid transition without force
        response = client.post("/lifecycle/transition", json={
            "business_id": "test_biz_5",
            "to_stage": "scale",
            "force": False
        })
        assert response.status_code == 400

    def test_update_milestone(self, client, reset_engine):
        """Test updating milestone progress."""
        # Initialize first
        init_response = client.post("/lifecycle/init", json={
            "business_id": "test_biz_6",
            "initial_stage": "ideation"
        })
        
        # Get state to find a milestone
        state_response = client.get("/lifecycle/state/test_biz_6")
        milestones = state_response.json()["milestones"]
        assert len(milestones) > 0
        
        milestone_id = milestones[0]["id"]
        
        # Update milestone
        response = client.post("/lifecycle/milestones/update", json={
            "business_id": "test_biz_6",
            "milestone_id": milestone_id,
            "current_value": 0.5
        })
        assert response.status_code == 200
        data = response.json()
        assert "progress" in data

    def test_add_milestone(self, client, reset_engine):
        """Test adding a custom milestone."""
        # Initialize first
        client.post("/lifecycle/init", json={
            "business_id": "test_biz_7",
            "initial_stage": "ideation"
        })
        
        # Add milestone
        response = client.post("/lifecycle/milestones/add", json={
            "business_id": "test_biz_7",
            "name": "Custom Goal",
            "milestone_type": "revenue",
            "target_value": 10000
        })
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Custom Goal"

    def test_get_health(self, client, reset_engine):
        """Test getting business health score."""
        # Initialize first
        client.post("/lifecycle/init", json={
            "business_id": "test_biz_8",
            "initial_stage": "ideation"
        })
        
        # Get health
        response = client.get("/lifecycle/health/test_biz_8")
        assert response.status_code == 200
        data = response.json()
        assert "health_score" in data
        assert 0 <= data["health_score"] <= 100

    def test_get_recommendations(self, client, reset_engine):
        """Test getting recommendations."""
        # Initialize first
        client.post("/lifecycle/init", json={
            "business_id": "test_biz_9",
            "initial_stage": "ideation"
        })
        
        # Get recommendations
        response = client.get("/lifecycle/recommendations/test_biz_9")
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "count" in data

    def test_full_workflow(self, client, reset_engine):
        """Test a complete workflow from init to transition."""
        business_id = "test_biz_workflow"
        
        # 1. Initialize
        init_resp = client.post("/lifecycle/init", json={
            "business_id": business_id,
            "initial_stage": "ideation"
        })
        assert init_resp.status_code == 200
        
        # 2. Get state
        state_resp = client.get(f"/lifecycle/state/{business_id}")
        assert state_resp.status_code == 200
        state = state_resp.json()
        
        # 3. Complete a milestone
        milestone_id = state["milestones"][0]["id"]
        target = state["milestones"][0]["target"]
        update_resp = client.post("/lifecycle/milestones/update", json={
            "business_id": business_id,
            "milestone_id": milestone_id,
            "current_value": target
        })
        assert update_resp.status_code == 200
        
        # 4. Check health
        health_resp = client.get(f"/lifecycle/health/{business_id}")
        assert health_resp.status_code == 200
        
        # 5. Get recommendations
        rec_resp = client.get(f"/lifecycle/recommendations/{business_id}")
        assert rec_resp.status_code == 200
        
        # 6. Transition
        trans_resp = client.post("/lifecycle/transition", json={
            "business_id": business_id,
            "to_stage": "validation",
            "force": True
        })
        assert trans_resp.status_code == 200
