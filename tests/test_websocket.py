"""Tests for WebSocket and Monitoring."""
import pytest
from src.api.websocket import ConnectionManager
from src.api.events import EventBroadcaster, EventType
from src.monitoring.monitor import SystemMonitor


@pytest.fixture
def connection_manager():
    return ConnectionManager()


@pytest.fixture
def broadcaster():
    return EventBroadcaster()


@pytest.fixture
def monitor():
    return SystemMonitor()


class TestConnectionManager:
    def test_get_stats_empty(self, connection_manager):
        stats = connection_manager.get_stats()
        assert stats["total_connections"] == 0

    @pytest.mark.asyncio
    async def test_subscribe(self, connection_manager):
        await connection_manager.subscribe("conn_1", "test_channel")
        assert "conn_1" in connection_manager._subscriptions["test_channel"]

    @pytest.mark.asyncio
    async def test_unsubscribe(self, connection_manager):
        await connection_manager.subscribe("conn_1", "test_channel")
        await connection_manager.unsubscribe("conn_1", "test_channel")
        assert "conn_1" not in connection_manager._subscriptions.get("test_channel", set())


class TestSystemMonitor:
    @pytest.mark.asyncio
    async def test_collect_metrics(self, monitor):
        metrics = await monitor.collect_metrics()
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.disk_percent >= 0

    def test_register_health_check(self, monitor):
        async def dummy_check():
            return "healthy", "OK"
        
        monitor.register_health_check("test", dummy_check)
        assert "test" in monitor._health_checks

    @pytest.mark.asyncio
    async def test_check_health(self, monitor):
        async def dummy_check():
            return "healthy", "Test passed"
        
        monitor.register_health_check("dummy", dummy_check)
        results = await monitor.check_health()
        
        assert "dummy" in results
        assert results["dummy"].status == "healthy"

    def test_get_recent_alerts(self, monitor):
        monitor._alerts = [{"level": "warning", "message": "test"}]
        alerts = monitor.get_recent_alerts()
        assert len(alerts) == 1


class TestEventBroadcaster:
    @pytest.mark.asyncio
    async def test_emit_event(self, broadcaster):
        # Just test it doesn't throw
        await broadcaster.emit(
            EventType.BUSINESS_UPDATED,
            {"test": "data"},
            business_id="biz_1",
        )

    @pytest.mark.asyncio
    async def test_emit_system_alert(self, broadcaster):
        await broadcaster.emit_system_alert(
            "warning",
            "Test alert",
            {"detail": "test"},
        )
