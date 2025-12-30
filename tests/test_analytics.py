"""Tests for Analytics Agent."""
import pytest
from datetime import datetime, timedelta
from src.analytics.models import KPI, TimeSeries, TimeGranularity, MetricValue
from src.analytics.collector import MetricsCollector, STANDARD_METRICS
from src.agents.analytics import AnalyticsAgent


class TestKPI:
    def test_progress_percent(self):
        kpi = KPI(name="Revenue", current_value=8000, target_value=10000, previous_value=7000)
        assert kpi.progress_percent == 80.0

    def test_change_percent(self):
        kpi = KPI(name="Revenue", current_value=8000, target_value=10000, previous_value=7000)
        assert round(kpi.change_percent, 2) == 14.29


class TestTimeSeries:
    def test_average(self):
        ts = TimeSeries(
            metric_name="test",
            values=[(datetime.now(), 10), (datetime.now(), 20), (datetime.now(), 30)],
            granularity=TimeGranularity.DAILY,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert ts.average() == 20.0

    def test_trend(self):
        ts = TimeSeries(
            metric_name="test",
            values=[(datetime.now(), 100), (datetime.now(), 150)],
            granularity=TimeGranularity.DAILY,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert ts.trend() == 50.0


class TestMetricsCollector:
    @pytest.fixture
    def collector(self):
        return MetricsCollector()

    def test_standard_metrics_loaded(self, collector):
        assert "revenue_total" in collector.metrics
        assert "conversion_rate" in collector.metrics

    @pytest.mark.asyncio
    async def test_record_metric(self, collector):
        await collector.record("revenue_total", 1000)
        assert len(collector._cache["revenue_total"]) == 1

    def test_list_metrics_by_category(self, collector):
        from src.analytics.models import MetricCategory
        revenue_metrics = collector.list_metrics(MetricCategory.REVENUE)
        assert len(revenue_metrics) > 0


class TestAnalyticsAgent:
    @pytest.fixture
    def agent(self):
        return AnalyticsAgent()

    def test_set_kpi_target(self, agent):
        agent.set_kpi_target("biz_1", "revenue_total", 100000)
        assert agent._kpi_targets["biz_1"]["revenue_total"] == 100000

    @pytest.mark.asyncio
    async def test_get_dashboard_metrics(self, agent):
        result = await agent.get_dashboard_metrics("biz_1")
        assert result["success"]
        assert "revenue" in result["data"]
        assert "traffic" in result["data"]

    @pytest.mark.asyncio
    async def test_generate_report(self, agent):
        result = await agent.generate_report("biz_1", 30, "Test Report")
        assert result["success"]
        assert result["data"]["title"] == "Test Report"
        assert "sections" in result["data"]
        assert "insights" in result["data"]

    @pytest.mark.asyncio
    async def test_record_metric(self, agent):
        result = await agent.record_metric("revenue_total", 5000)
        assert result["success"]

    @pytest.mark.asyncio
    async def test_execute_legacy_format(self, agent):
        """Test backward compatibility with legacy task format."""
        task = {
            "description": "Analyze sales data",
            "input_data": {"sales": 1000, "profit": 200}
        }
        result = await agent.execute(task)
        # Should use LLM for general analysis
        assert "success" in result

    @pytest.mark.asyncio
    async def test_execute_action_format(self, agent):
        """Test new action-based task format."""
        task = {
            "action": "dashboard",
            "business_id": "test_biz"
        }
        result = await agent.execute(task)
        assert result["success"]
        assert "data" in result
