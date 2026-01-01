"""
Agent Performance Metrics.
Per-agent performance tracking and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum
import statistics
import time

from src.utils.structured_logging import get_logger

logger = get_logger("agent_metrics")


class MetricAggregation(str, Enum):
    """Aggregation methods."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_50 = "p50"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


@dataclass
class TaskExecution:
    """Record of a single task execution."""
    task_id: str
    agent_id: str
    task_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None
    
    # Resource usage
    tokens_used: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    
    # Quality metrics
    confidence: float = 0.0
    user_rating: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Aggregated metrics for an agent."""
    agent_id: str
    agent_type: str
    
    # Execution stats
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    
    # Performance
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    
    # Resource usage
    total_tokens: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    
    # Quality
    avg_confidence: float = 0.0
    avg_user_rating: float = 0.0
    
    # Calculated
    success_rate: float = 0.0
    tokens_per_task: float = 0.0
    
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": round(self.success_rate * 100, 1),
            "avg_duration_ms": round(self.avg_duration_ms, 1),
            "p95_duration_ms": round(self.p95_duration_ms, 1),
            "total_tokens": self.total_tokens,
            "tokens_per_task": round(self.tokens_per_task, 1),
            "avg_confidence": round(self.avg_confidence, 2),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
        }


@dataclass
class PerformanceTrend:
    """Performance trend over time."""
    metric_name: str
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    trend_direction: str = "stable"  # up, down, stable
    trend_percentage: float = 0.0


@dataclass
class PerformanceAlert:
    """Alert for performance issues."""
    alert_id: str
    agent_id: str
    alert_type: str
    severity: str  # warning, critical
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    triggered_at: datetime = field(default_factory=datetime.utcnow)


class MetricsStore:
    """Store for agent execution metrics."""
    
    def __init__(self, max_history_hours: int = 168):  # 7 days default
        self.executions: Dict[str, List[TaskExecution]] = {}
        self.max_history_hours = max_history_hours
    
    def record(self, execution: TaskExecution) -> None:
        """Record a task execution."""
        if execution.agent_id not in self.executions:
            self.executions[execution.agent_id] = []
        
        self.executions[execution.agent_id].append(execution)
        self._cleanup_old_entries(execution.agent_id)
    
    def get_executions(
        self,
        agent_id: str,
        hours: int = 24,
    ) -> List[TaskExecution]:
        """Get executions for an agent within time window."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            e for e in self.executions.get(agent_id, [])
            if e.start_time >= cutoff
        ]
    
    def get_all_executions(self, hours: int = 24) -> List[TaskExecution]:
        """Get all executions within time window."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        all_executions = []
        for executions in self.executions.values():
            all_executions.extend([
                e for e in executions if e.start_time >= cutoff
            ])
        
        return all_executions
    
    def _cleanup_old_entries(self, agent_id: str) -> None:
        """Remove old entries beyond max history."""
        cutoff = datetime.utcnow() - timedelta(hours=self.max_history_hours)
        
        self.executions[agent_id] = [
            e for e in self.executions[agent_id]
            if e.start_time >= cutoff
        ]


class AgentPerformanceTracker:
    """
    Agent Performance Tracking.
    
    Features:
    - Task execution recording
    - Performance aggregation
    - Trend analysis
    - Alerting
    - Comparisons
    """
    
    def __init__(self):
        self.store = MetricsStore()
        self.agent_types: Dict[str, str] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.alerts: List[PerformanceAlert] = []
        
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self) -> None:
        """Set up default alert thresholds."""
        self.thresholds["default"] = {
            "success_rate_min": 0.90,
            "avg_duration_ms_max": 10000,
            "p95_duration_ms_max": 30000,
            "tokens_per_task_max": 5000,
        }
    
    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        thresholds: Dict[str, float] = None,
    ) -> None:
        """
        Register an agent for tracking.
        
        Args:
            agent_id: Agent identifier
            agent_type: Type of agent
            thresholds: Custom thresholds for this agent
        """
        self.agent_types[agent_id] = agent_type
        
        if thresholds:
            self.thresholds[agent_id] = thresholds
        
        logger.info(f"Registered agent for tracking: {agent_id}")
    
    def start_task(
        self,
        agent_id: str,
        task_type: str,
        task_id: str = None,
    ) -> TaskExecution:
        """
        Start tracking a task execution.
        
        Args:
            agent_id: Agent identifier
            task_type: Type of task
            task_id: Optional task ID
            
        Returns:
            Task execution record
        """
        import uuid
        
        execution = TaskExecution(
            task_id=task_id or str(uuid.uuid4())[:8],
            agent_id=agent_id,
            task_type=task_type,
            start_time=datetime.utcnow(),
        )
        
        return execution
    
    def complete_task(
        self,
        execution: TaskExecution,
        success: bool = True,
        error: str = None,
        tokens_used: int = 0,
        llm_calls: int = 0,
        tool_calls: int = 0,
        confidence: float = 0.0,
    ) -> None:
        """
        Complete a task execution.
        
        Args:
            execution: The task execution
            success: Whether task succeeded
            error: Error message if failed
            tokens_used: Tokens consumed
            llm_calls: Number of LLM calls
            tool_calls: Number of tool calls
            confidence: Confidence score
        """
        execution.end_time = datetime.utcnow()
        execution.duration_ms = int(
            (execution.end_time - execution.start_time).total_seconds() * 1000
        )
        execution.success = success
        execution.error = error
        execution.tokens_used = tokens_used
        execution.llm_calls = llm_calls
        execution.tool_calls = tool_calls
        execution.confidence = confidence
        
        self.store.record(execution)
        
        # Check for alerts
        self._check_alerts(execution)
    
    def track_execution(self, agent_id: str, task_type: str = "default"):
        """
        Context manager for tracking task execution.
        
        Usage:
            async with tracker.track_execution("agent_1", "query") as task:
                # Do work
                task.tokens_used = 100
        """
        class ExecutionContext:
            def __init__(ctx, tracker, agent_id, task_type):
                ctx.tracker = tracker
                ctx.execution = tracker.start_task(agent_id, task_type)
                ctx.success = True
                ctx.error = None
            
            def __enter__(ctx):
                return ctx.execution
            
            def __exit__(ctx, exc_type, exc_val, exc_tb):
                if exc_type:
                    ctx.success = False
                    ctx.error = str(exc_val)
                
                ctx.tracker.complete_task(
                    ctx.execution,
                    success=ctx.success,
                    error=ctx.error,
                    tokens_used=ctx.execution.tokens_used,
                    llm_calls=ctx.execution.llm_calls,
                    tool_calls=ctx.execution.tool_calls,
                    confidence=ctx.execution.confidence,
                )
                return False
        
        return ExecutionContext(self, agent_id, task_type)
    
    def get_agent_metrics(
        self,
        agent_id: str,
        hours: int = 24,
    ) -> AgentMetrics:
        """
        Get aggregated metrics for an agent.
        
        Args:
            agent_id: Agent identifier
            hours: Time window in hours
            
        Returns:
            Aggregated metrics
        """
        executions = self.store.get_executions(agent_id, hours)
        
        if not executions:
            return AgentMetrics(
                agent_id=agent_id,
                agent_type=self.agent_types.get(agent_id, "unknown"),
            )
        
        durations = [e.duration_ms for e in executions]
        successful = [e for e in executions if e.success]
        ratings = [e.user_rating for e in executions if e.user_rating is not None]
        
        metrics = AgentMetrics(
            agent_id=agent_id,
            agent_type=self.agent_types.get(agent_id, "unknown"),
            total_tasks=len(executions),
            successful_tasks=len(successful),
            failed_tasks=len(executions) - len(successful),
            avg_duration_ms=statistics.mean(durations),
            p50_duration_ms=self._percentile(durations, 50),
            p95_duration_ms=self._percentile(durations, 95),
            p99_duration_ms=self._percentile(durations, 99),
            total_tokens=sum(e.tokens_used for e in executions),
            total_llm_calls=sum(e.llm_calls for e in executions),
            total_tool_calls=sum(e.tool_calls for e in executions),
            avg_confidence=statistics.mean([e.confidence for e in executions]) if executions else 0,
            avg_user_rating=statistics.mean(ratings) if ratings else 0,
            success_rate=len(successful) / len(executions),
            tokens_per_task=sum(e.tokens_used for e in executions) / len(executions),
            period_start=min(e.start_time for e in executions),
            period_end=max(e.end_time or e.start_time for e in executions),
        )
        
        return metrics
    
    def get_all_metrics(self, hours: int = 24) -> List[AgentMetrics]:
        """Get metrics for all tracked agents."""
        metrics = []
        
        for agent_id in set(self.agent_types.keys()) | set(self.store.executions.keys()):
            agent_metrics = self.get_agent_metrics(agent_id, hours)
            if agent_metrics.total_tasks > 0:
                metrics.append(agent_metrics)
        
        return metrics
    
    def get_performance_trend(
        self,
        agent_id: str,
        metric_name: str,
        hours: int = 168,
        interval_hours: int = 24,
    ) -> PerformanceTrend:
        """
        Get performance trend for a metric.
        
        Args:
            agent_id: Agent identifier
            metric_name: Metric to trend (e.g., "avg_duration_ms", "success_rate")
            hours: Total time window
            interval_hours: Interval for data points
            
        Returns:
            Performance trend
        """
        data_points = []
        now = datetime.utcnow()
        
        for i in range(hours // interval_hours):
            end = now - timedelta(hours=i * interval_hours)
            start = end - timedelta(hours=interval_hours)
            
            # Get executions for this interval
            executions = [
                e for e in self.store.get_executions(agent_id, hours)
                if start <= e.start_time < end
            ]
            
            if executions:
                metrics = self._calculate_interval_metrics(executions)
                value = metrics.get(metric_name, 0)
                
                data_points.append({
                    "timestamp": start.isoformat(),
                    "value": value,
                    "sample_size": len(executions),
                })
        
        # Calculate trend
        data_points.reverse()  # Chronological order
        
        if len(data_points) >= 2:
            values = [dp["value"] for dp in data_points]
            first_half = statistics.mean(values[:len(values)//2])
            second_half = statistics.mean(values[len(values)//2:])
            
            if first_half > 0:
                change = (second_half - first_half) / first_half
                if change > 0.1:
                    direction = "up"
                elif change < -0.1:
                    direction = "down"
                else:
                    direction = "stable"
            else:
                direction = "stable"
                change = 0
        else:
            direction = "stable"
            change = 0
        
        return PerformanceTrend(
            metric_name=metric_name,
            data_points=data_points,
            trend_direction=direction,
            trend_percentage=change * 100,
        )
    
    def _calculate_interval_metrics(
        self,
        executions: List[TaskExecution],
    ) -> Dict[str, float]:
        """Calculate metrics for an interval."""
        if not executions:
            return {}
        
        durations = [e.duration_ms for e in executions]
        successful = len([e for e in executions if e.success])
        
        return {
            "avg_duration_ms": statistics.mean(durations),
            "p95_duration_ms": self._percentile(durations, 95),
            "success_rate": successful / len(executions),
            "total_tokens": sum(e.tokens_used for e in executions),
            "avg_confidence": statistics.mean([e.confidence for e in executions]),
        }
    
    def _check_alerts(self, execution: TaskExecution) -> None:
        """Check if execution triggers any alerts."""
        agent_id = execution.agent_id
        thresholds = self.thresholds.get(agent_id, self.thresholds["default"])
        
        # Check for slow execution
        if execution.duration_ms > thresholds.get("avg_duration_ms_max", 10000) * 2:
            self._create_alert(
                agent_id=agent_id,
                alert_type="slow_execution",
                severity="warning",
                message=f"Slow task execution: {execution.duration_ms}ms",
                metric_name="duration_ms",
                current_value=execution.duration_ms,
                threshold_value=thresholds.get("avg_duration_ms_max", 10000),
            )
        
        # Check for high token usage
        if execution.tokens_used > thresholds.get("tokens_per_task_max", 5000):
            self._create_alert(
                agent_id=agent_id,
                alert_type="high_tokens",
                severity="warning",
                message=f"High token usage: {execution.tokens_used}",
                metric_name="tokens_used",
                current_value=execution.tokens_used,
                threshold_value=thresholds.get("tokens_per_task_max", 5000),
            )
        
        # Check success rate periodically
        recent = self.store.get_executions(agent_id, 1)
        if len(recent) >= 10:
            success_rate = sum(1 for e in recent if e.success) / len(recent)
            if success_rate < thresholds.get("success_rate_min", 0.90):
                self._create_alert(
                    agent_id=agent_id,
                    alert_type="low_success_rate",
                    severity="critical",
                    message=f"Low success rate: {success_rate:.1%}",
                    metric_name="success_rate",
                    current_value=success_rate,
                    threshold_value=thresholds.get("success_rate_min", 0.90),
                )
    
    def _create_alert(
        self,
        agent_id: str,
        alert_type: str,
        severity: str,
        message: str,
        metric_name: str,
        current_value: float,
        threshold_value: float,
    ) -> None:
        """Create a performance alert."""
        import uuid
        
        # Check for duplicate recent alerts
        recent_alerts = [
            a for a in self.alerts[-20:]
            if a.agent_id == agent_id
            and a.alert_type == alert_type
            and (datetime.utcnow() - a.triggered_at).seconds < 300
        ]
        
        if recent_alerts:
            return  # Skip duplicate
        
        alert = PerformanceAlert(
            alert_id=str(uuid.uuid4())[:8],
            agent_id=agent_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
        )
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        logger.warning(
            f"Performance alert: {message}",
            extra={
                "agent_id": agent_id,
                "alert_type": alert_type,
                "severity": severity,
            },
        )
    
    def get_alerts(
        self,
        agent_id: str = None,
        severity: str = None,
        hours: int = 24,
    ) -> List[PerformanceAlert]:
        """Get performance alerts."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        alerts = [a for a in self.alerts if a.triggered_at >= cutoff]
        
        if agent_id:
            alerts = [a for a in alerts if a.agent_id == agent_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
    
    def compare_agents(
        self,
        agent_ids: List[str],
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Compare performance across agents.
        
        Args:
            agent_ids: Agents to compare
            hours: Time window
            
        Returns:
            Comparison data
        """
        metrics = {
            aid: self.get_agent_metrics(aid, hours)
            for aid in agent_ids
        }
        
        # Find best performers
        comparisons = {}
        
        # Success rate comparison
        if any(m.total_tasks > 0 for m in metrics.values()):
            best_success = max(
                (aid for aid, m in metrics.items() if m.total_tasks > 0),
                key=lambda x: metrics[x].success_rate,
            )
            comparisons["best_success_rate"] = {
                "agent_id": best_success,
                "value": metrics[best_success].success_rate,
            }
        
        # Speed comparison
        if any(m.total_tasks > 0 for m in metrics.values()):
            fastest = min(
                (aid for aid, m in metrics.items() if m.total_tasks > 0),
                key=lambda x: metrics[x].avg_duration_ms,
            )
            comparisons["fastest"] = {
                "agent_id": fastest,
                "value": metrics[fastest].avg_duration_ms,
            }
        
        # Efficiency comparison (success per token)
        if any(m.total_tokens > 0 for m in metrics.values()):
            most_efficient = max(
                (aid for aid, m in metrics.items() if m.total_tokens > 0),
                key=lambda x: metrics[x].successful_tasks / max(metrics[x].total_tokens, 1),
            )
            comparisons["most_efficient"] = {
                "agent_id": most_efficient,
                "value": metrics[most_efficient].tokens_per_task,
            }
        
        return {
            "agents": {aid: m.to_dict() for aid, m in metrics.items()},
            "comparisons": comparisons,
        }
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])
    
    def get_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get overall performance summary."""
        all_metrics = self.get_all_metrics(hours)
        
        if not all_metrics:
            return {
                "total_agents": 0,
                "total_tasks": 0,
                "avg_success_rate": 0,
                "avg_duration_ms": 0,
                "total_tokens": 0,
                "alerts": [],
            }
        
        return {
            "total_agents": len(all_metrics),
            "total_tasks": sum(m.total_tasks for m in all_metrics),
            "avg_success_rate": round(
                statistics.mean([m.success_rate for m in all_metrics]) * 100, 1
            ),
            "avg_duration_ms": round(
                statistics.mean([m.avg_duration_ms for m in all_metrics]), 1
            ),
            "total_tokens": sum(m.total_tokens for m in all_metrics),
            "alerts": [
                {
                    "agent_id": a.agent_id,
                    "type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                }
                for a in self.get_alerts(hours=hours)[:5]
            ],
        }


# Global tracker instance
performance_tracker = AgentPerformanceTracker()


def get_performance_tracker() -> AgentPerformanceTracker:
    """Get the global performance tracker."""
    return performance_tracker
