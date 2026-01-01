"""
Log Aggregator.
Centralized log collection and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Pattern
from enum import Enum
from collections import defaultdict
import re
import json
import asyncio
from asyncio import Queue

from src.utils.structured_logging import get_logger

logger = get_logger("log_aggregator")


class LogLevel(str, Enum):
    """Log levels."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    
    @property
    def severity(self) -> int:
        """Get numeric severity."""
        levels = {
            "trace": 0,
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }
        return levels.get(self.value, 20)


@dataclass
class LogEntry:
    """A log entry."""
    id: str
    timestamp: datetime
    level: LogLevel
    message: str
    source: str = ""  # service/component name
    logger_name: str = ""
    
    # Context
    trace_id: str = ""
    span_id: str = ""
    user_id: str = ""
    request_id: str = ""
    
    # Extra data
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # Exception info
    exception_type: str = ""
    exception_message: str = ""
    stacktrace: str = ""
    
    # Metadata
    host: str = ""
    environment: str = ""
    version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "logger_name": self.logger_name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "extra": self.extra,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "host": self.host,
            "environment": self.environment,
        }
    
    def matches(self, query: 'LogQuery') -> bool:
        """Check if entry matches a query."""
        # Level filter
        if query.min_level and self.level.severity < query.min_level.severity:
            return False
        
        # Time filter
        if query.start_time and self.timestamp < query.start_time:
            return False
        if query.end_time and self.timestamp > query.end_time:
            return False
        
        # Source filter
        if query.sources and self.source not in query.sources:
            return False
        
        # Text search
        if query.search_text:
            text_lower = query.search_text.lower()
            if text_lower not in self.message.lower():
                if not any(text_lower in str(v).lower() for v in self.extra.values()):
                    return False
        
        # Regex search
        if query.search_pattern:
            if not query.search_pattern.search(self.message):
                return False
        
        # Trace/request filter
        if query.trace_id and self.trace_id != query.trace_id:
            return False
        if query.request_id and self.request_id != query.request_id:
            return False
        
        # Error only
        if query.errors_only and self.level.severity < LogLevel.ERROR.severity:
            return False
        
        return True


@dataclass
class LogQuery:
    """Query parameters for searching logs."""
    min_level: Optional[LogLevel] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    sources: Optional[List[str]] = None
    search_text: Optional[str] = None
    search_pattern: Optional[Pattern] = None
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    errors_only: bool = False
    limit: int = 100
    offset: int = 0


@dataclass
class LogStats:
    """Statistics about logs."""
    total_count: int = 0
    by_level: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0
    logs_per_minute: float = 0.0
    top_errors: List[Dict[str, Any]] = field(default_factory=list)
    time_range: Optional[timedelta] = None


@dataclass
class LogAlert:
    """An alert triggered by log patterns."""
    id: str
    name: str
    pattern: str
    threshold: int = 1  # Trigger if pattern appears this many times
    window_minutes: int = 5
    level: LogLevel = LogLevel.ERROR
    enabled: bool = True
    callback: Optional[Callable] = None
    
    # State
    trigger_count: int = 0
    last_triggered: Optional[datetime] = None


class RingBuffer:
    """Ring buffer for log entries."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: List[LogEntry] = []
        self.index = 0
    
    def append(self, entry: LogEntry) -> None:
        """Add an entry."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(entry)
        else:
            self.buffer[self.index] = entry
        self.index = (self.index + 1) % self.max_size
    
    def get_all(self) -> List[LogEntry]:
        """Get all entries in order."""
        if len(self.buffer) < self.max_size:
            return self.buffer.copy()
        
        # Reorder circular buffer
        return self.buffer[self.index:] + self.buffer[:self.index]
    
    def search(self, query: LogQuery) -> List[LogEntry]:
        """Search entries."""
        results = []
        for entry in self.get_all():
            if entry.matches(query):
                results.append(entry)
                if len(results) >= query.limit + query.offset:
                    break
        
        return results[query.offset:query.offset + query.limit]
    
    def clear(self) -> None:
        """Clear all entries."""
        self.buffer = []
        self.index = 0


class LogAggregator:
    """
    Centralized Log Aggregator.
    
    Features:
    - Log collection from multiple sources
    - In-memory ring buffer storage
    - Query and search
    - Statistics and analysis
    - Alert on patterns
    - Real-time streaming
    """
    
    def __init__(self, max_entries: int = 10000):
        self.buffer = RingBuffer(max_entries)
        self.alerts: Dict[str, LogAlert] = {}
        self.subscribers: List[Queue] = []
        self.error_patterns: Dict[str, int] = defaultdict(int)
        
        self._log_counter = 0
        self._start_time = datetime.utcnow()
    
    def ingest(
        self,
        level: str,
        message: str,
        source: str = "",
        **kwargs: Any,
    ) -> LogEntry:
        """
        Ingest a log entry.
        
        Args:
            level: Log level
            message: Log message
            source: Source service/component
            **kwargs: Additional fields
            
        Returns:
            Created log entry
        """
        self._log_counter += 1
        
        entry = LogEntry(
            id=f"log_{self._log_counter}",
            timestamp=datetime.utcnow(),
            level=LogLevel(level.lower()) if isinstance(level, str) else level,
            message=message,
            source=source,
            logger_name=kwargs.get("logger_name", ""),
            trace_id=kwargs.get("trace_id", ""),
            span_id=kwargs.get("span_id", ""),
            user_id=kwargs.get("user_id", ""),
            request_id=kwargs.get("request_id", ""),
            extra=kwargs.get("extra", {}),
            exception_type=kwargs.get("exception_type", ""),
            exception_message=kwargs.get("exception_message", ""),
            stacktrace=kwargs.get("stacktrace", ""),
            host=kwargs.get("host", ""),
            environment=kwargs.get("environment", ""),
            version=kwargs.get("version", ""),
        )
        
        self.buffer.append(entry)
        
        # Track error patterns
        if entry.level.severity >= LogLevel.ERROR.severity:
            pattern = self._extract_error_pattern(entry)
            self.error_patterns[pattern] += 1
        
        # Check alerts
        self._check_alerts(entry)
        
        # Notify subscribers
        self._notify_subscribers(entry)
        
        return entry
    
    def _extract_error_pattern(self, entry: LogEntry) -> str:
        """Extract error pattern from log entry."""
        if entry.exception_type:
            return f"{entry.exception_type}: {entry.exception_message[:50]}"
        
        # Normalize message (remove numbers, IDs, etc.)
        pattern = re.sub(r'\b[0-9a-f-]{36}\b', '<UUID>', entry.message)
        pattern = re.sub(r'\b\d+\b', '<N>', pattern)
        pattern = re.sub(r'"[^"]*"', '"<STR>"', pattern)
        
        return pattern[:100]
    
    def _check_alerts(self, entry: LogEntry) -> None:
        """Check if entry triggers any alerts."""
        for alert in self.alerts.values():
            if not alert.enabled:
                continue
            
            if entry.level.severity < alert.level.severity:
                continue
            
            if re.search(alert.pattern, entry.message, re.IGNORECASE):
                alert.trigger_count += 1
                
                if alert.trigger_count >= alert.threshold:
                    alert.last_triggered = datetime.utcnow()
                    
                    if alert.callback:
                        try:
                            alert.callback(alert, entry)
                        except Exception as e:
                            logger.error(f"Alert callback error: {e}")
                    
                    logger.warning(f"Alert triggered: {alert.name}")
                    alert.trigger_count = 0
    
    def _notify_subscribers(self, entry: LogEntry) -> None:
        """Notify real-time subscribers."""
        for queue in self.subscribers:
            try:
                queue.put_nowait(entry)
            except asyncio.QueueFull:
                pass  # Drop if subscriber is slow
    
    def query(self, query: LogQuery) -> List[LogEntry]:
        """
        Query logs.
        
        Args:
            query: Query parameters
            
        Returns:
            Matching log entries
        """
        return self.buffer.search(query)
    
    def search(
        self,
        text: str,
        level: Optional[str] = None,
        source: Optional[str] = None,
        hours: int = 1,
        limit: int = 100,
    ) -> List[LogEntry]:
        """
        Simple search interface.
        
        Args:
            text: Text to search for
            level: Minimum log level
            source: Source to filter by
            hours: Hours to look back
            limit: Maximum results
            
        Returns:
            Matching log entries
        """
        query = LogQuery(
            min_level=LogLevel(level.lower()) if level else None,
            start_time=datetime.utcnow() - timedelta(hours=hours),
            sources=[source] if source else None,
            search_text=text,
            limit=limit,
        )
        return self.query(query)
    
    def get_by_trace(self, trace_id: str) -> List[LogEntry]:
        """Get all logs for a trace."""
        query = LogQuery(trace_id=trace_id, limit=1000)
        return self.query(query)
    
    def get_by_request(self, request_id: str) -> List[LogEntry]:
        """Get all logs for a request."""
        query = LogQuery(request_id=request_id, limit=1000)
        return self.query(query)
    
    def get_errors(
        self,
        hours: int = 1,
        limit: int = 100,
    ) -> List[LogEntry]:
        """Get recent errors."""
        query = LogQuery(
            start_time=datetime.utcnow() - timedelta(hours=hours),
            errors_only=True,
            limit=limit,
        )
        return self.query(query)
    
    def get_stats(self, hours: int = 1) -> LogStats:
        """
        Get log statistics.
        
        Args:
            hours: Hours to analyze
            
        Returns:
            Log statistics
        """
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        by_level: Dict[str, int] = defaultdict(int)
        by_source: Dict[str, int] = defaultdict(int)
        total = 0
        errors = 0
        
        for entry in self.buffer.get_all():
            if entry.timestamp >= start_time:
                total += 1
                by_level[entry.level.value] += 1
                by_source[entry.source or "unknown"] += 1
                
                if entry.level.severity >= LogLevel.ERROR.severity:
                    errors += 1
        
        # Top error patterns
        top_errors = sorted(
            [{"pattern": k, "count": v} for k, v in self.error_patterns.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:10]
        
        return LogStats(
            total_count=total,
            by_level=dict(by_level),
            by_source=dict(by_source),
            error_rate=errors / total if total > 0 else 0,
            logs_per_minute=total / (hours * 60),
            top_errors=top_errors,
            time_range=timedelta(hours=hours),
        )
    
    def add_alert(
        self,
        id: str,
        name: str,
        pattern: str,
        threshold: int = 1,
        window_minutes: int = 5,
        level: str = "error",
        callback: Optional[Callable] = None,
    ) -> LogAlert:
        """
        Add a log alert.
        
        Args:
            id: Alert ID
            name: Alert name
            pattern: Regex pattern to match
            threshold: Number of matches to trigger
            window_minutes: Time window
            level: Minimum log level
            callback: Function to call when triggered
            
        Returns:
            Created alert
        """
        alert = LogAlert(
            id=id,
            name=name,
            pattern=pattern,
            threshold=threshold,
            window_minutes=window_minutes,
            level=LogLevel(level.lower()),
            callback=callback,
        )
        
        self.alerts[id] = alert
        logger.info(f"Added log alert: {name}")
        
        return alert
    
    def remove_alert(self, id: str) -> bool:
        """Remove an alert."""
        if id in self.alerts:
            del self.alerts[id]
            return True
        return False
    
    def subscribe(self) -> Queue:
        """Subscribe to real-time log stream."""
        queue: Queue = Queue(maxsize=100)
        self.subscribers.append(queue)
        return queue
    
    def unsubscribe(self, queue: Queue) -> None:
        """Unsubscribe from log stream."""
        if queue in self.subscribers:
            self.subscribers.remove(queue)
    
    async def stream(
        self,
        query: Optional[LogQuery] = None,
    ):
        """
        Stream logs in real-time.
        
        Args:
            query: Optional filter
            
        Yields:
            Log entries
        """
        queue = self.subscribe()
        try:
            while True:
                entry = await queue.get()
                if query is None or entry.matches(query):
                    yield entry
        finally:
            self.unsubscribe(queue)
    
    def export(
        self,
        format: str = "json",
        hours: int = 1,
    ) -> str:
        """
        Export logs.
        
        Args:
            format: Export format (json, ndjson)
            hours: Hours to export
            
        Returns:
            Exported logs
        """
        query = LogQuery(
            start_time=datetime.utcnow() - timedelta(hours=hours),
            limit=10000,
        )
        entries = self.query(query)
        
        if format == "ndjson":
            return "\n".join(json.dumps(e.to_dict()) for e in entries)
        else:
            return json.dumps([e.to_dict() for e in entries], indent=2)
    
    def clear(self) -> None:
        """Clear all logs."""
        self.buffer.clear()
        self.error_patterns.clear()
        logger.info("Cleared log buffer")


# Global log aggregator
log_aggregator = LogAggregator()


def get_log_aggregator() -> LogAggregator:
    """Get the global log aggregator."""
    return log_aggregator


# Handler for integrating with Python logging
class AggregatorHandler:
    """Handler to send logs to the aggregator."""
    
    def __init__(self, source: str = ""):
        self.source = source
    
    def emit(self, record: Dict[str, Any]) -> None:
        """Emit a log record."""
        log_aggregator.ingest(
            level=record.get("level", "info"),
            message=record.get("message", ""),
            source=self.source,
            logger_name=record.get("logger", ""),
            extra=record.get("extra", {}),
        )
