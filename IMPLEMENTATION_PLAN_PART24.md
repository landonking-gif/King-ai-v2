# Implementation Plan Part 24: Dashboard WebSocket & Monitoring

| Field | Value |
|-------|-------|
| Module | Real-Time WebSocket & System Monitoring |
| Priority | High |
| Estimated Effort | 5-6 hours |
| Dependencies | Part 3 (API), Part 22 (Dashboard Components) |

---

## 1. Scope

This module implements real-time communication and monitoring:

- **WebSocket Server** - Real-time event streaming
- **Event Broadcasting** - Push updates to connected clients
- **System Monitoring** - Health checks, metrics, alerts
- **Live Dashboard** - Real-time UI updates
- **Connection Management** - Handle reconnects, heartbeats

---

## 2. Tasks

### Task 24.1: WebSocket Manager (Backend)

**File: `src/api/websocket.py`**

```python
"""
WebSocket Manager for real-time communication.
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, Callable, Any
from fastapi import WebSocket, WebSocketDisconnect
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}
        self._subscriptions: dict[str, set[str]] = {}  # channel -> connection_ids
        self._connection_meta: dict[str, dict] = {}
        self._message_handlers: dict[str, Callable] = {}

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str = None,
        business_id: str = None,
    ) -> str:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        self._connections[connection_id] = websocket
        self._connection_meta[connection_id] = {
            "user_id": user_id,
            "business_id": business_id,
            "connected_at": datetime.utcnow().isoformat(),
            "last_ping": datetime.utcnow().isoformat(),
        }

        # Auto-subscribe to user and business channels
        if user_id:
            await self.subscribe(connection_id, f"user:{user_id}")
        if business_id:
            await self.subscribe(connection_id, f"business:{business_id}")

        # Subscribe to global channel
        await self.subscribe(connection_id, "global")

        logger.info(f"WebSocket connected: {connection_id}")
        
        # Send welcome message
        await self.send_to_connection(connection_id, {
            "type": "connected",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return connection_id

    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket."""
        if connection_id in self._connections:
            del self._connections[connection_id]
        if connection_id in self._connection_meta:
            del self._connection_meta[connection_id]

        # Remove from all subscriptions
        for channel, subs in self._subscriptions.items():
            subs.discard(connection_id)

        logger.info(f"WebSocket disconnected: {connection_id}")

    async def subscribe(self, connection_id: str, channel: str):
        """Subscribe a connection to a channel."""
        if channel not in self._subscriptions:
            self._subscriptions[channel] = set()
        self._subscriptions[channel].add(connection_id)

    async def unsubscribe(self, connection_id: str, channel: str):
        """Unsubscribe from a channel."""
        if channel in self._subscriptions:
            self._subscriptions[channel].discard(connection_id)

    async def send_to_connection(self, connection_id: str, message: dict):
        """Send a message to a specific connection."""
        ws = self._connections.get(connection_id)
        if ws:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to {connection_id}: {e}")
                await self.disconnect(connection_id)

    async def broadcast_to_channel(self, channel: str, message: dict):
        """Broadcast a message to all connections in a channel."""
        subscribers = self._subscriptions.get(channel, set())
        for conn_id in list(subscribers):
            await self.send_to_connection(conn_id, message)

    async def broadcast_all(self, message: dict):
        """Broadcast to all connections."""
        for conn_id in list(self._connections.keys()):
            await self.send_to_connection(conn_id, message)

    def register_handler(self, message_type: str, handler: Callable):
        """Register a handler for a message type."""
        self._message_handlers[message_type] = handler

    async def handle_message(self, connection_id: str, data: dict):
        """Handle an incoming message."""
        msg_type = data.get("type", "")
        
        # Handle ping/pong
        if msg_type == "ping":
            self._connection_meta[connection_id]["last_ping"] = datetime.utcnow().isoformat()
            await self.send_to_connection(connection_id, {"type": "pong"})
            return

        # Handle subscription requests
        if msg_type == "subscribe":
            channel = data.get("channel")
            if channel:
                await self.subscribe(connection_id, channel)
                await self.send_to_connection(connection_id, {
                    "type": "subscribed",
                    "channel": channel,
                })
            return

        if msg_type == "unsubscribe":
            channel = data.get("channel")
            if channel:
                await self.unsubscribe(connection_id, channel)
            return

        # Call registered handler
        handler = self._message_handlers.get(msg_type)
        if handler:
            try:
                await handler(connection_id, data)
            except Exception as e:
                logger.error(f"Handler error for {msg_type}: {e}")

    def get_stats(self) -> dict:
        """Get connection statistics."""
        return {
            "total_connections": len(self._connections),
            "channels": {
                ch: len(subs) for ch, subs in self._subscriptions.items()
            },
        }


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str = None,
    business_id: str = None,
):
    """WebSocket endpoint handler."""
    connection_id = await manager.connect(websocket, user_id, business_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            await manager.handle_message(connection_id, data)
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(connection_id)
```

---

### Task 24.2: Event Broadcaster

**File: `src/api/events.py`**

```python
"""
Event Broadcasting System.
"""
from datetime import datetime
from enum import Enum
from typing import Any
from src.api.websocket import manager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events that can be broadcast."""
    # Business events
    BUSINESS_CREATED = "business.created"
    BUSINESS_UPDATED = "business.updated"
    BUSINESS_STAGE_CHANGED = "business.stage_changed"
    
    # Task events
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    
    # Playbook events
    PLAYBOOK_STARTED = "playbook.started"
    PLAYBOOK_COMPLETED = "playbook.completed"
    PLAYBOOK_FAILED = "playbook.failed"
    
    # Approval events
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"
    
    # System events
    SYSTEM_ALERT = "system.alert"
    SYSTEM_METRIC = "system.metric"
    
    # Analytics events
    KPI_UPDATE = "analytics.kpi_update"
    ANOMALY_DETECTED = "analytics.anomaly"


class EventBroadcaster:
    """Broadcast events to WebSocket clients."""

    async def emit(
        self,
        event_type: EventType,
        data: dict,
        business_id: str = None,
        user_id: str = None,
    ):
        """Emit an event to relevant subscribers."""
        event = {
            "type": "event",
            "event": event_type.value,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Broadcast to specific channels
        if business_id:
            await manager.broadcast_to_channel(f"business:{business_id}", event)
        
        if user_id:
            await manager.broadcast_to_channel(f"user:{user_id}", event)

        # Always broadcast to global for system events
        if event_type.value.startswith("system."):
            await manager.broadcast_to_channel("global", event)

        logger.debug(f"Emitted event: {event_type.value}")

    async def emit_business_update(self, business_id: str, changes: dict):
        """Emit a business update event."""
        await self.emit(
            EventType.BUSINESS_UPDATED,
            {"business_id": business_id, "changes": changes},
            business_id=business_id,
        )

    async def emit_task_progress(
        self,
        business_id: str,
        task_id: str,
        status: str,
        progress: float = None,
        result: Any = None,
    ):
        """Emit task progress event."""
        event_type = {
            "started": EventType.TASK_STARTED,
            "completed": EventType.TASK_COMPLETED,
            "failed": EventType.TASK_FAILED,
        }.get(status, EventType.TASK_STARTED)

        await self.emit(
            event_type,
            {
                "task_id": task_id,
                "status": status,
                "progress": progress,
                "result": result,
            },
            business_id=business_id,
        )

    async def emit_approval_required(
        self,
        business_id: str,
        request_id: str,
        title: str,
        risk_level: str,
    ):
        """Emit approval required event."""
        await self.emit(
            EventType.APPROVAL_REQUIRED,
            {
                "request_id": request_id,
                "title": title,
                "risk_level": risk_level,
            },
            business_id=business_id,
        )

    async def emit_kpi_update(self, business_id: str, kpis: dict):
        """Emit KPI update event."""
        await self.emit(
            EventType.KPI_UPDATE,
            {"kpis": kpis},
            business_id=business_id,
        )

    async def emit_system_alert(self, level: str, message: str, details: dict = None):
        """Emit system alert."""
        await self.emit(
            EventType.SYSTEM_ALERT,
            {"level": level, "message": message, "details": details or {}},
        )


# Global broadcaster instance
broadcaster = EventBroadcaster()
```

---

### Task 24.3: System Monitor

**File: `src/monitoring/monitor.py`**

```python
"""
System Health Monitoring.
"""
import asyncio
import psutil
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, field
from src.api.events import broadcaster, EventType
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: float = 0
    message: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_connections: int
    pending_tasks: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class SystemMonitor:
    """Monitor system health and resources."""

    def __init__(self):
        self._health_checks: dict[str, Callable] = {}
        self._last_metrics: Optional[SystemMetrics] = None
        self._alerts: list[dict] = []
        self._running = False
        self._check_interval = 30  # seconds

    def register_health_check(self, name: str, check_fn: Callable):
        """Register a health check function."""
        self._health_checks[name] = check_fn

    async def check_health(self) -> dict[str, HealthCheck]:
        """Run all health checks."""
        results = {}
        
        for name, check_fn in self._health_checks.items():
            start = datetime.utcnow()
            try:
                status, message = await check_fn()
                latency = (datetime.utcnow() - start).total_seconds() * 1000
                results[name] = HealthCheck(
                    name=name,
                    status=status,
                    latency_ms=latency,
                    message=message,
                )
            except Exception as e:
                results[name] = HealthCheck(
                    name=name,
                    status="unhealthy",
                    message=str(e),
                )

        return results

    async def collect_metrics(self) -> SystemMetrics:
        """Collect system metrics."""
        from src.api.websocket import manager
        
        metrics = SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage('/').percent,
            active_connections=manager.get_stats()["total_connections"],
            pending_tasks=0,  # Would be populated from task queue
        )
        
        self._last_metrics = metrics
        return metrics

    async def check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against thresholds and alert if needed."""
        alerts = []
        
        if metrics.cpu_percent > 90:
            alerts.append(("critical", f"CPU usage critical: {metrics.cpu_percent}%"))
        elif metrics.cpu_percent > 75:
            alerts.append(("warning", f"CPU usage high: {metrics.cpu_percent}%"))

        if metrics.memory_percent > 90:
            alerts.append(("critical", f"Memory usage critical: {metrics.memory_percent}%"))
        elif metrics.memory_percent > 80:
            alerts.append(("warning", f"Memory usage high: {metrics.memory_percent}%"))

        if metrics.disk_percent > 90:
            alerts.append(("critical", f"Disk usage critical: {metrics.disk_percent}%"))
        elif metrics.disk_percent > 80:
            alerts.append(("warning", f"Disk usage high: {metrics.disk_percent}%"))

        for level, message in alerts:
            await broadcaster.emit_system_alert(level, message, {
                "cpu": metrics.cpu_percent,
                "memory": metrics.memory_percent,
                "disk": metrics.disk_percent,
            })
            self._alerts.append({
                "level": level,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            })

    async def start_monitoring(self):
        """Start background monitoring loop."""
        self._running = True
        logger.info("System monitoring started")
        
        while self._running:
            try:
                metrics = await self.collect_metrics()
                await self.check_thresholds(metrics)
                
                # Broadcast metrics
                await broadcaster.emit(
                    EventType.SYSTEM_METRIC,
                    {
                        "cpu": metrics.cpu_percent,
                        "memory": metrics.memory_percent,
                        "disk": metrics.disk_percent,
                        "connections": metrics.active_connections,
                    },
                )
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

            await asyncio.sleep(self._check_interval)

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self._running = False
        logger.info("System monitoring stopped")

    def get_recent_alerts(self, limit: int = 20) -> list[dict]:
        """Get recent alerts."""
        return self._alerts[-limit:]

    def get_last_metrics(self) -> Optional[SystemMetrics]:
        """Get last collected metrics."""
        return self._last_metrics


# Global monitor instance
monitor = SystemMonitor()


# Default health checks
async def check_database() -> tuple[str, str]:
    """Check database connectivity."""
    try:
        # Would actually check DB connection
        return "healthy", "Database connected"
    except Exception as e:
        return "unhealthy", str(e)


async def check_redis() -> tuple[str, str]:
    """Check Redis connectivity."""
    try:
        # Would actually check Redis
        return "healthy", "Redis connected"
    except Exception:
        return "degraded", "Redis unavailable, using fallback"


async def check_llm() -> tuple[str, str]:
    """Check LLM service."""
    try:
        # Would check Ollama/LLM service
        return "healthy", "LLM service available"
    except Exception as e:
        return "unhealthy", str(e)


# Register default checks
monitor.register_health_check("database", check_database)
monitor.register_health_check("redis", check_redis)
monitor.register_health_check("llm", check_llm)
```

---

### Task 24.4: Monitoring API Routes

**File: `src/api/routes/monitoring.py`**

```python
"""
Monitoring API Routes.
"""
from fastapi import APIRouter, WebSocket
from src.api.websocket import manager, websocket_endpoint
from src.monitoring.monitor import monitor
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.websocket("/ws")
async def websocket_route(
    websocket: WebSocket,
    user_id: str = None,
    business_id: str = None,
):
    """WebSocket endpoint for real-time updates."""
    await websocket_endpoint(websocket, user_id, business_id)


@router.get("/health")
async def health_check():
    """Get system health status."""
    checks = await monitor.check_health()
    
    overall = "healthy"
    for check in checks.values():
        if check.status == "unhealthy":
            overall = "unhealthy"
            break
        elif check.status == "degraded" and overall == "healthy":
            overall = "degraded"

    return {
        "status": overall,
        "checks": {
            name: {
                "status": c.status,
                "latency_ms": round(c.latency_ms, 2),
                "message": c.message,
            }
            for name, c in checks.items()
        },
    }


@router.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    metrics = await monitor.collect_metrics()
    
    return {
        "cpu_percent": metrics.cpu_percent,
        "memory_percent": metrics.memory_percent,
        "disk_percent": metrics.disk_percent,
        "active_connections": metrics.active_connections,
        "pending_tasks": metrics.pending_tasks,
        "timestamp": metrics.timestamp.isoformat(),
    }


@router.get("/alerts")
async def get_alerts(limit: int = 20):
    """Get recent system alerts."""
    return {
        "alerts": monitor.get_recent_alerts(limit),
    }


@router.get("/connections")
async def get_connections():
    """Get WebSocket connection stats."""
    return manager.get_stats()
```

---

### Task 24.5: React WebSocket Hook

**File: `dashboard/src/hooks/useWebSocket.js`**

```javascript
import { useEffect, useRef, useState, useCallback } from 'react';

const RECONNECT_DELAY = 3000;
const HEARTBEAT_INTERVAL = 30000;

export function useWebSocket(url, options = {}) {
  const { userId, businessId, onEvent, autoReconnect = true } = options;
  
  const wsRef = useRef(null);
  const heartbeatRef = useRef(null);
  const reconnectRef = useRef(null);
  
  const [connected, setConnected] = useState(false);
  const [connectionId, setConnectionId] = useState(null);
  const [lastEvent, setLastEvent] = useState(null);

  const connect = useCallback(() => {
    const params = new URLSearchParams();
    if (userId) params.set('user_id', userId);
    if (businessId) params.set('business_id', businessId);
    
    const wsUrl = `${url}?${params.toString()}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      
      // Start heartbeat
      heartbeatRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, HEARTBEAT_INTERVAL);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'connected') {
          setConnectionId(data.connection_id);
        } else if (data.type === 'event') {
          setLastEvent(data);
          onEvent?.(data.event, data.data, data.timestamp);
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
      setConnectionId(null);
      
      if (heartbeatRef.current) {
        clearInterval(heartbeatRef.current);
      }
      
      // Auto reconnect
      if (autoReconnect) {
        reconnectRef.current = setTimeout(connect, RECONNECT_DELAY);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    wsRef.current = ws;
  }, [url, userId, businessId, onEvent, autoReconnect]);

  const disconnect = useCallback(() => {
    if (reconnectRef.current) {
      clearTimeout(reconnectRef.current);
    }
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
  }, []);

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const subscribe = useCallback((channel) => {
    send({ type: 'subscribe', channel });
  }, [send]);

  const unsubscribe = useCallback((channel) => {
    send({ type: 'unsubscribe', channel });
  }, [send]);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    connected,
    connectionId,
    lastEvent,
    send,
    subscribe,
    unsubscribe,
    reconnect: connect,
  };
}
```

---

### Task 24.6: React Monitoring Components

**File: `dashboard/src/components/Monitoring/SystemStatus.jsx`**

```jsx
import { useState, useEffect } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import './Monitoring.css';

export function SystemStatus() {
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  const { connected } = useWebSocket(
    `ws://${window.location.host}/api/monitoring/ws`,
    {
      onEvent: (event, data) => {
        if (event === 'system.metric') {
          setMetrics(data);
        }
      },
    }
  );

  useEffect(() => {
    fetchHealth();
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/monitoring/health');
      const data = await res.json();
      setHealth(data);
    } catch (err) {
      console.error('Failed to fetch health:', err);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch('/api/monitoring/metrics');
      const data = await res.json();
      setMetrics(data);
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
    }
    setLoading(false);
  };

  if (loading) {
    return <div className="loading">Loading system status...</div>;
  }

  const statusColor = {
    healthy: '#27ae60',
    degraded: '#f39c12',
    unhealthy: '#e74c3c',
  };

  return (
    <div className="system-status">
      <div className="status-header">
        <h2>System Status</h2>
        <span className={`connection-indicator ${connected ? 'connected' : ''}`}>
          {connected ? '● Live' : '○ Disconnected'}
        </span>
      </div>

      {health && (
        <div className="health-section">
          <div
            className="overall-status"
            style={{ backgroundColor: statusColor[health.status] }}
          >
            {health.status.toUpperCase()}
          </div>

          <div className="health-checks">
            {Object.entries(health.checks).map(([name, check]) => (
              <div key={name} className="health-check">
                <span
                  className="check-indicator"
                  style={{ backgroundColor: statusColor[check.status] }}
                />
                <span className="check-name">{name}</span>
                <span className="check-latency">{check.latency_ms}ms</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {metrics && (
        <div className="metrics-section">
          <h3>Resources</h3>
          <div className="metrics-grid">
            <MetricGauge
              label="CPU"
              value={metrics.cpu_percent}
              max={100}
              unit="%"
            />
            <MetricGauge
              label="Memory"
              value={metrics.memory_percent}
              max={100}
              unit="%"
            />
            <MetricGauge
              label="Disk"
              value={metrics.disk_percent}
              max={100}
              unit="%"
            />
            <div className="metric-box">
              <span className="metric-label">Connections</span>
              <span className="metric-value">{metrics.active_connections}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricGauge({ label, value, max, unit }) {
  const percent = (value / max) * 100;
  const color = percent > 90 ? '#e74c3c' : percent > 75 ? '#f39c12' : '#27ae60';

  return (
    <div className="metric-gauge">
      <span className="gauge-label">{label}</span>
      <div className="gauge-bar">
        <div
          className="gauge-fill"
          style={{ width: `${percent}%`, backgroundColor: color }}
        />
      </div>
      <span className="gauge-value">
        {value.toFixed(1)}{unit}
      </span>
    </div>
  );
}
```

**File: `dashboard/src/components/Monitoring/LiveFeed.jsx`**

```jsx
import { useState, useEffect, useRef } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import './Monitoring.css';

export function LiveFeed({ businessId, maxItems = 50 }) {
  const [events, setEvents] = useState([]);
  const feedRef = useRef(null);

  const { connected } = useWebSocket(
    `ws://${window.location.host}/api/monitoring/ws`,
    {
      businessId,
      onEvent: (event, data, timestamp) => {
        setEvents((prev) => {
          const newEvent = { event, data, timestamp, id: Date.now() };
          const updated = [newEvent, ...prev].slice(0, maxItems);
          return updated;
        });
      },
    }
  );

  const eventIcons = {
    'business.updated': '🏢',
    'task.started': '▶️',
    'task.completed': '✅',
    'task.failed': '❌',
    'playbook.started': '📋',
    'playbook.completed': '🎉',
    'approval.required': '⚠️',
    'approval.granted': '✓',
    'system.alert': '🚨',
    'analytics.kpi_update': '📊',
  };

  return (
    <div className="live-feed">
      <div className="feed-header">
        <h3>Live Activity</h3>
        <span className={`live-indicator ${connected ? 'active' : ''}`}>
          {connected ? '● LIVE' : '○ OFFLINE'}
        </span>
      </div>

      <div className="feed-list" ref={feedRef}>
        {events.length === 0 ? (
          <div className="feed-empty">
            <p>Waiting for events...</p>
          </div>
        ) : (
          events.map((e) => (
            <div key={e.id} className="feed-item">
              <span className="feed-icon">
                {eventIcons[e.event] || '📌'}
              </span>
              <div className="feed-content">
                <span className="feed-event">{e.event}</span>
                <span className="feed-time">
                  {new Date(e.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
```

**File: `dashboard/src/components/Monitoring/AlertBanner.jsx`**

```jsx
import { useState, useEffect } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import './Monitoring.css';

export function AlertBanner() {
  const [alerts, setAlerts] = useState([]);

  useWebSocket(
    `ws://${window.location.host}/api/monitoring/ws`,
    {
      onEvent: (event, data) => {
        if (event === 'system.alert') {
          setAlerts((prev) => [...prev, { ...data, id: Date.now() }]);
        }
      },
    }
  );

  const dismissAlert = (id) => {
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  };

  if (alerts.length === 0) return null;

  return (
    <div className="alert-banner-container">
      {alerts.map((alert) => (
        <div
          key={alert.id}
          className={`alert-banner alert-${alert.level}`}
        >
          <span className="alert-icon">
            {alert.level === 'critical' ? '🚨' : '⚠️'}
          </span>
          <span className="alert-message">{alert.message}</span>
          <button
            className="alert-dismiss"
            onClick={() => dismissAlert(alert.id)}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
```

---

### Task 24.7: Monitoring Styles

**File: `dashboard/src/components/Monitoring/Monitoring.css`**

```css
/* System Status */
.system-status {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.status-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.connection-indicator {
  font-size: 14px;
  color: #999;
}

.connection-indicator.connected {
  color: #27ae60;
}

.overall-status {
  padding: 12px 24px;
  border-radius: 8px;
  color: white;
  font-weight: 600;
  text-align: center;
  margin-bottom: 20px;
}

.health-checks {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.health-check {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px;
  background: #f9f9f9;
  border-radius: 4px;
}

.check-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.check-name {
  flex: 1;
  font-weight: 500;
}

.check-latency {
  color: #999;
  font-size: 14px;
}

/* Metrics */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
}

.metric-gauge {
  background: #f9f9f9;
  padding: 12px;
  border-radius: 8px;
}

.gauge-label {
  display: block;
  font-size: 12px;
  color: #666;
  margin-bottom: 8px;
}

.gauge-bar {
  height: 8px;
  background: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 4px;
}

.gauge-fill {
  height: 100%;
  transition: width 0.3s ease;
}

.gauge-value {
  font-weight: 600;
}

/* Live Feed */
.live-feed {
  background: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.feed-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid #eee;
}

.live-indicator {
  font-size: 12px;
  font-weight: 600;
  color: #999;
}

.live-indicator.active {
  color: #e74c3c;
  animation: pulse 2s infinite;
}

.feed-list {
  max-height: 400px;
  overflow-y: auto;
}

.feed-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid #f5f5f5;
}

.feed-icon {
  font-size: 18px;
}

.feed-content {
  flex: 1;
  display: flex;
  justify-content: space-between;
}

.feed-event {
  font-size: 14px;
}

.feed-time {
  font-size: 12px;
  color: #999;
}

/* Alert Banner */
.alert-banner-container {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
}

.alert-banner {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 20px;
  color: white;
}

.alert-banner.alert-critical {
  background: #c0392b;
}

.alert-banner.alert-warning {
  background: #e67e22;
}

.alert-message {
  flex: 1;
}

.alert-dismiss {
  background: none;
  border: none;
  color: white;
  font-size: 20px;
  cursor: pointer;
  opacity: 0.8;
}

.alert-dismiss:hover {
  opacity: 1;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
```

---

### Task 24.8: Tests

**File: `tests/test_websocket.py`**

```python
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
```

---

## 3. Acceptance Criteria

| Criteria | Validation |
|----------|------------|
| WebSocket connects | Clients can connect/disconnect |
| Events broadcast | Events sent to subscribers |
| Health checks work | Returns status for all services |
| Metrics collected | CPU, memory, disk monitored |
| Alerts fire | Threshold breaches trigger alerts |
| Live UI updates | React components update in real-time |

---

## 4. File Summary

| File | Purpose |
|------|---------|
| `src/api/websocket.py` | WebSocket connection manager |
| `src/api/events.py` | Event broadcasting system |
| `src/monitoring/monitor.py` | System health monitoring |
| `src/api/routes/monitoring.py` | Monitoring API endpoints |
| `hooks/useWebSocket.js` | React WebSocket hook |
| `components/Monitoring/SystemStatus.jsx` | Health/metrics display |
| `components/Monitoring/LiveFeed.jsx` | Real-time event feed |
| `components/Monitoring/AlertBanner.jsx` | System alerts |
| `components/Monitoring/Monitoring.css` | Component styles |
| `tests/test_websocket.py` | Unit tests |
