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

    async def broadcast(self, message: dict):
        """Legacy method for backward compatibility."""
        await self.broadcast_all(message)

    def register_handler(self, message_type: str, handler: Callable):
        """Register a handler for a message type."""
        self._message_handlers[message_type] = handler

    async def handle_message(self, connection_id: str, data: dict):
        """Handle an incoming message."""
        msg_type = data.get("type", "")
        
        # Handle ping/pong
        if msg_type == "ping":
            if connection_id in self._connection_meta:
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


# Optional Redis pubsub integration for horizontal scaling
try:
    import redis.asyncio as aioredis
    redis_client = None
    async def _ensure_redis():
        global redis_client
        if redis_client is None:
            from config.settings import settings
            redis_client = aioredis.from_url(settings.redis_url)
        return redis_client

    async def _redis_listener():
        client = await _ensure_redis()
        pubsub = client.pubsub()
        await pubsub.psubscribe("ws_events:*")
        async for message in pubsub.listen():
            if message is None or message.get("type") != "pmessage":
                continue
            try:
                payload = json.loads(message.get("data", "{}"))
                channel = message.get("channel", b"").decode() if isinstance(message.get("channel"), bytes) else message.get("channel")
                # Broadcast to local subscribers
                await manager.broadcast_to_channel(channel, payload)
            except Exception as e:
                logger.error(f"Redis listener error: {e}")

    # Start background redis listener
    import asyncio
    asyncio.create_task(_redis_listener())
except Exception:
    # Redis not available; continue without pubsub
    redis_client = None


async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str = None,
    business_id: str = None,
    initial_channels: list[str] | None = None,
):
    """WebSocket endpoint handler."""
    connection_id = await manager.connect(websocket, user_id, business_id)

    # Subscribe to any initial channels requested
    if initial_channels:
        for ch in initial_channels:
            await manager.subscribe(connection_id, ch)

    try:
        while True:
            data = await websocket.receive_json()
            await manager.handle_message(connection_id, data)
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(connection_id)
