"""
WebSocket Manager - Handles real-time communication with the dashboard.
Broadcasts system events, task updates, and financial metrics.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import List
import json

class ConnectionManager:
    """
    Manages active WebSocket connections for the King AI dashboard.
    """
    def __init__(self):
        # Tracking active clients
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accepts a new connection and adds it to the pool."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Removes a connection from the pool."""
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """
        Sends a JSON message to all connected clients (e.g., status updates).
        """
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Handle stale connections
                continue

# Global manager instance
manager = ConnectionManager()
