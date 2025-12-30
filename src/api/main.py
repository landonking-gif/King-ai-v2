"""FastAPI application entry point."""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.routes import chat, businesses, approvals, evolution, research
from src.master_ai.brain import MasterAI
from src.database.connection import init_db

# Global MasterAI instance - initialized during startup
master_ai: MasterAI | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Initializes the database schema and the Master AI brain.
    """
    global master_ai
    
    # --- Startup ---
    # Create database tables if they don't exist
    await init_db()
    # Initialize the central brain
    master_ai = MasterAI()
    
    yield
    
    # Shutdown
    # Cleanup if needed

app = FastAPI(
    title="King AI v2",
    description="Autonomous Business Empire API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(businesses.router, prefix="/api/businesses", tags=["businesses"])
app.include_router(approvals.router, prefix="/api/approvals", tags=["approvals"])
app.include_router(evolution.router, prefix="/api/evolution", tags=["evolution"])
app.include_router(research.router, prefix="/api", tags=["research"])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time update stream for the dashboard.
    """
    from src.api.websocket import manager
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception:
        manager.disconnect(websocket)

@app.get("/api/health")
async def health():
    return {"status": "ok", "master_ai": master_ai is not None}

def get_master_ai() -> MasterAI:
    """Dependency to get MasterAI instance."""
    if master_ai is None:
        raise RuntimeError("MasterAI not initialized")
    return master_ai
