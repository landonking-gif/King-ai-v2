"""FastAPI application entry point."""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.routes import chat, businesses, approvals, evolution, health, playbook, portfolio, system
from src.api.routes import scheduler as scheduler_routes
from src.api.middleware import RateLimitMiddleware, RateLimitConfig
from src.master_ai.brain import MasterAI
from src.database.connection import init_db
from src.services.scheduler import scheduler, TaskFrequency
from src.approvals.risk_profile import get_risk_manager
from src.utils.structured_logging import get_logger
from config.settings import settings

logger = get_logger("api")

# Global MasterAI instance - initialized during startup
master_ai: MasterAI | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Initializes the database schema, the Master AI brain, and the scheduler.
    """
    global master_ai
    
    # --- Startup ---
    logger.info("Starting King AI v2...")
    
    # Create database tables if they don't exist
    await init_db()
    
    # Initialize the central brain
    master_ai = MasterAI()
    
    # Initialize risk profile manager
    risk_manager = get_risk_manager()
    active_profile = risk_manager.get_active_profile()
    logger.info(f"Active risk profile: {active_profile.name}")
    
    # Store in app state for access in routes
    app.state.master_ai = master_ai
    app.state.risk_manager = risk_manager
    
    # Register and start scheduled tasks
    if getattr(settings, 'enable_scheduler', True):
        # KPI Review - configurable interval (default 6 hours)
        kpi_interval = getattr(settings, 'kpi_review_interval_hours', 6)
        frequency_map = {
            1: TaskFrequency.HOURLY,
            6: TaskFrequency.EVERY_6_HOURS,
            12: TaskFrequency.EVERY_12_HOURS,
            24: TaskFrequency.DAILY,
        }
        kpi_frequency = frequency_map.get(kpi_interval, TaskFrequency.EVERY_6_HOURS)
        
        async def kpi_review_task():
            """Periodic KPI review and optimization suggestions."""
            try:
                if master_ai:
                    await master_ai._consider_evolution("Scheduled KPI review")
                    logger.info("Completed scheduled KPI review")
            except Exception as e:
                logger.error(f"KPI review task failed: {e}")
        
        scheduler.register_task(
            name="kpi_review",
            callback=kpi_review_task,
            frequency=kpi_frequency,
            enabled=getattr(settings, 'enable_autonomous_mode', False),
            run_immediately=False,
            metadata={"description": "Periodic KPI analysis and optimization"}
        )
        
        # Business Health Check - configurable (default hourly)
        async def health_check_task():
            """Check health of all business units."""
            try:
                if master_ai and hasattr(master_ai, '_check_business_health'):
                    await master_ai._check_business_health("Scheduled health check")
                    logger.info("Completed scheduled health check")
            except Exception as e:
                logger.error(f"Health check task failed: {e}")
        
        scheduler.register_task(
            name="business_health_check",
            callback=health_check_task,
            frequency=TaskFrequency.HOURLY,
            enabled=getattr(settings, 'enable_autonomous_mode', False),
            run_immediately=False,
            metadata={"description": "Monitor business unit health"}
        )
        
        # Evolution Check - Daily (spec: 1 proposal/day limit)
        async def evolution_check_task():
            """Daily check for system evolution opportunities."""
            try:
                if master_ai and getattr(settings, 'enable_self_modification', True):
                    await master_ai._consider_evolution("Daily evolution check")
                    logger.info("Completed daily evolution check")
            except Exception as e:
                logger.error(f"Evolution check task failed: {e}")
        
        scheduler.register_task(
            name="evolution_check",
            callback=evolution_check_task,
            frequency=TaskFrequency.DAILY,
            enabled=getattr(settings, 'enable_self_modification', True),
            run_immediately=False,
            metadata={"description": "Check for system evolution opportunities"}
        )
        
        # Approval Expiry Check - Every 6 hours
        async def approval_expiry_task():
            """Check and expire old approval requests."""
            try:
                from src.approvals.manager import approval_manager
                expired = await approval_manager.expire_old_requests()
                if expired:
                    logger.info(f"Expired {expired} old approval requests")
            except Exception as e:
                logger.error(f"Approval expiry task failed: {e}")
        
        scheduler.register_task(
            name="approval_expiry",
            callback=approval_expiry_task,
            frequency=TaskFrequency.EVERY_6_HOURS,
            enabled=True,
            run_immediately=False,
            metadata={"description": "Expire old approval requests"}
        )
        
        # Start the scheduler
        await scheduler.start()
        logger.info(f"Scheduler started with {len(scheduler.list_tasks())} tasks")
    
    logger.info("King AI v2 startup complete")
    
    yield
    
    # --- Shutdown ---
    logger.info("Shutting down King AI v2...")
    await scheduler.stop()
    logger.info("Scheduler stopped")

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

# Rate Limiting
if getattr(settings, 'enable_rate_limiting', True):
    app.add_middleware(
        RateLimitMiddleware,
        default_limit=RateLimitConfig(
            requests=getattr(settings, 'rate_limit_requests', 100),
            window_seconds=getattr(settings, 'rate_limit_window', 60)
        ),
    )

# Routes
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(businesses.router, prefix="/api/businesses", tags=["businesses"])
app.include_router(approvals.router, prefix="/api/approvals", tags=["approvals"])
app.include_router(evolution.router, prefix="/api/evolution", tags=["evolution"])
app.include_router(playbook.router, prefix="/api", tags=["playbooks"])
app.include_router(portfolio.router, prefix="/api", tags=["portfolios"])
app.include_router(scheduler_routes.router, prefix="/api/scheduler", tags=["scheduler"])
app.include_router(system.router, tags=["system"])

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
