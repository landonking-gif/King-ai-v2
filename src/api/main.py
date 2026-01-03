"""FastAPI application entry point."""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime

from src.api.routes import (
    chat, businesses, approvals, evolution, health, playbook, portfolio, system,
    analytics, banking, codegen, commerce, content, finance, legal, lifecycle,
    monitoring, research, sandbox, supplier, webhooks, dev_dashboard
)
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
    if settings.enable_scheduler:
        # KPI Review - configurable interval (default 6 hours)
        kpi_interval = settings.kpi_review_interval_hours
        frequency_map = {
            1: TaskFrequency.HOURLY,
            6: TaskFrequency.EVERY_6_HOURS,
            12: TaskFrequency.DAILY,  # 12 hours = daily
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
            enabled=settings.enable_autonomous_mode,
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
            enabled=settings.enable_autonomous_mode,
            run_immediately=False,
            metadata={"description": "Monitor business unit health"}
        )
        
        # Evolution Check - Daily (spec: 1 proposal/day limit)
        async def evolution_check_task():
            """Daily check for system evolution opportunities."""
            try:
                if master_ai and settings.enable_self_modification:
                    await master_ai._consider_evolution("Daily evolution check")
                    logger.info("Completed daily evolution check")
            except Exception as e:
                logger.error(f"Evolution check task failed: {e}")
        
        scheduler.register_task(
            name="evolution_check",
            callback=evolution_check_task,
            frequency=TaskFrequency.DAILY,
            enabled=settings.enable_self_modification,
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
if settings.enable_rate_limiting:
    app.add_middleware(
        RateLimitMiddleware,
        default_limit=RateLimitConfig(
            requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window
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

# Additional feature routes
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(banking.router, prefix="/api/banking", tags=["banking"])
app.include_router(codegen.router, prefix="/api/codegen", tags=["codegen"])
app.include_router(commerce.router, prefix="/api/commerce", tags=["commerce"])
app.include_router(content.router, prefix="/api/content", tags=["content"])
app.include_router(finance.router, prefix="/api/finance", tags=["finance"])
app.include_router(legal.router, prefix="/api/legal", tags=["legal"])
app.include_router(lifecycle.router, prefix="/api/lifecycle", tags=["lifecycle"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["monitoring"])
app.include_router(research.router, prefix="/api/research", tags=["research"])
app.include_router(sandbox.router, prefix="/api/sandbox", tags=["sandbox"])
app.include_router(supplier.router, prefix="/api/supplier", tags=["supplier"])
app.include_router(webhooks.router, prefix="/api/webhooks", tags=["webhooks"])
app.include_router(dev_dashboard.router, prefix="/dev", tags=["dev-dashboard"])

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

def get_master_ai() -> MasterAI:
    """Dependency to get MasterAI instance."""
    if master_ai is None:
        raise RuntimeError("MasterAI not initialized")
    return master_ai

@app.get("/api/health")
async def simple_health_check():
    """Simple health check endpoint for load balancer monitoring."""
    return {"status": "healthy", "service": "king-ai-v2", "timestamp": datetime.now().isoformat()}
