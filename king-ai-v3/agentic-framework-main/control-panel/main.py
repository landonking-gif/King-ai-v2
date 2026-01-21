"""
Master Control Panel Backend
FastAPI server providing REST API and WebSocket support for the King AI v3 Agentic Framework control panel.
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import secrets
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
from typing import List, Dict, Any
import asyncio
import json
from datetime import datetime
import httpx

# Import settings
import sys
sys.path.append('../..')
from config.settings import Settings

# Stub implementations for missing modules
class WebSocketManager:
    def __init__(self):
        self.connections = {}

    def connect(self, websocket, channel):
        if channel not in self.connections:
            self.connections[channel] = []
        self.connections[channel].append(websocket)
        return websocket.accept()

    def disconnect(self, websocket, channel):
        if channel in self.connections:
            self.connections[channel].remove(websocket)

    async def broadcast(self, channel, message):
        if channel in self.connections:
            for websocket in self.connections[channel]:
                try:
                    await websocket.send_json(message)
                except:
                    pass

class SystemMonitor:
    async def get_system_health(self): return {"status": "healthy"}
    async def get_active_workflows(self): return []

class ApprovalManager:
    async def get_pending_count(self): return 0

# Load settings
settings = Settings()

# Authentication configuration
SECRET_KEY = settings.secret_key if hasattr(settings, 'secret_key') else "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User models
class User(BaseModel):
    username: str
    email: str
    full_name: str
    role: str  # admin, operator, analyst, auditor, developer
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class UserCredentials(BaseModel):
    username: str
    password: str

# Mock user database (in production, use proper database)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@kingai.com",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
        "disabled": False,
    },
    "operator": {
        "username": "operator",
        "full_name": "System Operator",
        "email": "operator@kingai.com",
        "hashed_password": pwd_context.hash("operator123"),
        "role": "operator",
        "disabled": False,
    },
    "analyst": {
        "username": "analyst",
        "full_name": "Data Analyst",
        "email": "analyst@kingai.com",
        "hashed_password": pwd_context.hash("analyst123"),
        "role": "analyst",
        "disabled": False,
    }
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

app = FastAPI(
    title="King AI v3 Master Control Panel",
    description="Comprehensive dashboard for monitoring and controlling the Agentic Framework",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Mount static files for frontend (when built)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # TODO: Implement proper JWT validation
    # For now, accept any token
    return {"user": "admin", "role": "admin"}

# Health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Dashboard overview
@app.get("/api/dashboard/overview")
async def get_dashboard_overview(user: dict = Depends(get_current_user)):
    """Get main dashboard KPIs"""
    try:
        # Get data from system monitor
        health = await system_monitor.get_system_health()
        workflows = await system_monitor.get_active_workflows()
        approvals = await approval_manager.get_pending_count()

        return {
            "health": health,
            "active_workflows": len(workflows),
            "pending_approvals": approvals,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Service health
@app.get("/api/dashboard/health")
async def get_service_health(user: dict = Depends(get_current_user)):
    """Get health status of all services"""
    services = ["orchestrator", "subagent-manager", "memory-service", "mcp-gateway", "code-exec"]
    health_status = {}

    for service in services:
        try:
            # TODO: Implement actual health checks
            health_status[service] = {"status": "healthy", "port": 8000 + services.index(service)}
        except:
            health_status[service] = {"status": "unhealthy", "port": 8000 + services.index(service)}

    return health_status

# Dashboard status endpoint
@app.get("/api/dashboard/status")
async def get_dashboard_status(user: dict = Depends(get_current_user)):
    """Get dashboard status summary"""
    return {
        "agents": 3,
        "workflows": 2,
        "approvals": 1,
        "alerts": 0
    }

# Dashboard agents endpoint
@app.get("/api/dashboard/agents")
async def get_dashboard_agents(user: dict = Depends(get_current_user)):
    """Get list of agents"""
    return [
        {
            "id": "agent-1",
            "name": "Business Analyst",
            "status": "running",
            "type": "analysis",
            "lastActivity": "2024-01-15T10:30:00Z"
        },
        {
            "id": "agent-2",
            "name": "Code Generator",
            "status": "running",
            "type": "development",
            "lastActivity": "2024-01-15T10:25:00Z"
        },
        {
            "id": "agent-3",
            "name": "Workflow Orchestrator",
            "status": "paused",
            "type": "orchestration",
            "lastActivity": "2024-01-15T10:20:00Z"
        }
    ]

# Workflow endpoints
@app.get("/api/workflows")
async def get_workflows(user: dict = Depends(get_current_user)):
    """Get list of workflows"""
    return [
        {
            "id": "wf-1",
            "name": "Business Analysis Workflow",
            "description": "Complete business analysis and planning",
            "steps": [
                {
                    "id": "step-1",
                    "name": "Market Research",
                    "status": "completed",
                    "agent": "Research Agent",
                    "duration": 45
                },
                {
                    "id": "step-2",
                    "name": "Financial Analysis",
                    "status": "running",
                    "agent": "Finance Agent",
                    "duration": 30
                },
                {
                    "id": "step-3",
                    "name": "Strategy Development",
                    "status": "pending",
                    "agent": "Strategy Agent"
                }
            ],
            "status": "running",
            "createdAt": "2024-01-15T09:00:00Z",
            "updatedAt": "2024-01-15T10:30:00Z"
        },
        {
            "id": "wf-2",
            "name": "Code Generation Pipeline",
            "description": "Automated code generation and testing",
            "steps": [
                {
                    "id": "step-4",
                    "name": "Requirements Analysis",
                    "status": "completed",
                    "agent": "Analysis Agent",
                    "duration": 25
                },
                {
                    "id": "step-5",
                    "name": "Code Generation",
                    "status": "completed",
                    "agent": "Code Generator",
                    "duration": 120
                },
                {
                    "id": "step-6",
                    "name": "Testing",
                    "status": "running",
                    "agent": "Test Agent",
                    "duration": 60
                }
            ],
            "status": "running",
            "createdAt": "2024-01-15T08:30:00Z",
            "updatedAt": "2024-01-15T10:25:00Z"
        }
    ]

@app.get("/api/workflows/{workflow_id}")
async def get_workflow(workflow_id: str, user: dict = Depends(get_current_user)):
    """Get workflow details"""
    return {
        "id": workflow_id,
        "name": "Business Analysis Pipeline",
        "description": "Analyze business data and generate insights",
        "status": "running",
        "steps": [
            {"id": "step-1", "name": "Data Collection", "type": "data", "status": "completed"},
            {"id": "step-2", "name": "Analysis", "type": "llm", "status": "running"},
            {"id": "step-3", "name": "Report Generation", "type": "tool", "status": "pending"}
        ],
        "createdAt": "2024-01-10T09:00:00Z"
    }

# Approval endpoints
@app.get("/api/approvals")
async def get_approvals(user: dict = Depends(get_current_user)):
    """Get list of approval requests"""
    return [
        {
            "id": "apr-1",
            "type": "workflow_execution",
            "title": "Production Data Analysis Workflow",
            "description": "Execute comprehensive data analysis on production customer database",
            "requester": "Data Analytics Team",
            "priority": "high",
            "status": "pending",
            "submittedAt": "2024-01-15T09:30:00Z",
            "dueDate": "2024-01-16T17:00:00Z",
            "riskLevel": "medium",
            "category": "Data Processing"
        },
        {
            "id": "apr-2",
            "type": "agent_deployment",
            "title": "New Customer Support Agent",
            "description": "Deploy AI agent for automated customer support responses",
            "requester": "Customer Service Team",
            "priority": "medium",
            "status": "approved",
            "submittedAt": "2024-01-14T14:20:00Z",
            "riskLevel": "low",
            "category": "Agent Deployment"
        }
    ]

@app.post("/api/approvals/{approval_id}/approve")
async def approve_request(approval_id: str, user: dict = Depends(get_current_user)):
    """Approve an approval request"""
    return {"status": "approved", "message": f"Approval {approval_id} has been approved"}

@app.post("/api/approvals/{approval_id}/reject")
async def reject_request(approval_id: str, user: dict = Depends(get_current_user)):
    """Reject an approval request"""
    return {"status": "rejected", "message": f"Approval {approval_id} has been rejected"}

# Analytics endpoints
@app.get("/api/analytics/metrics")
async def get_analytics_metrics(user: dict = Depends(get_current_user)):
    """Get analytics metrics"""
    return {
        "totalRequests": 1247,
        "activeAgents": 23,
        "successRate": 94.5,
        "avgResponseTime": 2.3,
        "costSavings": 45000,
        "period": "7d"
    }

@app.get("/api/analytics/chart-data")
async def get_chart_data(user: dict = Depends(get_current_user)):
    """Get chart data for analytics"""
    return [
        {"date": "2024-01-08", "requests": 120, "success": 115},
        {"date": "2024-01-09", "requests": 135, "success": 128},
        {"date": "2024-01-10", "requests": 142, "success": 138},
        {"date": "2024-01-11", "requests": 158, "success": 152},
        {"date": "2024-01-12", "requests": 167, "success": 160},
        {"date": "2024-01-13", "requests": 145, "success": 140},
        {"date": "2024-01-14", "requests": 178, "success": 172}
    ]

# Analytics endpoints
@app.get("/api/analytics")
async def get_analytics(range: str = "7d", user: dict = Depends(get_current_user)):
    """Get analytics data"""
    return {
        "totalAgents": 5,
        "activeWorkflows": 3,
        "completedTasks": 127,
        "averageResponseTime": 2.3,
        "systemUptime": 99.2,
        "errorRate": 0.8,
        "performanceMetrics": [
            {"period": "2024-01-10", "agents": 4, "workflows": 2, "tasks": 45},
            {"period": "2024-01-11", "agents": 5, "workflows": 3, "tasks": 52},
            {"period": "2024-01-12", "agents": 5, "workflows": 4, "tasks": 38}
        ],
        "agentPerformance": [
            {"name": "Business Analyst", "tasksCompleted": 45, "averageTime": 1.8, "successRate": 98},
            {"name": "Code Generator", "tasksCompleted": 32, "averageTime": 3.2, "successRate": 95},
            {"name": "Data Processor", "tasksCompleted": 28, "averageTime": 2.1, "successRate": 97},
            {"name": "Research Agent", "tasksCompleted": 22, "averageTime": 2.8, "successRate": 92}
        ]
    }

# Settings endpoints
@app.get("/api/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    """Get system settings"""
    return {
        "system": {
            "name": "King AI v3 Production",
            "debugMode": False,
            "maxConcurrentWorkflows": 10
        },
        "ai": {
            "ollamaUrl": "http://localhost:11434",
            "model": "llama3.1:8b",
            "temperature": 0.7
        },
        "security": {
            "sessionTimeout": 3600,
            "maxLoginAttempts": 5
        }
    }

@app.put("/api/settings")
async def update_settings(settings: dict, user: dict = Depends(get_current_user)):
    """Update system settings"""
    return {"status": "updated", "message": "Settings have been updated successfully"}

# Business P&L endpoints
@app.get("/api/business/pl/summary")
async def get_pl_summary(user: dict = Depends(get_current_user)):
    """Get P&L summary"""
    # TODO: Implement business logic
    return {
        "total_revenue": 0,
        "total_expenses": 0,
        "net_profit": 0,
        "margin_percent": 0
    }

@app.get("/api/business/pl/trend")
async def get_pl_trend(user: dict = Depends(get_current_user)):
    """Get P&L trend data"""
    # TODO: Implement trend calculation
    return []

@app.post("/api/business/transactions")
async def add_transaction(transaction: dict, user: dict = Depends(get_current_user)):
    """Add business transaction"""
    # TODO: Implement transaction recording
    return {"status": "recorded"}

# Analytics endpoints
@app.get("/api/analytics/workflows/throughput")
async def get_workflow_throughput(user: dict = Depends(get_current_user)):
    """Get workflow throughput metrics"""
    # TODO: Implement analytics
    return []

@app.get("/api/analytics/agents/utilization")
async def get_agent_utilization(user: dict = Depends(get_current_user)):
    """Get agent utilization metrics"""
    # TODO: Implement analytics
    return []

@app.get("/api/analytics/models/usage")
async def get_model_usage(user: dict = Depends(get_current_user)):
    """Get model usage statistics"""
    # TODO: Implement analytics
    return []

# Conversational interface
@app.post("/api/chat/message")
async def send_chat_message(message: dict, user: dict = Depends(get_current_user)):
    """Send message to King AI"""
    # TODO: Integrate with conversational AI
    return {"response": "This feature is under development"}

@app.get("/api/chat/history")
async def get_chat_history(user: dict = Depends(get_current_user)):
    """Get chat history"""
    # TODO: Implement chat history
    return []

# WebSocket endpoint for real-time updates
@app.websocket("/ws/activity-feed")
async def activity_feed(websocket):
    """WebSocket for real-time activity feed"""
    await websocket.accept()
    ws_manager.connect(websocket, "activity")
    try:
        while True:
            # Send periodic updates
            data = {
                "type": "activity",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "System activity update"
            }
            await websocket.send_json(data)
            await asyncio.sleep(5)
    except:
        ws_manager.disconnect(websocket, "activity")

@app.websocket("/ws/approvals")
async def approvals_feed(websocket):
    """WebSocket for approval updates"""
    await websocket.accept()
    ws_manager.connect(websocket, "approvals")
    try:
        while True:
            # Send approval updates
            await websocket.send_json({
                "type": "approvals",
                "pending_count": 2,
                "timestamp": datetime.utcnow().isoformat()
            })
            await asyncio.sleep(10)
    except:
        ws_manager.disconnect(websocket, "approvals")

@app.websocket("/ws/workflows/{workflow_id}")
async def workflow_feed(websocket, workflow_id: str):
    """WebSocket for specific workflow updates"""
    await ws_manager.connect(websocket, f"workflow_{workflow_id}")
    try:
        while True:
            # Send workflow status updates
            await websocket.send_json({
                "type": "workflow",
                "workflow_id": workflow_id,
                "status": "running",
                "timestamp": datetime.utcnow().isoformat()
            })
            await asyncio.sleep(2)
    except:
        ws_manager.disconnect(websocket, f"workflow_{workflow_id}")

# Settings endpoints
@app.get("/api/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    """Get system settings"""
    return [
        {
            "id": "system-name",
            "category": "System",
            "name": "System Name",
            "description": "Display name for this King AI instance",
            "type": "text",
            "value": "King AI v3 Control Panel",
            "requiresRestart": False
        },
        {
            "id": "max-concurrent-workflows",
            "category": "Performance",
            "name": "Max Concurrent Workflows",
            "description": "Maximum number of workflows that can run simultaneously",
            "type": "number",
            "value": 10,
            "requiresRestart": True
        },
        {
            "id": "enable-notifications",
            "category": "Notifications",
            "name": "Enable Notifications",
            "description": "Send notifications for important events",
            "type": "boolean",
            "value": True,
            "requiresRestart": False
        },
        {
            "id": "ollama-endpoint",
            "category": "AI Models",
            "name": "Ollama Endpoint",
            "description": "URL for the Ollama API server",
            "type": "text",
            "value": "http://localhost:11434",
            "requiresRestart": True
        }
    ]

@app.put("/api/settings")
async def update_settings(settings: dict, user: dict = Depends(get_current_user)):
    """Update system settings"""
    return {"status": "updated", "message": "Settings have been updated successfully"}

# Authentication endpoints
@app.post("/api/auth/login", response_model=Token)
async def login_for_access_token(credentials: UserCredentials):
    """Authenticate user and return JWT token"""
    user = authenticate_user(fake_users_db, credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/auth/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """User logout"""
    return {"status": "logged_out"}

@app.get("/api/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user info"""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role
    }

@app.get("/api/auth/users")
async def get_users(current_user: User = Depends(get_current_active_user)):
    """Get all users (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    users = []
    for username, user_data in fake_users_db.items():
        users.append({
            "username": user_data["username"],
            "full_name": user_data["full_name"],
            "email": user_data["email"],
            "role": user_data["role"],
            "disabled": user_data["disabled"]
        })
    return users

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8100,
        reload=True,
        log_level="info"
    )