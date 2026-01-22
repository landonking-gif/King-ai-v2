from fastapi import FastAPI, WebSocket, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DECIMAL, DateTime, Integer, Text, ForeignKey, select
from redis.asyncio import Redis
from jose import jwt
import os
from datetime import datetime, timedelta
from typing import List, Optional, Literal
from pydantic import BaseModel
import uvicorn
import httpx
import asyncio
from passlib.context import CryptContext

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./control_panel.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Database setup
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(20))  # Admin, Operator, Analyst, Auditor, Developer
    is_active: Mapped[bool] = mapped_column(default=True)

# class FinancialTransaction(Base):
#     __tablename__ = "financial_transactions"
#     id: Mapped[str] = mapped_column(String(50), primary_key=True)
#     business_unit_id: Mapped[str] = mapped_column(String(50), ForeignKey("business_units.id"))
#     workflow_id: Mapped[Optional[str]] = mapped_column(String(50))
#     transaction_type: Mapped[str] = mapped_column(String(10))  # revenue or expense
#     category: Mapped[str] = mapped_column(String(50))
#     amount: Mapped[DECIMAL] = mapped_column(DECIMAL(10, 2))
#     timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Redis setup
redis = Redis.from_url(REDIS_URL)

# FastAPI app
app = FastAPI(title="King AI Control Panel", version="1.0.0")

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models
class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class BusinessUnitModel(BaseModel):
    id: str
    name: str
    description: Optional[str] = None

# class FinancialTransactionModel(BaseModel):
#     id: str
#     business_unit_id: str
#     workflow_id: Optional[str] = None
#     transaction_type: Literal["revenue", "expense"]
#     category: str
#     amount: float
#     timestamp: datetime

class PLSummary(BaseModel):
    period: str
    total_revenue: float
    total_expenses: float
    net_profit: float
    margin_percent: float

async def get_db():
    async with async_session() as session:
        yield session

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
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
        role: str = payload.get("role")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except jwt.PyJWTError:
        raise credentials_exception
    return token_data

# Routes
@app.post("/api/auth/login", response_model=Token)
async def login(user_credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    user = await db.execute(select(User).where(User.username == user_credentials.username))
    user = user.scalar_one_or_none()
    if not user or not verify_password(user_credentials.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/auth/logout")
async def logout(current_user: TokenData = Depends(get_current_user)):
    # In a real app, you might blacklist the token
    return {"message": "Successfully logged out"}

@app.get("/api/auth/me")
async def read_users_me(current_user: TokenData = Depends(get_current_user)):
    return {"username": current_user.username}

@app.get("/api/dashboard/overview")
async def get_dashboard_overview(current_user: TokenData = Depends(get_current_user)):
    # Mock data - in real app, aggregate from all services
    return {
        "active_workflows": 5,
        "pending_approvals": 3,
        "total_tokens": 125000,
        "system_health": "good"
    }

@app.get("/api/dashboard/health")
async def get_system_health(current_user: TokenData = Depends(get_current_user)):
    # Mock system health data
    return {
        "orchestrator": {"status": "healthy", "port": 8000},
        "subagent_manager": {"status": "healthy", "port": 8001},
        "memory_service": {"status": "healthy", "port": 8002},
        "mcp_gateway": {"status": "healthy", "port": 8080},
        "code_executor": {"status": "healthy", "port": 8004}
    }

@app.get("/api/business/pl/summary", response_model=PLSummary)
async def get_pl_summary(period: str = "30d", db: AsyncSession = Depends(get_db), current_user: TokenData = Depends(get_current_user)):
    # Mock data for now
    return PLSummary(
        period=period,
        total_revenue=10000.0,
        total_expenses=7000.0,
        net_profit=3000.0,
        margin_percent=30.0
    )

@app.get("/api/business/pl/trend")
async def get_pl_trend(current_user: TokenData = Depends(get_current_user)):
    # Mock trend data
    return [
        {"date": "2024-01-01", "revenue": 8000, "expenses": 6000, "profit": 2000},
        {"date": "2024-01-02", "revenue": 9000, "expenses": 6500, "profit": 2500},
        {"date": "2024-01-03", "revenue": 10000, "expenses": 7000, "profit": 3000}
    ]

@app.post("/api/business/transactions")
# async def create_transaction(transaction: FinancialTransactionModel, current_user: TokenData = Depends(get_current_user)):
#     # Create transaction in DB
#     # db_transaction = FinancialTransaction(**transaction.dict())
#     # db.add(db_transaction)
#     # await db.commit()
#     # await db.refresh(db_transaction)
#     return transaction

# Analytics
@app.get("/api/analytics/workflows/throughput")
async def get_workflow_throughput(current_user: TokenData = Depends(get_current_user)):
    # Mock data
    return [
        {"hour": "00", "completed": 5},
        {"hour": "01", "completed": 3},
        {"hour": "02", "completed": 7}
    ]

@app.get("/api/analytics/agents/utilization")
async def get_agent_utilization(current_user: TokenData = Depends(get_current_user)):
    # Mock data
    return [
        {"agent": "orchestrator", "utilization": 85},
        {"agent": "subagent", "utilization": 70},
        {"agent": "memory", "utilization": 60}
    ]

@app.get("/api/analytics/models/usage")
async def get_model_usage(current_user: TokenData = Depends(get_current_user)):
    # Mock data
    return [
        {"model": "gpt-4", "requests": 1000, "cost": 50.0},
        {"model": "claude-3", "requests": 800, "cost": 40.0}
    ]
@app.websocket("/ws/activity-feed")
async def activity_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Mock activity data - in real app, subscribe to Redis pubsub
            data = {"type": "activity", "message": "Workflow completed", "timestamp": datetime.utcnow().isoformat()}
            await websocket.send_json(data)
            await asyncio.sleep(5)  # Send every 5 seconds
    except Exception as e:
        print(f"WebSocket error: {e}")

# Conversational
@app.post("/api/chat/message")
async def send_chat_message(message: dict, current_user: TokenData = Depends(get_current_user)):
    # Mock response - in real app, call orchestrator
    return {"response": f"Echo: {message.get('text', '')}", "type": "text"}

@app.get("/api/chat/history")
async def get_chat_history(current_user: TokenData = Depends(get_current_user)):
    # Mock history
    return [{"text": "Hello", "response": "Hi there!", "timestamp": datetime.utcnow().isoformat()}]

# Proxy routes
SERVICE_URLS = {
    "orchestrator": "http://localhost:8000",
    "subagent-manager": "http://localhost:8001",
    "memory": "http://localhost:8002",
    "mcp": "http://localhost:8080",
    "code-exec": "http://localhost:8004"
}

async def proxy_request(service: str, path: str, request: Request, method: str):
    target_url = f"{SERVICE_URLS[service]}/{path}"
    async with httpx.AsyncClient() as client:
        # Forward headers, body, etc.
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ['host', 'authorization']}
        body = await request.body()
        try:
            response = await client.request(method, target_url, headers=headers, content=body, params=request.query_params)
            # Return FastAPI Response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Service {service} unavailable: {str(e)}")

@app.api_route("/api/orchestrator/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_orchestrator(path: str, request: Request, current_user: TokenData = Depends(get_current_user)):
    return await proxy_request("orchestrator", path, request, request.method)

@app.api_route("/api/subagent-manager/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_subagent_manager(path: str, request: Request, current_user: TokenData = Depends(get_current_user)):
    return await proxy_request("subagent-manager", path, request, request.method)

@app.api_route("/api/memory/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_memory(path: str, request: Request, current_user: TokenData = Depends(get_current_user)):
    return await proxy_request("memory", path, request, request.method)

@app.api_route("/api/mcp/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_mcp(path: str, request: Request, current_user: TokenData = Depends(get_current_user)):
    return await proxy_request("mcp", path, request, request.method)

@app.api_route("/api/code-exec/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_code_exec(path: str, request: Request, current_user: TokenData = Depends(get_current_user)):
    return await proxy_request("code-exec", path, request, request.method)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)