from fastapi import FastAPI, WebSocket, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from redis.asyncio import Redis
import jwt
import os
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel
import uvicorn

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/control_panel")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Database setup
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=Session, expire_on_commit=False)

# Redis setup
redis = Redis.from_url(REDIS_URL)

# FastAPI app
app = FastAPI(title="King AI Control Panel", version="1.0.0")

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

# Database models (simplified)
class User:
    id: int
    username: str
    hashed_password: str
    role: str

# Helper functions
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
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    return token_data

# Routes
@app.post("/api/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    # Simplified authentication - in real app, check against database
    if user_credentials.username == "admin" and user_credentials.password == "password":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_credentials.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Incorrect username or password")

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
    # Mock health check for all 5 services
    return {
        "orchestrator": {"status": "healthy", "port": 8000},
        "subagent_manager": {"status": "healthy", "port": 8001},
        "memory_service": {"status": "healthy", "port": 8002},
        "mcp_gateway": {"status": "healthy", "port": 8080},
        "code_executor": {"status": "healthy", "port": 8004}
    }

# WebSocket endpoints
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

@app.websocket("/ws/approvals")
async def approvals_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Mock approvals data
            data = {"type": "approval", "id": "123", "status": "pending", "timestamp": datetime.utcnow().isoformat()}
            await websocket.send_json(data)
            await asyncio.sleep(10)
    except Exception as e:
        print(f"WebSocket error: {e}")

# Proxy routes (simplified - in real app, use httpx or similar)
@app.api_route("/api/orchestrator/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_orchestrator(path: str, request: Request, current_user: TokenData = Depends(get_current_user)):
    # Proxy to localhost:8000
    target_url = f"http://localhost:8000/{path}"
    # In real implementation, forward the request
    return {"message": f"Proxy to orchestrator: {target_url}"}

# Similar proxy routes for other services...

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)