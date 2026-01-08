"""
Banking API Routes - Plaid banking integration endpoints.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from src.agents.banking import BankingAgent
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["banking"])

_agent: Optional[BankingAgent] = None


def get_agent() -> BankingAgent:
    global _agent
    if _agent is None:
        _agent = BankingAgent()
    return _agent


class InitPlaidRequest(BaseModel):
    business_id: str
    client_id: str
    secret: str
    environment: str = "sandbox"


class LinkTokenRequest(BaseModel):
    business_id: str
    user_id: str
    products: Optional[list[str]] = None


class ExchangeTokenRequest(BaseModel):
    business_id: str
    public_token: str


class DisconnectRequest(BaseModel):
    business_id: str
    item_id: str


@router.post("/init")
async def init_plaid(req: InitPlaidRequest, agent: BankingAgent = Depends(get_agent)):
    """Initialize Plaid connection."""
    success = await agent.initialize_plaid(
        req.business_id, req.client_id, req.secret, req.environment
    )
    if not success:
        raise HTTPException(500, "Failed to initialize Plaid")
    return {"status": "ok"}


@router.post("/link-token")
async def create_link_token(req: LinkTokenRequest, agent: BankingAgent = Depends(get_agent)):
    """Create a link token for Plaid Link."""
    result = await agent.create_link_token(req.business_id, req.user_id, req.products)
    if not result["success"]:
        raise HTTPException(400, result.get("error", "Failed to create link token"))
    return result["output"]


@router.post("/exchange-token")
async def exchange_token(req: ExchangeTokenRequest, agent: BankingAgent = Depends(get_agent)):
    """Exchange public token for access token."""
    result = await agent.exchange_token(req.business_id, req.public_token)
    if not result["success"]:
        raise HTTPException(400, result.get("error", "Failed to exchange token"))
    return result.get("metadata", {})


@router.get("/accounts/{business_id}")
async def get_accounts(business_id: str, agent: BankingAgent = Depends(get_agent)):
    """Get all connected bank accounts."""
    result = await agent.get_accounts(business_id)
    if not result["success"]:
        raise HTTPException(400, result.get("error", "Failed to get accounts"))
    return result["output"]


@router.get("/balances/{business_id}")
async def get_balances(business_id: str, agent: BankingAgent = Depends(get_agent)):
    """Get real-time balances."""
    result = await agent.get_balances(business_id)
    if not result["success"]:
        raise HTTPException(400, result.get("error", "Failed to get balances"))
    return result["output"]


@router.get("/transactions/{business_id}")
async def get_transactions(
    business_id: str,
    days: int = 30,
    account_id: Optional[str] = None,
    agent: BankingAgent = Depends(get_agent)
):
    """Get recent transactions."""
    result = await agent.get_transactions(business_id, days, account_id)
    if not result["success"]:
        raise HTTPException(400, result.get("error", "Failed to get transactions"))
    return result["output"]


@router.get("/cash-flow/{business_id}")
async def analyze_cash_flow(
    business_id: str,
    days: int = 30,
    agent: BankingAgent = Depends(get_agent)
):
    """Analyze cash flow."""
    result = await agent.analyze_cash_flow(business_id, days)
    if not result["success"]:
        raise HTTPException(400, result.get("error", "Failed to analyze cash flow"))
    return result["output"]


@router.get("/health/{business_id}")
async def get_financial_health(business_id: str, agent: BankingAgent = Depends(get_agent)):
    """Get financial health score."""
    result = await agent.get_financial_health(business_id)
    if not result["success"]:
        raise HTTPException(400, result.get("error", "Failed to get financial health"))
    return result["output"]


@router.post("/disconnect")
async def disconnect_bank(req: DisconnectRequest, agent: BankingAgent = Depends(get_agent)):
    """Disconnect a bank account."""
    result = await agent.disconnect_bank(req.business_id, req.item_id)
    if not result["success"]:
        raise HTTPException(400, result.get("error", "Failed to disconnect bank"))
    return {"status": "disconnected"}
