"""
Supplier API Routes - REST endpoints for supplier operations.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from src.agents.supplier import SupplierAgent
from src.integrations.supplier_client import SupplierType
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/supplier", tags=["supplier"])

# Shared agent instance
_agent: Optional[SupplierAgent] = None


def get_agent() -> SupplierAgent:
    global _agent
    if _agent is None:
        _agent = SupplierAgent()
    return _agent


class InitSupplierRequest(BaseModel):
    supplier_type: str = Field(..., description="Supplier type: aliexpress, cj_dropshipping")
    credentials: dict = Field(..., description="API credentials for supplier")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    supplier: Optional[str] = None
    min_margin: float = Field(30.0, ge=0, le=100)
    max_price: float = Field(100.0, ge=0)
    limit: int = Field(20, ge=1, le=50)


class ImportRequest(BaseModel):
    business_id: str
    product_id: str
    supplier: str
    price_override: Optional[float] = None
    title_override: Optional[str] = None


class FulfillRequest(BaseModel):
    business_id: str
    order_id: str


@router.post("/init")
async def init_supplier(req: InitSupplierRequest, agent: SupplierAgent = Depends(get_agent)):
    """Initialize a supplier connection."""
    try:
        supplier_type = SupplierType(req.supplier_type)
    except ValueError:
        raise HTTPException(400, f"Invalid supplier: {req.supplier_type}")
    
    success = await agent.initialize_supplier(supplier_type, req.credentials)
    if not success:
        raise HTTPException(500, "Failed to initialize supplier")
    
    return {"status": "ok", "supplier": req.supplier_type}


@router.post("/search")
async def search_products(req: SearchRequest, agent: SupplierAgent = Depends(get_agent)):
    """Search for products across suppliers."""
    supplier_type = SupplierType(req.supplier) if req.supplier else None
    result = await agent.search_products(
        query=req.query,
        supplier_type=supplier_type,
        min_margin=req.min_margin,
        max_price=req.max_price,
        limit=req.limit,
    )
    
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Search failed"))
    
    return result.get("output", {})


@router.post("/import")
async def import_product(req: ImportRequest, agent: SupplierAgent = Depends(get_agent)):
    """Import a supplier product to Shopify store."""
    result = await agent.import_to_store(
        business_id=req.business_id,
        supplier_product_id=req.product_id,
        supplier_type=SupplierType(req.supplier),
        price_override=req.price_override,
        title_override=req.title_override,
    )
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Import failed"))
    
    return result.get("output", {})


@router.post("/sync-inventory/{business_id}")
async def sync_inventory(business_id: str, agent: SupplierAgent = Depends(get_agent)):
    """Sync inventory from suppliers to store."""
    result = await agent.sync_inventory(business_id)
    
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Sync failed"))
    
    return result.get("output", {})


@router.post("/fulfill")
async def fulfill_order(req: FulfillRequest, agent: SupplierAgent = Depends(get_agent)):
    """Forward order to supplier for fulfillment."""
    result = await agent.fulfill_order(req.business_id, req.order_id)
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Fulfillment failed"))
    
    return result.get("output", {})


@router.get("/price-changes/{business_id}")
async def check_price_changes(business_id: str, agent: SupplierAgent = Depends(get_agent)):
    """Check for supplier price changes."""
    result = await agent.check_price_changes(business_id)
    
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Price check failed"))
    
    return result.get("output", {})
