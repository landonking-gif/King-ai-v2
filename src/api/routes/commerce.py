"""
Commerce API Routes - REST endpoints for e-commerce operations.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel

from src.agents.commerce import (
    CommerceAgent,
    ProductData,
    OrderStatus,
    FulfillmentStatus
)
from src.integrations.shopify_client import ShopifyConfig
from src.utils.structured_logging import get_logger
from config.settings import settings

logger = get_logger("commerce_api")
router = APIRouter(prefix="/commerce", tags=["commerce"])


class StoreConfigRequest(BaseModel):
    """Request to configure store connection."""
    shop_name: str
    access_token: str
    webhook_secret: Optional[str] = None


class CreateProductRequest(BaseModel):
    """Request to create a product."""
    title: str
    description: str = ""
    vendor: str = ""
    product_type: str = ""
    tags: List[str] = []
    price: Optional[float] = None
    sku: Optional[str] = None
    inventory_quantity: Optional[int] = None


class GenerateProductRequest(BaseModel):
    """Request to generate product listing."""
    product_info: str
    style: str = "professional"


class InventoryUpdateRequest(BaseModel):
    """Request to update inventory."""
    inventory_item_id: str
    location_id: str
    quantity: int
    adjustment: bool = False


class FulfillOrderRequest(BaseModel):
    """Request to fulfill an order."""
    tracking_number: Optional[str] = None
    tracking_company: Optional[str] = None
    notify_customer: bool = True


class ProductResponse(BaseModel):
    """Response with product data."""
    id: str
    title: str
    status: str
    vendor: str
    product_type: str
    created_at: str


class MetricsResponse(BaseModel):
    """Response with store metrics."""
    total_orders: int
    total_revenue: float
    average_order_value: float
    orders_by_status: dict
    top_products: List[dict]
    period: str


# Global agent instance
_commerce_agent: Optional[CommerceAgent] = None


def get_commerce_agent() -> CommerceAgent:
    """Get or create commerce agent."""
    global _commerce_agent
    if _commerce_agent is None:
        _commerce_agent = CommerceAgent()
        
        # Configure from settings if available
        if settings.SHOPIFY_SHOP_NAME and settings.SHOPIFY_ACCESS_TOKEN:
            config = ShopifyConfig(
                shop_name=settings.SHOPIFY_SHOP_NAME,
                access_token=settings.SHOPIFY_ACCESS_TOKEN,
                webhook_secret=settings.SHOPIFY_WEBHOOK_SECRET
            )
            _commerce_agent.set_store(config)
    
    return _commerce_agent


@router.post("/configure")
async def configure_store(request: StoreConfigRequest):
    """Configure Shopify store connection."""
    try:
        agent = get_commerce_agent()
        
        config = ShopifyConfig(
            shop_name=request.shop_name,
            access_token=request.access_token,
            webhook_secret=request.webhook_secret
        )
        
        agent.set_store(config)
        
        # Test connection
        shop = await agent.shopify.get_shop()
        
        return {
            "status": "connected",
            "shop_name": shop.get("name"),
            "domain": shop.get("domain")
        }
        
    except Exception as e:
        logger.error(f"Store configuration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/shop")
async def get_shop_info():
    """Get connected shop information."""
    agent = get_commerce_agent()
    
    if not agent.shopify:
        raise HTTPException(status_code=400, detail="Shopify not configured")
    
    shop = await agent.shopify.get_shop()
    return shop


# ===== Product Endpoints =====

@router.get("/products")
async def list_products(
    status: Optional[str] = None,
    limit: int = 50
):
    """List all products."""
    agent = get_commerce_agent()
    
    if not agent.shopify:
        raise HTTPException(status_code=400, detail="Shopify not configured")
    
    products = await agent.get_products(status=status, limit=limit)
    return {"products": products, "count": len(products)}


@router.post("/products", response_model=ProductResponse)
async def create_product(request: CreateProductRequest):
    """Create a new product."""
    agent = get_commerce_agent()
    
    if not agent.shopify:
        raise HTTPException(status_code=400, detail="Shopify not configured")
    
    try:
        variants = []
        if request.price or request.sku or request.inventory_quantity:
            variant = {}
            if request.price:
                variant["price"] = str(request.price)
            if request.sku:
                variant["sku"] = request.sku
            if request.inventory_quantity:
                variant["inventory_quantity"] = request.inventory_quantity
            variants.append(variant)
        
        product = ProductData(
            title=request.title,
            description=request.description,
            vendor=request.vendor,
            product_type=request.product_type,
            tags=request.tags,
            variants=variants
        )
        
        result = await agent.create_product(product)
        
        return ProductResponse(
            id=str(result.get("id")),
            title=result.get("title"),
            status=result.get("status"),
            vendor=result.get("vendor", ""),
            product_type=result.get("product_type", ""),
            created_at=result.get("created_at", "")
        )
        
    except Exception as e:
        logger.error(f"Product creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/products/generate")
async def generate_product_listing(request: GenerateProductRequest):
    """Generate product listing using AI."""
    agent = get_commerce_agent()
    
    try:
        result = await agent.generate_product_listing(
            product_info=request.product_info,
            style=request.style
        )
        return result
        
    except Exception as e:
        logger.error(f"Product generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Inventory Endpoints =====

@router.post("/inventory/update")
async def update_inventory(updates: List[InventoryUpdateRequest]):
    """Update inventory levels."""
    agent = get_commerce_agent()
    
    if not agent.shopify:
        raise HTTPException(status_code=400, detail="Shopify not configured")
    
    try:
        update_dicts = [u.dict() for u in updates]
        result = await agent.update_inventory(update_dicts)
        return result
        
    except Exception as e:
        logger.error(f"Inventory update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inventory/low-stock")
async def get_low_stock(threshold: int = 10):
    """Get products with low inventory."""
    agent = get_commerce_agent()
    
    if not agent.shopify:
        raise HTTPException(status_code=400, detail="Shopify not configured")
    
    products = await agent.get_low_stock_products(threshold)
    return {"low_stock_products": products, "count": len(products)}


# ===== Order Endpoints =====

@router.get("/orders")
async def list_orders(
    status: str = "any",
    fulfillment_status: Optional[str] = None,
    days: int = 30
):
    """List orders with filters."""
    agent = get_commerce_agent()
    
    if not agent.shopify:
        raise HTTPException(status_code=400, detail="Shopify not configured")
    
    orders = await agent.get_orders(
        status=OrderStatus(status),
        fulfillment_status=FulfillmentStatus(fulfillment_status) if fulfillment_status else None,
        days=days
    )
    
    return {"orders": orders, "count": len(orders)}


@router.get("/orders/unfulfilled")
async def get_unfulfilled_orders():
    """Get all unfulfilled orders."""
    agent = get_commerce_agent()
    
    if not agent.shopify:
        raise HTTPException(status_code=400, detail="Shopify not configured")
    
    orders = await agent.get_unfulfilled_orders()
    return {"orders": orders, "count": len(orders)}


@router.post("/orders/{order_id}/fulfill")
async def fulfill_order(order_id: str, request: FulfillOrderRequest):
    """Fulfill an order."""
    agent = get_commerce_agent()
    
    if not agent.shopify:
        raise HTTPException(status_code=400, detail="Shopify not configured")
    
    try:
        result = await agent.fulfill_order(
            order_id=order_id,
            tracking_number=request.tracking_number,
            tracking_company=request.tracking_company,
            notify_customer=request.notify_customer
        )
        return result
        
    except Exception as e:
        logger.error(f"Order fulfillment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Analytics Endpoints =====

@router.get("/metrics", response_model=MetricsResponse)
async def get_store_metrics(days: int = 30):
    """Get store performance metrics."""
    agent = get_commerce_agent()
    
    if not agent.shopify:
        raise HTTPException(status_code=400, detail="Shopify not configured")
    
    metrics = await agent.get_store_metrics(days=days)
    
    return MetricsResponse(
        total_orders=metrics.total_orders,
        total_revenue=metrics.total_revenue,
        average_order_value=metrics.average_order_value,
        orders_by_status=metrics.orders_by_status,
        top_products=metrics.top_products,
        period=metrics.period
    )


# ===== Webhook Endpoint =====

@router.post("/webhooks/{topic}")
async def handle_webhook(
    topic: str,
    request: Request,
    x_shopify_hmac_sha256: str = Header(None)
):
    """Handle Shopify webhooks."""
    agent = get_commerce_agent()
    
    if not agent.shopify:
        raise HTTPException(status_code=400, detail="Shopify not configured")
    
    body = await request.body()
    
    # Verify webhook
    if x_shopify_hmac_sha256:
        if not agent.shopify.verify_webhook(body, x_shopify_hmac_sha256):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
    
    data = await request.json()
    
    logger.info(f"Webhook received: {topic}", data_keys=list(data.keys()))
    
    # Handle different webhook topics
    if topic == "orders/create":
        # New order created
        pass
    elif topic == "orders/fulfilled":
        # Order fulfilled
        pass
    elif topic == "products/update":
        # Product updated
        pass
    elif topic == "inventory_levels/update":
        # Inventory changed
        pass
    
    return {"status": "received", "topic": topic}
