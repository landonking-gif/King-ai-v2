"""
Commerce Agent - E-commerce operations via Shopify.
Manages products, inventory, orders, and store analytics.
"""

import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.agents.base import SubAgent
from src.integrations.shopify_client import (
    ShopifyClient,
    ShopifyConfig,
    ShopifyAPIError
)
from src.utils.structured_logging import get_logger
from src.utils.metrics import TASKS_EXECUTED

logger = get_logger("commerce_agent")


# Function schema for LLM function-calling
FUNCTION_SCHEMA = {
    "name": "commerce_agent",
    "description": "E-commerce operations via Shopify - manages products, inventory, orders, and store analytics",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create_product", "update_product", "delete_product", "get_product", 
                         "list_products", "sync_inventory", "get_orders", "fulfill_order",
                         "get_store_metrics", "update_pricing"],
                "description": "The commerce action to perform"
            },
            "product_id": {
                "type": "string",
                "description": "Shopify product ID for product-specific operations"
            },
            "product_data": {
                "type": "object",
                "description": "Product data including title, description, variants, pricing",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "vendor": {"type": "string"},
                    "product_type": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "variants": {"type": "array", "items": {"type": "object"}},
                    "images": {"type": "array", "items": {"type": "object"}},
                    "status": {"type": "string", "enum": ["active", "draft", "archived"]}
                }
            },
            "order_id": {
                "type": "string",
                "description": "Order ID for order operations"
            },
            "order_status": {
                "type": "string",
                "enum": ["open", "closed", "cancelled", "any"],
                "description": "Filter orders by status"
            },
            "metrics_period": {
                "type": "string",
                "enum": ["today", "week", "month", "quarter", "year"],
                "description": "Time period for store metrics"
            },
            "inventory_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "variant_id": {"type": "string"},
                        "quantity": {"type": "integer"}
                    }
                },
                "description": "Inventory updates to apply"
            }
        },
        "required": ["action"]
    }
}


class OrderStatus(str, Enum):
    """Order status types."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    ANY = "any"


class FulfillmentStatus(str, Enum):
    """Fulfillment status types."""
    UNFULFILLED = "unfulfilled"
    PARTIAL = "partial"
    FULFILLED = "fulfilled"


@dataclass
class ProductData:
    """Product data for creation/update."""
    title: str
    description: str = ""
    vendor: str = ""
    product_type: str = ""
    tags: List[str] = field(default_factory=list)
    variants: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    status: str = "active"
    
    def to_shopify_dict(self) -> Dict[str, Any]:
        """Convert to Shopify API format."""
        data = {
            "title": self.title,
            "body_html": self.description,
            "vendor": self.vendor,
            "product_type": self.product_type,
            "tags": ",".join(self.tags),
            "status": self.status
        }
        
        if self.variants:
            data["variants"] = self.variants
        
        if self.images:
            data["images"] = self.images
        
        return data


@dataclass
class StoreMetrics:
    """Store performance metrics."""
    total_orders: int
    total_revenue: float
    average_order_value: float
    orders_by_status: Dict[str, int]
    top_products: List[Dict[str, Any]]
    period: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_orders": self.total_orders,
            "total_revenue": self.total_revenue,
            "average_order_value": self.average_order_value,
            "orders_by_status": self.orders_by_status,
            "top_products": self.top_products,
            "period": self.period
        }


class CommerceAgent(SubAgent):
    """
    Agent for e-commerce operations through Shopify.
    Handles products, inventory, orders, and analytics.
    """
    
    name = "commerce"
    description = "Handles e-commerce operations including product management, inventory, orders, and analytics via Shopify."
    
    def __init__(self, shopify_config: ShopifyConfig = None):
        """
        Initialize commerce agent.
        
        Args:
            shopify_config: Shopify store configuration
        """
        super().__init__()
        
        self.shopify: Optional[ShopifyClient] = None
        if shopify_config:
            self.shopify = ShopifyClient(shopify_config)
    
    def set_store(self, config: ShopifyConfig):
        """Configure the Shopify store connection."""
        self.shopify = ShopifyClient(config)
    
    async def execute(self, task: dict) -> dict:
        """Execute a commerce task."""
        if not self.shopify:
            # Fall back to LLM-based advice if Shopify not configured
            return await self._execute_llm_advice(task)
        
        action = task.get("action", "")
        
        try:
            if action == "create_product":
                result = await self.create_product(ProductData(**task.get("product", {})))
            elif action == "update_inventory":
                result = await self.update_inventory(task.get("updates", []))
            elif action == "get_orders":
                result = await self.get_orders(
                    status=OrderStatus(task.get("status", "any")),
                    days=task.get("days", 30)
                )
            elif action == "fulfill_order":
                result = await self.fulfill_order(
                    task.get("order_id"),
                    task.get("tracking_number"),
                    task.get("tracking_company")
                )
            elif action == "get_metrics":
                result = await self.get_store_metrics(days=task.get("days", 30))
            elif action == "generate_product":
                result = await self.generate_product_listing(
                    task.get("product_info", ""),
                    task.get("style", "professional")
                )
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
            
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True,
                "output": result if isinstance(result, dict) else result.to_dict() if hasattr(result, 'to_dict') else result,
                "metadata": {"type": "commerce_op", "action": action}
            }
            
        except ShopifyAPIError as e:
            logger.error(f"Shopify API error: {e}")
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {
                "success": False,
                "error": str(e),
                "metadata": {"errors": e.errors}
            }
        except Exception as e:
            logger.error(f"Commerce action failed: {e}")
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_llm_advice(self, task: dict) -> dict:
        """Execute using LLM advice when Shopify is not configured."""
        description = task.get("description", "Commerce task")
        input_data = task.get("input_data", {})
        
        prompt = f"""
        ### TASK: COMMERCE OPERATION
        {description}
        
        ### DETAILS:
        {input_data}
        
        ### INSTRUCTION:
        Recommend the most efficient way to handle this commerce task.
        Include pricing strategies, sourcing options, or fulfillment steps.
        """
        
        try:
            result = await self._ask_llm(prompt)
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True, 
                "output": result, 
                "metadata": {"type": "commerce_advice"}
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}
    
    # ===== Product Operations =====
    
    async def create_product(self, product: ProductData) -> Dict[str, Any]:
        """
        Create a new product in Shopify.
        
        Args:
            product: Product data
            
        Returns:
            Created product data
        """
        logger.info(f"Creating product: {product.title}")
        
        result = await self.shopify.create_product(product.to_shopify_dict())
        
        logger.info(f"Product created: {result.get('id')}")
        return result
    
    async def update_product(
        self,
        product_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an existing product."""
        logger.info(f"Updating product: {product_id}")
        return await self.shopify.update_product(product_id, updates)
    
    async def get_products(
        self,
        status: str = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all products with optional filtering."""
        response = await self.shopify.get_products(limit=limit, status=status)
        return response.items
    
    async def generate_product_listing(
        self,
        product_info: str,
        style: str = "professional"
    ) -> Dict[str, Any]:
        """
        Generate product listing using AI.
        
        Args:
            product_info: Raw product information
            style: Writing style (professional, casual, luxury)
            
        Returns:
            Generated product data
        """
        prompt = f"""Create a compelling Shopify product listing based on this information:

Product Info: {product_info}
Style: {style}

Generate:
1. A catchy, SEO-friendly title (max 70 chars)
2. An engaging product description with HTML formatting
3. 5-10 relevant tags
4. Suggested product type category

Format your response as:
TITLE: [title]
DESCRIPTION: [HTML description]
TAGS: [comma-separated tags]
PRODUCT_TYPE: [category]
"""
        
        response = await self._ask_llm(prompt)
        
        # Parse response
        product_data = self._parse_generated_product(response)
        
        return product_data
    
    def _parse_generated_product(self, response: str) -> Dict[str, Any]:
        """Parse AI-generated product listing."""
        result = {
            "title": "",
            "body_html": "",
            "tags": "",
            "product_type": ""
        }
        
        title_match = re.search(r'TITLE:\s*(.+?)(?:\n|$)', response)
        if title_match:
            result["title"] = title_match.group(1).strip()
        
        desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?:TAGS:|$)', response, re.DOTALL)
        if desc_match:
            result["body_html"] = desc_match.group(1).strip()
        
        tags_match = re.search(r'TAGS:\s*(.+?)(?:\n|PRODUCT_TYPE:|$)', response)
        if tags_match:
            result["tags"] = tags_match.group(1).strip()
        
        type_match = re.search(r'PRODUCT_TYPE:\s*(.+?)(?:\n|$)', response)
        if type_match:
            result["product_type"] = type_match.group(1).strip()
        
        return result
    
    # ===== Inventory Operations =====
    
    async def update_inventory(
        self,
        updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Batch update inventory levels.
        
        Args:
            updates: List of inventory updates
            
        Returns:
            Summary of updates
        """
        results = {
            "success": [],
            "failed": []
        }
        
        for update in updates:
            try:
                if update.get("adjustment", False):
                    await self.shopify.adjust_inventory_level(
                        inventory_item_id=update["inventory_item_id"],
                        location_id=update["location_id"],
                        adjustment=update["quantity"]
                    )
                else:
                    await self.shopify.set_inventory_level(
                        inventory_item_id=update["inventory_item_id"],
                        location_id=update["location_id"],
                        available=update["quantity"]
                    )
                
                results["success"].append(update.get("inventory_item_id"))
                
            except Exception as e:
                results["failed"].append({
                    "inventory_item_id": update.get("inventory_item_id"),
                    "error": str(e)
                })
        
        logger.info(
            f"Inventory updated",
            success=len(results["success"]),
            failed=len(results["failed"])
        )
        
        return results
    
    # ===== Order Operations =====
    
    async def get_orders(
        self,
        status: OrderStatus = OrderStatus.ANY,
        fulfillment_status: FulfillmentStatus = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get orders with filters."""
        created_at_min = datetime.now() - timedelta(days=days)
        
        response = await self.shopify.get_orders(
            status=status.value,
            fulfillment_status=fulfillment_status.value if fulfillment_status else None,
            created_at_min=created_at_min
        )
        
        return response.items
    
    async def get_unfulfilled_orders(self) -> List[Dict[str, Any]]:
        """Get all unfulfilled orders."""
        return await self.get_orders(
            status=OrderStatus.OPEN,
            fulfillment_status=FulfillmentStatus.UNFULFILLED
        )
    
    async def fulfill_order(
        self,
        order_id: str,
        tracking_number: str = None,
        tracking_company: str = None,
        notify_customer: bool = True
    ) -> Dict[str, Any]:
        """
        Fulfill an order.
        
        Args:
            order_id: Order ID to fulfill
            tracking_number: Shipment tracking number
            tracking_company: Shipping carrier name
            notify_customer: Send notification email
            
        Returns:
            Fulfillment data
        """
        logger.info(f"Fulfilling order: {order_id}")
        
        fulfillment_data = {
            "notify_customer": notify_customer
        }
        
        if tracking_number:
            fulfillment_data["tracking_number"] = tracking_number
        if tracking_company:
            fulfillment_data["tracking_company"] = tracking_company
        
        result = await self.shopify.create_fulfillment(order_id, fulfillment_data)
        
        logger.info(f"Order fulfilled: {order_id}")
        return result
    
    # ===== Analytics Operations =====
    
    async def get_store_metrics(self, days: int = 30) -> StoreMetrics:
        """
        Get store performance metrics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Store metrics
        """
        orders = await self.get_orders(days=days)
        
        total_revenue = 0
        orders_by_status = {}
        product_sales = {}
        
        for order in orders:
            # Revenue
            total_revenue += float(order.get("total_price", 0))
            
            # Status count
            status = order.get("financial_status", "unknown")
            orders_by_status[status] = orders_by_status.get(status, 0) + 1
            
            # Product sales
            for item in order.get("line_items", []):
                product_id = item.get("product_id")
                if product_id:
                    if product_id not in product_sales:
                        product_sales[product_id] = {
                            "title": item.get("title"),
                            "quantity": 0,
                            "revenue": 0
                        }
                    product_sales[product_id]["quantity"] += item.get("quantity", 0)
                    product_sales[product_id]["revenue"] += float(item.get("price", 0)) * item.get("quantity", 0)
        
        # Top products
        top_products = sorted(
            [{"product_id": k, **v} for k, v in product_sales.items()],
            key=lambda x: x["revenue"],
            reverse=True
        )[:10]
        
        return StoreMetrics(
            total_orders=len(orders),
            total_revenue=total_revenue,
            average_order_value=total_revenue / len(orders) if orders else 0,
            orders_by_status=orders_by_status,
            top_products=top_products,
            period=f"last_{days}_days"
        )
    
    async def get_low_stock_products(self, threshold: int = 10) -> List[Dict[str, Any]]:
        """Get products with low inventory."""
        products = await self.get_products()
        
        low_stock = []
        for product in products:
            for variant in product.get("variants", []):
                inventory = variant.get("inventory_quantity", 0)
                if inventory <= threshold:
                    low_stock.append({
                        "product_id": product.get("id"),
                        "product_title": product.get("title"),
                        "variant_id": variant.get("id"),
                        "variant_title": variant.get("title"),
                        "sku": variant.get("sku"),
                        "inventory": inventory
                    })
        
        return low_stock
    
    async def close(self):
        """Clean up resources."""
        if self.shopify:
            await self.shopify.close()
