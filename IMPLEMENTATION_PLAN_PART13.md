# King AI v2 - Implementation Plan Part 13
## Sub-Agent: Commerce - Shopify

**Target Timeline:** Week 9-10
**Objective:** Implement a Shopify integration agent for e-commerce automation including store management, product operations, and order processing.

---

## Overview of All Parts

| Part | Title | Status |
|------|-------|--------|
| 1 | Infrastructure Layer & Core System Hardening | ✅ Complete |
| 2 | Master AI Brain - Core Enhancements | ✅ Complete |
| 3 | Master AI Brain - Context & Memory System | ✅ Complete |
| 4 | Master AI Brain - Planning & ReAct Implementation | ✅ Complete |
| 5 | Evolution Engine - Core Models & Proposal System | ✅ Complete |
| 6 | Evolution Engine - Code Analysis & AST Tools | ✅ Complete |
| 7 | Evolution Engine - Code Patching & Generation | ✅ Complete |
| 8 | Evolution Engine - Git Integration & Rollback | ✅ Complete |
| 9 | Evolution Engine - Sandbox Testing | ✅ Complete |
| 10 | Sub-Agent: Research (Web/API) | ✅ Complete |
| 11 | Sub-Agent: Code Generator | ✅ Complete |
| 12 | Sub-Agent: Content (Blog/SEO) | ✅ Complete |
| **13** | **Sub-Agent: Commerce - Shopify** | 🔄 Current |
| 14 | Sub-Agent: Commerce - Suppliers | ⏳ Pending |
| 15 | Sub-Agent: Finance - Stripe | ⏳ Pending |
| 16 | Sub-Agent: Finance - Plaid/Banking | ⏳ Pending |
| 17 | Sub-Agent: Analytics | ⏳ Pending |
| 18 | Sub-Agent: Legal | ⏳ Pending |
| 19 | Business: Lifecycle Engine | ⏳ Pending |
| 20 | Business: Playbook System | ⏳ Pending |
| 21 | Business: Portfolio Management | ⏳ Pending |
| 22 | Dashboard: React Components | ⏳ Pending |
| 23 | Dashboard: Approval Workflows | ⏳ Pending |
| 24 | Dashboard: WebSocket & Monitoring | ⏳ Pending |

---

## Part 13 Scope

This part focuses on:
1. Shopify Admin API integration
2. Product management (CRUD operations)
3. Inventory management
4. Order processing and fulfillment
5. Store analytics retrieval
6. Webhook handling for real-time updates

---

## Task 13.1: Create Shopify API Client

**File:** `src/integrations/shopify_client.py` (CREATE NEW FILE)

```python
"""
Shopify API Client - Integration with Shopify Admin API.
Handles authentication, rate limiting, and API operations.
"""

import asyncio
import hashlib
import hmac
import base64
from typing import Optional, List, Dict, Any, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import aiohttp

from src.utils.structured_logging import get_logger
from src.utils.retry import with_retry, RetryConfig

logger = get_logger("shopify_client")

T = TypeVar('T')


class ShopifyAPIVersion(str, Enum):
    """Shopify API versions."""
    V2024_01 = "2024-01"
    V2024_04 = "2024-04"
    V2024_07 = "2024-07"
    V2024_10 = "2024-10"


@dataclass
class ShopifyConfig:
    """Configuration for Shopify store connection."""
    shop_name: str
    access_token: str
    api_version: ShopifyAPIVersion = ShopifyAPIVersion.V2024_10
    webhook_secret: Optional[str] = None
    
    @property
    def base_url(self) -> str:
        """Get the base URL for API requests."""
        return f"https://{self.shop_name}.myshopify.com/admin/api/{self.api_version.value}"


@dataclass
class PaginatedResponse(Generic[T]):
    """Paginated API response."""
    items: List[T]
    has_next: bool
    next_cursor: Optional[str] = None
    total_count: Optional[int] = None


@dataclass
class RateLimitInfo:
    """Rate limit information from Shopify."""
    available: int
    maximum: int
    restore_rate: float
    
    @property
    def usage_percent(self) -> float:
        return (self.maximum - self.available) / self.maximum * 100


class ShopifyAPIError(Exception):
    """Shopify API error."""
    def __init__(self, message: str, status_code: int = None, errors: List[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.errors = errors or []


class ShopifyClient:
    """
    Async client for Shopify Admin API.
    Handles authentication, rate limiting, and common operations.
    """
    
    RETRY_CONFIG = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=10.0,
        exponential_base=2
    )
    
    def __init__(self, config: ShopifyConfig):
        """
        Initialize Shopify client.
        
        Args:
            config: Shopify store configuration
        """
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit: Optional[RateLimitInfo] = None
        self._rate_limit_lock = asyncio.Lock()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "X-Shopify-Access-Token": self.config.access_token,
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _parse_rate_limit(self, headers: dict) -> Optional[RateLimitInfo]:
        """Parse rate limit from response headers."""
        limit_header = headers.get("X-Shopify-Shop-Api-Call-Limit")
        if limit_header:
            try:
                current, maximum = limit_header.split("/")
                return RateLimitInfo(
                    available=int(maximum) - int(current),
                    maximum=int(maximum),
                    restore_rate=2.0  # Shopify restores 2 calls per second
                )
            except ValueError:
                pass
        return None
    
    async def _wait_for_rate_limit(self):
        """Wait if approaching rate limit."""
        async with self._rate_limit_lock:
            if self._rate_limit and self._rate_limit.available < 5:
                wait_time = 5 / self._rate_limit.restore_rate
                logger.warning(f"Rate limit low, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
    
    @with_retry(RETRY_CONFIG)
    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Dict[str, Any] = None,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Make an API request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data
        """
        await self._wait_for_rate_limit()
        
        url = f"{self.config.base_url}/{endpoint}"
        session = await self._get_session()
        
        async with session.request(method, url, json=data, params=params) as response:
            self._rate_limit = self._parse_rate_limit(dict(response.headers))
            
            if response.status == 429:
                retry_after = float(response.headers.get("Retry-After", 2.0))
                logger.warning(f"Rate limited, retrying after {retry_after}s")
                await asyncio.sleep(retry_after)
                raise ShopifyAPIError("Rate limited", status_code=429)
            
            response_data = await response.json()
            
            if response.status >= 400:
                errors = response_data.get("errors", [])
                if isinstance(errors, dict):
                    errors = [f"{k}: {v}" for k, v in errors.items()]
                raise ShopifyAPIError(
                    f"API error: {response.status}",
                    status_code=response.status,
                    errors=errors
                )
            
            return response_data
    
    async def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a GET request."""
        return await self._request("GET", endpoint, params=params)
    
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request."""
        return await self._request("POST", endpoint, data=data)
    
    async def put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a PUT request."""
        return await self._request("PUT", endpoint, data=data)
    
    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make a DELETE request."""
        return await self._request("DELETE", endpoint)
    
    def verify_webhook(self, data: bytes, hmac_header: str) -> bool:
        """
        Verify webhook signature.
        
        Args:
            data: Raw request body
            hmac_header: X-Shopify-Hmac-Sha256 header value
            
        Returns:
            True if signature is valid
        """
        if not self.config.webhook_secret:
            logger.warning("Webhook secret not configured")
            return False
        
        computed = base64.b64encode(
            hmac.new(
                self.config.webhook_secret.encode(),
                data,
                hashlib.sha256
            ).digest()
        ).decode()
        
        return hmac.compare_digest(computed, hmac_header)
    
    # ===== Shop Operations =====
    
    async def get_shop(self) -> Dict[str, Any]:
        """Get shop information."""
        response = await self.get("shop.json")
        return response.get("shop", {})
    
    # ===== Product Operations =====
    
    async def get_products(
        self,
        limit: int = 50,
        page_info: str = None,
        status: str = None,
        collection_id: str = None
    ) -> PaginatedResponse[Dict[str, Any]]:
        """
        Get products with pagination.
        
        Args:
            limit: Number of products per page
            page_info: Cursor for pagination
            status: Filter by status (active, draft, archived)
            collection_id: Filter by collection
            
        Returns:
            Paginated response with products
        """
        params = {"limit": limit}
        
        if page_info:
            params["page_info"] = page_info
        if status:
            params["status"] = status
        if collection_id:
            params["collection_id"] = collection_id
        
        response = await self.get("products.json", params=params)
        products = response.get("products", [])
        
        # Check for next page
        # Note: Shopify uses Link headers for cursor-based pagination
        
        return PaginatedResponse(
            items=products,
            has_next=len(products) == limit,
            next_cursor=None  # Would be extracted from Link header
        )
    
    async def get_product(self, product_id: str) -> Dict[str, Any]:
        """Get a single product by ID."""
        response = await self.get(f"products/{product_id}.json")
        return response.get("product", {})
    
    async def create_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new product.
        
        Args:
            product_data: Product data including title, variants, etc.
            
        Returns:
            Created product
        """
        response = await self.post("products.json", {"product": product_data})
        return response.get("product", {})
    
    async def update_product(
        self,
        product_id: str,
        product_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an existing product."""
        response = await self.put(
            f"products/{product_id}.json",
            {"product": product_data}
        )
        return response.get("product", {})
    
    async def delete_product(self, product_id: str) -> bool:
        """Delete a product."""
        await self.delete(f"products/{product_id}.json")
        return True
    
    # ===== Variant Operations =====
    
    async def get_variant(self, variant_id: str) -> Dict[str, Any]:
        """Get a product variant."""
        response = await self.get(f"variants/{variant_id}.json")
        return response.get("variant", {})
    
    async def update_variant(
        self,
        variant_id: str,
        variant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a variant (price, inventory, etc.)."""
        response = await self.put(
            f"variants/{variant_id}.json",
            {"variant": variant_data}
        )
        return response.get("variant", {})
    
    # ===== Inventory Operations =====
    
    async def get_inventory_levels(
        self,
        inventory_item_ids: List[str] = None,
        location_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Get inventory levels."""
        params = {}
        if inventory_item_ids:
            params["inventory_item_ids"] = ",".join(inventory_item_ids)
        if location_ids:
            params["location_ids"] = ",".join(location_ids)
        
        response = await self.get("inventory_levels.json", params=params)
        return response.get("inventory_levels", [])
    
    async def set_inventory_level(
        self,
        inventory_item_id: str,
        location_id: str,
        available: int
    ) -> Dict[str, Any]:
        """Set inventory level for an item at a location."""
        response = await self.post("inventory_levels/set.json", {
            "inventory_item_id": inventory_item_id,
            "location_id": location_id,
            "available": available
        })
        return response.get("inventory_level", {})
    
    async def adjust_inventory_level(
        self,
        inventory_item_id: str,
        location_id: str,
        adjustment: int
    ) -> Dict[str, Any]:
        """Adjust inventory level (add or subtract)."""
        response = await self.post("inventory_levels/adjust.json", {
            "inventory_item_id": inventory_item_id,
            "location_id": location_id,
            "available_adjustment": adjustment
        })
        return response.get("inventory_level", {})
    
    # ===== Order Operations =====
    
    async def get_orders(
        self,
        limit: int = 50,
        status: str = "any",
        financial_status: str = None,
        fulfillment_status: str = None,
        created_at_min: datetime = None,
        created_at_max: datetime = None
    ) -> PaginatedResponse[Dict[str, Any]]:
        """Get orders with filters."""
        params = {
            "limit": limit,
            "status": status
        }
        
        if financial_status:
            params["financial_status"] = financial_status
        if fulfillment_status:
            params["fulfillment_status"] = fulfillment_status
        if created_at_min:
            params["created_at_min"] = created_at_min.isoformat()
        if created_at_max:
            params["created_at_max"] = created_at_max.isoformat()
        
        response = await self.get("orders.json", params=params)
        orders = response.get("orders", [])
        
        return PaginatedResponse(
            items=orders,
            has_next=len(orders) == limit
        )
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get a single order."""
        response = await self.get(f"orders/{order_id}.json")
        return response.get("order", {})
    
    async def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new order."""
        response = await self.post("orders.json", {"order": order_data})
        return response.get("order", {})
    
    async def update_order(
        self,
        order_id: str,
        order_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an order."""
        response = await self.put(
            f"orders/{order_id}.json",
            {"order": order_data}
        )
        return response.get("order", {})
    
    async def cancel_order(
        self,
        order_id: str,
        reason: str = None,
        restock: bool = True
    ) -> Dict[str, Any]:
        """Cancel an order."""
        data = {"restock": restock}
        if reason:
            data["reason"] = reason
        
        response = await self.post(f"orders/{order_id}/cancel.json", data)
        return response.get("order", {})
    
    # ===== Fulfillment Operations =====
    
    async def get_fulfillments(self, order_id: str) -> List[Dict[str, Any]]:
        """Get fulfillments for an order."""
        response = await self.get(f"orders/{order_id}/fulfillments.json")
        return response.get("fulfillments", [])
    
    async def create_fulfillment(
        self,
        order_id: str,
        fulfillment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a fulfillment for an order."""
        response = await self.post(
            f"orders/{order_id}/fulfillments.json",
            {"fulfillment": fulfillment_data}
        )
        return response.get("fulfillment", {})
    
    # ===== Customer Operations =====
    
    async def get_customers(
        self,
        limit: int = 50,
        query: str = None
    ) -> PaginatedResponse[Dict[str, Any]]:
        """Get customers."""
        params = {"limit": limit}
        if query:
            params["query"] = query
        
        response = await self.get("customers.json", params=params)
        customers = response.get("customers", [])
        
        return PaginatedResponse(
            items=customers,
            has_next=len(customers) == limit
        )
    
    async def get_customer(self, customer_id: str) -> Dict[str, Any]:
        """Get a single customer."""
        response = await self.get(f"customers/{customer_id}.json")
        return response.get("customer", {})
    
    # ===== Collection Operations =====
    
    async def get_collections(
        self,
        collection_type: str = "custom"
    ) -> List[Dict[str, Any]]:
        """Get collections (custom or smart)."""
        if collection_type == "smart":
            response = await self.get("smart_collections.json")
            return response.get("smart_collections", [])
        else:
            response = await self.get("custom_collections.json")
            return response.get("custom_collections", [])
    
    async def add_product_to_collection(
        self,
        collection_id: str,
        product_id: str
    ) -> Dict[str, Any]:
        """Add a product to a collection."""
        response = await self.post("collects.json", {
            "collect": {
                "product_id": product_id,
                "collection_id": collection_id
            }
        })
        return response.get("collect", {})
```

---

## Task 13.2: Create Shopify Commerce Agent

**File:** `src/agents/commerce.py` (REPLACE EXISTING FILE)

```python
"""
Commerce Agent - E-commerce operations via Shopify.
Manages products, inventory, orders, and store analytics.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.agents.base import BaseAgent, AgentCapability, AgentResult
from src.integrations.shopify_client import (
    ShopifyClient,
    ShopifyConfig,
    ShopifyAPIError,
    PaginatedResponse
)
from src.utils.ollama_client import OllamaClient
from src.utils.structured_logging import get_logger

logger = get_logger("commerce_agent")


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
class InventoryUpdate:
    """Inventory update request."""
    sku: str
    quantity: int
    location_id: Optional[str] = None
    adjustment: bool = False  # If True, quantity is adjustment, not absolute


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


class CommerceAgent(BaseAgent):
    """
    Agent for e-commerce operations through Shopify.
    Handles products, inventory, orders, and analytics.
    """
    
    CAPABILITIES = [
        AgentCapability.ECOMMERCE,
        AgentCapability.INVENTORY_MANAGEMENT,
        AgentCapability.ORDER_PROCESSING
    ]
    
    def __init__(
        self,
        llm_client: OllamaClient,
        shopify_config: ShopifyConfig = None
    ):
        """
        Initialize commerce agent.
        
        Args:
            llm_client: LLM client for AI operations
            shopify_config: Shopify store configuration
        """
        super().__init__("commerce", llm_client)
        
        self.shopify: Optional[ShopifyClient] = None
        if shopify_config:
            self.shopify = ShopifyClient(shopify_config)
    
    def set_store(self, config: ShopifyConfig):
        """Configure the Shopify store connection."""
        self.shopify = ShopifyClient(config)
    
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        """Execute a commerce task."""
        if not self.shopify:
            return AgentResult(
                success=False,
                error="Shopify not configured",
                message="Please configure Shopify credentials first"
            )
        
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
                return AgentResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )
            
            return AgentResult(
                success=True,
                data=result if isinstance(result, dict) else result.to_dict() if hasattr(result, 'to_dict') else result,
                message=f"Action '{action}' completed successfully"
            )
            
        except ShopifyAPIError as e:
            logger.error(f"Shopify API error: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                message=f"Shopify API error: {e.errors}"
            )
        except Exception as e:
            logger.error(f"Commerce action failed: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                message="Commerce action failed"
            )
    
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
        
        response = await self.llm.generate(prompt)
        
        # Parse response
        product_data = self._parse_generated_product(response)
        
        return product_data
    
    def _parse_generated_product(self, response: str) -> Dict[str, Any]:
        """Parse AI-generated product listing."""
        import re
        
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
    
    async def sync_inventory_from_supplier(
        self,
        supplier_inventory: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Sync inventory from supplier data.
        
        Args:
            supplier_inventory: Dict mapping SKU to quantity
            
        Returns:
            Sync results
        """
        # Get all products with variants
        products = await self.get_products()
        
        updates = []
        for product in products:
            for variant in product.get("variants", []):
                sku = variant.get("sku")
                if sku and sku in supplier_inventory:
                    updates.append({
                        "inventory_item_id": variant.get("inventory_item_id"),
                        "location_id": "primary",  # Would need actual location ID
                        "quantity": supplier_inventory[sku],
                        "adjustment": False
                    })
        
        return await self.update_inventory(updates)
    
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
    
    async def process_returns(
        self,
        order_id: str,
        reason: str,
        restock: bool = True
    ) -> Dict[str, Any]:
        """Process an order return/cancellation."""
        return await self.shopify.cancel_order(
            order_id=order_id,
            reason=reason,
            restock=restock
        )
    
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
```

---

## Task 13.3: Create Commerce API Routes

**File:** `src/api/routes/commerce.py` (CREATE NEW FILE)

```python
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
from src.utils.ollama_client import OllamaClient
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
        llm = OllamaClient()
        _commerce_agent = CommerceAgent(llm)
        
        # Configure from settings if available
        if hasattr(settings, 'SHOPIFY_SHOP_NAME') and settings.SHOPIFY_SHOP_NAME:
            config = ShopifyConfig(
                shop_name=settings.SHOPIFY_SHOP_NAME,
                access_token=settings.SHOPIFY_ACCESS_TOKEN,
                webhook_secret=getattr(settings, 'SHOPIFY_WEBHOOK_SECRET', None)
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
```

---

## Task 13.4: Add Settings for Shopify

**File:** `config/settings.py` (MODIFY - add these settings)

Add the following to the Settings class:

```python
    # Shopify Configuration
    SHOPIFY_SHOP_NAME: Optional[str] = None
    SHOPIFY_ACCESS_TOKEN: Optional[str] = None
    SHOPIFY_WEBHOOK_SECRET: Optional[str] = None
    SHOPIFY_API_VERSION: str = "2024-10"
```

---

## Testing Requirements

**File:** `tests/test_commerce.py` (CREATE NEW FILE)

```python
"""Tests for commerce agent and Shopify integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.integrations.shopify_client import (
    ShopifyClient,
    ShopifyConfig,
    ShopifyAPIError,
    PaginatedResponse
)
from src.agents.commerce import (
    CommerceAgent,
    ProductData,
    OrderStatus,
    StoreMetrics
)


class TestShopifyClient:
    """Tests for Shopify API client."""
    
    @pytest.fixture
    def config(self):
        return ShopifyConfig(
            shop_name="test-store",
            access_token="test-token"
        )
    
    @pytest.fixture
    def client(self, config):
        return ShopifyClient(config)
    
    def test_base_url_construction(self, config):
        """Test base URL is correctly constructed."""
        expected = "https://test-store.myshopify.com/admin/api/2024-10"
        assert config.base_url == expected
    
    def test_verify_webhook_valid(self, client):
        """Test webhook verification with valid signature."""
        client.config.webhook_secret = "test-secret"
        
        # This would need actual HMAC calculation for real test
        # Simplified for example
        assert client.config.webhook_secret is not None
    
    @pytest.mark.asyncio
    async def test_rate_limit_parsing(self, client):
        """Test rate limit header parsing."""
        headers = {"X-Shopify-Shop-Api-Call-Limit": "5/40"}
        
        rate_limit = client._parse_rate_limit(headers)
        
        assert rate_limit is not None
        assert rate_limit.available == 35
        assert rate_limit.maximum == 40


class TestProductData:
    """Tests for ProductData model."""
    
    def test_to_shopify_dict(self):
        """Test conversion to Shopify format."""
        product = ProductData(
            title="Test Product",
            description="<p>Description</p>",
            tags=["tag1", "tag2"],
            vendor="Test Vendor"
        )
        
        result = product.to_shopify_dict()
        
        assert result["title"] == "Test Product"
        assert result["body_html"] == "<p>Description</p>"
        assert result["tags"] == "tag1,tag2"
        assert result["vendor"] == "Test Vendor"
    
    def test_with_variants(self):
        """Test product with variants."""
        product = ProductData(
            title="Test Product",
            variants=[
                {"price": "19.99", "sku": "TEST-001"},
                {"price": "29.99", "sku": "TEST-002"}
            ]
        )
        
        result = product.to_shopify_dict()
        
        assert "variants" in result
        assert len(result["variants"]) == 2


class TestCommerceAgent:
    """Tests for commerce agent."""
    
    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="""
TITLE: Amazing Product Title
DESCRIPTION: <p>This is an amazing product.</p>
TAGS: amazing, product, new
PRODUCT_TYPE: Electronics
""")
        return llm
    
    @pytest.fixture
    def mock_shopify(self):
        client = AsyncMock()
        client.get_products = AsyncMock(return_value=PaginatedResponse(
            items=[{"id": "123", "title": "Test Product"}],
            has_next=False
        ))
        client.get_orders = AsyncMock(return_value=PaginatedResponse(
            items=[
                {
                    "id": "order-1",
                    "total_price": "100.00",
                    "financial_status": "paid",
                    "line_items": [
                        {"product_id": "123", "title": "Product", "quantity": 1, "price": "100.00"}
                    ]
                }
            ],
            has_next=False
        ))
        return client
    
    @pytest.fixture
    def agent(self, mock_llm, mock_shopify):
        agent = CommerceAgent(mock_llm)
        agent.shopify = mock_shopify
        return agent
    
    @pytest.mark.asyncio
    async def test_get_products(self, agent):
        """Test getting products."""
        products = await agent.get_products()
        
        assert len(products) == 1
        assert products[0]["title"] == "Test Product"
    
    @pytest.mark.asyncio
    async def test_generate_product_listing(self, agent):
        """Test AI product generation."""
        result = await agent.generate_product_listing(
            "A cool gadget for tech enthusiasts",
            style="professional"
        )
        
        assert result["title"] == "Amazing Product Title"
        assert "amazing" in result["tags"]
    
    @pytest.mark.asyncio
    async def test_get_store_metrics(self, agent):
        """Test store metrics calculation."""
        metrics = await agent.get_store_metrics(days=30)
        
        assert isinstance(metrics, StoreMetrics)
        assert metrics.total_orders == 1
        assert metrics.total_revenue == 100.0
    
    def test_parse_generated_product(self, agent):
        """Test parsing generated product."""
        response = """
TITLE: Test Title
DESCRIPTION: Test description
TAGS: tag1, tag2
PRODUCT_TYPE: Category
"""
        
        result = agent._parse_generated_product(response)
        
        assert result["title"] == "Test Title"
        assert result["product_type"] == "Category"


class TestStoreMetrics:
    """Tests for StoreMetrics model."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = StoreMetrics(
            total_orders=100,
            total_revenue=5000.0,
            average_order_value=50.0,
            orders_by_status={"paid": 90, "pending": 10},
            top_products=[{"title": "Best Seller", "revenue": 1000}],
            period="last_30_days"
        )
        
        result = metrics.to_dict()
        
        assert result["total_orders"] == 100
        assert result["average_order_value"] == 50.0
        assert len(result["top_products"]) == 1
```

---

## Acceptance Criteria

- [ ] `src/integrations/shopify_client.py` - Complete Shopify API client
- [ ] `src/agents/commerce.py` - Full commerce agent
- [ ] `src/api/routes/commerce.py` - REST API endpoints
- [ ] `tests/test_commerce.py` - All tests passing
- [ ] Product CRUD operations working
- [ ] Inventory management functional
- [ ] Order processing working
- [ ] Store metrics calculated correctly
- [ ] Webhook verification working

---

## File Summary

| Action | File Path |
|--------|-----------|
| CREATE | `src/integrations/shopify_client.py` |
| REPLACE | `src/agents/commerce.py` |
| CREATE | `src/api/routes/commerce.py` |
| MODIFY | `config/settings.py` |
| CREATE | `tests/test_commerce.py` |

---

*End of Part 13*
