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
