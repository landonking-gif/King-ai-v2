"""
Shopify API Client - Integration with Shopify Admin API.
Simplified implementation for supplier integration dependency.
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import httpx

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ShopifyAPIError(Exception):
    """Exception raised for Shopify API errors."""
    
    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
    
    def __str__(self):
        if self.status_code:
            return f"ShopifyAPIError ({self.status_code}): {super().__str__()}"
        return f"ShopifyAPIError: {super().__str__()}"


@dataclass
class PaginatedResponse:
    """Response wrapper for paginated Shopify API calls."""
    items: List[Dict[str, Any]]
    has_next_page: bool = False
    next_page_info: Optional[str] = None
    total_count: Optional[int] = None


@dataclass
class ShopifyConfig:
    """Configuration for Shopify store connection."""
    shop_url: str  # e.g., "mystore.myshopify.com"
    shop_name: str = ""  # Alias for compatibility
    access_token: str = ""
    api_version: str = "2024-10"
    
    def __post_init__(self):
        # Allow shop_name as alias for shop_url
        if not self.shop_url and self.shop_name:
            self.shop_url = f"{self.shop_name}.myshopify.com"


class ShopifyClient:
    """
    Async client for Shopify Admin API.
    Handles products, inventory, and orders.
    """

    def __init__(self, shop_url: str, access_token: str, api_version: str = "2024-10"):
        """
        Initialize Shopify client.
        
        Args:
            shop_url: Store URL (e.g., "mystore.myshopify.com")
            access_token: Admin API access token
            api_version: API version (default: 2024-10)
        """
        self.shop_url = shop_url.replace("https://", "").replace("http://", "")
        self.access_token = access_token
        self.api_version = api_version
        self.base_url = f"https://{self.shop_url}/admin/api/{api_version}"
        self.client = httpx.AsyncClient(timeout=30.0)
        self._rate_limit_delay = 0.5

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make authenticated request to Shopify API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data as dictionary
        """
        await asyncio.sleep(self._rate_limit_delay)
        
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json",
        }

        try:
            if method == "GET":
                resp = await self.client.get(url, headers=headers, params=params)
            elif method == "POST":
                resp = await self.client.post(url, headers=headers, json=data)
            elif method == "PUT":
                resp = await self.client.put(url, headers=headers, json=data)
            elif method == "DELETE":
                resp = await self.client.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            resp.raise_for_status()
            
            # Handle empty responses
            if resp.status_code == 204 or not resp.content:
                return {}
                
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Shopify API error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Shopify request error: {e}")
            raise

    async def get_products(self, limit: int = 50, page_info: Optional[str] = None) -> List[Dict]:
        """
        Get products from store.
        
        Args:
            limit: Number of products to retrieve (max 250)
            page_info: Pagination cursor for next page
            
        Returns:
            List of product dictionaries
        """
        params = {"limit": min(limit, 250)}
        if page_info:
            params["page_info"] = page_info

        data = await self._request("GET", "products.json", params=params)
        return data.get("products", [])

    async def get_product(self, product_id: str) -> Optional[Dict]:
        """
        Get a single product by ID.
        
        Args:
            product_id: Shopify product ID
            
        Returns:
            Product dictionary or None if not found
        """
        try:
            data = await self._request("GET", f"products/{product_id}.json")
            return data.get("product")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def create_product(self, product_data: Dict) -> Dict:
        """
        Create a new product.
        
        Args:
            product_data: Product data including title, variants, images, etc.
            
        Returns:
            Created product data
        """
        data = await self._request("POST", "products.json", data={"product": product_data})
        return data.get("product", {})

    async def update_product(self, product_id: str, product_data: Dict) -> Dict:
        """
        Update an existing product.
        
        Args:
            product_id: Shopify product ID
            product_data: Updated product data
            
        Returns:
            Updated product data
        """
        data = await self._request(
            "PUT", 
            f"products/{product_id}.json",
            data={"product": product_data}
        )
        return data.get("product", {})

    async def delete_product(self, product_id: str) -> bool:
        """
        Delete a product.
        
        Args:
            product_id: Shopify product ID
            
        Returns:
            True if deleted successfully
        """
        try:
            await self._request("DELETE", f"products/{product_id}.json")
            return True
        except Exception as e:
            logger.error(f"Failed to delete product {product_id}: {e}")
            return False

    async def get_order(self, order_id: str) -> Optional[Dict]:
        """
        Get order details.
        
        Args:
            order_id: Shopify order ID
            
        Returns:
            Order dictionary or None if not found
        """
        try:
            data = await self._request("GET", f"orders/{order_id}.json")
            return data.get("order")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_orders(
        self, 
        status: str = "any", 
        limit: int = 50,
        financial_status: Optional[str] = None,
        fulfillment_status: Optional[str] = None
    ) -> List[Dict]:
        """
        Get orders from store.
        
        Args:
            status: Order status filter (any, open, closed, cancelled)
            limit: Number of orders to retrieve
            financial_status: Filter by payment status
            fulfillment_status: Filter by fulfillment status
            
        Returns:
            List of order dictionaries
        """
        params = {
            "status": status,
            "limit": min(limit, 250),
        }
        if financial_status:
            params["financial_status"] = financial_status
        if fulfillment_status:
            params["fulfillment_status"] = fulfillment_status

        data = await self._request("GET", "orders.json", params=params)
        return data.get("orders", [])

    async def update_inventory(
        self, 
        inventory_item_id: str, 
        available: int,
        location_id: Optional[str] = None
    ) -> Dict:
        """
        Update inventory level for a product variant.
        
        Args:
            inventory_item_id: Inventory item ID
            available: Available quantity
            location_id: Location ID (uses first location if not provided)
            
        Returns:
            Updated inventory level data
        """
        # Get location if not provided
        if not location_id:
            locations = await self._request("GET", "locations.json")
            if locations.get("locations"):
                location_id = str(locations["locations"][0]["id"])
            else:
                raise ValueError("No locations found for store")

        data = await self._request(
            "POST",
            f"inventory_levels/set.json",
            data={
                "location_id": location_id,
                "inventory_item_id": inventory_item_id,
                "available": available
            }
        )
        return data.get("inventory_level", {})

    async def create_fulfillment(
        self,
        order_id: str,
        line_items: List[Dict],
        tracking_number: Optional[str] = None,
        tracking_url: Optional[str] = None,
        tracking_company: Optional[str] = None
    ) -> Dict:
        """
        Create a fulfillment for an order.
        
        Args:
            order_id: Shopify order ID
            line_items: List of line items to fulfill
            tracking_number: Shipment tracking number
            tracking_url: Tracking URL
            tracking_company: Carrier name
            
        Returns:
            Fulfillment data
        """
        fulfillment_data = {
            "line_items": line_items,
            "notify_customer": True,
        }
        
        if tracking_number:
            fulfillment_data["tracking_number"] = tracking_number
        if tracking_url:
            fulfillment_data["tracking_url"] = tracking_url
        if tracking_company:
            fulfillment_data["tracking_company"] = tracking_company

        data = await self._request(
            "POST",
            f"orders/{order_id}/fulfillments.json",
            data={"fulfillment": fulfillment_data}
        )
        return data.get("fulfillment", {})
