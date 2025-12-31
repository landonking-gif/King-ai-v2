"""
Supplier Integration Client - Multi-provider dropshipping/wholesale API integration.
"""
import asyncio
import hashlib
import hmac
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import httpx
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SupplierType(Enum):
    """Supported supplier types."""
    ALIEXPRESS = "aliexpress"
    CJ_DROPSHIPPING = "cj_dropshipping"
    SPOCKET = "spocket"
    WHOLESALE = "wholesale_generic"


@dataclass
class SupplierProduct:
    """Standardized supplier product data."""
    supplier_id: str
    supplier_type: SupplierType
    product_id: str
    title: str
    description: str
    price: float
    currency: str = "USD"
    shipping_cost: float = 0.0
    shipping_days: tuple[int, int] = (7, 21)
    stock_quantity: int = 0
    images: list[str] = field(default_factory=list)
    variants: list[dict] = field(default_factory=list)
    category: str = ""
    supplier_url: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_cost(self) -> float:
        return self.price + self.shipping_cost

    def calculate_margin(self, sell_price: float) -> float:
        if sell_price <= 0:
            return 0.0
        return ((sell_price - self.total_cost) / sell_price) * 100


@dataclass
class SupplierOrder:
    """Supplier order for fulfillment."""
    order_id: str
    supplier_type: SupplierType
    supplier_order_id: Optional[str] = None
    status: str = "pending"
    tracking_number: Optional[str] = None
    tracking_url: Optional[str] = None
    shipped_at: Optional[datetime] = None
    items: list[dict] = field(default_factory=list)
    shipping_address: dict = field(default_factory=dict)
    total_cost: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


class BaseSupplierClient(ABC):
    """Abstract base class for supplier integrations."""

    def __init__(self, api_key: str, api_secret: str = "", **kwargs):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = httpx.AsyncClient(timeout=30.0)
        self._rate_limit_delay = 1.0

    async def close(self):
        await self.client.aclose()

    @abstractmethod
    async def search_products(
        self, query: str, category: str = "", page: int = 1, limit: int = 20
    ) -> list[SupplierProduct]:
        """Search for products."""
        pass

    @abstractmethod
    async def get_product(self, product_id: str) -> Optional[SupplierProduct]:
        """Get product details."""
        pass

    @abstractmethod
    async def get_stock(self, product_id: str) -> int:
        """Get current stock level."""
        pass

    @abstractmethod
    async def place_order(
        self, items: list[dict], shipping_address: dict
    ) -> SupplierOrder:
        """Place order with supplier."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> SupplierOrder:
        """Get order status and tracking."""
        pass

    async def _rate_limit(self):
        await asyncio.sleep(self._rate_limit_delay)


class AliExpressClient(BaseSupplierClient):
    """AliExpress Affiliate/Dropshipping API client."""

    BASE_URL = "https://api.aliexpress.com/v2"
    DEFAULT_STOCK_QUANTITY = 999  # Default stock for AliExpress products (API doesn't provide actual stock)

    def __init__(self, app_key: str, app_secret: str, tracking_id: str = ""):
        super().__init__(app_key, app_secret)
        self.tracking_id = tracking_id
        self._rate_limit_delay = 0.5

    def _sign_request(self, params: dict) -> str:
        """
        Sign request using MD5 as required by AliExpress API.
        Note: MD5 is used here because it's mandated by the AliExpress API specification.
        """
        sorted_params = sorted(params.items())
        sign_str = self.api_secret + "".join(f"{k}{v}" for k, v in sorted_params) + self.api_secret
        return hashlib.md5(sign_str.encode()).hexdigest().upper()

    async def _request(self, method: str, params: dict) -> dict:
        await self._rate_limit()
        params.update({
            "app_key": self.api_key,
            "timestamp": str(int(time.time() * 1000)),
            "sign_method": "md5",
            "method": method,
        })
        params["sign"] = self._sign_request(params)
        
        try:
            resp = await self.client.post(self.BASE_URL, data=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"AliExpress API error: {e}")
            raise

    async def search_products(
        self, query: str, category: str = "", page: int = 1, limit: int = 20
    ) -> list[SupplierProduct]:
        params = {
            "keywords": query,
            "page_no": str(page),
            "page_size": str(min(limit, 50)),
            "sort": "SALE_PRICE_ASC",
        }
        if category:
            params["category_ids"] = category

        data = await self._request("aliexpress.affiliate.product.query", params)
        products = []
        
        for item in data.get("resp_result", {}).get("result", {}).get("products", []):
            products.append(SupplierProduct(
                supplier_id="aliexpress",
                supplier_type=SupplierType.ALIEXPRESS,
                product_id=str(item.get("product_id", "")),
                title=item.get("product_title", ""),
                description=item.get("product_description", ""),
                price=float(item.get("target_sale_price", 0)),
                currency=item.get("target_sale_price_currency", "USD"),
                shipping_cost=0.0,
                stock_quantity=self.DEFAULT_STOCK_QUANTITY,
                images=[item.get("product_main_image_url", "")],
                category=str(item.get("first_level_category_id", "")),
                supplier_url=item.get("product_detail_url", ""),
            ))
        return products

    async def get_product(self, product_id: str) -> Optional[SupplierProduct]:
        params = {"product_ids": product_id}
        data = await self._request("aliexpress.affiliate.product.query", params)
        items = data.get("resp_result", {}).get("result", {}).get("products", [])
        
        if not items:
            return None
        item = items[0]
        return SupplierProduct(
            supplier_id="aliexpress",
            supplier_type=SupplierType.ALIEXPRESS,
            product_id=str(item.get("product_id", "")),
            title=item.get("product_title", ""),
            description=item.get("product_description", ""),
            price=float(item.get("target_sale_price", 0)),
            currency=item.get("target_sale_price_currency", "USD"),
            images=[item.get("product_main_image_url", "")],
            supplier_url=item.get("product_detail_url", ""),
        )

    async def get_stock(self, product_id: str) -> int:
        product = await self.get_product(product_id)
        return product.stock_quantity if product else 0

    async def place_order(
        self, items: list[dict], shipping_address: dict
    ) -> SupplierOrder:
        # AliExpress DS API order placement
        params = {
            "product_items": str(items),
            "logistics_address": str(shipping_address),
        }
        data = await self._request("aliexpress.ds.order.create", params)
        order_data = data.get("result", {})
        
        return SupplierOrder(
            order_id=str(order_data.get("order_id", "")),
            supplier_type=SupplierType.ALIEXPRESS,
            supplier_order_id=str(order_data.get("order_id", "")),
            status="placed",
            items=items,
            shipping_address=shipping_address,
        )

    async def get_order_status(self, order_id: str) -> SupplierOrder:
        params = {"order_id": order_id}
        data = await self._request("aliexpress.ds.order.get", params)
        order = data.get("result", {})
        
        return SupplierOrder(
            order_id=order_id,
            supplier_type=SupplierType.ALIEXPRESS,
            supplier_order_id=order.get("order_id"),
            status=order.get("order_status", "unknown"),
            tracking_number=order.get("logistics_tracking_number"),
            items=[],
        )


class CJDropshippingClient(BaseSupplierClient):
    """CJ Dropshipping API client."""

    BASE_URL = "https://developers.cjdropshipping.com/api2.0"

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._access_token: Optional[str] = None
        self._token_expires: float = 0

    async def _get_token(self) -> str:
        if self._access_token and time.time() < self._token_expires:
            return self._access_token
        
        resp = await self.client.post(
            f"{self.BASE_URL}/v1/authentication/getAccessToken",
            json={"email": "", "password": ""},
            headers={"CJ-Access-Token": self.api_key}
        )
        data = resp.json()
        self._access_token = data.get("data", {}).get("accessToken", "")
        self._token_expires = time.time() + 3600
        return self._access_token

    async def _request(self, endpoint: str, method: str = "GET", data: dict = None) -> dict:
        await self._rate_limit()
        token = await self._get_token()
        headers = {"CJ-Access-Token": token}
        
        try:
            if method == "GET":
                resp = await self.client.get(f"{self.BASE_URL}{endpoint}", headers=headers, params=data)
            else:
                resp = await self.client.post(f"{self.BASE_URL}{endpoint}", headers=headers, json=data)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"CJ Dropshipping API error: {e}")
            raise

    async def search_products(
        self, query: str, category: str = "", page: int = 1, limit: int = 20
    ) -> list[SupplierProduct]:
        data = await self._request("/v1/product/list", "GET", {
            "productNameEn": query,
            "pageNum": page,
            "pageSize": limit,
        })
        
        products = []
        for item in data.get("data", {}).get("list", []):
            products.append(SupplierProduct(
                supplier_id="cj_dropshipping",
                supplier_type=SupplierType.CJ_DROPSHIPPING,
                product_id=item.get("pid", ""),
                title=item.get("productNameEn", ""),
                description=item.get("description", ""),
                price=float(item.get("sellPrice", 0)),
                stock_quantity=int(item.get("productStock", 0)),
                images=item.get("productImage", "").split(";"),
                category=item.get("categoryName", ""),
            ))
        return products

    async def get_product(self, product_id: str) -> Optional[SupplierProduct]:
        data = await self._request(f"/v1/product/query", "GET", {"pid": product_id})
        item = data.get("data")
        if not item:
            return None
        
        return SupplierProduct(
            supplier_id="cj_dropshipping",
            supplier_type=SupplierType.CJ_DROPSHIPPING,
            product_id=item.get("pid", ""),
            title=item.get("productNameEn", ""),
            description=item.get("description", ""),
            price=float(item.get("sellPrice", 0)),
            stock_quantity=int(item.get("productStock", 0)),
            images=item.get("productImage", "").split(";"),
        )

    async def get_stock(self, product_id: str) -> int:
        product = await self.get_product(product_id)
        return product.stock_quantity if product else 0

    async def place_order(
        self, items: list[dict], shipping_address: dict
    ) -> SupplierOrder:
        order_data = {
            "orderNumber": f"ORD-{int(time.time())}",
            "shippingCountryCode": shipping_address.get("country_code", "US"),
            "shippingCountry": shipping_address.get("country", "United States"),
            "shippingProvince": shipping_address.get("province", ""),
            "shippingCity": shipping_address.get("city", ""),
            "shippingAddress": shipping_address.get("address1", ""),
            "shippingCustomerName": shipping_address.get("name", ""),
            "shippingPhone": shipping_address.get("phone", ""),
            "shippingZip": shipping_address.get("zip", ""),
            "products": [{"vid": i["variant_id"], "quantity": i["quantity"]} for i in items],
        }
        
        data = await self._request("/v1/shopping/order/createOrder", "POST", order_data)
        result = data.get("data", {})
        
        return SupplierOrder(
            order_id=result.get("orderId", ""),
            supplier_type=SupplierType.CJ_DROPSHIPPING,
            supplier_order_id=result.get("orderId"),
            status="placed",
            items=items,
            shipping_address=shipping_address,
        )

    async def get_order_status(self, order_id: str) -> SupplierOrder:
        data = await self._request("/v1/shopping/order/getOrderDetail", "GET", {"orderId": order_id})
        order = data.get("data", {})
        
        return SupplierOrder(
            order_id=order_id,
            supplier_type=SupplierType.CJ_DROPSHIPPING,
            supplier_order_id=order.get("orderId"),
            status=order.get("orderStatus", "unknown"),
            tracking_number=order.get("trackNumber"),
            tracking_url=order.get("trackUrl"),
        )


class SpocketClient(BaseSupplierClient):
    """
    Spocket API client for US/EU dropshipping.
    
    Spocket specializes in US and EU suppliers with faster shipping times
    compared to Asian suppliers. Premium products with 30-60% margins.
    """

    BASE_URL = "https://app.spocket.co/api/v1"

    def __init__(self, api_key: str, store_id: str = "", **kwargs):
        super().__init__(api_key, "")
        self.store_id = store_id
        self._rate_limit_delay = 0.5
        
        # Update headers for Spocket
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    async def _request(
        self, endpoint: str, method: str = "GET", params: dict = None, json_data: dict = None
    ) -> dict:
        """Make authenticated request to Spocket API."""
        await self._rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            if method == "GET":
                resp = await self.client.get(url, params=params)
            elif method == "POST":
                resp = await self.client.post(url, json=json_data)
            elif method == "PUT":
                resp = await self.client.put(url, json=json_data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Spocket API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Spocket request failed: {e}")
            raise

    async def search_products(
        self, query: str, category: str = "", page: int = 1, limit: int = 20
    ) -> list[SupplierProduct]:
        """
        Search for products in Spocket catalog.
        
        Spocket offers:
        - US/EU suppliers with 2-5 day shipping
        - Pre-vetted suppliers
        - Branded invoicing
        """
        params = {
            "search": query,
            "page": page,
            "per_page": min(limit, 50),
        }
        
        if category:
            params["category"] = category
        
        # Add filters for quality suppliers
        params["supplier_location"] = "us,eu,ca,uk"  # Fast shipping regions
        params["min_reviews"] = 4  # Minimum rating
        
        data = await self._request("/products", "GET", params)
        products = []
        
        for item in data.get("products", []):
            # Calculate shipping based on supplier location
            supplier_location = item.get("supplier_location", "").lower()
            if supplier_location in ["us", "ca"]:
                shipping_days = (2, 5)
                shipping_cost = 4.99
            elif supplier_location in ["eu", "uk"]:
                shipping_days = (3, 7)
                shipping_cost = 5.99
            else:
                shipping_days = (7, 14)
                shipping_cost = 3.99
            
            products.append(SupplierProduct(
                supplier_id="spocket",
                supplier_type=SupplierType.SPOCKET,
                product_id=str(item.get("id", "")),
                title=item.get("title", ""),
                description=item.get("description", ""),
                price=float(item.get("retail_price", 0)),
                currency="USD",
                shipping_cost=shipping_cost,
                shipping_days=shipping_days,
                stock_quantity=item.get("inventory_quantity", 100),
                images=[img.get("src", "") for img in item.get("images", [])],
                variants=item.get("variants", []),
                category=item.get("category", ""),
                supplier_url=item.get("spocket_url", ""),
            ))
        
        return products

    async def get_product(self, product_id: str) -> Optional[SupplierProduct]:
        """Get detailed product information."""
        try:
            data = await self._request(f"/products/{product_id}", "GET")
            product = data.get("product", {})
            
            if not product:
                return None
            
            supplier_location = product.get("supplier_location", "").lower()
            if supplier_location in ["us", "ca"]:
                shipping_days = (2, 5)
                shipping_cost = 4.99
            elif supplier_location in ["eu", "uk"]:
                shipping_days = (3, 7)
                shipping_cost = 5.99
            else:
                shipping_days = (7, 14)
                shipping_cost = 3.99
            
            return SupplierProduct(
                supplier_id="spocket",
                supplier_type=SupplierType.SPOCKET,
                product_id=str(product.get("id", "")),
                title=product.get("title", ""),
                description=product.get("description", ""),
                price=float(product.get("retail_price", 0)),
                currency="USD",
                shipping_cost=shipping_cost,
                shipping_days=shipping_days,
                stock_quantity=product.get("inventory_quantity", 100),
                images=[img.get("src", "") for img in product.get("images", [])],
                variants=product.get("variants", []),
                category=product.get("category", ""),
                supplier_url=product.get("spocket_url", ""),
            )
        except Exception as e:
            logger.error(f"Failed to get Spocket product {product_id}: {e}")
            return None

    async def get_stock(self, product_id: str) -> int:
        """Get current stock level for a product."""
        product = await self.get_product(product_id)
        return product.stock_quantity if product else 0

    async def place_order(
        self, items: list[dict], shipping_address: dict
    ) -> SupplierOrder:
        """
        Place an order with Spocket for fulfillment.
        
        Args:
            items: List of items with product_id, variant_id, quantity
            shipping_address: Shipping destination details
        """
        order_data = {
            "order": {
                "line_items": [
                    {
                        "product_id": item["product_id"],
                        "variant_id": item.get("variant_id"),
                        "quantity": item.get("quantity", 1),
                    }
                    for item in items
                ],
                "shipping_address": {
                    "first_name": shipping_address.get("first_name", ""),
                    "last_name": shipping_address.get("last_name", ""),
                    "address1": shipping_address.get("address1", ""),
                    "address2": shipping_address.get("address2", ""),
                    "city": shipping_address.get("city", ""),
                    "province": shipping_address.get("province", ""),
                    "country": shipping_address.get("country", "US"),
                    "zip": shipping_address.get("zip", ""),
                    "phone": shipping_address.get("phone", ""),
                },
                "branded_invoice": True,  # Spocket feature
            }
        }
        
        if self.store_id:
            order_data["order"]["store_id"] = self.store_id
        
        data = await self._request("/orders", "POST", json_data=order_data)
        result = data.get("order", {})
        
        return SupplierOrder(
            order_id=str(result.get("id", "")),
            supplier_type=SupplierType.SPOCKET,
            supplier_order_id=result.get("spocket_order_id"),
            status=result.get("status", "processing"),
            items=items,
            shipping_address=shipping_address,
            total_cost=float(result.get("total", 0)),
        )

    async def get_order_status(self, order_id: str) -> SupplierOrder:
        """Get order status and tracking information."""
        data = await self._request(f"/orders/{order_id}", "GET")
        order = data.get("order", {})
        
        return SupplierOrder(
            order_id=order_id,
            supplier_type=SupplierType.SPOCKET,
            supplier_order_id=order.get("spocket_order_id"),
            status=order.get("status", "unknown"),
            tracking_number=order.get("tracking_number"),
            tracking_url=order.get("tracking_url"),
            shipped_at=datetime.fromisoformat(order["shipped_at"]) if order.get("shipped_at") else None,
        )

    async def import_to_store(self, product_id: str, store_id: str = None) -> dict:
        """
        Import a product to connected store (Shopify, WooCommerce, etc.).
        
        Spocket handles the product sync automatically.
        """
        data = {
            "product_id": product_id,
            "store_id": store_id or self.store_id,
        }
        
        result = await self._request("/products/import", "POST", json_data=data)
        return result

    async def get_shipping_rates(
        self, product_id: str, destination_country: str = "US"
    ) -> list[dict]:
        """Get available shipping rates for a product."""
        params = {
            "product_id": product_id,
            "destination": destination_country,
        }
        
        data = await self._request("/shipping/rates", "GET", params)
        return data.get("rates", [])


class SupplierClientFactory:
    """Factory for creating supplier clients."""

    _clients: dict[SupplierType, type[BaseSupplierClient]] = {
        SupplierType.ALIEXPRESS: AliExpressClient,
        SupplierType.CJ_DROPSHIPPING: CJDropshippingClient,
        SupplierType.SPOCKET: SpocketClient,
    }

    @classmethod
    def create(cls, supplier_type: SupplierType, **credentials) -> BaseSupplierClient:
        client_class = cls._clients.get(supplier_type)
        if not client_class:
            raise ValueError(f"Unsupported supplier: {supplier_type}")
        return client_class(**credentials)

    @classmethod
    def register(cls, supplier_type: SupplierType, client_class: type[BaseSupplierClient]):
        cls._clients[supplier_type] = client_class
