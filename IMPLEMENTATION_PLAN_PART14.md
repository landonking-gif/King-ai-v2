# Implementation Plan Part 14: Commerce - Supplier Integration Sub-Agent

| Field | Value |
|-------|-------|
| Module | Supplier Integration for Dropshipping & Wholesale |
| Priority | High |
| Estimated Effort | 5-6 hours |
| Dependencies | Part 13 (Shopify Commerce), Part 3 (Database) |

---

## 1. Scope

This module implements supplier integration for dropshipping and wholesale operations:

- **Supplier Client** - Multi-provider API integration (AliExpress, CJ Dropshipping, wholesale APIs)
- **Product Sourcing** - Find products from suppliers, compare prices, margins
- **Order Fulfillment** - Auto-forward orders to suppliers for fulfillment
- **Inventory Sync** - Sync supplier stock levels with store inventory
- **Price Monitoring** - Track supplier price changes, update store prices

---

## 2. Tasks

### Task 14.1: Supplier Client Base

**File: `src/integrations/supplier_client.py`**

```python
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

    def __init__(self, app_key: str, app_secret: str, tracking_id: str = ""):
        super().__init__(app_key, app_secret)
        self.tracking_id = tracking_id
        self._rate_limit_delay = 0.5

    def _sign_request(self, params: dict) -> str:
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
                stock_quantity=999,
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


class SupplierClientFactory:
    """Factory for creating supplier clients."""

    _clients: dict[SupplierType, type[BaseSupplierClient]] = {
        SupplierType.ALIEXPRESS: AliExpressClient,
        SupplierType.CJ_DROPSHIPPING: CJDropshippingClient,
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
```

---

### Task 14.2: Supplier Agent

**File: `src/agents/supplier.py`**

```python
"""
Supplier Agent - Product sourcing, order fulfillment, inventory sync.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from src.agents.base import BaseAgent, AgentCapability, AgentResult
from src.integrations.supplier_client import (
    BaseSupplierClient, SupplierClientFactory, SupplierType,
    SupplierProduct, SupplierOrder
)
from src.integrations.shopify_client import ShopifyClient
from src.database.connection import get_db_session
from src.database.models import BusinessUnit
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProductMatch:
    """Match between store product and supplier product."""
    store_product_id: str
    supplier_product: SupplierProduct
    current_price: float
    recommended_price: float
    margin_percent: float
    price_change: float = 0.0


@dataclass
class SourcedProduct:
    """Product ready for import to store."""
    supplier_product: SupplierProduct
    recommended_price: float
    margin_percent: float
    tags: list[str]


class SupplierAgent(BaseAgent):
    """Agent for supplier integration and dropshipping operations."""

    def __init__(self):
        super().__init__(
            name="Supplier Agent",
            capabilities=[
                AgentCapability.COMMERCE,
                AgentCapability.ANALYSIS,
            ]
        )
        self.suppliers: dict[str, BaseSupplierClient] = {}
        self.default_margin = 2.5  # 250% markup
        self.min_margin_percent = 30.0

    async def initialize_supplier(
        self, supplier_type: SupplierType, credentials: dict
    ) -> bool:
        """Initialize a supplier client."""
        try:
            client = SupplierClientFactory.create(supplier_type, **credentials)
            self.suppliers[supplier_type.value] = client
            logger.info(f"Initialized supplier: {supplier_type.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize supplier {supplier_type}: {e}")
            return False

    async def search_products(
        self,
        query: str,
        supplier_type: Optional[SupplierType] = None,
        min_margin: float = 30.0,
        max_price: float = 100.0,
        limit: int = 20,
    ) -> AgentResult:
        """Search for products across suppliers."""
        all_products: list[SourcedProduct] = []
        
        suppliers_to_search = (
            [self.suppliers.get(supplier_type.value)]
            if supplier_type else list(self.suppliers.values())
        )
        suppliers_to_search = [s for s in suppliers_to_search if s]

        if not suppliers_to_search:
            return AgentResult(
                success=False,
                message="No suppliers configured",
                data={"products": []}
            )

        for client in suppliers_to_search:
            try:
                products = await client.search_products(query, limit=limit)
                for p in products:
                    if p.total_cost > max_price:
                        continue
                    
                    rec_price = self._calculate_price(p.total_cost)
                    margin = p.calculate_margin(rec_price)
                    
                    if margin >= min_margin:
                        all_products.append(SourcedProduct(
                            supplier_product=p,
                            recommended_price=rec_price,
                            margin_percent=margin,
                            tags=self._generate_tags(p, query),
                        ))
            except Exception as e:
                logger.error(f"Search error: {e}")

        all_products.sort(key=lambda x: x.margin_percent, reverse=True)
        
        return AgentResult(
            success=True,
            message=f"Found {len(all_products)} products",
            data={"products": [self._sourced_to_dict(p) for p in all_products[:limit]]}
        )

    async def import_to_store(
        self,
        business_id: str,
        supplier_product_id: str,
        supplier_type: SupplierType,
        price_override: Optional[float] = None,
        title_override: Optional[str] = None,
    ) -> AgentResult:
        """Import supplier product to Shopify store."""
        client = self.suppliers.get(supplier_type.value)
        if not client:
            return AgentResult(success=False, message="Supplier not configured")

        product = await client.get_product(supplier_product_id)
        if not product:
            return AgentResult(success=False, message="Product not found")

        # Get Shopify client for business
        shopify = await self._get_shopify_client(business_id)
        if not shopify:
            return AgentResult(success=False, message="Shopify not configured")

        sell_price = price_override or self._calculate_price(product.total_cost)
        
        shopify_product = await shopify.create_product({
            "title": title_override or product.title,
            "body_html": product.description,
            "vendor": product.supplier_type.value,
            "product_type": product.category,
            "tags": f"dropship,{supplier_type.value},{product.product_id}",
            "variants": [{
                "price": str(sell_price),
                "sku": f"{supplier_type.value}:{product.product_id}",
                "inventory_quantity": min(product.stock_quantity, 100),
                "inventory_management": "shopify",
            }],
            "images": [{"src": img} for img in product.images[:5]],
        })

        return AgentResult(
            success=True,
            message="Product imported successfully",
            data={
                "shopify_product_id": shopify_product.get("id"),
                "supplier_product_id": product.product_id,
                "price": sell_price,
                "margin": product.calculate_margin(sell_price),
            }
        )

    async def sync_inventory(self, business_id: str) -> AgentResult:
        """Sync inventory levels from suppliers to store."""
        shopify = await self._get_shopify_client(business_id)
        if not shopify:
            return AgentResult(success=False, message="Shopify not configured")

        updated = 0
        errors = 0
        
        products = await shopify.get_products(limit=250)
        for product in products:
            tags = product.get("tags", "")
            if "dropship" not in tags:
                continue

            for variant in product.get("variants", []):
                sku = variant.get("sku", "")
                if ":" not in sku:
                    continue

                supplier_key, product_id = sku.split(":", 1)
                client = self.suppliers.get(supplier_key)
                if not client:
                    continue

                try:
                    stock = await client.get_stock(product_id)
                    stock = min(stock, 100)  # Cap display stock
                    
                    await shopify.update_inventory(
                        variant.get("inventory_item_id"),
                        stock
                    )
                    updated += 1
                except Exception as e:
                    logger.error(f"Inventory sync error for {sku}: {e}")
                    errors += 1

        return AgentResult(
            success=True,
            message=f"Synced {updated} products, {errors} errors",
            data={"updated": updated, "errors": errors}
        )

    async def fulfill_order(
        self,
        business_id: str,
        shopify_order_id: str,
    ) -> AgentResult:
        """Forward order to supplier for fulfillment."""
        shopify = await self._get_shopify_client(business_id)
        if not shopify:
            return AgentResult(success=False, message="Shopify not configured")

        order = await shopify.get_order(shopify_order_id)
        if not order:
            return AgentResult(success=False, message="Order not found")

        # Group items by supplier
        supplier_items: dict[str, list[dict]] = {}
        for item in order.get("line_items", []):
            sku = item.get("sku", "")
            if ":" not in sku:
                continue
            supplier_key, product_id = sku.split(":", 1)
            if supplier_key not in supplier_items:
                supplier_items[supplier_key] = []
            supplier_items[supplier_key].append({
                "product_id": product_id,
                "variant_id": item.get("variant_id"),
                "quantity": item.get("quantity", 1),
            })

        shipping = order.get("shipping_address", {})
        shipping_address = {
            "name": f"{shipping.get('first_name', '')} {shipping.get('last_name', '')}",
            "address1": shipping.get("address1", ""),
            "address2": shipping.get("address2", ""),
            "city": shipping.get("city", ""),
            "province": shipping.get("province", ""),
            "country": shipping.get("country", ""),
            "country_code": shipping.get("country_code", ""),
            "zip": shipping.get("zip", ""),
            "phone": shipping.get("phone", ""),
        }

        supplier_orders = []
        for supplier_key, items in supplier_items.items():
            client = self.suppliers.get(supplier_key)
            if not client:
                continue
            
            try:
                supplier_order = await client.place_order(items, shipping_address)
                supplier_orders.append({
                    "supplier": supplier_key,
                    "order_id": supplier_order.supplier_order_id,
                    "status": supplier_order.status,
                })
            except Exception as e:
                logger.error(f"Fulfillment error with {supplier_key}: {e}")
                supplier_orders.append({
                    "supplier": supplier_key,
                    "error": str(e),
                })

        return AgentResult(
            success=True,
            message=f"Forwarded to {len(supplier_orders)} suppliers",
            data={"supplier_orders": supplier_orders, "shopify_order_id": shopify_order_id}
        )

    async def check_price_changes(self, business_id: str) -> AgentResult:
        """Check for supplier price changes and recommend updates."""
        shopify = await self._get_shopify_client(business_id)
        if not shopify:
            return AgentResult(success=False, message="Shopify not configured")

        changes: list[ProductMatch] = []
        products = await shopify.get_products(limit=250)
        
        for product in products:
            tags = product.get("tags", "")
            if "dropship" not in tags:
                continue

            for variant in product.get("variants", []):
                sku = variant.get("sku", "")
                if ":" not in sku:
                    continue

                supplier_key, product_id = sku.split(":", 1)
                client = self.suppliers.get(supplier_key)
                if not client:
                    continue

                try:
                    supplier_product = await client.get_product(product_id)
                    if not supplier_product:
                        continue

                    current_price = float(variant.get("price", 0))
                    new_rec_price = self._calculate_price(supplier_product.total_cost)
                    price_diff = abs(current_price - new_rec_price)
                    
                    if price_diff > 0.50:  # Only flag significant changes
                        changes.append(ProductMatch(
                            store_product_id=str(product.get("id")),
                            supplier_product=supplier_product,
                            current_price=current_price,
                            recommended_price=new_rec_price,
                            margin_percent=supplier_product.calculate_margin(current_price),
                            price_change=new_rec_price - current_price,
                        ))
                except Exception as e:
                    logger.error(f"Price check error for {sku}: {e}")

        return AgentResult(
            success=True,
            message=f"Found {len(changes)} price changes",
            data={"changes": [self._match_to_dict(c) for c in changes]}
        )

    def _calculate_price(self, cost: float) -> float:
        """Calculate selling price with margin."""
        base_price = cost * self.default_margin
        # Round to .99 pricing
        return round(base_price) - 0.01 if base_price > 1 else round(base_price, 2)

    def _generate_tags(self, product: SupplierProduct, query: str) -> list[str]:
        """Generate product tags."""
        tags = ["dropship", product.supplier_type.value]
        if product.category:
            tags.append(product.category.lower().replace(" ", "-"))
        tags.extend(query.lower().split()[:3])
        return tags

    def _sourced_to_dict(self, p: SourcedProduct) -> dict:
        return {
            "product_id": p.supplier_product.product_id,
            "supplier": p.supplier_product.supplier_type.value,
            "title": p.supplier_product.title,
            "cost": p.supplier_product.total_cost,
            "recommended_price": p.recommended_price,
            "margin_percent": round(p.margin_percent, 1),
            "stock": p.supplier_product.stock_quantity,
            "images": p.supplier_product.images[:3],
            "tags": p.tags,
        }

    def _match_to_dict(self, m: ProductMatch) -> dict:
        return {
            "store_product_id": m.store_product_id,
            "supplier_product_id": m.supplier_product.product_id,
            "current_price": m.current_price,
            "recommended_price": m.recommended_price,
            "margin_percent": round(m.margin_percent, 1),
            "price_change": round(m.price_change, 2),
        }

    async def _get_shopify_client(self, business_id: str) -> Optional[ShopifyClient]:
        """Get Shopify client for business."""
        async with get_db_session() as session:
            business = await session.get(BusinessUnit, business_id)
            if not business or not business.config:
                return None
            shopify_config = business.config.get("shopify", {})
            if not shopify_config.get("shop_url") or not shopify_config.get("access_token"):
                return None
            return ShopifyClient(
                shop_url=shopify_config["shop_url"],
                access_token=shopify_config["access_token"]
            )

    async def execute(self, task: str, context: dict) -> AgentResult:
        """Execute supplier task based on context."""
        action = context.get("action", "search")
        
        if action == "search":
            return await self.search_products(
                query=context.get("query", ""),
                min_margin=context.get("min_margin", 30.0),
                max_price=context.get("max_price", 100.0),
            )
        elif action == "import":
            return await self.import_to_store(
                business_id=context["business_id"],
                supplier_product_id=context["product_id"],
                supplier_type=SupplierType(context["supplier"]),
            )
        elif action == "sync_inventory":
            return await self.sync_inventory(context["business_id"])
        elif action == "fulfill":
            return await self.fulfill_order(
                business_id=context["business_id"],
                shopify_order_id=context["order_id"],
            )
        elif action == "check_prices":
            return await self.check_price_changes(context["business_id"])
        
        return AgentResult(success=False, message=f"Unknown action: {action}")
```

---

### Task 14.3: Supplier API Routes

**File: `src/api/routes/supplier.py`**

```python
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
    return result.data


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
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.post("/sync-inventory/{business_id}")
async def sync_inventory(business_id: str, agent: SupplierAgent = Depends(get_agent)):
    """Sync inventory from suppliers to store."""
    result = await agent.sync_inventory(business_id)
    return result.data


@router.post("/fulfill")
async def fulfill_order(req: FulfillRequest, agent: SupplierAgent = Depends(get_agent)):
    """Forward order to supplier for fulfillment."""
    result = await agent.fulfill_order(req.business_id, req.order_id)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/price-changes/{business_id}")
async def check_price_changes(business_id: str, agent: SupplierAgent = Depends(get_agent)):
    """Check for supplier price changes."""
    result = await agent.check_price_changes(business_id)
    return result.data
```

---

### Task 14.4: Tests

**File: `tests/test_supplier.py`**

```python
"""Tests for Supplier Agent and Client."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.integrations.supplier_client import (
    SupplierType, SupplierProduct, SupplierOrder,
    AliExpressClient, CJDropshippingClient, SupplierClientFactory
)
from src.agents.supplier import SupplierAgent, SourcedProduct


@pytest.fixture
def sample_supplier_product():
    return SupplierProduct(
        supplier_id="test",
        supplier_type=SupplierType.ALIEXPRESS,
        product_id="12345",
        title="Test Product",
        description="A test product",
        price=10.00,
        shipping_cost=2.00,
        stock_quantity=100,
        images=["https://example.com/img.jpg"],
        category="Electronics",
    )


@pytest.fixture
def supplier_agent():
    return SupplierAgent()


class TestSupplierProduct:
    def test_total_cost(self, sample_supplier_product):
        assert sample_supplier_product.total_cost == 12.00

    def test_calculate_margin(self, sample_supplier_product):
        margin = sample_supplier_product.calculate_margin(30.00)
        assert margin == 60.0  # (30 - 12) / 30 * 100

    def test_calculate_margin_zero_price(self, sample_supplier_product):
        assert sample_supplier_product.calculate_margin(0) == 0.0


class TestSupplierClientFactory:
    def test_create_aliexpress(self):
        client = SupplierClientFactory.create(
            SupplierType.ALIEXPRESS,
            app_key="key",
            app_secret="secret"
        )
        assert isinstance(client, AliExpressClient)

    def test_create_cj(self):
        client = SupplierClientFactory.create(
            SupplierType.CJ_DROPSHIPPING,
            api_key="key"
        )
        assert isinstance(client, CJDropshippingClient)

    def test_create_unsupported(self):
        with pytest.raises(ValueError):
            SupplierClientFactory.create(SupplierType.WHOLESALE, api_key="key")


class TestSupplierAgent:
    @pytest.mark.asyncio
    async def test_initialize_supplier(self, supplier_agent):
        result = await supplier_agent.initialize_supplier(
            SupplierType.ALIEXPRESS,
            {"app_key": "key", "app_secret": "secret"}
        )
        assert result is True
        assert "aliexpress" in supplier_agent.suppliers

    @pytest.mark.asyncio
    async def test_search_no_suppliers(self, supplier_agent):
        result = await supplier_agent.search_products("phone case")
        assert not result.success
        assert "No suppliers" in result.message

    @pytest.mark.asyncio
    async def test_search_with_results(self, supplier_agent, sample_supplier_product):
        mock_client = AsyncMock()
        mock_client.search_products = AsyncMock(return_value=[sample_supplier_product])
        supplier_agent.suppliers["aliexpress"] = mock_client
        
        result = await supplier_agent.search_products("test", min_margin=20.0)
        assert result.success
        assert len(result.data["products"]) > 0

    @pytest.mark.asyncio
    async def test_calculate_price(self, supplier_agent):
        price = supplier_agent._calculate_price(10.00)
        assert price == 24.99  # 10 * 2.5 = 25, rounded to 24.99

    def test_generate_tags(self, supplier_agent, sample_supplier_product):
        tags = supplier_agent._generate_tags(sample_supplier_product, "phone case")
        assert "dropship" in tags
        assert "aliexpress" in tags


class TestAliExpressClient:
    @pytest.mark.asyncio
    async def test_sign_request(self):
        client = AliExpressClient("appkey", "secret")
        params = {"a": "1", "b": "2"}
        signature = client._sign_request(params)
        assert len(signature) == 32  # MD5 hex length


class TestCJDropshippingClient:
    @pytest.mark.asyncio
    async def test_init(self):
        client = CJDropshippingClient("api_key")
        assert client.api_key == "api_key"
        assert client._access_token is None
```

---

## 3. Acceptance Criteria

| Criteria | Validation |
|----------|------------|
| Supplier clients connect to APIs | AliExpress and CJ Dropshipping clients functional |
| Product search returns results | Search across suppliers with margin filtering |
| Products import to Shopify | Supplier products create in store with pricing |
| Inventory syncs correctly | Stock levels update from supplier to store |
| Orders forward to suppliers | Customer orders placed with suppliers |
| Price monitoring works | Detects supplier price changes |

---

## 4. File Summary

| File | Purpose |
|------|---------|
| `src/integrations/supplier_client.py` | Multi-provider supplier API clients |
| `src/agents/supplier.py` | Supplier agent for sourcing/fulfillment |
| `src/api/routes/supplier.py` | REST API endpoints |
| `tests/test_supplier.py` | Unit tests |

---

## 5. Integration Notes

**Register routes in main.py:**
```python
from src.api.routes.supplier import router as supplier_router
app.include_router(supplier_router, prefix="/api/v1")
```

**Environment variables needed:**
```
ALIEXPRESS_APP_KEY=your_app_key
ALIEXPRESS_APP_SECRET=your_secret
ALIEXPRESS_TRACKING_ID=your_tracking_id
CJ_API_KEY=your_cj_api_key
```
