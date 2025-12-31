"""
Supplier Agent - Product sourcing, order fulfillment, inventory sync.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from src.agents.base import SubAgent
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


# Function schema for LLM function-calling
FUNCTION_SCHEMA = {
    "name": "supplier_agent",
    "description": "Supplier integration for dropshipping - product sourcing, order fulfillment, inventory sync",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search_products", "get_product_details", "import_product",
                         "sync_inventory", "place_order", "track_order", 
                         "check_stock", "update_pricing", "list_suppliers"],
                "description": "The supplier action to perform"
            },
            "supplier_type": {
                "type": "string",
                "enum": ["aliexpress", "spocket", "cjdropshipping", "printful"],
                "description": "Which supplier platform to use"
            },
            "search_query": {
                "type": "string",
                "description": "Product search query"
            },
            "product_id": {
                "type": "string",
                "description": "Supplier product ID"
            },
            "order_id": {
                "type": "string",
                "description": "Order ID for tracking"
            },
            "min_margin": {
                "type": "number",
                "description": "Minimum profit margin percentage (default 30%)"
            },
            "max_price": {
                "type": "number",
                "description": "Maximum product cost from supplier"
            },
            "shipping_info": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"},
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                    "postal_code": {"type": "string"}
                },
                "description": "Shipping details for order fulfillment"
            },
            "quantity": {
                "type": "integer",
                "description": "Quantity to order or check stock for"
            }
        },
        "required": ["action"]
    }
}


class SupplierAgent(SubAgent):
    """Agent for supplier integration and dropshipping operations."""

    name = "supplier"
    description = "Handles supplier integration, product sourcing, and dropshipping operations."
    FUNCTION_SCHEMA = FUNCTION_SCHEMA
    
    # Configuration constants
    MAX_DISPLAY_STOCK = 100  # Cap stock display to avoid showing excessive quantities
    SIGNIFICANT_PRICE_CHANGE_THRESHOLD = 0.50  # Minimum price change to flag (in currency units)

    def __init__(self):
        super().__init__()
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
    ) -> dict:
        """Search for products across suppliers."""
        all_products: list[SourcedProduct] = []
        
        suppliers_to_search = (
            [self.suppliers.get(supplier_type.value)]
            if supplier_type else list(self.suppliers.values())
        )
        suppliers_to_search = [s for s in suppliers_to_search if s]

        if not suppliers_to_search:
            return {
                "success": False,
                "output": {"products": []},
                "error": "No suppliers configured",
                "metadata": {}
            }

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
        
        return {
            "success": True,
            "output": {"products": [self._sourced_to_dict(p) for p in all_products[:limit]]},
            "metadata": {"found": len(all_products), "query": query}
        }

    async def import_to_store(
        self,
        business_id: str,
        supplier_product_id: str,
        supplier_type: SupplierType,
        price_override: Optional[float] = None,
        title_override: Optional[str] = None,
    ) -> dict:
        """Import supplier product to Shopify store."""
        client = self.suppliers.get(supplier_type.value)
        if not client:
            return {"success": False, "error": "Supplier not configured", "metadata": {}}

        product = await client.get_product(supplier_product_id)
        if not product:
            return {"success": False, "error": "Product not found", "metadata": {}}

        # Get Shopify client for business
        shopify = await self._get_shopify_client(business_id)
        if not shopify:
            return {"success": False, "error": "Shopify not configured", "metadata": {}}

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
                "inventory_quantity": min(product.stock_quantity, self.MAX_DISPLAY_STOCK),
                "inventory_management": "shopify",
            }],
            "images": [{"src": img} for img in product.images[:5]],
        })

        return {
            "success": True,
            "output": {
                "shopify_product_id": shopify_product.get("id"),
                "supplier_product_id": product.product_id,
                "price": sell_price,
                "margin": product.calculate_margin(sell_price),
            },
            "metadata": {"business_id": business_id}
        }

    async def sync_inventory(self, business_id: str) -> dict:
        """Sync inventory levels from suppliers to store."""
        shopify = await self._get_shopify_client(business_id)
        if not shopify:
            return {"success": False, "error": "Shopify not configured", "metadata": {}}

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
                    stock = min(stock, self.MAX_DISPLAY_STOCK)  # Cap display stock
                    
                    await shopify.update_inventory(
                        variant.get("inventory_item_id"),
                        stock
                    )
                    updated += 1
                except Exception as e:
                    logger.error(f"Inventory sync error for {sku}: {e}")
                    errors += 1

        return {
            "success": True,
            "output": {"updated": updated, "errors": errors},
            "metadata": {"business_id": business_id}
        }

    async def fulfill_order(
        self,
        business_id: str,
        shopify_order_id: str,
    ) -> dict:
        """Forward order to supplier for fulfillment."""
        shopify = await self._get_shopify_client(business_id)
        if not shopify:
            return {"success": False, "error": "Shopify not configured", "metadata": {}}

        order = await shopify.get_order(shopify_order_id)
        if not order:
            return {"success": False, "error": "Order not found", "metadata": {}}

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

        return {
            "success": True,
            "output": {"supplier_orders": supplier_orders, "shopify_order_id": shopify_order_id},
            "metadata": {"business_id": business_id}
        }

    async def check_price_changes(self, business_id: str) -> dict:
        """Check for supplier price changes and recommend updates."""
        shopify = await self._get_shopify_client(business_id)
        if not shopify:
            return {"success": False, "error": "Shopify not configured", "metadata": {}}

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
                    
                    if price_diff > self.SIGNIFICANT_PRICE_CHANGE_THRESHOLD:
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

        return {
            "success": True,
            "output": {"changes": [self._match_to_dict(c) for c in changes]},
            "metadata": {"business_id": business_id, "found": len(changes)}
        }

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

    async def execute(self, task: dict) -> dict:
        """
        Execute supplier task based on task definition.
        
        Args:
            task: {
                "name": str,
                "description": str,
                "input": dict with action-specific parameters
            }
            
        Returns:
            {
                "success": bool,
                "output": Any,
                "error": str | None,
                "metadata": dict
            }
        """
        input_data = task.get("input", {})
        action = input_data.get("action", "search")
        
        try:
            if action == "search":
                return await self.search_products(
                    query=input_data.get("query", ""),
                    min_margin=input_data.get("min_margin", 30.0),
                    max_price=input_data.get("max_price", 100.0),
                )
            elif action == "import":
                return await self.import_to_store(
                    business_id=input_data["business_id"],
                    supplier_product_id=input_data["product_id"],
                    supplier_type=SupplierType(input_data["supplier"]),
                )
            elif action == "sync_inventory":
                return await self.sync_inventory(input_data["business_id"])
            elif action == "fulfill":
                return await self.fulfill_order(
                    business_id=input_data["business_id"],
                    shopify_order_id=input_data["order_id"],
                )
            elif action == "check_prices":
                return await self.check_price_changes(input_data["business_id"])
            
            return {"success": False, "error": f"Unknown action: {action}", "metadata": {}}
        except Exception as e:
            logger.error(f"Supplier agent error: {e}")
            return {"success": False, "error": str(e), "metadata": {}}
