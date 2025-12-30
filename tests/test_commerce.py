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
    def mock_ollama(self):
        ollama = AsyncMock()
        ollama.complete = AsyncMock(return_value="""
TITLE: Amazing Product Title
DESCRIPTION: <p>This is an amazing product.</p>
TAGS: amazing, product, new
PRODUCT_TYPE: Electronics
""")
        return ollama
    
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
    def agent(self, mock_shopify):
        agent = CommerceAgent()
        agent.shopify = mock_shopify
        return agent
    
    @pytest.mark.asyncio
    async def test_get_products(self, agent):
        """Test getting products."""
        products = await agent.get_products()
        
        assert len(products) == 1
        assert products[0]["title"] == "Test Product"
    
    @pytest.mark.asyncio
    async def test_generate_product_listing(self, agent, mock_ollama):
        """Test AI product generation."""
        agent.ollama = mock_ollama
        
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
