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
        assert not result.get("success")
        assert "No suppliers" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_search_with_results(self, supplier_agent, sample_supplier_product):
        mock_client = AsyncMock()
        mock_client.search_products = AsyncMock(return_value=[sample_supplier_product])
        supplier_agent.suppliers["aliexpress"] = mock_client
        
        result = await supplier_agent.search_products("test", min_margin=20.0)
        assert result.get("success")
        assert len(result.get("output", {}).get("products", [])) > 0

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
