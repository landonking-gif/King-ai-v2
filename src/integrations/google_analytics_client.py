"""
Google Analytics 4 Client - Analytics data retrieval and reporting.

Integrates with GA4 Data API for retrieving website analytics,
user behavior, and e-commerce metrics for business units.
"""

import os
from datetime import date, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import (
        RunReportRequest, DateRange, Dimension, Metric,
        OrderBy, FilterExpression, Filter, RunRealtimeReportRequest
    )
    from google.oauth2 import service_account
    GA_AVAILABLE = True
except ImportError:
    GA_AVAILABLE = False
    BetaAnalyticsDataClient = None

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Common GA4 metrics."""
    SESSIONS = "sessions"
    USERS = "totalUsers"
    NEW_USERS = "newUsers"
    PAGE_VIEWS = "screenPageViews"
    BOUNCE_RATE = "bounceRate"
    AVG_SESSION_DURATION = "averageSessionDuration"
    CONVERSIONS = "conversions"
    REVENUE = "totalRevenue"
    TRANSACTIONS = "transactions"
    ITEMS_PURCHASED = "itemsPurchased"
    PURCHASE_REVENUE = "purchaseRevenue"
    ADD_TO_CARTS = "addToCarts"
    CHECKOUTS = "checkouts"
    ECOMMERCE_PURCHASES = "ecommercePurchases"


class DimensionType(str, Enum):
    """Common GA4 dimensions."""
    DATE = "date"
    COUNTRY = "country"
    CITY = "city"
    DEVICE = "deviceCategory"
    SOURCE = "sessionSource"
    MEDIUM = "sessionMedium"
    CAMPAIGN = "sessionCampaignName"
    PAGE_PATH = "pagePath"
    PAGE_TITLE = "pageTitle"
    LANDING_PAGE = "landingPage"
    EVENT_NAME = "eventName"
    ITEM_NAME = "itemName"
    ITEM_CATEGORY = "itemCategory"


@dataclass
class AnalyticsMetric:
    """A single analytics metric with metadata."""
    name: str
    value: float
    change_percent: Optional[float] = None  # vs previous period
    trend: Optional[str] = None  # "up", "down", "stable"


@dataclass
class AnalyticsReport:
    """Complete analytics report."""
    property_id: str
    date_range: tuple[date, date]
    metrics: Dict[str, AnalyticsMetric]
    dimensions: Dict[str, List[Dict[str, Any]]]
    generated_at: str = field(default_factory=lambda: date.today().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property_id": self.property_id,
            "date_range": {
                "start": self.date_range[0].isoformat(),
                "end": self.date_range[1].isoformat(),
            },
            "metrics": {
                k: {"value": v.value, "change": v.change_percent, "trend": v.trend}
                for k, v in self.metrics.items()
            },
            "dimensions": self.dimensions,
            "generated_at": self.generated_at,
        }


@dataclass
class EcommerceReport:
    """E-commerce specific analytics."""
    total_revenue: float
    transactions: int
    average_order_value: float
    conversion_rate: float
    top_products: List[Dict[str, Any]]
    top_categories: List[Dict[str, Any]]
    cart_abandonment_rate: float
    revenue_by_source: Dict[str, float]


class GoogleAnalyticsClient:
    """
    Client for Google Analytics 4 Data API.
    
    Provides methods for retrieving analytics data including:
    - Traffic metrics (sessions, users, page views)
    - E-commerce data (revenue, transactions, products)
    - User behavior (bounce rate, session duration)
    - Real-time data
    """

    def __init__(
        self,
        property_id: str = None,
        credentials_path: str = None,
        credentials_json: str = None,
    ):
        """
        Initialize GA4 client.
        
        Args:
            property_id: GA4 property ID (e.g., "properties/123456789")
            credentials_path: Path to service account JSON file
            credentials_json: Service account JSON as string (alternative to path)
        """
        self.property_id = property_id or os.getenv("GA4_PROPERTY_ID")
        self._client: Optional[BetaAnalyticsDataClient] = None
        self._credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self._credentials_json = credentials_json or os.getenv("GA4_CREDENTIALS_JSON")
        
        if not GA_AVAILABLE:
            logger.warning("Google Analytics SDK not installed. Install with: pip install google-analytics-data")

    def _get_client(self) -> Optional[BetaAnalyticsDataClient]:
        """Get or create the GA4 client."""
        if not GA_AVAILABLE:
            return None
            
        if self._client is None:
            try:
                if self._credentials_json:
                    import json
                    credentials = service_account.Credentials.from_service_account_info(
                        json.loads(self._credentials_json)
                    )
                    self._client = BetaAnalyticsDataClient(credentials=credentials)
                elif self._credentials_path:
                    self._client = BetaAnalyticsDataClient.from_service_account_file(
                        self._credentials_path
                    )
                else:
                    # Use default credentials
                    self._client = BetaAnalyticsDataClient()
            except Exception as e:
                logger.error(f"Failed to initialize GA4 client: {e}")
                return None
        
        return self._client

    async def get_traffic_metrics(
        self,
        start_date: date = None,
        end_date: date = None,
        compare_previous: bool = True,
    ) -> Dict[str, Any]:
        """
        Get traffic metrics for the date range.
        
        Returns sessions, users, page views, bounce rate, etc.
        """
        client = self._get_client()
        if not client:
            return self._mock_traffic_metrics()
        
        start = start_date or (date.today() - timedelta(days=30))
        end = end_date or date.today()
        
        try:
            request = RunReportRequest(
                property=self.property_id,
                date_ranges=[DateRange(start_date=start.isoformat(), end_date=end.isoformat())],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="totalUsers"),
                    Metric(name="newUsers"),
                    Metric(name="screenPageViews"),
                    Metric(name="bounceRate"),
                    Metric(name="averageSessionDuration"),
                ],
            )
            response = client.run_report(request)
            
            metrics = {}
            if response.rows:
                row = response.rows[0]
                for i, metric in enumerate(response.metric_headers):
                    metrics[metric.name] = float(row.metric_values[i].value)
            
            return {
                "success": True,
                "metrics": metrics,
                "date_range": {"start": start.isoformat(), "end": end.isoformat()},
            }
        except Exception as e:
            logger.error(f"GA4 traffic metrics error: {e}")
            return {"success": False, "error": str(e)}

    async def get_ecommerce_metrics(
        self,
        start_date: date = None,
        end_date: date = None,
    ) -> Dict[str, Any]:
        """
        Get e-commerce metrics including revenue, transactions, and products.
        """
        client = self._get_client()
        if not client:
            return self._mock_ecommerce_metrics()
        
        start = start_date or (date.today() - timedelta(days=30))
        end = end_date or date.today()
        
        try:
            request = RunReportRequest(
                property=self.property_id,
                date_ranges=[DateRange(start_date=start.isoformat(), end_date=end.isoformat())],
                metrics=[
                    Metric(name="totalRevenue"),
                    Metric(name="transactions"),
                    Metric(name="ecommercePurchases"),
                    Metric(name="addToCarts"),
                    Metric(name="checkouts"),
                    Metric(name="itemsPurchased"),
                    Metric(name="purchaseRevenue"),
                ],
            )
            response = client.run_report(request)
            
            metrics = {}
            if response.rows:
                row = response.rows[0]
                for i, metric in enumerate(response.metric_headers):
                    metrics[metric.name] = float(row.metric_values[i].value)
            
            # Calculate derived metrics
            transactions = metrics.get("transactions", 0)
            revenue = metrics.get("totalRevenue", 0)
            add_to_carts = metrics.get("addToCarts", 0)
            checkouts = metrics.get("checkouts", 0)
            
            metrics["averageOrderValue"] = revenue / transactions if transactions > 0 else 0
            metrics["cartAbandonmentRate"] = (
                (add_to_carts - checkouts) / add_to_carts * 100 if add_to_carts > 0 else 0
            )
            
            return {
                "success": True,
                "metrics": metrics,
                "date_range": {"start": start.isoformat(), "end": end.isoformat()},
            }
        except Exception as e:
            logger.error(f"GA4 ecommerce metrics error: {e}")
            return {"success": False, "error": str(e)}

    async def get_top_products(
        self,
        start_date: date = None,
        end_date: date = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Get top performing products by revenue."""
        client = self._get_client()
        if not client:
            return self._mock_top_products()
        
        start = start_date or (date.today() - timedelta(days=30))
        end = end_date or date.today()
        
        try:
            request = RunReportRequest(
                property=self.property_id,
                date_ranges=[DateRange(start_date=start.isoformat(), end_date=end.isoformat())],
                dimensions=[
                    Dimension(name="itemName"),
                    Dimension(name="itemCategory"),
                ],
                metrics=[
                    Metric(name="itemRevenue"),
                    Metric(name="itemsPurchased"),
                    Metric(name="itemsViewed"),
                ],
                order_bys=[
                    OrderBy(metric=OrderBy.MetricOrderBy(metric_name="itemRevenue"), desc=True)
                ],
                limit=limit,
            )
            response = client.run_report(request)
            
            products = []
            for row in response.rows:
                products.append({
                    "name": row.dimension_values[0].value,
                    "category": row.dimension_values[1].value,
                    "revenue": float(row.metric_values[0].value),
                    "quantity": int(float(row.metric_values[1].value)),
                    "views": int(float(row.metric_values[2].value)),
                })
            
            return {"success": True, "products": products}
        except Exception as e:
            logger.error(f"GA4 top products error: {e}")
            return {"success": False, "error": str(e)}

    async def get_traffic_sources(
        self,
        start_date: date = None,
        end_date: date = None,
    ) -> Dict[str, Any]:
        """Get traffic acquisition breakdown by source/medium."""
        client = self._get_client()
        if not client:
            return self._mock_traffic_sources()
        
        start = start_date or (date.today() - timedelta(days=30))
        end = end_date or date.today()
        
        try:
            request = RunReportRequest(
                property=self.property_id,
                date_ranges=[DateRange(start_date=start.isoformat(), end_date=end.isoformat())],
                dimensions=[
                    Dimension(name="sessionSource"),
                    Dimension(name="sessionMedium"),
                ],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="totalUsers"),
                    Metric(name="conversions"),
                    Metric(name="totalRevenue"),
                ],
                order_bys=[
                    OrderBy(metric=OrderBy.MetricOrderBy(metric_name="sessions"), desc=True)
                ],
                limit=20,
            )
            response = client.run_report(request)
            
            sources = []
            for row in response.rows:
                sources.append({
                    "source": row.dimension_values[0].value,
                    "medium": row.dimension_values[1].value,
                    "sessions": int(float(row.metric_values[0].value)),
                    "users": int(float(row.metric_values[1].value)),
                    "conversions": int(float(row.metric_values[2].value)),
                    "revenue": float(row.metric_values[3].value),
                })
            
            return {"success": True, "sources": sources}
        except Exception as e:
            logger.error(f"GA4 traffic sources error: {e}")
            return {"success": False, "error": str(e)}

    async def get_realtime_users(self) -> Dict[str, Any]:
        """Get real-time active users count."""
        client = self._get_client()
        if not client:
            return {"success": True, "active_users": 0, "is_mock": True}
        
        try:
            request = RunRealtimeReportRequest(
                property=self.property_id,
                metrics=[Metric(name="activeUsers")],
            )
            response = client.run_realtime_report(request)
            
            active_users = 0
            if response.rows:
                active_users = int(float(response.rows[0].metric_values[0].value))
            
            return {"success": True, "active_users": active_users}
        except Exception as e:
            logger.error(f"GA4 realtime error: {e}")
            return {"success": False, "error": str(e)}

    async def get_conversion_funnel(
        self,
        start_date: date = None,
        end_date: date = None,
    ) -> Dict[str, Any]:
        """Get e-commerce conversion funnel metrics."""
        client = self._get_client()
        if not client:
            return self._mock_conversion_funnel()
        
        start = start_date or (date.today() - timedelta(days=30))
        end = end_date or date.today()
        
        try:
            request = RunReportRequest(
                property=self.property_id,
                date_ranges=[DateRange(start_date=start.isoformat(), end_date=end.isoformat())],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="itemsViewed"),
                    Metric(name="addToCarts"),
                    Metric(name="checkouts"),
                    Metric(name="ecommercePurchases"),
                ],
            )
            response = client.run_report(request)
            
            funnel = {"stages": []}
            if response.rows:
                row = response.rows[0]
                stages = [
                    ("Sessions", float(row.metric_values[0].value)),
                    ("Product Views", float(row.metric_values[1].value)),
                    ("Add to Cart", float(row.metric_values[2].value)),
                    ("Checkout", float(row.metric_values[3].value)),
                    ("Purchase", float(row.metric_values[4].value)),
                ]
                
                for i, (name, value) in enumerate(stages):
                    conversion_rate = None
                    if i > 0 and stages[i-1][1] > 0:
                        conversion_rate = (value / stages[i-1][1]) * 100
                    
                    funnel["stages"].append({
                        "name": name,
                        "value": int(value),
                        "conversion_rate": round(conversion_rate, 2) if conversion_rate else None,
                    })
            
            return {"success": True, "funnel": funnel}
        except Exception as e:
            logger.error(f"GA4 funnel error: {e}")
            return {"success": False, "error": str(e)}

    # Mock methods for development without GA4 credentials
    def _mock_traffic_metrics(self) -> Dict[str, Any]:
        return {
            "success": True,
            "is_mock": True,
            "metrics": {
                "sessions": 15234,
                "totalUsers": 8456,
                "newUsers": 4523,
                "screenPageViews": 45678,
                "bounceRate": 42.5,
                "averageSessionDuration": 185.3,
            },
        }

    def _mock_ecommerce_metrics(self) -> Dict[str, Any]:
        return {
            "success": True,
            "is_mock": True,
            "metrics": {
                "totalRevenue": 125678.50,
                "transactions": 423,
                "ecommercePurchases": 423,
                "addToCarts": 1256,
                "checkouts": 567,
                "itemsPurchased": 892,
                "averageOrderValue": 297.11,
                "cartAbandonmentRate": 54.86,
            },
        }

    def _mock_top_products(self) -> Dict[str, Any]:
        return {
            "success": True,
            "is_mock": True,
            "products": [
                {"name": "Premium Widget", "category": "Electronics", "revenue": 12500.00, "quantity": 125, "views": 3456},
                {"name": "Smart Gadget Pro", "category": "Electronics", "revenue": 9800.00, "quantity": 98, "views": 2890},
                {"name": "Eco Friendly Bag", "category": "Accessories", "revenue": 5600.00, "quantity": 280, "views": 1567},
            ],
        }

    def _mock_traffic_sources(self) -> Dict[str, Any]:
        return {
            "success": True,
            "is_mock": True,
            "sources": [
                {"source": "google", "medium": "organic", "sessions": 5234, "users": 4123, "conversions": 234, "revenue": 45678.00},
                {"source": "facebook", "medium": "cpc", "sessions": 2345, "users": 2100, "conversions": 89, "revenue": 12345.00},
                {"source": "direct", "medium": "(none)", "sessions": 1890, "users": 1654, "conversions": 67, "revenue": 9876.00},
            ],
        }

    def _mock_conversion_funnel(self) -> Dict[str, Any]:
        return {
            "success": True,
            "is_mock": True,
            "funnel": {
                "stages": [
                    {"name": "Sessions", "value": 15234, "conversion_rate": None},
                    {"name": "Product Views", "value": 8567, "conversion_rate": 56.24},
                    {"name": "Add to Cart", "value": 1256, "conversion_rate": 14.67},
                    {"name": "Checkout", "value": 567, "conversion_rate": 45.14},
                    {"name": "Purchase", "value": 423, "conversion_rate": 74.60},
                ],
            },
        }


# Global singleton instance
_ga_client: Optional[GoogleAnalyticsClient] = None


def get_ga_client(property_id: str = None) -> GoogleAnalyticsClient:
    """Get or create the global GA4 client instance."""
    global _ga_client
    if _ga_client is None or (property_id and _ga_client.property_id != property_id):
        _ga_client = GoogleAnalyticsClient(property_id=property_id)
    return _ga_client
