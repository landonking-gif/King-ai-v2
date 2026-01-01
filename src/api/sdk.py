"""
API Client SDK.
Python SDK for consuming King AI APIs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from enum import Enum
import asyncio
import json

from src.utils.structured_logging import get_logger

logger = get_logger("api_sdk")


T = TypeVar("T")


class HTTPMethod(str, Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


@dataclass
class APIResponse(Generic[T]):
    """API response wrapper."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], model: Type[T] = None) -> "APIResponse[T]":
        """Create response from dict."""
        parsed_data = None
        if data.get("data") is not None:
            if model and hasattr(model, "from_dict"):
                parsed_data = model.from_dict(data["data"])
            else:
                parsed_data = data.get("data")
        
        return cls(
            success=data.get("success", True),
            data=parsed_data,
            error=data.get("error"),
            status_code=data.get("status_code", 200),
        )


@dataclass
class PaginatedResponse(Generic[T]):
    """Paginated API response."""
    items: List[T] = field(default_factory=list)
    total: int = 0
    page: int = 1
    per_page: int = 20
    has_next: bool = False
    has_prev: bool = False
    
    @property
    def total_pages(self) -> int:
        if self.per_page == 0:
            return 0
        return (self.total + self.per_page - 1) // self.per_page


@dataclass
class SDKConfig:
    """SDK configuration."""
    base_url: str = "http://localhost:8000"
    api_version: str = "v1"
    api_key: Optional[str] = None
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True


class BaseResource:
    """Base class for API resources."""
    
    def __init__(self, client: "KingAIClient"):
        self.client = client
    
    async def _request(
        self,
        method: HTTPMethod,
        path: str,
        data: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Make an API request."""
        return await self.client._request(method, path, data, params)


class BusinessResource(BaseResource):
    """Business management API."""
    
    async def list(
        self,
        page: int = 1,
        per_page: int = 20,
    ) -> PaginatedResponse[Dict]:
        """List all businesses."""
        response = await self._request(
            HTTPMethod.GET,
            "/businesses",
            params={"page": page, "per_page": per_page},
        )
        
        return PaginatedResponse(
            items=response.get("data", []),
            total=response.get("total", 0),
            page=page,
            per_page=per_page,
            has_next=response.get("has_next", False),
        )
    
    async def get(self, business_id: str) -> APIResponse[Dict]:
        """Get a business by ID."""
        response = await self._request(
            HTTPMethod.GET,
            f"/businesses/{business_id}",
        )
        return APIResponse.from_dict(response)
    
    async def create(self, data: Dict[str, Any]) -> APIResponse[Dict]:
        """Create a new business."""
        response = await self._request(
            HTTPMethod.POST,
            "/businesses",
            data=data,
        )
        return APIResponse.from_dict(response)
    
    async def update(
        self,
        business_id: str,
        data: Dict[str, Any],
    ) -> APIResponse[Dict]:
        """Update a business."""
        response = await self._request(
            HTTPMethod.PUT,
            f"/businesses/{business_id}",
            data=data,
        )
        return APIResponse.from_dict(response)
    
    async def delete(self, business_id: str) -> APIResponse[None]:
        """Delete a business."""
        response = await self._request(
            HTTPMethod.DELETE,
            f"/businesses/{business_id}",
        )
        return APIResponse.from_dict(response)
    
    async def get_health(self, business_id: str) -> APIResponse[Dict]:
        """Get business health score."""
        response = await self._request(
            HTTPMethod.GET,
            f"/businesses/{business_id}/health",
        )
        return APIResponse.from_dict(response)
    
    async def get_analytics(
        self,
        business_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> APIResponse[Dict]:
        """Get business analytics."""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        response = await self._request(
            HTTPMethod.GET,
            f"/businesses/{business_id}/analytics",
            params=params,
        )
        return APIResponse.from_dict(response)


class AgentResource(BaseResource):
    """Agent management API."""
    
    async def list(self) -> APIResponse[List[Dict]]:
        """List all agents."""
        response = await self._request(HTTPMethod.GET, "/agents")
        return APIResponse.from_dict(response)
    
    async def get(self, agent_id: str) -> APIResponse[Dict]:
        """Get agent details."""
        response = await self._request(
            HTTPMethod.GET,
            f"/agents/{agent_id}",
        )
        return APIResponse.from_dict(response)
    
    async def execute(
        self,
        agent_id: str,
        task: str,
        context: Dict[str, Any] = None,
    ) -> APIResponse[Dict]:
        """Execute an agent task."""
        response = await self._request(
            HTTPMethod.POST,
            f"/agents/{agent_id}/execute",
            data={
                "task": task,
                "context": context or {},
            },
        )
        return APIResponse.from_dict(response)
    
    async def get_capabilities(self, agent_id: str) -> APIResponse[List[str]]:
        """Get agent capabilities."""
        response = await self._request(
            HTTPMethod.GET,
            f"/agents/{agent_id}/capabilities",
        )
        return APIResponse.from_dict(response)


class ApprovalResource(BaseResource):
    """Approval system API."""
    
    async def list(
        self,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> PaginatedResponse[Dict]:
        """List approval requests."""
        params = {"page": page, "per_page": per_page}
        if status:
            params["status"] = status
        
        response = await self._request(
            HTTPMethod.GET,
            "/approvals",
            params=params,
        )
        
        return PaginatedResponse(
            items=response.get("data", []),
            total=response.get("total", 0),
            page=page,
            per_page=per_page,
        )
    
    async def get(self, approval_id: str) -> APIResponse[Dict]:
        """Get approval request details."""
        response = await self._request(
            HTTPMethod.GET,
            f"/approvals/{approval_id}",
        )
        return APIResponse.from_dict(response)
    
    async def approve(
        self,
        approval_id: str,
        comment: str = "",
    ) -> APIResponse[Dict]:
        """Approve a request."""
        response = await self._request(
            HTTPMethod.POST,
            f"/approvals/{approval_id}/approve",
            data={"comment": comment},
        )
        return APIResponse.from_dict(response)
    
    async def reject(
        self,
        approval_id: str,
        reason: str,
    ) -> APIResponse[Dict]:
        """Reject a request."""
        response = await self._request(
            HTTPMethod.POST,
            f"/approvals/{approval_id}/reject",
            data={"reason": reason},
        )
        return APIResponse.from_dict(response)
    
    async def escalate(
        self,
        approval_id: str,
        to_user: str,
        reason: str = "",
    ) -> APIResponse[Dict]:
        """Escalate a request."""
        response = await self._request(
            HTTPMethod.POST,
            f"/approvals/{approval_id}/escalate",
            data={"to_user": to_user, "reason": reason},
        )
        return APIResponse.from_dict(response)


class AnalyticsResource(BaseResource):
    """Analytics API."""
    
    async def get_summary(
        self,
        period: str = "day",
    ) -> APIResponse[Dict]:
        """Get analytics summary."""
        response = await self._request(
            HTTPMethod.GET,
            "/analytics/summary",
            params={"period": period},
        )
        return APIResponse.from_dict(response)
    
    async def get_revenue(
        self,
        start_date: str,
        end_date: str,
        granularity: str = "day",
    ) -> APIResponse[List[Dict]]:
        """Get revenue analytics."""
        response = await self._request(
            HTTPMethod.GET,
            "/analytics/revenue",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "granularity": granularity,
            },
        )
        return APIResponse.from_dict(response)
    
    async def get_customers(
        self,
        segment: Optional[str] = None,
    ) -> APIResponse[Dict]:
        """Get customer analytics."""
        params = {}
        if segment:
            params["segment"] = segment
        
        response = await self._request(
            HTTPMethod.GET,
            "/analytics/customers",
            params=params,
        )
        return APIResponse.from_dict(response)


class GoalsResource(BaseResource):
    """Goals and OKR API."""
    
    async def list(
        self,
        team: Optional[str] = None,
        status: Optional[str] = None,
    ) -> APIResponse[List[Dict]]:
        """List goals."""
        params = {}
        if team:
            params["team"] = team
        if status:
            params["status"] = status
        
        response = await self._request(
            HTTPMethod.GET,
            "/goals",
            params=params,
        )
        return APIResponse.from_dict(response)
    
    async def get(self, goal_id: str) -> APIResponse[Dict]:
        """Get goal details."""
        response = await self._request(
            HTTPMethod.GET,
            f"/goals/{goal_id}",
        )
        return APIResponse.from_dict(response)
    
    async def create(self, data: Dict[str, Any]) -> APIResponse[Dict]:
        """Create a new goal."""
        response = await self._request(
            HTTPMethod.POST,
            "/goals",
            data=data,
        )
        return APIResponse.from_dict(response)
    
    async def update_progress(
        self,
        goal_id: str,
        value: float,
        note: str = "",
    ) -> APIResponse[Dict]:
        """Update goal progress."""
        response = await self._request(
            HTTPMethod.POST,
            f"/goals/{goal_id}/progress",
            data={"value": value, "note": note},
        )
        return APIResponse.from_dict(response)


class MasterAIResource(BaseResource):
    """Master AI API."""
    
    async def query(
        self,
        message: str,
        context: Dict[str, Any] = None,
        session_id: Optional[str] = None,
    ) -> APIResponse[Dict]:
        """Send a query to the Master AI."""
        response = await self._request(
            HTTPMethod.POST,
            "/master-ai/query",
            data={
                "message": message,
                "context": context or {},
                "session_id": session_id,
            },
        )
        return APIResponse.from_dict(response)
    
    async def analyze(
        self,
        topic: str,
        data: Dict[str, Any] = None,
    ) -> APIResponse[Dict]:
        """Request analysis from Master AI."""
        response = await self._request(
            HTTPMethod.POST,
            "/master-ai/analyze",
            data={"topic": topic, "data": data or {}},
        )
        return APIResponse.from_dict(response)
    
    async def recommend(
        self,
        category: str,
        constraints: Dict[str, Any] = None,
    ) -> APIResponse[List[Dict]]:
        """Get recommendations from Master AI."""
        response = await self._request(
            HTTPMethod.POST,
            "/master-ai/recommend",
            data={
                "category": category,
                "constraints": constraints or {},
            },
        )
        return APIResponse.from_dict(response)


class KingAIClient:
    """
    King AI API Client SDK.
    
    Features:
    - Type-safe API access
    - Automatic retries
    - Error handling
    - Pagination support
    
    Example:
        client = KingAIClient(api_key="your-api-key")
        
        # List businesses
        businesses = await client.businesses.list()
        
        # Execute agent task
        result = await client.agents.execute(
            agent_id="research_agent",
            task="Analyze market trends for Q4",
        )
        
        # Query Master AI
        response = await client.master_ai.query(
            message="What are our top priorities?",
        )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        config: SDKConfig = None,
    ):
        self.config = config or SDKConfig(
            base_url=base_url,
            api_key=api_key,
        )
        
        # Initialize resources
        self.businesses = BusinessResource(self)
        self.agents = AgentResource(self)
        self.approvals = ApprovalResource(self)
        self.analytics = AnalyticsResource(self)
        self.goals = GoalsResource(self)
        self.master_ai = MasterAIResource(self)
    
    def _build_url(self, path: str) -> str:
        """Build full URL for request."""
        base = self.config.base_url.rstrip("/")
        version = self.config.api_version
        path = path.lstrip("/")
        return f"{base}/api/{version}/{path}"
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        return headers
    
    async def _request(
        self,
        method: HTTPMethod,
        path: str,
        data: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.
        
        Note: This is a mock implementation.
        In production, use aiohttp or httpx.
        """
        url = self._build_url(path)
        headers = self._build_headers()
        
        logger.debug(
            f"API request: {method.value} {url}",
            extra={"params": params, "has_body": data is not None},
        )
        
        # Mock implementation - replace with actual HTTP client
        # In production:
        # async with aiohttp.ClientSession() as session:
        #     async with session.request(
        #         method.value,
        #         url,
        #         json=data,
        #         params=params,
        #         headers=headers,
        #         timeout=self.config.timeout,
        #     ) as response:
        #         return await response.json()
        
        # For now, return mock response
        return {
            "success": True,
            "data": None,
            "message": "Mock response - implement HTTP client",
        }
    
    async def health_check(self) -> bool:
        """Check API health."""
        try:
            response = await self._request(HTTPMethod.GET, "/health")
            return response.get("status") == "healthy"
        except Exception:
            return False
    
    async def get_version(self) -> str:
        """Get API version."""
        response = await self._request(HTTPMethod.GET, "/version")
        return response.get("version", "unknown")


# Synchronous wrapper for non-async contexts
class SyncKingAIClient:
    """Synchronous wrapper for KingAIClient."""
    
    def __init__(self, *args, **kwargs):
        self._client = KingAIClient(*args, **kwargs)
        self._loop = asyncio.new_event_loop()
    
    def _run(self, coro):
        """Run coroutine synchronously."""
        return self._loop.run_until_complete(coro)
    
    def health_check(self) -> bool:
        return self._run(self._client.health_check())
    
    def close(self):
        """Close the client."""
        self._loop.close()


# Factory functions
def create_client(
    api_key: Optional[str] = None,
    base_url: str = "http://localhost:8000",
) -> KingAIClient:
    """Create an async API client."""
    return KingAIClient(api_key=api_key, base_url=base_url)


def create_sync_client(
    api_key: Optional[str] = None,
    base_url: str = "http://localhost:8000",
) -> SyncKingAIClient:
    """Create a synchronous API client."""
    return SyncKingAIClient(api_key=api_key, base_url=base_url)
