"""
API Versioning System.
Manages API versions with deprecation support.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Callable, Dict, List, Optional, Set, Type
from enum import Enum
from functools import wraps
import re

from src.utils.structured_logging import get_logger

logger = get_logger("api_versioning")


class VersionStatus(str, Enum):
    """API version lifecycle status."""
    ALPHA = "alpha"  # Experimental, may change
    BETA = "beta"  # Testing, mostly stable
    STABLE = "stable"  # Production-ready
    DEPRECATED = "deprecated"  # Still works, but will be removed
    SUNSET = "sunset"  # No longer available


class VersioningScheme(str, Enum):
    """API versioning schemes."""
    URL_PATH = "url_path"  # /v1/resource
    HEADER = "header"  # X-API-Version: 1
    QUERY_PARAM = "query_param"  # ?version=1
    ACCEPT_HEADER = "accept_header"  # Accept: application/vnd.api.v1+json


@dataclass
class APIVersion:
    """Represents an API version."""
    major: int
    minor: int = 0
    patch: int = 0
    status: VersionStatus = VersionStatus.STABLE
    release_date: Optional[date] = None
    deprecation_date: Optional[date] = None
    sunset_date: Optional[date] = None
    changelog: List[str] = field(default_factory=list)
    
    @classmethod
    def from_string(cls, version_str: str) -> "APIVersion":
        """Parse version from string like 'v1', 'v1.0', 'v1.0.0'."""
        match = re.match(r"v?(\d+)(?:\.(\d+))?(?:\.(\d+))?", version_str)
        if not match:
            raise ValueError(f"Invalid version string: {version_str}")
        
        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) else 0
        patch = int(match.group(3)) if match.group(3) else 0
        
        return cls(major=major, minor=minor, patch=patch)
    
    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"
    
    @property
    def short_version(self) -> str:
        """Get short version like 'v1' or 'v1.0'."""
        if self.patch == 0:
            if self.minor == 0:
                return f"v{self.major}"
            return f"v{self.major}.{self.minor}"
        return str(self)
    
    @property
    def is_deprecated(self) -> bool:
        if self.status == VersionStatus.DEPRECATED:
            return True
        if self.deprecation_date and date.today() >= self.deprecation_date:
            return True
        return False
    
    @property
    def is_sunset(self) -> bool:
        if self.status == VersionStatus.SUNSET:
            return True
        if self.sunset_date and date.today() >= self.sunset_date:
            return True
        return False
    
    def __lt__(self, other: "APIVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, APIVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))
    
    def compatible_with(self, other: "APIVersion") -> bool:
        """Check if this version is backwards compatible with other."""
        return self.major == other.major and self >= other


@dataclass
class VersionedEndpoint:
    """Configuration for a versioned endpoint."""
    path: str
    methods: List[str]
    min_version: APIVersion
    max_version: Optional[APIVersion] = None
    handler: Optional[Callable] = None
    deprecated_in: Optional[APIVersion] = None
    removed_in: Optional[APIVersion] = None
    replacement: Optional[str] = None  # Path to replacement endpoint
    breaking_changes: List[str] = field(default_factory=list)


@dataclass
class VersionInfo:
    """Information about available versions."""
    current: APIVersion
    available: List[APIVersion]
    deprecated: List[APIVersion]
    sunset: List[APIVersion]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current": str(self.current),
            "available": [str(v) for v in self.available],
            "deprecated": [str(v) for v in self.deprecated],
            "sunset": [str(v) for v in self.sunset],
        }


class VersionRegistry:
    """Registry of API versions and their endpoints."""
    
    def __init__(self):
        self._versions: Dict[str, APIVersion] = {}
        self._endpoints: Dict[str, List[VersionedEndpoint]] = {}
        self._default_version: Optional[APIVersion] = None
    
    def register_version(
        self,
        version: APIVersion,
        is_default: bool = False,
    ) -> None:
        """Register an API version."""
        key = version.short_version
        self._versions[key] = version
        
        if is_default:
            self._default_version = version
        
        logger.info(f"Registered API version: {version} (status: {version.status.value})")
    
    def register_endpoint(self, endpoint: VersionedEndpoint) -> None:
        """Register a versioned endpoint."""
        if endpoint.path not in self._endpoints:
            self._endpoints[endpoint.path] = []
        
        self._endpoints[endpoint.path].append(endpoint)
    
    def get_version(self, version_str: str) -> Optional[APIVersion]:
        """Get version by string."""
        # Try exact match first
        if version_str in self._versions:
            return self._versions[version_str]
        
        # Try parsing
        try:
            parsed = APIVersion.from_string(version_str)
            return self._versions.get(parsed.short_version)
        except ValueError:
            return None
    
    def get_endpoint_for_version(
        self,
        path: str,
        version: APIVersion,
    ) -> Optional[VersionedEndpoint]:
        """Get endpoint handler for a specific version."""
        endpoints = self._endpoints.get(path, [])
        
        for endpoint in endpoints:
            if endpoint.min_version <= version:
                if endpoint.max_version is None or version <= endpoint.max_version:
                    return endpoint
        
        return None
    
    @property
    def default_version(self) -> Optional[APIVersion]:
        return self._default_version
    
    @property
    def all_versions(self) -> List[APIVersion]:
        return sorted(self._versions.values(), reverse=True)
    
    @property
    def active_versions(self) -> List[APIVersion]:
        return [v for v in self._versions.values() if not v.is_sunset]


class APIVersioning:
    """
    API versioning system with deprecation support.
    
    Features:
    - Multiple versioning schemes
    - Version lifecycle management
    - Deprecation warnings
    - Version negotiation
    - Migration helpers
    """
    
    def __init__(
        self,
        scheme: VersioningScheme = VersioningScheme.URL_PATH,
        default_version: str = "v1",
        header_name: str = "X-API-Version",
        query_param: str = "version",
    ):
        self.scheme = scheme
        self.registry = VersionRegistry()
        self._header_name = header_name
        self._query_param = query_param
        
        # Register default version
        self.registry.register_version(
            APIVersion.from_string(default_version),
            is_default=True,
        )
    
    def add_version(
        self,
        version: str,
        status: VersionStatus = VersionStatus.STABLE,
        release_date: Optional[date] = None,
        deprecation_date: Optional[date] = None,
        sunset_date: Optional[date] = None,
        changelog: List[str] = None,
    ) -> APIVersion:
        """Add a new API version."""
        api_version = APIVersion.from_string(version)
        api_version.status = status
        api_version.release_date = release_date
        api_version.deprecation_date = deprecation_date
        api_version.sunset_date = sunset_date
        api_version.changelog = changelog or []
        
        self.registry.register_version(api_version)
        return api_version
    
    def deprecate_version(
        self,
        version: str,
        sunset_date: date,
        replacement: Optional[str] = None,
    ) -> None:
        """Mark a version as deprecated."""
        api_version = self.registry.get_version(version)
        if api_version:
            api_version.status = VersionStatus.DEPRECATED
            api_version.deprecation_date = date.today()
            api_version.sunset_date = sunset_date
            
            logger.warning(
                f"API version {version} deprecated, sunset: {sunset_date}"
            )
    
    def extract_version(
        self,
        request_path: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
    ) -> Optional[APIVersion]:
        """Extract version from request based on scheme."""
        version_str = None
        
        if self.scheme == VersioningScheme.URL_PATH and request_path:
            match = re.match(r"/v(\d+(?:\.\d+)?)/", request_path)
            if match:
                version_str = f"v{match.group(1)}"
        
        elif self.scheme == VersioningScheme.HEADER and headers:
            version_str = headers.get(self._header_name)
        
        elif self.scheme == VersioningScheme.QUERY_PARAM and query_params:
            version_str = query_params.get(self._query_param)
        
        elif self.scheme == VersioningScheme.ACCEPT_HEADER and headers:
            accept = headers.get("Accept", "")
            match = re.search(r"vnd\.api\.v(\d+)", accept)
            if match:
                version_str = f"v{match.group(1)}"
        
        if version_str:
            return self.registry.get_version(version_str)
        
        return self.registry.default_version
    
    def get_deprecation_headers(
        self,
        version: APIVersion,
    ) -> Dict[str, str]:
        """Get deprecation warning headers."""
        headers = {}
        
        if version.is_deprecated:
            headers["Deprecation"] = "true"
            
            if version.sunset_date:
                headers["Sunset"] = version.sunset_date.isoformat()
            
            # Find newest non-deprecated version
            for v in self.registry.all_versions:
                if not v.is_deprecated:
                    headers["Link"] = f'</api/{v.short_version}/>; rel="successor-version"'
                    break
        
        return headers
    
    def get_version_info(self) -> VersionInfo:
        """Get information about all versions."""
        all_versions = self.registry.all_versions
        
        return VersionInfo(
            current=self.registry.default_version or all_versions[0],
            available=[v for v in all_versions if not v.is_sunset],
            deprecated=[v for v in all_versions if v.is_deprecated and not v.is_sunset],
            sunset=[v for v in all_versions if v.is_sunset],
        )
    
    def version_route(
        self,
        min_version: str = "v1",
        max_version: Optional[str] = None,
        deprecated_in: Optional[str] = None,
    ):
        """Decorator for versioned routes."""
        min_v = APIVersion.from_string(min_version)
        max_v = APIVersion.from_string(max_version) if max_version else None
        dep_v = APIVersion.from_string(deprecated_in) if deprecated_in else None
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Version checking would happen in middleware
                return await func(*args, **kwargs)
            
            # Store version info on function
            wrapper._version_info = {
                "min": min_v,
                "max": max_v,
                "deprecated_in": dep_v,
            }
            
            return wrapper
        
        return decorator


# FastAPI integration
def create_versioned_app(versioning: APIVersioning):
    """Create versioned FastAPI application structure."""
    from fastapi import FastAPI, APIRouter, Request, Response
    from fastapi.responses import JSONResponse
    
    app = FastAPI()
    
    @app.middleware("http")
    async def version_middleware(request: Request, call_next) -> Response:
        """Extract and validate API version."""
        # Extract version
        version = versioning.extract_version(
            request_path=str(request.url.path),
            headers=dict(request.headers),
            query_params=dict(request.query_params),
        )
        
        if not version:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid or missing API version"},
            )
        
        # Check if sunset
        if version.is_sunset:
            return JSONResponse(
                status_code=410,
                content={
                    "error": f"API version {version} is no longer available",
                    "sunset_date": version.sunset_date.isoformat() if version.sunset_date else None,
                },
            )
        
        # Store version in request state
        request.state.api_version = version
        
        # Process request
        response = await call_next(request)
        
        # Add deprecation headers
        deprecation_headers = versioning.get_deprecation_headers(version)
        for header, value in deprecation_headers.items():
            response.headers[header] = value
        
        # Add current version header
        response.headers["X-API-Version"] = str(version)
        
        return response
    
    @app.get("/api/versions")
    async def get_versions():
        """Get available API versions."""
        return versioning.get_version_info().to_dict()
    
    return app


# Global versioning instance
api_versioning = APIVersioning(
    scheme=VersioningScheme.URL_PATH,
    default_version="v1",
)


def get_api_versioning() -> APIVersioning:
    """Get the global API versioning instance."""
    return api_versioning


# Initialize with standard versions
def setup_default_versions():
    """Set up default API versions."""
    from datetime import date, timedelta
    
    # V1 - stable
    api_versioning.add_version(
        "v1",
        status=VersionStatus.STABLE,
        release_date=date(2024, 1, 1),
        changelog=[
            "Initial release",
            "Core business management APIs",
            "Agent execution endpoints",
            "Approval workflow",
        ],
    )
    
    # V2 - beta
    api_versioning.add_version(
        "v2",
        status=VersionStatus.BETA,
        release_date=date(2024, 6, 1),
        changelog=[
            "Improved response formats",
            "Batch operations",
            "WebSocket support",
            "Enhanced analytics",
        ],
    )


# Auto-setup
setup_default_versions()
