"""
OpenAPI Enhancements.
Enhanced OpenAPI documentation and schema generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Callable, Union
from enum import Enum
import inspect
import re

from src.utils.structured_logging import get_logger

logger = get_logger("openapi_enhancements")


class ParameterLocation(str, Enum):
    """OpenAPI parameter locations."""
    QUERY = "query"
    PATH = "path"
    HEADER = "header"
    COOKIE = "cookie"


class HTTPMethod(str, Enum):
    """HTTP methods."""
    GET = "get"
    POST = "post"
    PUT = "put"
    PATCH = "patch"
    DELETE = "delete"
    OPTIONS = "options"
    HEAD = "head"


@dataclass
class APIExample:
    """An example for API documentation."""
    name: str
    summary: str = ""
    description: str = ""
    value: Any = None
    external_value: str = ""


@dataclass
class APIParameter:
    """An API parameter."""
    name: str
    location: ParameterLocation
    description: str = ""
    required: bool = False
    deprecated: bool = False
    schema: Dict[str, Any] = field(default_factory=dict)
    example: Any = None
    examples: List[APIExample] = field(default_factory=list)


@dataclass
class APIResponse:
    """An API response definition."""
    status_code: int
    description: str
    content_type: str = "application/json"
    schema: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    examples: List[APIExample] = field(default_factory=list)


@dataclass
class APIEndpoint:
    """An API endpoint definition."""
    path: str
    method: HTTPMethod
    operation_id: str
    summary: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: List[APIParameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: List[APIResponse] = field(default_factory=list)
    security: List[Dict[str, List[str]]] = field(default_factory=list)
    deprecated: bool = False
    
    # Handler function
    handler: Optional[Callable] = None


@dataclass
class APITag:
    """An API tag for grouping endpoints."""
    name: str
    description: str = ""
    external_docs: Optional[Dict[str, str]] = None


@dataclass  
class SecurityScheme:
    """A security scheme definition."""
    name: str
    type: str  # apiKey, http, oauth2, openIdConnect
    description: str = ""
    
    # apiKey
    in_location: str = ""  # query, header, cookie
    key_name: str = ""
    
    # http
    scheme: str = ""  # bearer, basic
    bearer_format: str = ""
    
    # oauth2
    flows: Dict[str, Any] = field(default_factory=dict)


class SchemaGenerator:
    """Generate JSON Schema from Python types."""
    
    type_mappings = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        type(None): {"type": "null"},
    }
    
    @classmethod
    def from_type(cls, python_type: Type) -> Dict[str, Any]:
        """Generate schema from Python type."""
        # Handle basic types
        if python_type in cls.type_mappings:
            return cls.type_mappings[python_type].copy()
        
        # Handle Optional
        origin = getattr(python_type, "__origin__", None)
        if origin is Union:
            args = python_type.__args__
            if type(None) in args:
                # Optional type
                non_none_args = [a for a in args if a is not type(None)]
                if len(non_none_args) == 1:
                    schema = cls.from_type(non_none_args[0])
                    schema["nullable"] = True
                    return schema
        
        # Handle List[T]
        if origin is list:
            args = getattr(python_type, "__args__", ())
            item_type = args[0] if args else Any
            return {
                "type": "array",
                "items": cls.from_type(item_type) if item_type is not Any else {},
            }
        
        # Handle Dict[K, V]
        if origin is dict:
            args = getattr(python_type, "__args__", ())
            value_type = args[1] if len(args) > 1 else Any
            return {
                "type": "object",
                "additionalProperties": cls.from_type(value_type) if value_type is not Any else True,
            }
        
        # Handle Enum
        if inspect.isclass(python_type) and issubclass(python_type, Enum):
            return {
                "type": "string",
                "enum": [e.value for e in python_type],
            }
        
        # Handle dataclass
        if hasattr(python_type, "__dataclass_fields__"):
            return cls.from_dataclass(python_type)
        
        # Default to object
        return {"type": "object"}
    
    @classmethod
    def from_dataclass(cls, dataclass_type: Type) -> Dict[str, Any]:
        """Generate schema from dataclass."""
        properties = {}
        required = []
        
        for field_name, field_info in dataclass_type.__dataclass_fields__.items():
            field_type = field_info.type
            properties[field_name] = cls.from_type(field_type)
            
            # Check if required (no default value)
            if field_info.default is field_info.default_factory is type:
                required.append(field_name)
        
        schema = {
            "type": "object",
            "properties": properties,
        }
        
        if required:
            schema["required"] = required
        
        return schema
    
    @classmethod
    def from_pydantic(cls, model: Type) -> Dict[str, Any]:
        """Generate schema from Pydantic model."""
        if hasattr(model, "model_json_schema"):
            # Pydantic v2
            return model.model_json_schema()
        elif hasattr(model, "schema"):
            # Pydantic v1
            return model.schema()
        
        return {"type": "object"}


class OpenAPIEnhancer:
    """
    OpenAPI Documentation Enhancer.
    
    Features:
    - Schema generation from Python types
    - Endpoint documentation
    - Examples and descriptions
    - Security schemes
    - Tag organization
    """
    
    def __init__(
        self,
        title: str = "King AI API",
        version: str = "2.0.0",
        description: str = "",
    ):
        self.title = title
        self.version = version
        self.description = description
        
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.tags: Dict[str, APITag] = {}
        self.security_schemes: Dict[str, SecurityScheme] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}
        
        self._setup_default_schemas()
    
    def _setup_default_schemas(self) -> None:
        """Set up common schemas."""
        self.schemas["Error"] = {
            "type": "object",
            "properties": {
                "error": {"type": "string"},
                "message": {"type": "string"},
                "code": {"type": "integer"},
                "details": {"type": "object"},
            },
            "required": ["error", "message"],
        }
        
        self.schemas["Pagination"] = {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "minimum": 1},
                "page_size": {"type": "integer", "minimum": 1, "maximum": 100},
                "total": {"type": "integer"},
                "total_pages": {"type": "integer"},
            },
        }
        
        self.schemas["HealthStatus"] = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                "version": {"type": "string"},
                "uptime": {"type": "number"},
            },
        }
    
    def add_tag(
        self,
        name: str,
        description: str = "",
        external_docs_url: str = "",
    ) -> APITag:
        """Add an API tag."""
        tag = APITag(
            name=name,
            description=description,
            external_docs={"url": external_docs_url} if external_docs_url else None,
        )
        self.tags[name] = tag
        return tag
    
    def add_security_scheme(
        self,
        name: str,
        type: str,
        description: str = "",
        **kwargs: Any,
    ) -> SecurityScheme:
        """Add a security scheme."""
        scheme = SecurityScheme(
            name=name,
            type=type,
            description=description,
            in_location=kwargs.get("in_location", ""),
            key_name=kwargs.get("key_name", ""),
            scheme=kwargs.get("scheme", ""),
            bearer_format=kwargs.get("bearer_format", ""),
            flows=kwargs.get("flows", {}),
        )
        self.security_schemes[name] = scheme
        return scheme
    
    def add_schema(
        self,
        name: str,
        schema: Dict[str, Any],
    ) -> None:
        """Add a reusable schema."""
        self.schemas[name] = schema
    
    def register_schema(
        self,
        name: str,
        model: Type,
    ) -> Dict[str, Any]:
        """Register a schema from a Python type."""
        schema = SchemaGenerator.from_type(model)
        self.schemas[name] = schema
        return schema
    
    def document_endpoint(
        self,
        path: str,
        method: str,
        operation_id: str,
        summary: str,
        description: str = "",
        tags: List[str] = None,
        parameters: List[Dict[str, Any]] = None,
        request_body: Dict[str, Any] = None,
        responses: Dict[int, Dict[str, Any]] = None,
        security: List[str] = None,
        deprecated: bool = False,
    ) -> APIEndpoint:
        """
        Document an API endpoint.
        
        Args:
            path: API path
            method: HTTP method
            operation_id: Unique operation ID
            summary: Short summary
            description: Detailed description
            tags: Endpoint tags
            parameters: Path/query parameters
            request_body: Request body schema
            responses: Response definitions
            security: Security requirements
            deprecated: Whether endpoint is deprecated
            
        Returns:
            Documented endpoint
        """
        # Parse parameters
        api_params = []
        for param in (parameters or []):
            api_params.append(APIParameter(
                name=param["name"],
                location=ParameterLocation(param.get("in", "query")),
                description=param.get("description", ""),
                required=param.get("required", False),
                schema=param.get("schema", {}),
                example=param.get("example"),
            ))
        
        # Parse responses
        api_responses = []
        for status_code, resp in (responses or {}).items():
            api_responses.append(APIResponse(
                status_code=status_code,
                description=resp.get("description", ""),
                schema=resp.get("schema", {}),
                examples=[
                    APIExample(name=k, value=v)
                    for k, v in resp.get("examples", {}).items()
                ],
            ))
        
        # Add default error responses
        if not any(r.status_code >= 400 for r in api_responses):
            api_responses.append(APIResponse(
                status_code=400,
                description="Bad Request",
                schema={"$ref": "#/components/schemas/Error"},
            ))
            api_responses.append(APIResponse(
                status_code=500,
                description="Internal Server Error",
                schema={"$ref": "#/components/schemas/Error"},
            ))
        
        endpoint = APIEndpoint(
            path=path,
            method=HTTPMethod(method.lower()),
            operation_id=operation_id,
            summary=summary,
            description=description,
            tags=tags or [],
            parameters=api_params,
            request_body=request_body,
            responses=api_responses,
            security=[{s: []} for s in (security or [])],
            deprecated=deprecated,
        )
        
        endpoint_key = f"{method.upper()}:{path}"
        self.endpoints[endpoint_key] = endpoint
        
        return endpoint
    
    def endpoint(
        self,
        path: str,
        method: str = "GET",
        **kwargs: Any,
    ) -> Callable:
        """Decorator to document an endpoint."""
        def decorator(func: Callable) -> Callable:
            # Extract info from function
            operation_id = kwargs.get("operation_id", func.__name__)
            summary = kwargs.get("summary", func.__doc__.split("\n")[0] if func.__doc__ else "")
            description = kwargs.get("description", func.__doc__ or "")
            
            self.document_endpoint(
                path=path,
                method=method,
                operation_id=operation_id,
                summary=summary,
                description=description,
                **{k: v for k, v in kwargs.items() 
                   if k not in ["operation_id", "summary", "description"]},
            )
            
            return func
        return decorator
    
    def generate_spec(self) -> Dict[str, Any]:
        """
        Generate OpenAPI specification.
        
        Returns:
            OpenAPI 3.0 specification
        """
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": self.description,
            },
            "paths": {},
            "components": {
                "schemas": self.schemas,
                "securitySchemes": {},
            },
            "tags": [],
        }
        
        # Add tags
        for tag in self.tags.values():
            tag_obj = {
                "name": tag.name,
                "description": tag.description,
            }
            if tag.external_docs:
                tag_obj["externalDocs"] = tag.external_docs
            spec["tags"].append(tag_obj)
        
        # Add security schemes
        for scheme in self.security_schemes.values():
            scheme_obj = {
                "type": scheme.type,
                "description": scheme.description,
            }
            
            if scheme.type == "apiKey":
                scheme_obj["in"] = scheme.in_location
                scheme_obj["name"] = scheme.key_name
            elif scheme.type == "http":
                scheme_obj["scheme"] = scheme.scheme
                if scheme.bearer_format:
                    scheme_obj["bearerFormat"] = scheme.bearer_format
            elif scheme.type == "oauth2":
                scheme_obj["flows"] = scheme.flows
            
            spec["components"]["securitySchemes"][scheme.name] = scheme_obj
        
        # Add paths
        for endpoint in self.endpoints.values():
            if endpoint.path not in spec["paths"]:
                spec["paths"][endpoint.path] = {}
            
            operation = {
                "operationId": endpoint.operation_id,
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "deprecated": endpoint.deprecated,
                "responses": {},
            }
            
            # Parameters
            if endpoint.parameters:
                operation["parameters"] = [
                    {
                        "name": p.name,
                        "in": p.location.value,
                        "description": p.description,
                        "required": p.required,
                        "schema": p.schema,
                        **({"example": p.example} if p.example else {}),
                    }
                    for p in endpoint.parameters
                ]
            
            # Request body
            if endpoint.request_body:
                operation["requestBody"] = endpoint.request_body
            
            # Responses
            for resp in endpoint.responses:
                operation["responses"][str(resp.status_code)] = {
                    "description": resp.description,
                    "content": {
                        resp.content_type: {
                            "schema": resp.schema,
                            **({"examples": {
                                ex.name: {"value": ex.value}
                                for ex in resp.examples
                            }} if resp.examples else {}),
                        },
                    },
                }
            
            # Security
            if endpoint.security:
                operation["security"] = endpoint.security
            
            spec["paths"][endpoint.path][endpoint.method.value] = operation
        
        return spec
    
    def to_yaml(self) -> str:
        """Generate YAML OpenAPI spec."""
        try:
            import yaml
            return yaml.dump(self.generate_spec(), sort_keys=False)
        except ImportError:
            raise ImportError("PyYAML required for YAML output")
    
    def to_json(self, indent: int = 2) -> str:
        """Generate JSON OpenAPI spec."""
        import json
        return json.dumps(self.generate_spec(), indent=indent)


# Global OpenAPI enhancer
openapi_enhancer = OpenAPIEnhancer()


def get_openapi_enhancer() -> OpenAPIEnhancer:
    """Get the global OpenAPI enhancer."""
    return openapi_enhancer


# Convenience decorators
def api_endpoint(path: str, method: str = "GET", **kwargs):
    """Document an API endpoint."""
    return openapi_enhancer.endpoint(path, method, **kwargs)


def api_tag(name: str, description: str = ""):
    """Add an API tag."""
    return openapi_enhancer.add_tag(name, description)
