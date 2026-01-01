"""
Agent Capability Registry.
Dynamic discovery and registration of agent capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Type
from enum import Enum
import inspect
import asyncio

from src.utils.structured_logging import get_logger

logger = get_logger("capability_registry")


class CapabilityCategory(str, Enum):
    """Categories of agent capabilities."""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    RESEARCH = "research"
    COMMERCE = "commerce"
    FINANCE = "finance"
    LEGAL = "legal"
    OPERATIONS = "operations"
    COMMUNICATION = "communication"
    CODE = "code"
    DATA = "data"


class CapabilityStatus(str, Enum):
    """Status of a capability."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


class InputType(str, Enum):
    """Types of input a capability can accept."""
    TEXT = "text"
    JSON = "json"
    FILE = "file"
    URL = "url"
    STRUCTURED = "structured"
    BINARY = "binary"


class OutputType(str, Enum):
    """Types of output a capability can produce."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    FILE = "file"
    STRUCTURED = "structured"


@dataclass
class ParameterSchema:
    """Schema for a capability parameter."""
    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityMetrics:
    """Metrics for a capability."""
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    total_duration_ms: int = 0
    last_invoked: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_invocations == 0:
            return 0.0
        return self.successful_invocations / self.total_invocations
    
    @property
    def avg_duration_ms(self) -> float:
        if self.successful_invocations == 0:
            return 0.0
        return self.total_duration_ms / self.successful_invocations


@dataclass
class Capability:
    """A registered agent capability."""
    id: str
    name: str
    description: str
    category: CapabilityCategory
    agent_type: str
    handler: Callable
    input_schema: List[ParameterSchema] = field(default_factory=list)
    output_type: OutputType = OutputType.JSON
    status: CapabilityStatus = CapabilityStatus.ACTIVE
    version: str = "1.0.0"
    requires_approval: bool = False
    cost_estimate: Optional[float] = None  # Estimated cost per invocation
    timeout_seconds: float = 300.0
    retry_config: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: CapabilityMetrics = field(default_factory=CapabilityMetrics)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches_query(self, query: str) -> bool:
        """Check if capability matches a search query."""
        query_lower = query.lower()
        
        if query_lower in self.name.lower():
            return True
        if query_lower in self.description.lower():
            return True
        if any(query_lower in tag.lower() for tag in self.tags):
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "agent_type": self.agent_type,
            "input_schema": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                }
                for p in self.input_schema
            ],
            "output_type": self.output_type.value,
            "status": self.status.value,
            "version": self.version,
            "requires_approval": self.requires_approval,
            "tags": list(self.tags),
            "metrics": {
                "invocations": self.metrics.total_invocations,
                "success_rate": round(self.metrics.success_rate, 2),
                "avg_duration_ms": round(self.metrics.avg_duration_ms, 2),
            },
        }


@dataclass
class CapabilityInvocation:
    """Record of a capability invocation."""
    capability_id: str
    input_params: Dict[str, Any]
    output: Any
    success: bool
    duration_ms: int
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CapabilityRegistry:
    """
    Registry for agent capabilities with discovery and invocation.
    
    Features:
    - Dynamic registration
    - Capability discovery
    - Schema validation
    - Metrics tracking
    - Version management
    """
    
    def __init__(self):
        self._capabilities: Dict[str, Capability] = {}
        self._by_category: Dict[CapabilityCategory, List[str]] = {}
        self._by_agent: Dict[str, List[str]] = {}
        self._invocation_history: List[CapabilityInvocation] = []
        self._max_history = 1000
    
    def register(
        self,
        capability_id: str,
        name: str,
        description: str,
        category: CapabilityCategory,
        agent_type: str,
        handler: Callable,
        input_schema: List[Dict] = None,
        output_type: OutputType = OutputType.JSON,
        requires_approval: bool = False,
        tags: List[str] = None,
        **kwargs,
    ) -> Capability:
        """Register a new capability."""
        # Parse input schema
        parsed_schema = []
        for param in (input_schema or []):
            parsed_schema.append(ParameterSchema(
                name=param.get("name", ""),
                type=param.get("type", "string"),
                description=param.get("description", ""),
                required=param.get("required", True),
                default=param.get("default"),
                constraints=param.get("constraints", {}),
            ))
        
        capability = Capability(
            id=capability_id,
            name=name,
            description=description,
            category=category,
            agent_type=agent_type,
            handler=handler,
            input_schema=parsed_schema,
            output_type=output_type,
            requires_approval=requires_approval,
            tags=set(tags or []),
            **kwargs,
        )
        
        self._capabilities[capability_id] = capability
        
        # Index by category
        if category not in self._by_category:
            self._by_category[category] = []
        self._by_category[category].append(capability_id)
        
        # Index by agent
        if agent_type not in self._by_agent:
            self._by_agent[agent_type] = []
        self._by_agent[agent_type].append(capability_id)
        
        logger.info(f"Registered capability: {name} ({capability_id})")
        
        return capability
    
    def register_from_class(
        self,
        agent_class: Type,
        agent_type: str,
        category: CapabilityCategory,
    ) -> List[Capability]:
        """Auto-register capabilities from an agent class."""
        registered = []
        
        for name, method in inspect.getmembers(agent_class, predicate=inspect.isfunction):
            # Skip private methods
            if name.startswith("_"):
                continue
            
            # Check for capability marker
            if hasattr(method, "_capability_config"):
                config = method._capability_config
                
                # Extract parameters from signature
                sig = inspect.signature(method)
                input_schema = []
                
                for param_name, param in sig.parameters.items():
                    if param_name in ("self", "cls"):
                        continue
                    
                    input_schema.append({
                        "name": param_name,
                        "type": str(param.annotation.__name__) if param.annotation != inspect.Parameter.empty else "any",
                        "required": param.default == inspect.Parameter.empty,
                        "default": None if param.default == inspect.Parameter.empty else param.default,
                    })
                
                capability = self.register(
                    capability_id=f"{agent_type}.{name}",
                    name=config.get("name", name),
                    description=config.get("description", method.__doc__ or ""),
                    category=category,
                    agent_type=agent_type,
                    handler=method,
                    input_schema=input_schema,
                    **{k: v for k, v in config.items() if k not in ("name", "description")},
                )
                
                registered.append(capability)
        
        return registered
    
    def unregister(self, capability_id: str) -> bool:
        """Unregister a capability."""
        capability = self._capabilities.pop(capability_id, None)
        
        if capability:
            # Remove from indexes
            if capability.category in self._by_category:
                self._by_category[capability.category] = [
                    c for c in self._by_category[capability.category]
                    if c != capability_id
                ]
            
            if capability.agent_type in self._by_agent:
                self._by_agent[capability.agent_type] = [
                    c for c in self._by_agent[capability.agent_type]
                    if c != capability_id
                ]
            
            logger.info(f"Unregistered capability: {capability_id}")
            return True
        
        return False
    
    def get(self, capability_id: str) -> Optional[Capability]:
        """Get a capability by ID."""
        return self._capabilities.get(capability_id)
    
    def list_all(
        self,
        status: Optional[CapabilityStatus] = None,
    ) -> List[Capability]:
        """List all capabilities."""
        capabilities = list(self._capabilities.values())
        
        if status:
            capabilities = [c for c in capabilities if c.status == status]
        
        return capabilities
    
    def list_by_category(
        self,
        category: CapabilityCategory,
    ) -> List[Capability]:
        """List capabilities by category."""
        capability_ids = self._by_category.get(category, [])
        return [self._capabilities[cid] for cid in capability_ids if cid in self._capabilities]
    
    def list_by_agent(
        self,
        agent_type: str,
    ) -> List[Capability]:
        """List capabilities by agent type."""
        capability_ids = self._by_agent.get(agent_type, [])
        return [self._capabilities[cid] for cid in capability_ids if cid in self._capabilities]
    
    def search(
        self,
        query: str,
        category: Optional[CapabilityCategory] = None,
        limit: int = 10,
    ) -> List[Capability]:
        """Search capabilities."""
        results = []
        
        for capability in self._capabilities.values():
            if category and capability.category != category:
                continue
            
            if capability.matches_query(query):
                results.append(capability)
        
        # Sort by relevance (simple: name match first)
        results.sort(key=lambda c: query.lower() in c.name.lower(), reverse=True)
        
        return results[:limit]
    
    async def invoke(
        self,
        capability_id: str,
        params: Dict[str, Any],
    ) -> Any:
        """Invoke a capability."""
        capability = self.get(capability_id)
        if not capability:
            raise ValueError(f"Capability not found: {capability_id}")
        
        if capability.status != CapabilityStatus.ACTIVE:
            raise ValueError(f"Capability is not active: {capability_id}")
        
        # Validate parameters
        self._validate_params(capability, params)
        
        # Execute
        start = datetime.utcnow()
        
        try:
            if asyncio.iscoroutinefunction(capability.handler):
                result = await capability.handler(**params)
            else:
                result = capability.handler(**params)
            
            duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
            
            # Record success
            self._record_invocation(capability, params, result, True, duration_ms)
            
            return result
        
        except Exception as e:
            duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
            
            # Record failure
            self._record_invocation(capability, params, None, False, duration_ms, str(e))
            
            raise
    
    def _validate_params(
        self,
        capability: Capability,
        params: Dict[str, Any],
    ) -> None:
        """Validate input parameters."""
        for schema in capability.input_schema:
            if schema.required and schema.name not in params:
                if schema.default is None:
                    raise ValueError(f"Missing required parameter: {schema.name}")
    
    def _record_invocation(
        self,
        capability: Capability,
        params: Dict[str, Any],
        result: Any,
        success: bool,
        duration_ms: int,
        error: Optional[str] = None,
    ) -> None:
        """Record an invocation for metrics."""
        # Update capability metrics
        capability.metrics.total_invocations += 1
        if success:
            capability.metrics.successful_invocations += 1
            capability.metrics.total_duration_ms += duration_ms
        else:
            capability.metrics.failed_invocations += 1
        capability.metrics.last_invoked = datetime.utcnow()
        
        # Store invocation record
        invocation = CapabilityInvocation(
            capability_id=capability.id,
            input_params=params,
            output=result if success else None,
            success=success,
            duration_ms=duration_ms,
            error=error,
        )
        
        self._invocation_history.append(invocation)
        
        # Limit history size
        while len(self._invocation_history) > self._max_history:
            self._invocation_history.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        active_count = sum(
            1 for c in self._capabilities.values()
            if c.status == CapabilityStatus.ACTIVE
        )
        
        total_invocations = sum(
            c.metrics.total_invocations
            for c in self._capabilities.values()
        )
        
        by_category = {
            cat.value: len(ids)
            for cat, ids in self._by_category.items()
        }
        
        return {
            "total_capabilities": len(self._capabilities),
            "active_capabilities": active_count,
            "total_invocations": total_invocations,
            "by_category": by_category,
            "agent_types": list(self._by_agent.keys()),
        }


# Decorator for marking methods as capabilities
def capability(
    name: Optional[str] = None,
    description: Optional[str] = None,
    requires_approval: bool = False,
    output_type: OutputType = OutputType.JSON,
    tags: List[str] = None,
    **kwargs,
):
    """Decorator to mark a method as a capability."""
    def decorator(func: Callable) -> Callable:
        func._capability_config = {
            "name": name or func.__name__,
            "description": description or func.__doc__ or "",
            "requires_approval": requires_approval,
            "output_type": output_type,
            "tags": tags or [],
            **kwargs,
        }
        return func
    
    return decorator


# Global registry instance
capability_registry = CapabilityRegistry()


def get_capability_registry() -> CapabilityRegistry:
    """Get the global capability registry."""
    return capability_registry


# Register built-in capabilities
def register_builtin_capabilities():
    """Register built-in capabilities."""
    
    # Research capability
    async def web_search(query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search the web for information."""
        return {"query": query, "results": [], "source": "web_search"}
    
    capability_registry.register(
        capability_id="research.web_search",
        name="Web Search",
        description="Search the web for information",
        category=CapabilityCategory.RESEARCH,
        agent_type="research",
        handler=web_search,
        input_schema=[
            {"name": "query", "type": "string", "description": "Search query", "required": True},
            {"name": "max_results", "type": "integer", "description": "Maximum results", "required": False, "default": 10},
        ],
        tags=["search", "web", "research"],
    )
    
    # Analytics capability
    async def analyze_data(data: Dict, analysis_type: str = "summary") -> Dict[str, Any]:
        """Analyze data and generate insights."""
        return {"analysis_type": analysis_type, "insights": []}
    
    capability_registry.register(
        capability_id="analytics.analyze_data",
        name="Data Analysis",
        description="Analyze data and generate insights",
        category=CapabilityCategory.ANALYSIS,
        agent_type="analytics",
        handler=analyze_data,
        input_schema=[
            {"name": "data", "type": "object", "description": "Data to analyze", "required": True},
            {"name": "analysis_type", "type": "string", "description": "Type of analysis", "required": False, "default": "summary"},
        ],
        tags=["analytics", "data", "insights"],
    )
    
    # Content capability
    async def generate_content(prompt: str, content_type: str = "blog") -> Dict[str, Any]:
        """Generate content based on prompt."""
        return {"content_type": content_type, "content": ""}
    
    capability_registry.register(
        capability_id="content.generate",
        name="Content Generation",
        description="Generate various types of content",
        category=CapabilityCategory.GENERATION,
        agent_type="content",
        handler=generate_content,
        input_schema=[
            {"name": "prompt", "type": "string", "description": "Content prompt", "required": True},
            {"name": "content_type", "type": "string", "description": "Type of content", "required": False, "default": "blog"},
        ],
        requires_approval=False,
        tags=["content", "generation", "writing"],
    )


# Auto-register built-ins
register_builtin_capabilities()
