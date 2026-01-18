"""
Model Tier Selector.

Intelligent model selection based on task complexity and budget.
Based on mother-harness orchestrator patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import re

from src.utils.structured_logging import get_logger

logger = get_logger("model_selector")


class ModelTier(str, Enum):
    """Model performance tiers."""
    FAST = "fast"           # Fastest, cheapest, good for simple tasks
    BALANCED = "balanced"   # Balance of speed and quality
    QUALITY = "quality"     # Higher quality, more expensive
    PREMIUM = "premium"     # Best quality, most expensive


class TaskComplexity(str, Enum):
    """Task complexity levels."""
    TRIVIAL = "trivial"     # Simple formatting, extraction
    SIMPLE = "simple"       # Basic Q&A, summarization
    MODERATE = "moderate"   # Analysis, comparison
    COMPLEX = "complex"     # Multi-step reasoning
    EXPERT = "expert"       # Domain expertise, novel solutions


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    tier: ModelTier
    provider: str  # "ollama", "openai", "anthropic", etc.
    
    # Capabilities
    max_tokens: int = 4096
    supports_tools: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False
    
    # Performance
    avg_latency_ms: int = 1000
    tokens_per_second: float = 50.0
    quality_score: float = 0.7  # 0-1 scale
    
    # Cost
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    
    # Availability
    is_local: bool = False
    requires_api_key: bool = True
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a request."""
        return (
            (input_tokens / 1000) * self.cost_per_1k_input +
            (output_tokens / 1000) * self.cost_per_1k_output
        )
    
    def estimate_latency_ms(
        self,
        output_tokens: int,
    ) -> int:
        """Estimate latency for a request."""
        generation_time = (output_tokens / self.tokens_per_second) * 1000
        return int(self.avg_latency_ms + generation_time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier.value,
            "provider": self.provider,
            "max_tokens": self.max_tokens,
            "supports_tools": self.supports_tools,
            "quality_score": self.quality_score,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "is_local": self.is_local,
        }


@dataclass
class TaskProfile:
    """Profile of a task for model selection."""
    task_type: str
    prompt_length: int = 0
    expected_output_length: int = 500
    requires_reasoning: bool = False
    requires_creativity: bool = False
    requires_precision: bool = False
    requires_tools: bool = False
    requires_vision: bool = False
    domain: Optional[str] = None
    priority: str = "normal"  # "low", "normal", "high"
    budget_limit: Optional[float] = None
    latency_limit_ms: Optional[int] = None
    
    def estimate_complexity(self) -> TaskComplexity:
        """Estimate task complexity."""
        score = 0
        
        # Prompt length affects complexity
        if self.prompt_length > 10000:
            score += 2
        elif self.prompt_length > 3000:
            score += 1
        
        # Requirements affect complexity
        if self.requires_reasoning:
            score += 2
        if self.requires_creativity:
            score += 1
        if self.requires_precision:
            score += 1
        if self.requires_tools:
            score += 1
        
        # Domain expertise
        if self.domain in ["legal", "medical", "finance"]:
            score += 2
        
        # Map score to complexity
        if score <= 1:
            return TaskComplexity.TRIVIAL
        elif score <= 2:
            return TaskComplexity.SIMPLE
        elif score <= 4:
            return TaskComplexity.MODERATE
        elif score <= 6:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.EXPERT


@dataclass
class SelectionResult:
    """Result of model selection."""
    selected_model: ModelConfig
    reason: str
    fallback_models: List[ModelConfig] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_latency_ms: int = 0
    complexity: TaskComplexity = TaskComplexity.SIMPLE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_model": self.selected_model.to_dict(),
            "reason": self.reason,
            "fallback_models": [m.name for m in self.fallback_models],
            "estimated_cost": self.estimated_cost,
            "estimated_latency_ms": self.estimated_latency_ms,
            "complexity": self.complexity.value,
        }


class ModelTierSelector:
    """
    Intelligent model selection.
    
    Features:
    - Complexity-based tier selection
    - Budget-aware decisions
    - Fallback chains
    - Performance tracking
    - Automatic tier upgrade after failures
    """
    
    # Default models by tier
    DEFAULT_MODELS = {
        ModelTier.FAST: [
            ModelConfig(
                name="llama3.2",
                tier=ModelTier.FAST,
                provider="ollama",
                max_tokens=8192,
                avg_latency_ms=500,
                tokens_per_second=80,
                quality_score=0.65,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                is_local=True,
                requires_api_key=False,
            ),
            ModelConfig(
                name="gpt-4o-mini",
                tier=ModelTier.FAST,
                provider="openai",
                max_tokens=16384,
                supports_tools=True,
                supports_json_mode=True,
                avg_latency_ms=800,
                tokens_per_second=100,
                quality_score=0.75,
                cost_per_1k_input=0.00015,
                cost_per_1k_output=0.0006,
                is_local=False,
                requires_api_key=True,
            ),
        ],
        ModelTier.BALANCED: [
            ModelConfig(
                name="llama3.1:70b",
                tier=ModelTier.BALANCED,
                provider="ollama",
                max_tokens=8192,
                avg_latency_ms=2000,
                tokens_per_second=30,
                quality_score=0.8,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                is_local=True,
                requires_api_key=False,
            ),
            ModelConfig(
                name="claude-3-5-haiku-latest",
                tier=ModelTier.BALANCED,
                provider="anthropic",
                max_tokens=200000,
                supports_tools=True,
                avg_latency_ms=1000,
                tokens_per_second=80,
                quality_score=0.82,
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.005,
                is_local=False,
                requires_api_key=True,
            ),
        ],
        ModelTier.QUALITY: [
            ModelConfig(
                name="gpt-4o",
                tier=ModelTier.QUALITY,
                provider="openai",
                max_tokens=128000,
                supports_tools=True,
                supports_vision=True,
                supports_json_mode=True,
                avg_latency_ms=2000,
                tokens_per_second=60,
                quality_score=0.9,
                cost_per_1k_input=0.0025,
                cost_per_1k_output=0.01,
                is_local=False,
                requires_api_key=True,
            ),
            ModelConfig(
                name="claude-3-5-sonnet-latest",
                tier=ModelTier.QUALITY,
                provider="anthropic",
                max_tokens=200000,
                supports_tools=True,
                supports_vision=True,
                avg_latency_ms=2000,
                tokens_per_second=50,
                quality_score=0.92,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                is_local=False,
                requires_api_key=True,
            ),
        ],
        ModelTier.PREMIUM: [
            ModelConfig(
                name="claude-sonnet-4-20250514",
                tier=ModelTier.PREMIUM,
                provider="anthropic",
                max_tokens=200000,
                supports_tools=True,
                supports_vision=True,
                avg_latency_ms=5000,
                tokens_per_second=30,
                quality_score=0.98,
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.075,
                is_local=False,
                requires_api_key=True,
            ),
            ModelConfig(
                name="o1",
                tier=ModelTier.PREMIUM,
                provider="openai",
                max_tokens=128000,
                avg_latency_ms=10000,
                tokens_per_second=20,
                quality_score=0.95,
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.06,
                is_local=False,
                requires_api_key=True,
            ),
        ],
    }
    
    # Complexity to tier mapping
    COMPLEXITY_TIER_MAP = {
        TaskComplexity.TRIVIAL: ModelTier.FAST,
        TaskComplexity.SIMPLE: ModelTier.FAST,
        TaskComplexity.MODERATE: ModelTier.BALANCED,
        TaskComplexity.COMPLEX: ModelTier.QUALITY,
        TaskComplexity.EXPERT: ModelTier.PREMIUM,
    }
    
    def __init__(
        self,
        available_providers: Set[str] = None,
        prefer_local: bool = True,
        default_budget_per_request: float = 0.10,
    ):
        """
        Initialize the selector.
        
        Args:
            available_providers: Providers that are available
            prefer_local: Prefer local models when quality is similar
            default_budget_per_request: Default max budget per request
        """
        self._models: Dict[ModelTier, List[ModelConfig]] = dict(self.DEFAULT_MODELS)
        self._available_providers = available_providers or {"ollama"}
        self._prefer_local = prefer_local
        self._default_budget = default_budget_per_request
        
        # Track model performance
        self._performance: Dict[str, Dict[str, Any]] = {}
        self._failure_counts: Dict[str, int] = {}
    
    def add_model(self, model: ModelConfig) -> None:
        """Add a model configuration."""
        if model.tier not in self._models:
            self._models[model.tier] = []
        self._models[model.tier].append(model)
    
    def set_available_providers(self, providers: Set[str]) -> None:
        """Set available providers."""
        self._available_providers = providers
    
    def select(
        self,
        task: TaskProfile,
        min_tier: ModelTier = None,
        max_tier: ModelTier = None,
        require_local: bool = False,
    ) -> SelectionResult:
        """
        Select the best model for a task.
        
        Args:
            task: Task profile
            min_tier: Minimum tier to consider
            max_tier: Maximum tier to consider
            require_local: Require local model
            
        Returns:
            Selection result
        """
        # Estimate complexity
        complexity = task.estimate_complexity()
        
        # Determine target tier
        target_tier = self.COMPLEXITY_TIER_MAP[complexity]
        
        # Apply tier constraints
        tier_order = [ModelTier.FAST, ModelTier.BALANCED, ModelTier.QUALITY, ModelTier.PREMIUM]
        target_idx = tier_order.index(target_tier)
        
        if min_tier:
            min_idx = tier_order.index(min_tier)
            target_idx = max(target_idx, min_idx)
        
        if max_tier:
            max_idx = tier_order.index(max_tier)
            target_idx = min(target_idx, max_idx)
        
        target_tier = tier_order[target_idx]
        
        # Find candidates
        candidates = self._get_candidates(
            target_tier,
            task,
            require_local,
        )
        
        if not candidates:
            # Try lower tiers
            for i in range(target_idx - 1, -1, -1):
                candidates = self._get_candidates(
                    tier_order[i],
                    task,
                    require_local,
                )
                if candidates:
                    break
        
        if not candidates:
            # Use first available model
            for tier in tier_order:
                if self._models.get(tier):
                    candidates = [m for m in self._models[tier] if self._is_available(m)]
                    if candidates:
                        break
        
        if not candidates:
            raise ValueError("No models available")
        
        # Score and sort candidates
        scored = []
        for model in candidates:
            score = self._score_model(model, task, complexity)
            scored.append((score, model))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Select best
        best_model = scored[0][1]
        fallbacks = [m for _, m in scored[1:4]]  # Up to 3 fallbacks
        
        # Estimate metrics
        estimated_cost = best_model.estimate_cost(
            task.prompt_length // 4,  # Rough token estimate
            task.expected_output_length // 4,
        )
        estimated_latency = best_model.estimate_latency_ms(
            task.expected_output_length // 4,
        )
        
        return SelectionResult(
            selected_model=best_model,
            reason=self._generate_reason(best_model, task, complexity),
            fallback_models=fallbacks,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            complexity=complexity,
        )
    
    def select_for_prompt(
        self,
        prompt: str,
        task_type: str = "general",
        **kwargs,
    ) -> SelectionResult:
        """
        Select model based on prompt analysis.
        
        Args:
            prompt: The prompt text
            task_type: Type of task
            **kwargs: Additional task profile fields
            
        Returns:
            Selection result
        """
        # Analyze prompt
        profile = self._analyze_prompt(prompt, task_type)
        
        # Apply any overrides
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        return self.select(profile)
    
    def record_success(
        self,
        model_name: str,
        latency_ms: int,
        tokens_used: int,
    ) -> None:
        """Record successful model use."""
        if model_name not in self._performance:
            self._performance[model_name] = {
                "successes": 0,
                "failures": 0,
                "total_latency_ms": 0,
                "total_tokens": 0,
            }
        
        self._performance[model_name]["successes"] += 1
        self._performance[model_name]["total_latency_ms"] += latency_ms
        self._performance[model_name]["total_tokens"] += tokens_used
        
        # Reset failure count on success
        self._failure_counts[model_name] = 0
    
    def record_failure(
        self,
        model_name: str,
        error: str,
    ) -> Optional[str]:
        """
        Record model failure and suggest upgrade.
        
        Returns:
            Name of upgraded model, or None
        """
        self._failure_counts[model_name] = self._failure_counts.get(model_name, 0) + 1
        
        if model_name not in self._performance:
            self._performance[model_name] = {
                "successes": 0,
                "failures": 0,
                "total_latency_ms": 0,
                "total_tokens": 0,
            }
        
        self._performance[model_name]["failures"] += 1
        
        # Suggest upgrade if too many failures
        if self._failure_counts[model_name] >= 2:
            return self._suggest_upgrade(model_name)
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        model_stats = {}
        for name, perf in self._performance.items():
            total = perf["successes"] + perf["failures"]
            model_stats[name] = {
                "success_rate": perf["successes"] / total if total > 0 else 0,
                "avg_latency_ms": perf["total_latency_ms"] / perf["successes"] if perf["successes"] > 0 else 0,
                "total_requests": total,
            }
        
        return {
            "models": model_stats,
            "available_providers": list(self._available_providers),
            "prefer_local": self._prefer_local,
        }
    
    # Private methods
    
    def _get_candidates(
        self,
        tier: ModelTier,
        task: TaskProfile,
        require_local: bool,
    ) -> List[ModelConfig]:
        """Get candidate models for a tier."""
        candidates = []
        
        for model in self._models.get(tier, []):
            # Check availability
            if not self._is_available(model):
                continue
            
            # Check local requirement
            if require_local and not model.is_local:
                continue
            
            # Check capability requirements
            if task.requires_tools and not model.supports_tools:
                continue
            if task.requires_vision and not model.supports_vision:
                continue
            
            # Check budget
            if task.budget_limit:
                estimated = model.estimate_cost(
                    task.prompt_length // 4,
                    task.expected_output_length // 4,
                )
                if estimated > task.budget_limit:
                    continue
            
            # Check latency
            if task.latency_limit_ms:
                estimated = model.estimate_latency_ms(
                    task.expected_output_length // 4,
                )
                if estimated > task.latency_limit_ms:
                    continue
            
            candidates.append(model)
        
        return candidates
    
    def _is_available(self, model: ModelConfig) -> bool:
        """Check if a model is available."""
        return model.provider in self._available_providers
    
    def _score_model(
        self,
        model: ModelConfig,
        task: TaskProfile,
        complexity: TaskComplexity,
    ) -> float:
        """Score a model for a task."""
        score = 0.0
        
        # Quality score (weighted by complexity)
        complexity_weights = {
            TaskComplexity.TRIVIAL: 0.3,
            TaskComplexity.SIMPLE: 0.5,
            TaskComplexity.MODERATE: 0.7,
            TaskComplexity.COMPLEX: 0.9,
            TaskComplexity.EXPERT: 1.0,
        }
        quality_weight = complexity_weights[complexity]
        score += model.quality_score * quality_weight * 50
        
        # Cost score (lower is better)
        estimated_cost = model.estimate_cost(
            task.prompt_length // 4,
            task.expected_output_length // 4,
        )
        if estimated_cost == 0:
            score += 20  # Bonus for free/local
        else:
            cost_score = max(0, 20 - (estimated_cost * 100))
            score += cost_score
        
        # Speed score
        estimated_latency = model.estimate_latency_ms(
            task.expected_output_length // 4,
        )
        if task.priority == "high":
            speed_score = max(0, 20 - (estimated_latency / 500))
        else:
            speed_score = max(0, 10 - (estimated_latency / 1000))
        score += speed_score
        
        # Local preference
        if self._prefer_local and model.is_local:
            score += 10
        
        # Historical performance
        perf = self._performance.get(model.name)
        if perf and perf["successes"] + perf["failures"] > 0:
            success_rate = perf["successes"] / (perf["successes"] + perf["failures"])
            score += success_rate * 10
        
        # Penalize recent failures
        failures = self._failure_counts.get(model.name, 0)
        score -= failures * 15
        
        return score
    
    def _analyze_prompt(
        self,
        prompt: str,
        task_type: str,
    ) -> TaskProfile:
        """Analyze prompt to create task profile."""
        # Detect reasoning requirement
        reasoning_patterns = [
            r'\banalyze\b', r'\bcompare\b', r'\bevaluate\b',
            r'\bexplain why\b', r'\breason\b', r'\bthink through\b',
            r'\bstep by step\b', r'\bbreakdown\b',
        ]
        requires_reasoning = any(
            re.search(pattern, prompt, re.IGNORECASE)
            for pattern in reasoning_patterns
        )
        
        # Detect creativity requirement
        creativity_patterns = [
            r'\bcreate\b', r'\bgenerate\b', r'\binvent\b',
            r'\bwrite\b.*\bstory\b', r'\bbrainstorm\b', r'\bimagine\b',
        ]
        requires_creativity = any(
            re.search(pattern, prompt, re.IGNORECASE)
            for pattern in creativity_patterns
        )
        
        # Detect precision requirement
        precision_patterns = [
            r'\bexact\b', r'\bprecise\b', r'\baccurate\b',
            r'\bcorrect\b', r'\bvalid\b', r'\bjson\b', r'\bformat\b',
        ]
        requires_precision = any(
            re.search(pattern, prompt, re.IGNORECASE)
            for pattern in precision_patterns
        )
        
        # Detect domain
        domain = None
        if re.search(r'\blegal\b|\blaw\b|\bcontract\b', prompt, re.IGNORECASE):
            domain = "legal"
        elif re.search(r'\bmedical\b|\bhealth\b|\bdiagnos', prompt, re.IGNORECASE):
            domain = "medical"
        elif re.search(r'\bfinance\b|\binvest\b|\bbudget\b', prompt, re.IGNORECASE):
            domain = "finance"
        
        return TaskProfile(
            task_type=task_type,
            prompt_length=len(prompt),
            requires_reasoning=requires_reasoning,
            requires_creativity=requires_creativity,
            requires_precision=requires_precision,
            domain=domain,
        )
    
    def _suggest_upgrade(self, model_name: str) -> Optional[str]:
        """Suggest an upgraded model after failures."""
        # Find current model tier
        current_model = None
        current_tier = None
        
        for tier, models in self._models.items():
            for model in models:
                if model.name == model_name:
                    current_model = model
                    current_tier = tier
                    break
        
        if not current_tier:
            return None
        
        # Get next tier
        tier_order = [ModelTier.FAST, ModelTier.BALANCED, ModelTier.QUALITY, ModelTier.PREMIUM]
        current_idx = tier_order.index(current_tier)
        
        if current_idx >= len(tier_order) - 1:
            return None  # Already at top tier
        
        next_tier = tier_order[current_idx + 1]
        candidates = [
            m for m in self._models.get(next_tier, [])
            if self._is_available(m)
        ]
        
        if candidates:
            return candidates[0].name
        
        return None
    
    def _generate_reason(
        self,
        model: ModelConfig,
        task: TaskProfile,
        complexity: TaskComplexity,
    ) -> str:
        """Generate explanation for selection."""
        reasons = []
        
        reasons.append(f"Task complexity: {complexity.value}")
        reasons.append(f"Selected tier: {model.tier.value}")
        
        if model.is_local:
            reasons.append("Using local model (cost-free)")
        else:
            reasons.append(f"Estimated cost: ${model.estimate_cost(task.prompt_length // 4, task.expected_output_length // 4):.4f}")
        
        if task.requires_reasoning:
            reasons.append("Task requires reasoning")
        
        return "; ".join(reasons)


# Global selector instance
_model_selector: Optional[ModelTierSelector] = None


def get_model_selector() -> ModelTierSelector:
    """Get or create the global model tier selector."""
    global _model_selector
    if _model_selector is None:
        _model_selector = ModelTierSelector()
    return _model_selector
