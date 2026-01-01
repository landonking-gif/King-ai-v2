"""
A/B Testing Framework.
Statistical testing for marketing and product experiments.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import random
import math
import statistics
import hashlib

from src.utils.structured_logging import get_logger

logger = get_logger("ab_testing")


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class VariantType(str, Enum):
    """Type of variant."""
    CONTROL = "control"
    TREATMENT = "treatment"


class MetricType(str, Enum):
    """Type of metric."""
    CONVERSION = "conversion"  # Binary: converted or not
    COUNT = "count"  # Count of events
    CONTINUOUS = "continuous"  # Continuous values (revenue, time, etc.)
    RATIO = "ratio"  # Ratio metrics


class AssignmentStrategy(str, Enum):
    """How to assign users to variants."""
    RANDOM = "random"
    HASH = "hash"  # Deterministic based on user ID
    WEIGHTED = "weighted"


@dataclass
class Variant:
    """An experiment variant (control or treatment)."""
    id: str
    name: str
    type: VariantType
    weight: float = 0.5  # Traffic allocation
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    participants: int = 0
    conversions: int = 0
    total_value: float = 0.0
    values: List[float] = field(default_factory=list)
    
    @property
    def conversion_rate(self) -> float:
        if self.participants == 0:
            return 0.0
        return self.conversions / self.participants
    
    @property
    def mean_value(self) -> float:
        if not self.values:
            return 0.0
        return statistics.mean(self.values)
    
    @property
    def std_value(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return statistics.stdev(self.values)


@dataclass
class Metric:
    """A metric to track in the experiment."""
    name: str
    type: MetricType
    primary: bool = True
    minimum_detectable_effect: float = 0.05  # 5% minimum effect to detect
    
    def calculate(self, variant: Variant) -> float:
        """Calculate metric value for a variant."""
        if self.type == MetricType.CONVERSION:
            return variant.conversion_rate
        elif self.type == MetricType.CONTINUOUS:
            return variant.mean_value
        elif self.type == MetricType.COUNT:
            return variant.total_value / max(variant.participants, 1)
        else:
            return variant.conversion_rate


@dataclass
class StatisticalResult:
    """Statistical analysis result."""
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float  # Percentage improvement
    p_value: float
    confidence_interval: tuple  # (lower, upper)
    is_significant: bool
    confidence_level: float
    sample_size_control: int
    sample_size_treatment: int
    power: float = 0.0
    

@dataclass
class Experiment:
    """An A/B test experiment."""
    id: str
    name: str
    description: str = ""
    hypothesis: str = ""
    
    # Configuration
    variants: List[Variant] = field(default_factory=list)
    metrics: List[Metric] = field(default_factory=list)
    assignment_strategy: AssignmentStrategy = AssignmentStrategy.HASH
    
    # Targeting
    audience_filter: Optional[Callable[[Dict], bool]] = None
    traffic_percentage: float = 1.0  # Percentage of eligible users
    
    # Timing
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Status
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Assignments
    user_assignments: Dict[str, str] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        if self.status != ExperimentStatus.RUNNING:
            return False
        now = datetime.utcnow()
        if self.start_date and now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False
        return True
    
    @property
    def control(self) -> Optional[Variant]:
        return next((v for v in self.variants if v.type == VariantType.CONTROL), None)
    
    @property
    def treatment(self) -> Optional[Variant]:
        return next((v for v in self.variants if v.type == VariantType.TREATMENT), None)


class StatisticalCalculator:
    """Statistical calculations for A/B testing."""
    
    @staticmethod
    def z_score(p: float) -> float:
        """Approximate inverse normal CDF (z-score)."""
        # Approximation for inverse normal CDF
        if p <= 0:
            return -4.0
        if p >= 1:
            return 4.0
        
        # Rational approximation
        a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ]
        b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ]
        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ]
        
        p_low = 0.02425
        p_high = 1 - p_low
        
        if p < p_low:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        else:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    
    @staticmethod
    def normal_cdf(z: float) -> float:
        """Cumulative distribution function for standard normal."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    @classmethod
    def two_proportion_z_test(
        cls,
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
    ) -> tuple:
        """
        Two-proportion z-test for conversion rate comparison.
        Returns (z_stat, p_value).
        """
        if control_total == 0 or treatment_total == 0:
            return 0.0, 1.0
        
        p1 = control_conversions / control_total
        p2 = treatment_conversions / treatment_total
        
        # Pooled proportion
        p_pool = (control_conversions + treatment_conversions) / (control_total + treatment_total)
        
        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total))
        
        if se == 0:
            return 0.0, 1.0
        
        # Z-statistic
        z = (p2 - p1) / se
        
        # Two-tailed p-value
        p_value = 2 * (1 - cls.normal_cdf(abs(z)))
        
        return z, p_value
    
    @classmethod
    def welch_t_test(
        cls,
        control_values: List[float],
        treatment_values: List[float],
    ) -> tuple:
        """
        Welch's t-test for continuous metrics.
        Returns (t_stat, p_value).
        """
        n1 = len(control_values)
        n2 = len(treatment_values)
        
        if n1 < 2 or n2 < 2:
            return 0.0, 1.0
        
        mean1 = statistics.mean(control_values)
        mean2 = statistics.mean(treatment_values)
        var1 = statistics.variance(control_values)
        var2 = statistics.variance(treatment_values)
        
        # Standard error
        se = math.sqrt(var1/n1 + var2/n2)
        
        if se == 0:
            return 0.0, 1.0
        
        # T-statistic
        t = (mean2 - mean1) / se
        
        # Degrees of freedom (Welch-Satterthwaite)
        num = (var1/n1 + var2/n2) ** 2
        denom = (var1/n1)**2 / (n1-1) + (var2/n2)**2 / (n2-1)
        df = num / denom if denom > 0 else 1
        
        # Approximate p-value using normal distribution for large samples
        p_value = 2 * (1 - cls.normal_cdf(abs(t)))
        
        return t, p_value
    
    @classmethod
    def calculate_sample_size(
        cls,
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> int:
        """Calculate required sample size per variant."""
        if baseline_rate <= 0 or baseline_rate >= 1:
            return 1000  # Default
        
        # Effect size
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        # Z-scores
        z_alpha = cls.z_score(1 - alpha/2)
        z_beta = cls.z_score(power)
        
        # Pooled variance
        p_avg = (p1 + p2) / 2
        
        # Sample size formula
        numerator = (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) + 
                    z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p2 - p1) ** 2
        
        if denominator == 0:
            return 1000
        
        n = numerator / denominator
        
        return int(math.ceil(n))


class ABTestingFramework:
    """
    A/B Testing Framework.
    
    Features:
    - Create and manage experiments
    - User assignment (random or hash-based)
    - Statistical analysis
    - Automatic significance detection
    """
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.calculator = StatisticalCalculator()
    
    def create_experiment(
        self,
        name: str,
        control_name: str = "Control",
        treatment_name: str = "Treatment",
        control_config: Dict[str, Any] = None,
        treatment_config: Dict[str, Any] = None,
        traffic_split: float = 0.5,
        minimum_detectable_effect: float = 0.05,
        **kwargs,
    ) -> Experiment:
        """
        Create a new A/B test experiment.
        
        Args:
            name: Experiment name
            control_name: Name for control variant
            treatment_name: Name for treatment variant
            control_config: Configuration for control
            treatment_config: Configuration for treatment
            traffic_split: Percentage of traffic to treatment (default 50%)
            minimum_detectable_effect: Minimum effect to detect
            
        Returns:
            Created experiment
        """
        exp_id = f"exp_{name.lower().replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d')}"
        
        variants = [
            Variant(
                id=f"{exp_id}_control",
                name=control_name,
                type=VariantType.CONTROL,
                weight=1 - traffic_split,
                config=control_config or {},
            ),
            Variant(
                id=f"{exp_id}_treatment",
                name=treatment_name,
                type=VariantType.TREATMENT,
                weight=traffic_split,
                config=treatment_config or {},
            ),
        ]
        
        metrics = [
            Metric(
                name="conversion_rate",
                type=MetricType.CONVERSION,
                primary=True,
                minimum_detectable_effect=minimum_detectable_effect,
            ),
        ]
        
        experiment = Experiment(
            id=exp_id,
            name=name,
            variants=variants,
            metrics=metrics,
            **kwargs,
        )
        
        self.experiments[exp_id] = experiment
        logger.info(f"Created experiment: {name}", extra={"experiment_id": exp_id})
        
        return experiment
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.utcnow()
        logger.info(f"Started experiment: {experiment.name}")
        return True
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.STOPPED
        experiment.end_date = datetime.utcnow()
        logger.info(f"Stopped experiment: {experiment.name}")
        return True
    
    def get_variant(
        self,
        experiment_id: str,
        user_id: str,
        user_attributes: Dict[str, Any] = None,
    ) -> Optional[Variant]:
        """
        Get variant assignment for a user.
        
        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            user_attributes: Optional user attributes for targeting
            
        Returns:
            Assigned variant or None if not eligible
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment or not experiment.is_active:
            return None
        
        # Check audience filter
        if experiment.audience_filter and user_attributes:
            if not experiment.audience_filter(user_attributes):
                return None
        
        # Check traffic percentage
        if experiment.traffic_percentage < 1.0:
            hash_val = int(hashlib.md5(
                f"{experiment_id}:{user_id}:traffic".encode()
            ).hexdigest(), 16) % 100
            if hash_val >= experiment.traffic_percentage * 100:
                return None
        
        # Check existing assignment
        if user_id in experiment.user_assignments:
            variant_id = experiment.user_assignments[user_id]
            return next((v for v in experiment.variants if v.id == variant_id), None)
        
        # Assign variant
        variant = self._assign_variant(experiment, user_id)
        if variant:
            experiment.user_assignments[user_id] = variant.id
            variant.participants += 1
        
        return variant
    
    def _assign_variant(
        self,
        experiment: Experiment,
        user_id: str,
    ) -> Optional[Variant]:
        """Assign user to a variant."""
        if experiment.assignment_strategy == AssignmentStrategy.HASH:
            # Deterministic hash-based assignment
            hash_val = int(hashlib.md5(
                f"{experiment.id}:{user_id}".encode()
            ).hexdigest(), 16) % 100
            
            cumulative = 0
            for variant in experiment.variants:
                cumulative += variant.weight * 100
                if hash_val < cumulative:
                    return variant
            
            return experiment.variants[-1]
        
        elif experiment.assignment_strategy == AssignmentStrategy.RANDOM:
            # Random assignment
            rand_val = random.random()
            cumulative = 0
            for variant in experiment.variants:
                cumulative += variant.weight
                if rand_val < cumulative:
                    return variant
            
            return experiment.variants[-1]
        
        return experiment.control
    
    def record_conversion(
        self,
        experiment_id: str,
        user_id: str,
        value: float = 1.0,
    ) -> bool:
        """
        Record a conversion for a user.
        
        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            value: Conversion value (default 1.0)
            
        Returns:
            True if recorded successfully
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return False
        
        variant_id = experiment.user_assignments.get(user_id)
        if not variant_id:
            return False
        
        variant = next((v for v in experiment.variants if v.id == variant_id), None)
        if not variant:
            return False
        
        variant.conversions += 1
        variant.total_value += value
        variant.values.append(value)
        
        return True
    
    def analyze_experiment(
        self,
        experiment_id: str,
        confidence_level: float = 0.95,
    ) -> Optional[StatisticalResult]:
        """
        Perform statistical analysis on an experiment.
        
        Args:
            experiment_id: Experiment ID
            confidence_level: Confidence level for significance
            
        Returns:
            Statistical analysis result
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return None
        
        control = experiment.control
        treatment = experiment.treatment
        
        if not control or not treatment:
            return None
        
        # Calculate z-test for conversion rate
        z_stat, p_value = self.calculator.two_proportion_z_test(
            control.conversions,
            control.participants,
            treatment.conversions,
            treatment.participants,
        )
        
        control_rate = control.conversion_rate
        treatment_rate = treatment.conversion_rate
        
        absolute_lift = treatment_rate - control_rate
        relative_lift = (absolute_lift / control_rate * 100) if control_rate > 0 else 0
        
        # Confidence interval
        alpha = 1 - confidence_level
        z_alpha = self.calculator.z_score(1 - alpha/2)
        
        # Standard error of difference
        if control.participants > 0 and treatment.participants > 0:
            se = math.sqrt(
                control_rate * (1 - control_rate) / control.participants +
                treatment_rate * (1 - treatment_rate) / treatment.participants
            )
            ci_lower = absolute_lift - z_alpha * se
            ci_upper = absolute_lift + z_alpha * se
        else:
            ci_lower = ci_upper = 0
        
        is_significant = p_value < alpha
        
        result = StatisticalResult(
            control_mean=control_rate,
            treatment_mean=treatment_rate,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            confidence_level=confidence_level,
            sample_size_control=control.participants,
            sample_size_treatment=treatment.participants,
        )
        
        logger.info(
            f"Analyzed experiment {experiment.name}",
            extra={
                "p_value": p_value,
                "is_significant": is_significant,
                "relative_lift": relative_lift,
            },
        )
        
        return result
    
    def get_required_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float = 0.05,
        power: float = 0.8,
    ) -> int:
        """Calculate required sample size per variant."""
        return self.calculator.calculate_sample_size(
            baseline_rate=baseline_rate,
            minimum_detectable_effect=minimum_detectable_effect,
            power=power,
        )
    
    def get_experiment_summary(
        self,
        experiment_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a summary of an experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return None
        
        result = self.analyze_experiment(experiment_id)
        
        return {
            "id": experiment.id,
            "name": experiment.name,
            "status": experiment.status.value,
            "start_date": experiment.start_date.isoformat() if experiment.start_date else None,
            "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
            "variants": [
                {
                    "id": v.id,
                    "name": v.name,
                    "type": v.type.value,
                    "participants": v.participants,
                    "conversions": v.conversions,
                    "conversion_rate": round(v.conversion_rate * 100, 2),
                }
                for v in experiment.variants
            ],
            "analysis": {
                "relative_lift": round(result.relative_lift, 2) if result else None,
                "p_value": round(result.p_value, 4) if result else None,
                "is_significant": result.is_significant if result else None,
            } if result else None,
        }


# Global A/B testing framework instance
ab_framework = ABTestingFramework()


def get_ab_framework() -> ABTestingFramework:
    """Get the global A/B testing framework."""
    return ab_framework
