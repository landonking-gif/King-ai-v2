"""
Database Query Profiler.
Profile and analyze database query performance.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from functools import wraps
import time
import statistics
import hashlib
import re

from src.utils.structured_logging import get_logger

logger = get_logger("query_profiler")


class QueryType(str, Enum):
    """Types of SQL queries."""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    OTHER = "OTHER"


class QueryPerformance(str, Enum):
    """Performance classifications."""
    FAST = "fast"  # < 10ms
    NORMAL = "normal"  # 10-100ms
    SLOW = "slow"  # 100-1000ms
    VERY_SLOW = "very_slow"  # > 1000ms


@dataclass
class QueryProfile:
    """Profile data for a single query execution."""
    query_hash: str
    query_text: str
    query_type: QueryType
    execution_time_ms: float
    rows_affected: int = 0
    rows_returned: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Context
    caller: str = ""
    connection_id: str = ""
    
    # Analysis
    explain_plan: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    @property
    def performance(self) -> QueryPerformance:
        if self.execution_time_ms < 10:
            return QueryPerformance.FAST
        elif self.execution_time_ms < 100:
            return QueryPerformance.NORMAL
        elif self.execution_time_ms < 1000:
            return QueryPerformance.SLOW
        else:
            return QueryPerformance.VERY_SLOW


@dataclass
class QueryStats:
    """Aggregated statistics for a query pattern."""
    query_hash: str
    query_pattern: str
    query_type: QueryType
    
    # Execution stats
    call_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    std_dev_ms: float = 0.0
    
    # Row stats
    total_rows: int = 0
    avg_rows: float = 0.0
    
    # Timing
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    
    # Analysis
    slow_count: int = 0
    is_n_plus_one: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "query_pattern": self.query_pattern[:100] + "..." if len(self.query_pattern) > 100 else self.query_pattern,
            "query_type": self.query_type.value,
            "call_count": self.call_count,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2),
            "max_time_ms": round(self.max_time_ms, 2),
            "slow_count": self.slow_count,
            "is_n_plus_one": self.is_n_plus_one,
        }


@dataclass
class ProfileReport:
    """Query profiling report."""
    period_start: datetime
    period_end: datetime
    
    # Summary
    total_queries: int = 0
    total_time_ms: float = 0.0
    slow_queries: int = 0
    
    # By type
    queries_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Top queries
    slowest_queries: List[QueryStats] = field(default_factory=list)
    most_frequent: List[QueryStats] = field(default_factory=list)
    n_plus_one_candidates: List[QueryStats] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "summary": {
                "total_queries": self.total_queries,
                "total_time_ms": round(self.total_time_ms, 2),
                "slow_queries": self.slow_queries,
                "queries_by_type": self.queries_by_type,
            },
            "slowest_queries": [q.to_dict() for q in self.slowest_queries],
            "most_frequent": [q.to_dict() for q in self.most_frequent],
            "n_plus_one_candidates": [q.to_dict() for q in self.n_plus_one_candidates],
            "recommendations": self.recommendations,
        }


class QueryNormalizer:
    """Normalize SQL queries for pattern matching."""
    
    @staticmethod
    def normalize(query: str) -> str:
        """Normalize a query by replacing literals with placeholders."""
        # Remove extra whitespace
        normalized = " ".join(query.split())
        
        # Replace string literals
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        
        # Replace numeric literals
        normalized = re.sub(r"\b\d+\b", "?", normalized)
        
        # Replace IN lists
        normalized = re.sub(r"IN\s*\([^)]+\)", "IN (?)", normalized)
        
        return normalized
    
    @staticmethod
    def hash_query(query: str) -> str:
        """Create a hash of a normalized query."""
        normalized = QueryNormalizer.normalize(query)
        return hashlib.md5(normalized.encode()).hexdigest()[:12]
    
    @staticmethod
    def detect_type(query: str) -> QueryType:
        """Detect the query type."""
        query_upper = query.strip().upper()
        
        if query_upper.startswith("SELECT"):
            return QueryType.SELECT
        elif query_upper.startswith("INSERT"):
            return QueryType.INSERT
        elif query_upper.startswith("UPDATE"):
            return QueryType.UPDATE
        elif query_upper.startswith("DELETE"):
            return QueryType.DELETE
        else:
            return QueryType.OTHER


class QueryAnalyzer:
    """Analyze queries for potential issues."""
    
    def analyze(self, query: str) -> List[str]:
        """Analyze a query for potential issues."""
        warnings = []
        query_upper = query.upper()
        
        # Check for SELECT *
        if "SELECT *" in query_upper:
            warnings.append("Avoid SELECT * - specify columns explicitly")
        
        # Check for missing WHERE on UPDATE/DELETE
        if ("UPDATE " in query_upper or "DELETE " in query_upper) and "WHERE" not in query_upper:
            warnings.append("UPDATE/DELETE without WHERE clause")
        
        # Check for LIKE with leading wildcard
        if re.search(r"LIKE\s+'%", query_upper):
            warnings.append("LIKE with leading wildcard prevents index usage")
        
        # Check for missing LIMIT on SELECT
        if query_upper.startswith("SELECT") and "LIMIT" not in query_upper:
            if "COUNT" not in query_upper and "SUM" not in query_upper:
                warnings.append("Consider adding LIMIT to prevent large result sets")
        
        # Check for OR conditions (potential index issues)
        if re.search(r"\bOR\b.*\bOR\b", query_upper):
            warnings.append("Multiple OR conditions may prevent index usage")
        
        # Check for functions on indexed columns
        if re.search(r"WHERE\s+\w+\s*\([^)]*\)\s*[=<>]", query_upper):
            warnings.append("Functions in WHERE clause prevent index usage")
        
        # Check for implicit type conversion
        if re.search(r"=\s*'?\d+'?", query):
            if "'" in query:
                warnings.append("Possible implicit type conversion")
        
        return warnings


class QueryProfiler:
    """
    Database Query Profiler.
    
    Features:
    - Query timing
    - Pattern detection
    - N+1 query detection
    - Slow query identification
    - Performance recommendations
    """
    
    def __init__(
        self,
        slow_threshold_ms: float = 100,
        max_history: int = 10000,
    ):
        self.slow_threshold_ms = slow_threshold_ms
        self.max_history = max_history
        
        self.profiles: List[QueryProfile] = []
        self.stats: Dict[str, QueryStats] = {}
        self.normalizer = QueryNormalizer()
        self.analyzer = QueryAnalyzer()
        
        # N+1 detection
        self.recent_patterns: List[tuple] = []
        self.n_plus_one_window_ms = 1000
    
    def profile(
        self,
        query: str,
        execution_time_ms: float,
        rows_affected: int = 0,
        rows_returned: int = 0,
        caller: str = "",
        explain_plan: str = None,
    ) -> QueryProfile:
        """
        Profile a query execution.
        
        Args:
            query: The SQL query text
            execution_time_ms: Execution time in milliseconds
            rows_affected: Number of rows affected
            rows_returned: Number of rows returned
            caller: Calling function/method
            explain_plan: Query execution plan
            
        Returns:
            Query profile
        """
        query_hash = self.normalizer.hash_query(query)
        query_type = self.normalizer.detect_type(query)
        
        # Analyze query
        warnings = self.analyzer.analyze(query)
        
        profile = QueryProfile(
            query_hash=query_hash,
            query_text=query,
            query_type=query_type,
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            rows_returned=rows_returned,
            caller=caller,
            explain_plan=explain_plan,
            warnings=warnings,
        )
        
        # Store profile
        self.profiles.append(profile)
        if len(self.profiles) > self.max_history:
            self.profiles = self.profiles[-self.max_history:]
        
        # Update stats
        self._update_stats(profile)
        
        # Check for N+1
        self._check_n_plus_one(profile)
        
        # Log slow queries
        if execution_time_ms >= self.slow_threshold_ms:
            logger.warning(
                f"Slow query detected: {execution_time_ms:.2f}ms",
                extra={
                    "query_hash": query_hash,
                    "query_type": query_type.value,
                    "caller": caller,
                },
            )
        
        return profile
    
    def _update_stats(self, profile: QueryProfile) -> None:
        """Update aggregated stats for a query pattern."""
        query_hash = profile.query_hash
        
        if query_hash not in self.stats:
            self.stats[query_hash] = QueryStats(
                query_hash=query_hash,
                query_pattern=self.normalizer.normalize(profile.query_text),
                query_type=profile.query_type,
                first_seen=profile.timestamp,
            )
        
        stats = self.stats[query_hash]
        stats.call_count += 1
        stats.total_time_ms += profile.execution_time_ms
        stats.avg_time_ms = stats.total_time_ms / stats.call_count
        stats.min_time_ms = min(stats.min_time_ms, profile.execution_time_ms)
        stats.max_time_ms = max(stats.max_time_ms, profile.execution_time_ms)
        stats.total_rows += profile.rows_returned or profile.rows_affected
        stats.avg_rows = stats.total_rows / stats.call_count
        stats.last_seen = profile.timestamp
        
        if profile.execution_time_ms >= self.slow_threshold_ms:
            stats.slow_count += 1
    
    def _check_n_plus_one(self, profile: QueryProfile) -> None:
        """Check for N+1 query pattern."""
        now = datetime.utcnow()
        
        # Clean old patterns
        cutoff = now - timedelta(milliseconds=self.n_plus_one_window_ms)
        self.recent_patterns = [
            (t, h) for t, h in self.recent_patterns
            if t > cutoff
        ]
        
        # Add current pattern
        self.recent_patterns.append((now, profile.query_hash))
        
        # Check for repeated pattern
        pattern_counts = {}
        for _, hash_val in self.recent_patterns:
            pattern_counts[hash_val] = pattern_counts.get(hash_val, 0) + 1
        
        # If same query executed 10+ times in window, flag as N+1
        for hash_val, count in pattern_counts.items():
            if count >= 10 and hash_val in self.stats:
                self.stats[hash_val].is_n_plus_one = True
    
    def profile_context(self, caller: str = ""):
        """Context manager for profiling a query."""
        class ProfileContext:
            def __init__(ctx, profiler, caller):
                ctx.profiler = profiler
                ctx.caller = caller
                ctx.start_time = None
                ctx.query = None
            
            def __enter__(ctx):
                ctx.start_time = time.perf_counter()
                return ctx
            
            def set_query(ctx, query: str):
                ctx.query = query
            
            def __exit__(ctx, exc_type, exc_val, exc_tb):
                if ctx.query and ctx.start_time:
                    elapsed_ms = (time.perf_counter() - ctx.start_time) * 1000
                    ctx.profiler.profile(
                        query=ctx.query,
                        execution_time_ms=elapsed_ms,
                        caller=ctx.caller,
                    )
                return False
        
        return ProfileContext(self, caller)
    
    def get_report(
        self,
        hours: int = 24,
        top_n: int = 10,
    ) -> ProfileReport:
        """
        Generate a profiling report.
        
        Args:
            hours: Time window in hours
            top_n: Number of top queries to include
            
        Returns:
            Profiling report
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter profiles
        recent_profiles = [p for p in self.profiles if p.timestamp >= cutoff]
        
        if not recent_profiles:
            return ProfileReport(
                period_start=cutoff,
                period_end=datetime.utcnow(),
            )
        
        # Calculate summary
        total_queries = len(recent_profiles)
        total_time = sum(p.execution_time_ms for p in recent_profiles)
        slow_queries = sum(
            1 for p in recent_profiles
            if p.execution_time_ms >= self.slow_threshold_ms
        )
        
        # Queries by type
        by_type: Dict[str, int] = {}
        for p in recent_profiles:
            by_type[p.query_type.value] = by_type.get(p.query_type.value, 0) + 1
        
        # Get relevant stats
        relevant_hashes = set(p.query_hash for p in recent_profiles)
        relevant_stats = [
            self.stats[h] for h in relevant_hashes
            if h in self.stats
        ]
        
        # Top slowest
        slowest = sorted(
            relevant_stats,
            key=lambda s: s.avg_time_ms,
            reverse=True,
        )[:top_n]
        
        # Most frequent
        most_frequent = sorted(
            relevant_stats,
            key=lambda s: s.call_count,
            reverse=True,
        )[:top_n]
        
        # N+1 candidates
        n_plus_one = [s for s in relevant_stats if s.is_n_plus_one][:top_n]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            recent_profiles, relevant_stats
        )
        
        return ProfileReport(
            period_start=cutoff,
            period_end=datetime.utcnow(),
            total_queries=total_queries,
            total_time_ms=total_time,
            slow_queries=slow_queries,
            queries_by_type=by_type,
            slowest_queries=slowest,
            most_frequent=most_frequent,
            n_plus_one_candidates=n_plus_one,
            recommendations=recommendations,
        )
    
    def _generate_recommendations(
        self,
        profiles: List[QueryProfile],
        stats: List[QueryStats],
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Check for N+1 patterns
        n_plus_one_count = sum(1 for s in stats if s.is_n_plus_one)
        if n_plus_one_count > 0:
            recommendations.append(
                f"Found {n_plus_one_count} potential N+1 query patterns. "
                "Consider using eager loading or batch queries."
            )
        
        # Check for high-frequency queries
        high_freq = [s for s in stats if s.call_count > 100]
        if high_freq:
            recommendations.append(
                f"{len(high_freq)} queries called >100 times. "
                "Consider caching frequently accessed data."
            )
        
        # Check for slow queries
        slow = [s for s in stats if s.avg_time_ms > self.slow_threshold_ms]
        if slow:
            recommendations.append(
                f"{len(slow)} queries averaging >{self.slow_threshold_ms}ms. "
                "Review EXPLAIN plans and add indexes."
            )
        
        # Check for queries with warnings
        warned = [p for p in profiles if p.warnings]
        if len(warned) > len(profiles) * 0.1:
            recommendations.append(
                "Over 10% of queries have potential issues. "
                "Review query patterns and optimize."
            )
        
        return recommendations
    
    def get_slow_queries(
        self,
        limit: int = 20,
    ) -> List[QueryProfile]:
        """Get recent slow queries."""
        slow = [
            p for p in self.profiles
            if p.execution_time_ms >= self.slow_threshold_ms
        ]
        return sorted(
            slow,
            key=lambda p: p.execution_time_ms,
            reverse=True,
        )[:limit]
    
    def get_stats_for_query(self, query: str) -> Optional[QueryStats]:
        """Get stats for a specific query pattern."""
        query_hash = self.normalizer.hash_query(query)
        return self.stats.get(query_hash)
    
    def clear(self) -> None:
        """Clear all profiling data."""
        self.profiles.clear()
        self.stats.clear()
        self.recent_patterns.clear()


# Decorator for profiling database functions
def profile_query(profiler: QueryProfiler):
    """Decorator to profile database queries."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                # Extract query from args if available
                query = args[0] if args and isinstance(args[0], str) else str(func.__name__)
                profiler.profile(
                    query=query,
                    execution_time_ms=elapsed_ms,
                    caller=func.__name__,
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                query = args[0] if args and isinstance(args[0], str) else str(func.__name__)
                profiler.profile(
                    query=query,
                    execution_time_ms=elapsed_ms,
                    caller=func.__name__,
                )
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Global profiler instance
query_profiler = QueryProfiler()


def get_query_profiler() -> QueryProfiler:
    """Get the global query profiler."""
    return query_profiler
