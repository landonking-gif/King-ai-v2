
import redis.asyncio as redis
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from config.settings import settings

class KPIMonitor:
    """
    The 'Pulse' of the Empire.
    Tracks real-time metrics to drive autonomous evolution.
    """
    
    def __init__(self):
        redis_url = getattr(settings, "redis_url", "redis://localhost:6379")
        if not isinstance(redis_url, str):
            redis_url = "redis://localhost:6379"

        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.metrics_prefix = "king_ai:metrics:"
    
    async def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """
        Stored as specific time-series or simple counters in Redis.
        """
        timestamp = int(time.time())
        key = f"{self.metrics_prefix}{name}"
        
        # simple list push for now, could be a sorted set for time-series
        # format: timestamp:value
        await self.redis.lpush(key, f"{timestamp}:{value}")
        await self.redis.ltrim(key, 0, 999) # Keep last 1000 points
        
    async def get_recent_stats(self, name: str, seconds: int = 3600) -> Dict[str, float]:
        """
        Get aggregation of a metric over the last N seconds.
        """
        key = f"{self.metrics_prefix}{name}"
        raw_data = await self.redis.lrange(key, 0, -1)
        
        cutoff = int(time.time()) - seconds
        values = []
        
        for item in raw_data:
            ts, val = item.split(":")
            if int(ts) < cutoff:
                break # sorted by lpush (newest first), so we can stop
            values.append(float(val))
            
        if not values:
            return {"avg": 0.0, "max": 0.0, "min": 0.0, "count": 0}
            
        return {
            "avg": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
            "count": len(values)
        }

    async def get_system_health(self) -> str:
        """
        Returns a summarized report for the Evolution Engine.
        """
        latency = await self.get_recent_stats("api_latency", 600) # last 10 mins
        errors = await self.get_recent_stats("error_rate", 3600)  # last hour
        revenue = await self.get_recent_stats("revenue", 86400)   # last 24h
        
        report = f"""
        [SYSTEM HEALTH REPORT]
        - API Latency (10m avg): {latency['avg']:.4f}s
        - Error Rate (1h count): {errors['count']}
        - Revenue (24h): ${revenue['avg'] * revenue['count']:.2f}
        """
        
        # Logic to determine status
        if latency['avg'] > 1.0 or errors['count'] > 5:
            report += "\nSTATUS: DEGRADED (High Latency/Errors detected)"
        else:
            report += "\nSTATUS: HEALTHY"
            
        return report

kpi_monitor = KPIMonitor()
