"""
Mock Objects for Testing.
Provides mock implementations of external services and dependencies.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union
from unittest.mock import MagicMock, AsyncMock
import json


class MockLLMClient:
    """
    Mock LLM client for testing.
    
    Usage:
        llm = MockLLMClient()
        llm.set_response("Hello, world!")
        result = await llm.generate("prompt")
    """
    
    def __init__(self):
        self.responses: List[str] = []
        self.response_index = 0
        self.calls: List[Dict[str, Any]] = []
        self.default_response = "Mock LLM response"
        self.raise_error: Optional[Exception] = None
        self.delay: float = 0.0
    
    def set_response(self, response: Union[str, List[str]]) -> None:
        """Set the response(s) to return."""
        if isinstance(response, str):
            self.responses = [response]
        else:
            self.responses = response
        self.response_index = 0
    
    def set_error(self, error: Exception) -> None:
        """Set an error to raise on next call."""
        self.raise_error = error
    
    def set_delay(self, delay: float) -> None:
        """Set delay before responding."""
        self.delay = delay
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> str:
        """Mock generate method."""
        # Record call
        self.calls.append({
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
            "timestamp": datetime.utcnow(),
        })
        
        # Simulate delay
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        # Raise error if set
        if self.raise_error:
            error = self.raise_error
            self.raise_error = None
            raise error
        
        # Return response
        if self.responses:
            response = self.responses[self.response_index % len(self.responses)]
            self.response_index += 1
            return response
        
        return self.default_response
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """Mock chat method."""
        prompt = "\n".join(m.get("content", "") for m in messages)
        return await self.generate(prompt, **kwargs)
    
    def get_call_count(self) -> int:
        """Get number of calls made."""
        return len(self.calls)
    
    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get the last call made."""
        return self.calls[-1] if self.calls else None
    
    def reset(self) -> None:
        """Reset the mock state."""
        self.responses = []
        self.response_index = 0
        self.calls = []
        self.raise_error = None
        self.delay = 0.0


class MockDatabaseSession:
    """
    Mock database session for testing.
    
    Usage:
        db = MockDatabaseSession()
        db.add_result({"id": "123", "name": "test"})
        result = await db.execute(query)
    """
    
    def __init__(self):
        self.results: List[Any] = []
        self.result_index = 0
        self.queries: List[str] = []
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self.raise_error: Optional[Exception] = None
    
    def add_result(self, result: Any) -> None:
        """Add a result to return."""
        self.results.append(result)
    
    def set_error(self, error: Exception) -> None:
        """Set an error to raise."""
        self.raise_error = error
    
    async def execute(self, query, *args, **kwargs) -> "MockResult":
        """Mock execute method."""
        self.queries.append(str(query))
        
        if self.raise_error:
            raise self.raise_error
        
        result_data = None
        if self.results and self.result_index < len(self.results):
            result_data = self.results[self.result_index]
            self.result_index += 1
        
        return MockResult(result_data)
    
    async def commit(self) -> None:
        """Mock commit."""
        if self.raise_error:
            raise self.raise_error
        self.committed = True
    
    async def rollback(self) -> None:
        """Mock rollback."""
        self.rolled_back = True
    
    async def close(self) -> None:
        """Mock close."""
        self.closed = True
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()
        await self.close()
    
    def reset(self) -> None:
        """Reset the mock state."""
        self.results = []
        self.result_index = 0
        self.queries = []
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self.raise_error = None


class MockResult:
    """Mock query result."""
    
    def __init__(self, data: Any = None):
        self.data = data
    
    def scalars(self) -> "MockResult":
        return self
    
    def all(self) -> List[Any]:
        if isinstance(self.data, list):
            return self.data
        return [self.data] if self.data else []
    
    def first(self) -> Optional[Any]:
        if isinstance(self.data, list):
            return self.data[0] if self.data else None
        return self.data
    
    def one(self) -> Any:
        result = self.first()
        if result is None:
            raise Exception("No result found")
        return result
    
    def one_or_none(self) -> Optional[Any]:
        return self.first()
    
    def fetchone(self) -> Optional[tuple]:
        data = self.first()
        if data is None:
            return None
        if isinstance(data, dict):
            return tuple(data.values())
        return (data,)
    
    def fetchall(self) -> List[tuple]:
        return [self.fetchone()] if self.data else []
    
    def close(self) -> None:
        pass


class MockRedisClient:
    """
    Mock Redis client for testing.
    
    Usage:
        redis = MockRedisClient()
        await redis.set("key", "value")
        result = await redis.get("key")
    """
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.expiry: Dict[str, datetime] = {}
        self.calls: List[Dict[str, Any]] = []
    
    async def get(self, key: str) -> Optional[str]:
        """Mock get."""
        self._record_call("get", key=key)
        self._check_expiry(key)
        value = self.data.get(key)
        if isinstance(value, bytes):
            return value.decode()
        return value
    
    async def set(
        self,
        key: str,
        value: str,
        ex: Optional[int] = None,
        px: Optional[int] = None,
    ) -> bool:
        """Mock set."""
        self._record_call("set", key=key, value=value, ex=ex)
        self.data[key] = value
        if ex:
            from datetime import timedelta
            self.expiry[key] = datetime.utcnow() + timedelta(seconds=ex)
        return True
    
    async def delete(self, *keys: str) -> int:
        """Mock delete."""
        self._record_call("delete", keys=keys)
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                self.expiry.pop(key, None)
                count += 1
        return count
    
    async def exists(self, *keys: str) -> int:
        """Mock exists."""
        self._record_call("exists", keys=keys)
        return sum(1 for k in keys if k in self.data)
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Mock expire."""
        self._record_call("expire", key=key, seconds=seconds)
        if key in self.data:
            from datetime import timedelta
            self.expiry[key] = datetime.utcnow() + timedelta(seconds=seconds)
            return True
        return False
    
    async def ttl(self, key: str) -> int:
        """Mock TTL."""
        self._record_call("ttl", key=key)
        if key not in self.data:
            return -2
        if key not in self.expiry:
            return -1
        remaining = (self.expiry[key] - datetime.utcnow()).total_seconds()
        return max(0, int(remaining))
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Mock hget."""
        self._record_call("hget", name=name, key=key)
        hash_data = self.data.get(name, {})
        return hash_data.get(key)
    
    async def hset(self, name: str, key: str, value: str) -> int:
        """Mock hset."""
        self._record_call("hset", name=name, key=key, value=value)
        if name not in self.data:
            self.data[name] = {}
        is_new = key not in self.data[name]
        self.data[name][key] = value
        return 1 if is_new else 0
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Mock hgetall."""
        self._record_call("hgetall", name=name)
        return self.data.get(name, {})
    
    async def lpush(self, name: str, *values: str) -> int:
        """Mock lpush."""
        self._record_call("lpush", name=name, values=values)
        if name not in self.data:
            self.data[name] = []
        for v in reversed(values):
            self.data[name].insert(0, v)
        return len(self.data[name])
    
    async def rpush(self, name: str, *values: str) -> int:
        """Mock rpush."""
        self._record_call("rpush", name=name, values=values)
        if name not in self.data:
            self.data[name] = []
        self.data[name].extend(values)
        return len(self.data[name])
    
    async def lrange(self, name: str, start: int, end: int) -> List[str]:
        """Mock lrange."""
        self._record_call("lrange", name=name, start=start, end=end)
        data = self.data.get(name, [])
        if end == -1:
            end = len(data)
        else:
            end += 1
        return data[start:end]
    
    async def incr(self, key: str) -> int:
        """Mock incr."""
        self._record_call("incr", key=key)
        current = int(self.data.get(key, 0))
        self.data[key] = str(current + 1)
        return current + 1
    
    async def close(self) -> None:
        """Mock close."""
        pass
    
    def _check_expiry(self, key: str) -> None:
        """Check and remove expired keys."""
        if key in self.expiry and datetime.utcnow() > self.expiry[key]:
            del self.data[key]
            del self.expiry[key]
    
    def _record_call(self, method: str, **kwargs) -> None:
        """Record a method call."""
        self.calls.append({
            "method": method,
            "timestamp": datetime.utcnow(),
            **kwargs,
        })
    
    def reset(self) -> None:
        """Reset the mock state."""
        self.data = {}
        self.expiry = {}
        self.calls = []


class MockHTTPClient:
    """
    Mock HTTP client for testing.
    
    Usage:
        http = MockHTTPClient()
        http.set_response({"data": "value"}, status=200)
        result = await http.get("https://api.example.com")
    """
    
    def __init__(self):
        self.responses: List[Dict[str, Any]] = []
        self.response_index = 0
        self.calls: List[Dict[str, Any]] = []
        self.default_response = {"status": "ok"}
        self.default_status = 200
    
    def set_response(
        self,
        data: Any,
        status: int = 200,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a response to return."""
        self.responses.append({
            "data": data,
            "status": status,
            "headers": headers or {},
        })
    
    def set_responses(self, responses: List[Dict[str, Any]]) -> None:
        """Set multiple responses."""
        self.responses = responses
        self.response_index = 0
    
    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> "MockHTTPResponse":
        """Mock GET request."""
        return await self._request("GET", url, headers=headers, **kwargs)
    
    async def post(
        self,
        url: str,
        data: Any = None,
        json: Any = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> "MockHTTPResponse":
        """Mock POST request."""
        return await self._request(
            "POST", url, data=data, json=json, headers=headers, **kwargs
        )
    
    async def put(
        self,
        url: str,
        data: Any = None,
        json: Any = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> "MockHTTPResponse":
        """Mock PUT request."""
        return await self._request(
            "PUT", url, data=data, json=json, headers=headers, **kwargs
        )
    
    async def delete(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> "MockHTTPResponse":
        """Mock DELETE request."""
        return await self._request("DELETE", url, headers=headers, **kwargs)
    
    async def _request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> "MockHTTPResponse":
        """Process a mock request."""
        self.calls.append({
            "method": method,
            "url": url,
            "timestamp": datetime.utcnow(),
            **kwargs,
        })
        
        # Get response
        if self.responses and self.response_index < len(self.responses):
            resp = self.responses[self.response_index]
            self.response_index += 1
        else:
            resp = {
                "data": self.default_response,
                "status": self.default_status,
                "headers": {},
            }
        
        return MockHTTPResponse(
            status=resp["status"],
            data=resp["data"],
            headers=resp.get("headers", {}),
        )
    
    def reset(self) -> None:
        """Reset the mock state."""
        self.responses = []
        self.response_index = 0
        self.calls = []


@dataclass
class MockHTTPResponse:
    """Mock HTTP response."""
    
    status: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    
    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300
    
    async def json(self) -> Any:
        return self.data
    
    async def text(self) -> str:
        if isinstance(self.data, str):
            return self.data
        return json.dumps(self.data)
    
    async def read(self) -> bytes:
        text = await self.text()
        return text.encode()


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""
    
    def __init__(self, name: str = "test"):
        self.name = name
        self.state = "closed"  # closed, open, half_open
        self.failure_count = 0
        self.success_count = 0
    
    def record_success(self) -> None:
        self.success_count += 1
        self.failure_count = 0
    
    def record_failure(self) -> None:
        self.failure_count += 1
    
    def is_open(self) -> bool:
        return self.state == "open"
    
    def open(self) -> None:
        self.state = "open"
    
    def close(self) -> None:
        self.state = "closed"
    
    def half_open(self) -> None:
        self.state = "half_open"
