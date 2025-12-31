"""Claude API Client.

Async client for Anthropic Claude, intended as a high-stakes fallback provider.

This module is safe to import even when Claude is not configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass(frozen=True)
class ClaudeConfig:
    api_key: str
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    base_url: str = "https://api.anthropic.com/v1"


class ClaudeClient:
    """Minimal async Claude Messages API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        timeout_seconds: float = 120.0,
        base_url: str = "https://api.anthropic.com/v1",
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = base_url
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")

        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system

        resp = await self._client.post(f"{self.base_url}/messages", json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Claude returns content as a list of blocks.
        content = data.get("content", [])
        if not content:
            return ""
        first = content[0]
        if isinstance(first, dict) and "text" in first:
            return first["text"]
        # Fallback
        return str(first)

    async def close(self) -> None:
        await self._client.aclose()
