from __future__ import annotations
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import diskcache as dc
from openai import AsyncOpenAI

class CachedAsyncOpenAI:
    """Async OpenAI client with file-system caching"""

    _NON_CACHEABLE_KEYS = {"stream", "user", "logit_bias"}

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_cache: bool = True,
        cache_dir: str | Path = ".cache",
        cache_size_bytes: int = 10_000_000_000,  # 10 GB
        debug: bool = False,
    ) -> None:

        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

        self.enable_cache = enable_cache
        self.debug = debug
        self.cache: Optional[dc.Cache] = None
        if enable_cache:
            path = Path(cache_dir).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            self.cache = dc.Cache(path, size_limit=cache_size_bytes)
            if self.debug:
                print(f"[cache] Ready at {path}")


    async def chat(self, **kwargs) -> Any:
        """Create a chat completion (cached unless `stream=True`)."""

        use_cache = self.enable_cache and not kwargs.get("stream")
        key = self._make_key(kwargs) if use_cache else None

        if key and self.cache and (cached := self.cache.get(key, default=None)) is not None:
            if self.debug:
                print(f"[cache] hit: {key[:12]}…")
            return cached

        if self.debug:
            print(f"[cache] miss → API: {key[:12] + '…' if key else 'stream'}")
        response = await self.client.chat.completions.create(**kwargs)

        if key and self.cache and self._looks_ok(response):
            self.cache.set(key, response)

        return response

    def cache_stats(self) -> Dict[str, Any]:
        """Get count using the stats method available on diskcache"""
        if self.cache is None:
            return {"enabled": False}
        try:
            item_count = self.cache.stats(enable=True)[0]  # hits, misses, loads, time
        except Exception:
            item_count = 0
        return {
            "enabled": True,
            "items": item_count,
            "disk_bytes": self.cache.volume(),
            "path": str(self.cache.directory),
        }

    def clear_cache(self) -> None:
        if self.cache is not None:
            self.cache.clear()

    @classmethod
    def _make_key(cls, params: Dict[str, Any]) -> str:
        """Strip out non-cacheable keys before hashing"""
        safe = {k: v for k, v in params.items() if k not in cls._NON_CACHEABLE_KEYS}
        payload = json.dumps(safe, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _looks_ok(resp: Any) -> bool:
        """Very light validation"""
        try:
            return bool(getattr(resp, "choices", []))
        except Exception:
            return False

    def __del__(self):
        if self.cache is not None:
            self.cache.close()

class RoutedAsyncOpenAI:
    
    def __init__(self, **kwargs) -> None:
        self.openai_client = CachedAsyncOpenAI(**kwargs)
        if os.environ.get("LOCAL_BASE_URL"):
            local_base_url = os.environ.get("LOCAL_BASE_URL")
        
        self.local_client = CachedAsyncOpenAI(base_url=local_base_url, api_key="dummy", **kwargs)

        self.together_client = CachedAsyncOpenAI(base_url=os.environ.get("TOGETHER_BASE_URL"), api_key=os.environ.get("TOGETHER_API_KEY"), **kwargs)
        self.local_models = set(["Qwen/QwQ-32B", "Qwen/Qwen2.5-32B-Instruct"])
        self.together_models = set(["deepseek-ai/DeepSeek-V3"])
    
    async def chat(self, **kwargs) -> Any:
        """Route to local or OpenAI based on model name."""
        model = kwargs.get("model", "")
        
        if model in self.local_models:
            return await self.local_client.chat(**kwargs)
        elif model in self.together_models:
            return await self.together_client.chat(**kwargs)
        else:
            return await self.openai_client.chat(**kwargs)