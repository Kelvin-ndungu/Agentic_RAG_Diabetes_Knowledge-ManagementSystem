"""LLM setup, embeddings, and lightweight utilities."""
from __future__ import annotations

import asyncio
import time
import os
from typing import Any, Dict, List, Union

import httpx
from langchain_anthropic import ChatAnthropic

from .config import CLAUDE_API_KEY, CLAUDE_MODEL, CLAUDE_TEMPERATURE


def create_llm() -> ChatAnthropic:
    """
    Create and return Claude LLM instance.
    """
    if not CLAUDE_API_KEY:
        raise ValueError(
            "CLAUDE_API_KEY environment variable is required. "
            "Set it in your .env file or environment."
        )

    return ChatAnthropic(
        model=CLAUDE_MODEL,
        api_key=CLAUDE_API_KEY,
        temperature=CLAUDE_TEMPERATURE,
    )


_llm = None


def get_llm() -> ChatAnthropic:
    """
    Get or create the global LLM instance.
    """
    global _llm
    if _llm is None:
        _llm = create_llm()
    return _llm


class JinaEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using Jina API.
    Uses async httpx with connection pooling for better performance.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jina-embeddings-v4",
        task: str = "text-matching",
        api_url: str = "https://api.jina.ai/v1/embeddings",
        batch_size: int = 10,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY environment variable is required. "
                "Set it with: export JINA_API_KEY=your_api_key"
            )
        self.model = model
        self.task = task
        self.api_url = api_url
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            headers=self.headers,
        )

    def name(self) -> str:
        return "jina-embeddings-v4"

    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for input text(s).
        Synchronous wrapper for ChromaDB compatibility.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._call_async(input))
                    return future.result()
            return loop.run_until_complete(self._call_async(input))
        except RuntimeError:
            return asyncio.run(self._call_async(input))

    async def _call_async(self, input: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input

        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        data = {
            "model": self.model,
            "task": self.task,
            "input": [{"text": text} for text in texts],
        }

        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(self.api_url, json=data)
                response.raise_for_status()

                result = response.json()
                embeddings = []
                if "data" in result:
                    for item in result["data"]:
                        if "embedding" in item:
                            embeddings.append(item["embedding"])
                    return embeddings
                raise ValueError(f"Unexpected API response format: {result}")
            except httpx.RequestError as exc:
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"⚠ API request failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(
                        f"Failed to get embeddings after {self.max_retries} attempts: {exc}"
                    )
            except httpx.HTTPStatusError as exc:
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"⚠ API request failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(
                        f"Failed to get embeddings after {self.max_retries} attempts: {exc}"
                    )

        return []

    async def close(self) -> None:
        await self.client.aclose()

    def __del__(self):
        if hasattr(self, "client"):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.client.aclose())
                else:
                    loop.run_until_complete(self.client.aclose())
            except Exception:
                pass


class Timer:
    """
    Simple timing context manager for operational logs.
    """

    def __init__(self, label: str, enabled: bool = True):
        # Justification: timing labels help identify slow stages without adding heavy dependencies.
        self.label = label
        self.enabled = enabled
        self._start = None

    def __enter__(self):
        if self.enabled:
            self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self._start is not None:
            elapsed_ms = (time.perf_counter() - self._start) * 1000
            print(f"⏱ {self.label}: {elapsed_ms:.1f}ms")


def invoke_with_retry(chain: Any, payload: Dict[str, Any], max_retries: int, op_name: str) -> Any:
    """
    Invoke a chain with simple exponential backoff retries.
    """
    # Justification: upstream API calls are intermittent; retries reduce user-visible failures.
    last_error = None
    for attempt in range(max_retries):
        try:
            return chain.invoke(payload)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries - 1:
                raise
            backoff = min(2**attempt, 8)
            print(f"⚠ {op_name} failed (attempt {attempt + 1}/{max_retries}); retrying in {backoff}s")
            time.sleep(backoff)
    if last_error:
        raise last_error
    raise RuntimeError(f"{op_name} failed unexpectedly without exception")
