"""
Model clients: OpenAI-compatible chat, embeddings and rerank, plus the `hash` development
embedder (docs/configuration.md, System models).

Every call goes through a per-model, per-event-loop semaphore (`max_concurrency`) and retries
408/409/425/429/5xx and connection errors with exponential backoff (1 s, 2 s, 4 s; `[models]
max_retries`). Embedding and LLM calls never run inside a database transaction. Each call, its
retries included, is a span with its token usage (agenticwf.kb.core.common.telemetry).

`hash` (provider = hash) needs no server: each token is hashed into one of `dimension` buckets
with a sign, and the vector is L2-normalised. It matches shared words, not meaning; it exists for
development, tests and CI, like the agent's `echo` model.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from agenticwf.kb.core.common.config import config_get, config_has_option
from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.core.common import telemetry

LOG = get_logger(__name__)

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
OPERATIONS = {"/chat/completions": "chat", "/embeddings": "embeddings", "/rerank": "rerank"}
EMBED_BATCH = 40


class ModelError(Exception):
    """A model call failed after its retries, or the model is not usable."""


def _cfg_int(option: str, default: int) -> int:
    return int(config_get("models", option)) if config_has_option("models", option) else default


@dataclass(frozen=True)
class ModelSpec:
    """A resolved model: what a client needs (the API key already decrypted or read from env)."""

    id: str
    name: str                       # the provider's model name
    type: str                       # KnowledgeQA | Embedding | Rerank | VLLM
    provider: str = "openai_compat"
    base_url: str = ""
    api_key: str = ""
    dimension: int = 0
    max_concurrency: int = 0
    context_window: int = 0
    max_output_tokens: int = 0
    supports_tools: bool = False
    supports_vision: bool = False
    headers: dict[str, str] = field(default_factory=dict)
    truncate_prompt_tokens: int = 0

    def concurrency(self) -> int:
        return self.max_concurrency or _cfg_int("max_concurrency", 4)


# --- concurrency and retries ------------------------------------------------------------------
_SEMAPHORES: dict[tuple[int, str], asyncio.Semaphore] = {}


def _semaphore(spec: ModelSpec) -> asyncio.Semaphore:
    key = (id(asyncio.get_running_loop()), spec.id)
    sem = _SEMAPHORES.get(key)
    if sem is None:
        sem = _SEMAPHORES[key] = asyncio.Semaphore(spec.concurrency())
    return sem


async def _post(spec: ModelSpec, path: str, body: dict, timeout: float) -> Any:
    operation = OPERATIONS.get(path, path.strip("/"))
    with telemetry.model_call(operation, spec.provider, spec.name) as current:
        data = await _send(spec, path, body, timeout)
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            telemetry.model_tokens(current, operation, spec.provider, spec.name, usage.get("prompt_tokens"),
                                   usage.get("completion_tokens"))
        return data


async def _send(spec: ModelSpec, path: str, body: dict, timeout: float) -> Any:
    url = spec.base_url.rstrip("/") + path
    headers = {"Content-Type": "application/json", **spec.headers}
    if spec.api_key:
        headers["Authorization"] = f"Bearer {spec.api_key}"
    retries = _cfg_int("max_retries", 3)
    delay = 1.0
    async with _semaphore(spec):
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
                    response = await client.post(url, json=body, headers=headers)
                if response.status_code in RETRY_STATUS and attempt < retries:
                    raise httpx.HTTPStatusError("retryable", request=response.request, response=response)
                if response.status_code >= 400:
                    raise ModelError(f"{spec.name}: HTTP {response.status_code}: {response.text[:300]}")
                return response.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if attempt >= retries:
                    raise ModelError(f"{spec.name}: {exc}") from exc
                LOG.warning("model %s: %s; retrying in %.0fs", spec.name, exc, delay)
                await asyncio.sleep(delay)
                delay *= 2
    raise ModelError(f"{spec.name}: no response")


# --- embeddings ---------------------------------------------------------------------------------
_TOKEN = re.compile(r"\w+", re.UNICODE)


def hash_embed(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for token in _TOKEN.findall(text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class Embedder:
    def __init__(self, spec: ModelSpec):
        if spec.type != "Embedding":
            raise ModelError(f"{spec.name} is not an embedding model")
        if spec.dimension <= 0:
            raise ModelError(f"{spec.name}: the embedding dimension is not set")
        self.spec = spec

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.spec.provider == "hash":
            return [hash_embed(t, self.spec.dimension) for t in texts]
        vectors: list[list[float]] = []
        timeout = float(_cfg_int("embedding_timeout", 60))
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]
            body: dict[str, Any] = {"model": self.spec.name, "input": batch}
            if self.spec.truncate_prompt_tokens:
                body["truncate_prompt_tokens"] = self.spec.truncate_prompt_tokens
            data = await _post(self.spec, "/embeddings", body, timeout)
            rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
            if len(rows) != len(batch):
                raise ModelError(f"{self.spec.name}: {len(rows)} embeddings for {len(batch)} texts")
            for row in rows:
                vector = row["embedding"]
                if len(vector) != self.spec.dimension:
                    raise ModelError(f"{self.spec.name}: got dimension {len(vector)}, configured {self.spec.dimension}")
                vectors.append(vector)
        return vectors


# --- rerank -------------------------------------------------------------------------------------
class Reranker:
    """`POST {base_url}/rerank` with {model, query, documents, top_n}; accepts the Cohere/Jina shape
    ({results: [{index, relevance_score}]}) and TEI's ([{index, score}])."""

    def __init__(self, spec: ModelSpec):
        if spec.type != "Rerank":
            raise ModelError(f"{spec.name} is not a rerank model")
        self.spec = spec

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        data = await _post(self.spec, "/rerank",
                           {"model": self.spec.name, "query": query, "documents": documents, "top_n": len(documents)},
                           float(_cfg_int("embedding_timeout", 60)))
        rows = data.get("results", data) if isinstance(data, dict) else data
        scores = [0.0] * len(documents)
        for row in rows or []:
            index = row.get("index")
            if isinstance(index, int) and 0 <= index < len(documents):
                scores[index] = float(row.get("relevance_score", row.get("score", 0.0)))
        return scores


# --- chat ---------------------------------------------------------------------------------------
class ChatModel:
    """OpenAI-compatible `/chat/completions`, with `tools` / `tool_calls` and image parts.

    `complete()` returns {content, reasoning, tool_calls: [{id, name, arguments (str)}], usage,
    finish_reason}. Calls are not streamed: agentic search emits its frames per step."""

    def __init__(self, spec: ModelSpec):
        if spec.type not in ("KnowledgeQA", "VLLM"):
            raise ModelError(f"{spec.name} is not a chat model")
        self.spec = spec

    async def complete(self, messages: list[dict], max_tokens: int | None = None, temperature: float = 0.3,
                       tools: list[dict] | None = None, timeout: float | None = None, **extra) -> dict:
        body: dict[str, Any] = {"model": self.spec.name, "messages": messages, "temperature": temperature, **extra}
        if tools:
            body["tools"] = tools
        if max_tokens or self.spec.max_output_tokens:
            body["max_tokens"] = max_tokens or self.spec.max_output_tokens
        data = await _post(self.spec, "/chat/completions", body,
                           float(timeout or _cfg_int("chat_stall_timeout", 120)))
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            calls.append({"id": call.get("id") or f"call_{index}", "name": function.get("name", ""),
                          "arguments": function.get("arguments") or "{}"})
        return {"content": message.get("content") or "", "reasoning": message.get("reasoning_content") or "",
                "tool_calls": calls, "usage": data.get("usage") or {}, "finish_reason": choice.get("finish_reason")}
