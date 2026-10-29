"""
Record-type plugins (docs/structured-records.md, Type plugins): system types with behaviour no
schema can express. Enabled with `[records] plugins` (e.g. `plugins = hpo`); at start their
types are synced to `kb_record_types` rows owned by `system`, as system models are.

    normalize(record, cfg)   canonical names (cfg: the KB's records_config)
    validate(record, cfg)    checks beyond JSON Schema
    derive(record)           computed values before storage: (record, per-item derived dicts)
    operations               {name: Operation}; read-only ones are also agentic-search tools
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agenticwf.kb.core.common.config import config_get, config_has_option


@dataclass
class Operation:
    name: str
    description: str
    read_only: bool
    run: Callable[..., Awaitable[dict]]          # async (env, body) -> dict
    parameters: dict = field(default_factory=lambda: {"type": "object"})
    tool_name: str | None = None                  # the agentic-search tool, for read-only operations


class RecordTypePlugin:
    name = ""
    types: list[dict] = []
    operations: dict[str, Operation] = {}

    def normalize(self, record: dict, cfg: dict) -> dict:
        return record

    def validate(self, record: dict, cfg: dict) -> list[str]:
        return []

    def derive(self, record: dict) -> tuple[dict, list[dict]]:
        return record, []


def available() -> dict[str, RecordTypePlugin]:
    from agenticwf.kb.runtime.records.plugins.hpo import HPOPlugin

    return {"hpo": HPOPlugin()}


def enabled() -> dict[str, RecordTypePlugin]:
    names = config_get("records", "plugins").replace(",", " ").split() if config_has_option("records", "plugins") \
        else ["hpo"]
    plugins = available()
    return {name: plugins[name] for name in names if name in plugins}


@dataclass
class OpEnv:
    """What an operation may use: the caller, the services, and the records service's scoped
    helpers (readable records KBs only)."""

    auth: Any
    services: Any
    records: Any
