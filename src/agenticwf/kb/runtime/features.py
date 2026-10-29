"""Deployment switches (`[features]` in kb.cfg, docs/configuration.md): a switched-off feature's
routes answer 404 and its KB type cannot be created."""

from __future__ import annotations

from agenticwf.kb.core.common.config import config_get, config_has_option

DEFAULTS = {"faq": True, "records": False, "wiki": False, "memory": False, "web": False, "remote": True}
KB_DEFAULTS = {"wiki_enabled": False, "memory_enabled": False}


def enabled(name: str) -> bool:
    if config_has_option("features", name):
        return config_get("features", name).strip().lower() in ("1", "true", "yes", "on")
    return DEFAULTS.get(name, False)


def kb_default(name: str) -> bool:
    """`[knowledge_bases] wiki_enabled / memory_enabled`: what a new document KB starts with when
    the create request does not say (it still needs `[features] wiki` / `memory`)."""
    if config_has_option("knowledge_bases", name):
        return config_get("knowledge_bases", name).strip().lower() in ("1", "true", "yes", "on")
    return KB_DEFAULTS.get(name, False)
