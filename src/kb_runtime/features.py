"""Deployment switches (`[features]` in kb.cfg, docs/configuration.md): a switched-off feature's
routes answer 404 and its KB type cannot be created."""

from __future__ import annotations

from kb_core.common.config import config_get, config_has_option

DEFAULTS = {"faq": True, "records": False, "wiki": False, "memory": False, "web": False}


def enabled(name: str) -> bool:
    if config_has_option("features", name):
        return config_get("features", name).strip().lower() in ("1", "true", "yes", "on")
    return DEFAULTS.get(name, False)
