"""
Prompt templates (docs/configuration.md, Prompt templates).

Templates live in `kb_runtime/defaults/prompts/<file>.yaml` (written for this service); a file of
the same name in `[prompts] dir` replaces the packaged one, key by key. `{{var}}` placeholders
are substituted; unknown ones become empty.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from kb_core.common.config import config_get, config_has_option

PACKAGED = Path(__file__).parent / "defaults" / "prompts"
_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@lru_cache(maxsize=32)
def _load(name: str) -> dict:
    data = yaml.safe_load((PACKAGED / f"{name}.yaml").read_text()) or {}
    if config_has_option("prompts", "dir"):
        override = Path(config_get("prompts", "dir")) / f"{name}.yaml"
        if override.is_file():
            data.update(yaml.safe_load(override.read_text()) or {})
    return data


def template(name: str, key: str) -> str:
    return str(_load(name).get(key, ""))


def render(name: str, key: str, **values) -> str:
    return _VAR.sub(lambda m: str(values.get(m.group(1), "")), template(name, key))
