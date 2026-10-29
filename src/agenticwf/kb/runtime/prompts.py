"""
Prompt templates (docs/configuration.md, Prompt templates).

Templates live in `agenticwf/kb/runtime/defaults/prompts/<file>.yaml` (written for this service).
Two layers can replace one, key by key, the first that has it winning:

    1. an admin's edit in the web page (kb_prompt_overrides; `/system/prompts`), read again
       every OVERRIDE_TTL seconds, so every process sees an edit within that time;
    2. a file of the same name in `[prompts] dir` (default /etc/agenticwf/kb/prompts), read
       again when it changes;
    3. the packaged template.

`{{var}}` placeholders are substituted; unknown ones become empty.
"""

from __future__ import annotations

import re
import threading
import time
from functools import lru_cache
from pathlib import Path

import yaml

from agenticwf.kb.core.common.config import config_get, config_has_option
from agenticwf.kb.core.common.logging import get_logger

LOG = get_logger(__name__)

PACKAGED = Path(__file__).parent / "defaults" / "prompts"
DEFAULT_DIR = "/etc/agenticwf/kb/prompts"
OVERRIDE_TTL = 10.0
_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")

_lock = threading.Lock()
_file_cache: dict[str, tuple[float, dict]] = {}
_db_cache: tuple[float, dict[tuple[str, str], str]] = (float("-inf"), {})


def files() -> list[str]:
    return sorted(p.stem for p in PACKAGED.glob("*.yaml"))


def variables(text: str) -> list[str]:
    return sorted(set(_VAR.findall(text or "")))


@lru_cache(maxsize=32)
def packaged(name: str) -> dict:
    path = PACKAGED / f"{name}.yaml"
    return (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}


def override_file(name: str) -> dict:
    """The `[prompts] dir` file's templates, read again when the file changes."""
    folder = config_get("prompts", "dir") if config_has_option("prompts", "dir") else DEFAULT_DIR
    path = Path(folder) / f"{name}.yaml"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    with _lock:
        cached = _file_cache.get(name)
        if cached and cached[0] == mtime:
            return cached[1]
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        LOG.warning("prompt file %s is unreadable: %s", path, exc)
        data = {}
    with _lock:
        _file_cache[name] = (mtime, data)
    return data


def edits() -> dict[tuple[str, str], str]:
    """The web page's edits, {(file, key): text}, at most OVERRIDE_TTL seconds old."""
    global _db_cache
    with _lock:
        loaded, data = _db_cache
    if time.monotonic() - loaded < OVERRIDE_TTL:
        return data
    try:
        import sqlalchemy as sa

        from agenticwf.kb.core.orm.base.session import get_engine
        from agenticwf.kb.core.orm.models.ext import PromptOverride

        with get_engine().connect() as conn:
            rows = conn.execute(sa.select(PromptOverride.file, PromptOverride.key, PromptOverride.text)).all()
        data = {(f, k): t for f, k, t in rows}
    except Exception as exc:          # no database (unit tests, tools): the files only
        LOG.debug("prompt overrides not read: %s", exc)
    with _lock:
        _db_cache = (time.monotonic(), data)
    return data


def invalidate() -> None:
    """Read the edits again on the next use (this process; the others within OVERRIDE_TTL)."""
    global _db_cache
    with _lock:
        _db_cache = (float("-inf"), _db_cache[1])


def resolve(name: str, key: str) -> tuple[str, str]:
    """(text, source): source is `edited`, `file` or `packaged` ("" when no template has it)."""
    edited = edits().get((name, key))
    if edited is not None:
        return edited, "edited"
    from_file = override_file(name)
    if key in from_file:
        return str(from_file[key]), "file"
    base = packaged(name)
    return (str(base[key]), "packaged") if key in base else ("", "")


def template(name: str, key: str) -> str:
    return resolve(name, key)[0]


def render(name: str, key: str, **values) -> str:
    return _VAR.sub(lambda m: str(values.get(m.group(1), "")), template(name, key))
