"""
SystemService: the database's health, its extensions and what the deployment offers.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import anyio
import sqlalchemy as sa

from kb_core.common.config import config_get, config_has_option
from kb_core.orm.base import session as db_session
from kb_core.orm.base import utils as db_utils

FEATURE_DEFAULTS = {"faq": True, "records": False, "wiki": False, "memory": False, "web": False}
EXTENSIONS = ("vector", "pg_search", "pg_trgm")


def kb_version() -> str:
    try:
        return version("agenticwf-kb")
    except PackageNotFoundError:
        return "0+unknown"


def _flag(section: str, option: str, default: bool) -> bool:
    if not config_has_option(section, option):
        return default
    return config_get(section, option).strip().lower() in ("1", "true", "yes", "on")


class SystemService:
    def features(self) -> dict[str, bool]:
        return {name: _flag("features", name, default) for name, default in FEATURE_DEFAULTS.items()}

    def bm25_tokenizer(self) -> str:
        return config_get("retrieval", "bm25_tokenizer") if config_has_option("retrieval", "bm25_tokenizer") else "icu"

    async def healthy(self) -> bool:
        try:
            async with db_session.async_session_scope(commit=False) as session:
                await session.execute(sa.text("SELECT 1"))
        except Exception:
            return False
        return True

    async def extensions(self) -> dict[str, str]:
        async with db_session.async_session_scope(commit=False) as session:
            rows = await session.execute(sa.text("SELECT extname, extversion FROM pg_extension"))
            found = {name: ext_version for name, ext_version in rows}
        return {name: found[name] for name in EXTENSIONS if name in found}

    async def info(self) -> dict:
        return {
            "version": kb_version(),
            "features": self.features(),
            "bm25_tokenizer": self.bm25_tokenizer(),
            "extensions": await self.extensions(),
        }

    async def check_database(self) -> dict[str, str]:
        """Raise DatabaseException unless the required extensions exist."""
        return await anyio.to_thread.run_sync(lambda: db_utils.check_extensions(db_session.get_engine()))
