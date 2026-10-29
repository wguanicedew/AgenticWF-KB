"""
PromptService: admins view and edit the prompt templates from the web page
(docs/configuration.md, Prompt templates; `/system/prompts`).

Only templates the package defines can be edited. An edit is stored in kb_prompt_overrides and
wins over the `[prompts] dir` file and the packaged text; resetting deletes it. Saving reports
`warnings` for placeholders the packaged template uses that the edit dropped (the code fills
them with what the template needs: the passage, the question …) and for placeholders the code
does not fill (they render empty).
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from agenticwf.auth.context import AuthContext

from agenticwf.kb.core.orm.base.session import async_session_scope
from agenticwf.kb.core.orm.models.ext import PromptOverride
from agenticwf.kb.core.orm.models.kb import utcnow
from agenticwf.kb.runtime import prompts
from agenticwf.kb.runtime.errors import Invalid, NotFound

MAX_TEXT = 50_000


def _known(name: str, key: str) -> str:
    """The packaged text, or NotFound."""
    base = prompts.packaged(name) if name in prompts.files() else {}
    if key not in base:
        raise NotFound(f"no prompt template {name}/{key}")
    return str(base[key])


def warnings(default: str, text: str) -> list[str]:
    expected, used = set(prompts.variables(default)), set(prompts.variables(text))
    out = [f"{{{{{v}}}}} is no longer used: the template will not get it" for v in sorted(expected - used)]
    out += [f"{{{{{v}}}}} is not filled by the code: it renders empty" for v in sorted(used - expected)]
    return out


class PromptService:
    @staticmethod
    def _entry(name: str, key: str, meta: dict | None = None) -> dict:
        text, source = prompts.resolve(name, key)
        default = str(prompts.packaged(name)[key])
        file_text = prompts.override_file(name).get(key)
        return {"file": name, "key": key, "text": text, "source": source, "default": default,
                "file_text": None if file_text is None else str(file_text), "variables": prompts.variables(default),
                "warnings": warnings(default, text), **(meta or {})}

    async def list(self) -> list[dict]:
        async with async_session_scope(commit=False) as session:
            meta = {(r.file, r.key): {"updated_by": r.updated_by, "updated_at": r.updated_at.isoformat()}
                    for r in (await session.execute(sa.select(PromptOverride))).scalars()}
        prompts.invalidate()
        return [self._entry(name, key, meta.get((name, key)))
                for name in prompts.files() for key in prompts.packaged(name)]

    async def set(self, ctx: AuthContext, name: str, key: str, text: str) -> dict:
        _known(name, key)
        if not isinstance(text, str) or not text.strip():
            raise Invalid("text is required (reset the template to go back to its default)")
        if len(text) > MAX_TEXT:
            raise Invalid(f"a template is at most {MAX_TEXT} characters")
        async with async_session_scope() as session:
            now = utcnow()
            await session.execute(insert(PromptOverride).values(
                file=name, key=key, text=text, updated_by=ctx.subject, updated_at=now).on_conflict_do_update(
                index_elements=["file", "key"], set_={"text": text, "updated_by": ctx.subject, "updated_at": now}))
        prompts.invalidate()
        return self._entry(name, key, {"updated_by": ctx.subject, "updated_at": now.isoformat()})

    async def reset(self, name: str, key: str) -> dict:
        _known(name, key)
        async with async_session_scope() as session:
            await session.execute(sa.delete(PromptOverride).where(PromptOverride.file == name,
                                                                  PromptOverride.key == key))
        prompts.invalidate()
        return self._entry(name, key)
