"""Prompt templates (docs/configuration.md, Prompt templates): admins view, edit and reset them.
Under /system, so authsvc's policy also limits them to `admin`."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from agenticwf.kb.api.deps import SCOPE_ADMIN, KBThread, thread_for
from agenticwf.kb.api.errors import ok

router = APIRouter(prefix="/api/v1/system/prompts")
ADMIN = thread_for(SCOPE_ADMIN, allow_machine=False)


@router.get("")
async def list_prompts(thread: KBThread = Depends(ADMIN)):
    return ok(await thread.prompts())


@router.put("/{file}/{key}")
async def set_prompt(file: str, key: str, body: dict[str, Any] = Body(...), thread: KBThread = Depends(ADMIN)):
    return ok(await thread.set_prompt(file, key, body.get("text")))


@router.delete("/{file}/{key}")
async def reset_prompt(file: str, key: str, thread: KBThread = Depends(ADMIN)):
    return ok(await thread.reset_prompt(file, key))
