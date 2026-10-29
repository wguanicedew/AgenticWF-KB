"""Prompt templates: the packaged text, the [prompts] dir file (read again when it changes) and
admins' edits from the web page (/system/prompts)."""

import os
import time

import pytest

from conftest import Caller, make_client, user
from agenticwf.kb.runtime import prompts

ADMIN = ("app:access", "knowledge:read", "knowledge:write", "admin")
RW = ("app:access", "knowledge:read", "knowledge:write")


def test_the_prompts_dir_file_is_read_again_when_it_changes(tmp_path, monkeypatch):
    from agenticwf.kb.runtime import prompts as module

    monkeypatch.setattr(module, "config_has_option", lambda section, option: section == "prompts")
    monkeypatch.setattr(module, "config_get", lambda section, option: str(tmp_path))
    monkeypatch.setattr(module, "edits", lambda: {})
    assert prompts.resolve("memory", "resolve_user")[1] == "packaged"
    path = tmp_path / "memory.yaml"
    path.write_text("resolve_user: 'first {{a}}'\n")
    assert prompts.render("memory", "resolve_user", a="x") == "first x"
    path.write_text("resolve_user: 'second {{a}}'\n")
    os.utime(path, (time.time() + 5, time.time() + 5))      # a new mtime even on a coarse clock
    assert prompts.resolve("memory", "resolve_user") == ("second {{a}}", "file")
    assert prompts.template("memory", "resolve_system") == str(prompts.packaged("memory")["resolve_system"])


@pytest.fixture
def client(database):
    with make_client("authsvc") as c:
        yield c


def test_admins_edit_and_reset_prompts(client):
    admin, alice = Caller(client, user("root", ADMIN)), Caller(client, user("alice", RW))
    assert alice.get("/kb/api/v1/system/prompts").status_code == 403
    assert alice.put("/kb/api/v1/system/prompts/memory/resolve_user", json={"text": "x"}).status_code == 403

    listed = admin.get("/kb/api/v1/system/prompts").json()["data"]
    entry = next(p for p in listed if p["file"] == "memory" and p["key"] == "resolve_user")
    assert entry["source"] == "packaged" and entry["variables"] == ["a", "b"] and entry["text"] == entry["default"]
    assert len(listed) == sum(len(prompts.packaged(f)) for f in prompts.files())

    saved = admin.put("/kb/api/v1/system/prompts/memory/resolve_user", json={"text": "Same? {{a}} vs {{c}}"})
    assert saved.status_code == 200, saved.text
    saved = saved.json()["data"]
    assert saved["source"] == "edited" and saved["updated_by"] == "root"
    assert any("{{b}}" in w for w in saved["warnings"]) and any("{{c}}" in w for w in saved["warnings"])
    assert prompts.render("memory", "resolve_user", a="A", b="B") == "Same? A vs "     # this process: at once

    assert admin.put("/kb/api/v1/system/prompts/memory/nope", json={"text": "x"}).status_code == 404
    assert admin.put("/kb/api/v1/system/prompts/passwd/x", json={"text": "x"}).status_code == 404     # packaged files only
    assert admin.put("/kb/api/v1/system/prompts/memory/resolve_user", json={"text": "  "}).status_code == 400

    reset = admin.delete("/kb/api/v1/system/prompts/memory/resolve_user").json()["data"]
    assert reset["source"] == "packaged" and reset["warnings"] == []
    assert prompts.render("memory", "resolve_user", a="A", b="B") == \
        prompts.packaged("memory")["resolve_user"].replace("{{a}}", "A").replace("{{b}}", "B")
