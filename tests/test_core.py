import pytest
import sqlalchemy as sa

from conftest import user


def test_init_is_idempotent_and_reaches_head(database):
    from agenticwf.kb.api.db import init_db
    from agenticwf.kb.core.orm.base.session import get_engine

    init_db()   # second time: nothing to do
    with get_engine().connect() as conn:
        assert conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar() == "0009"


def test_missing_extension_is_a_clear_error(database, monkeypatch):
    from agenticwf.kb.core.common.exceptions import DatabaseException
    from agenticwf.kb.core.orm.base import utils
    from agenticwf.kb.core.orm.base.session import get_engine

    monkeypatch.setattr(utils, "required_extensions", lambda: ("vector", "no_such_extension"))
    with pytest.raises(DatabaseException) as err:
        utils.check_extensions(get_engine())
    assert "no_such_extension" in str(err.value)
    assert "CREATE EXTENSION IF NOT EXISTS no_such_extension;" in str(err.value)


def test_async_session_runs_on_the_running_loop(database):
    import asyncio

    from agenticwf.kb.core.orm.base.session import async_session_scope, dispose_async_engine

    async def probe():
        async with async_session_scope(commit=False) as session:
            value = (await session.execute(sa.text("SELECT 41 + 1"))).scalar()
        await dispose_async_engine()
        return value

    assert asyncio.run(probe()) == 42


def test_psycopg2_url_is_refused():
    from agenticwf.kb.core.common.exceptions import AgenticWFException
    from agenticwf.kb.core.orm.base.session import _async_url

    with pytest.raises(AgenticWFException):
        _async_url("postgresql+psycopg2://kb@/kb")
    assert _async_url("postgresql://kb@h/kb") == "postgresql+psycopg://kb@h/kb"


def test_secret_box_round_trip_and_tamper():
    from agenticwf.kb.runtime.security.secrets import SecretBox, SecretsUnavailable

    box = SecretBox(SecretBox.generate_key())
    token = box.encrypt("sk-live-123")
    assert token != "sk-live-123" and box.decrypt(token) == "sk-live-123"
    with pytest.raises(SecretsUnavailable):
        SecretBox(SecretBox.generate_key()).decrypt(token)


def test_secret_box_from_environment(tmp_path, monkeypatch):
    from agenticwf.kb.runtime.security.secrets import SecretBox

    monkeypatch.delenv("KB_SECRETS_KEY", raising=False)
    monkeypatch.delenv("KB_SECRETS_KEY_FILE", raising=False)
    assert SecretBox.from_environment() is None
    path = tmp_path / "secrets_key"
    path.write_text(SecretBox.generate_key().decode())
    monkeypatch.setenv("KB_SECRETS_KEY_FILE", str(path))
    assert SecretBox.from_environment().decrypt(SecretBox.from_environment().encrypt("x")) == "x"


def test_principal_groups_prefixes_every_source():
    from agenticwf.kb.runtime.access import normalise_group, principal_groups

    ctx = user(org="eic", roles=("researcher",), groups=["atlas", "kb:ops"])
    assert principal_groups(ctx) == ["idp:atlas", "kb:ops", "org:eic", "role:researcher"]
    assert principal_groups(user()) == []
    assert normalise_group("atlas") == "kb:atlas" and normalise_group("org:eic") == "org:eic"


def test_principal_groups_include_every_organization():
    from dataclasses import replace

    from agenticwf.kb.runtime.access import group_names, principal_groups

    ctx = replace(user(org="eic"), organizations={"o1": "EIC", "o2": "Ops"})   # authsvc's memberships
    assert principal_groups(ctx) == ["org:eic", "org:o1", "org:o2"]
    assert group_names(ctx) == {"org:o1": "EIC", "org:o2": "Ops"}


def test_service_bind_guard(monkeypatch):
    from agenticwf.kb.api.settings import service_bind_allowed

    monkeypatch.delenv("KB_SERVICE_ALLOW_REMOTE", raising=False)
    assert service_bind_allowed("unix:/run/agenticwf-kb/kb.sock")
    assert service_bind_allowed("127.0.0.1:8021") and service_bind_allowed("[::1]:8021")
    assert not service_bind_allowed("0.0.0.0:8021")
    monkeypatch.setenv("KB_SERVICE_ALLOW_REMOTE", "true")
    assert service_bind_allowed("0.0.0.0:8021")


def test_settings_validate_mode_and_base_path():
    from agenticwf.kb.api.settings import Settings

    with pytest.raises(ValueError):
        Settings(mode="local")
    with pytest.raises(ValueError):
        Settings(base_path="/kb/")
