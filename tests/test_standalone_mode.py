"""standalone mode: one user, one access token (the agent's local-mode design)."""

import pytest

from conftest import SECRET, signed, user

ORIGIN = {"Origin": "http://kb.test"}


@pytest.fixture
def token(standalone_client):
    from agenticwf.kb.api.settings import token_path
    from agenticwf.kb.api.standalone import ensure_token

    return ensure_token(token_path())[0]


def bearer(value):
    return {"Authorization": f"Bearer {value}"}


def test_health_needs_no_token(standalone_client):
    assert standalone_client.get("/kb/health").status_code == 200


def test_token_file_is_created_once_and_rotates(tmp_path):
    from agenticwf.kb.api.standalone import ensure_token

    path = tmp_path / "secrets" / "token"
    first, created = ensure_token(path)
    assert created and len(first) >= 40 and oct(path.stat().st_mode)[-3:] == "600"
    assert ensure_token(path) == (first, False)
    second, rotated = ensure_token(path, rotate=True)
    assert rotated and second != first
    assert ensure_token(path, given="a-given-token-123") == ("a-given-token-123", True)
    with pytest.raises(ValueError):
        ensure_token(path, given="short")


def test_the_one_local_user(standalone_client, token):
    me = standalone_client.get("/kb/api/v1/me", headers=bearer(token)).json()["data"]
    assert me["subject"] == "local:user" and me["mode"] == "standalone" and me["auth_method"] == "bearer"
    assert set(me["scopes"]) == {"app:access", "knowledge:read", "knowledge:write", "admin"}
    same = standalone_client.get("/kb/api/v1/me", headers={"X-API-Key": token}).json()["data"]
    assert same["subject"] == "local:user"


def test_missing_or_wrong_token(standalone_client, token):
    response = standalone_client.get("/kb/api/v1/me")
    assert response.status_code == 401 and response.json()["error"]["code"] == 1001
    assert standalone_client.get("/kb/api/v1/me", headers=bearer("x" * 43)).status_code == 401
    page = standalone_client.get("/kb/", follow_redirects=False)
    assert page.status_code == 302 and page.headers["location"] == "/kb/auth/login?return_to=/kb/"


def test_client_identity_headers_are_ignored(standalone_client, token):
    forged = signed(user(subject="mallory", scopes=("app:access", "admin")), "GET", "/kb/api/v1/me", SECRET)
    assert standalone_client.get("/kb/api/v1/me", headers=forged).status_code == 401
    me = standalone_client.get("/kb/api/v1/me", headers={**forged, **bearer(token)}).json()["data"]
    assert me["subject"] == "local:user"


def test_sign_in_link_sets_a_cookie(standalone_client, token):
    response = standalone_client.get(f"/kb/?token={token}", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/kb/"
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/kb" in cookie
    assert token not in cookie   # the cookie is a signature, not the token
    assert standalone_client.get("/kb/api/v1/me").json()["data"]["auth_method"] == "session"


def test_rotation_applies_without_restart_and_signs_browsers_out(standalone_client, token):
    from agenticwf.kb.api.settings import token_path
    from agenticwf.kb.api.standalone import ensure_token

    standalone_client.get(f"/kb/?token={token}", follow_redirects=False)
    assert standalone_client.get("/kb/api/v1/me").status_code == 200
    new, _ = ensure_token(token_path(), rotate=True)
    assert standalone_client.get("/kb/api/v1/me").status_code == 401                     # cookie ended
    assert standalone_client.get("/kb/api/v1/me", headers=bearer(token)).status_code == 401
    assert standalone_client.get("/kb/api/v1/me", headers=bearer(new)).status_code == 200


def test_cookie_writes_must_be_same_origin(standalone_client, token):
    standalone_client.get(f"/kb/?token={token}", follow_redirects=False)
    cross = standalone_client.post("/kb/auth/logout", headers={"Origin": "http://evil.example"})
    assert cross.status_code == 403
    # a write to the API with the cookie from another origin
    assert standalone_client.post("/kb/api/v1/me", headers={"Origin": "http://evil.example"}).status_code == 403


def test_login_form_and_logout(standalone_client, token):
    assert standalone_client.get("/kb/auth/login").status_code == 200
    bad = standalone_client.post("/kb/auth/login", data={"token": "nope"}, headers=ORIGIN)
    assert bad.status_code == 401
    good = standalone_client.post("/kb/auth/login", data={"token": token, "return_to": "/kb/"}, headers=ORIGIN,
                                  follow_redirects=False)
    assert good.status_code == 303
    assert standalone_client.get("/kb/auth/session").json()["authenticated"] is True
    out = standalone_client.post("/kb/auth/logout", headers={**ORIGIN, "Accept": "application/json"})
    assert out.status_code == 200
    standalone_client.cookies.clear()
    assert standalone_client.get("/kb/auth/session").json()["authenticated"] is False


def test_open_redirects_are_refused(standalone_client, token):
    response = standalone_client.post("/kb/auth/login", data={"token": token, "return_to": "//evil.example/"},
                                      headers=ORIGIN, follow_redirects=False)
    assert response.headers["location"] == "/kb/"


def test_no_api_key_routes(standalone_client, token):
    assert standalone_client.get("/kb/api/v1/api-keys", headers=bearer(token)).status_code == 404


def test_cli_token(capsys, monkeypatch, tmp_path):
    from agenticwf.kb.api import cli

    monkeypatch.setenv("KB_TOKEN_FILE", str(tmp_path / "token"))
    monkeypatch.delenv("KB_TOKEN", raising=False)
    with pytest.raises(SystemExit) as done:
        cli.main(["token"])
    assert done.value.code == 1 and "no token yet" in capsys.readouterr().err
    with pytest.raises(SystemExit) as done:
        cli.main(["token", "--ensure"])
    assert done.value.code == 0 and "open:   http://localhost:8766/kb/?token=" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["token"])
    assert capsys.readouterr().out.strip() == (tmp_path / "token").read_text().strip()


def test_login_page_shows_how_to_get_the_token(monkeypatch):
    from agenticwf.kb.api.standalone import login_page

    assert "agenticwf-kb token" in login_page("/kb", "/kb/")               # run on the host
    monkeypatch.setenv("KB_TOKEN_HELP", "# By the container's id:\ndocker exec abc agenticwf-links kb")
    page = login_page("/kb", "/kb/")
    assert '<p class="h">By the container&#x27;s id:</p>' in page
    assert "<pre>docker exec abc agenticwf-links kb</pre>" in page and "agenticwf-kb token" not in page


def test_hint_shows_the_token(standalone_client, token, monkeypatch):
    assert 'action="/kb/auth/hint"' not in standalone_client.get("/kb/auth/login").text   # no hint: no form
    monkeypatch.setenv("KB_LOCAL_HINT", "short")                                          # too short: off
    assert 'action="/kb/auth/hint"' not in standalone_client.get("/kb/auth/login").text
    monkeypatch.setenv("KB_LOCAL_HINT", "blue whale 42")
    assert "<details>" in standalone_client.get("/kb/auth/login").text                    # the commands, folded
    assert standalone_client.post("/kb/auth/hint", data={"hint": "blue whale 42"},
                                  headers={"Origin": "https://evil.example"}).status_code == 403
    wrong = standalone_client.post("/kb/auth/hint", data={"hint": "red whale"}, headers=ORIGIN)
    assert wrong.status_code == 401 and "<details open>" in wrong.text and token not in wrong.text
    right = standalone_client.post("/kb/auth/hint", data={"hint": "blue whale 42", "return_to": "/kb/x"},
                                   headers=ORIGIN)
    assert right.status_code == 200 and f"<pre>{token}</pre>" in right.text and 'href="/kb/x"' in right.text
    assert "set-cookie" in right.headers


def test_wrong_hints_are_throttled(standalone_client, token, monkeypatch):
    import agenticwf.kb.api.standalone as module

    monkeypatch.setenv("KB_LOCAL_HINT", "blue whale 42")
    for _ in range(module.HINT_TRIES):
        assert standalone_client.post("/kb/auth/hint", data={"hint": "guess"}, headers=ORIGIN).status_code == 401
    held = standalone_client.post("/kb/auth/hint", data={"hint": "blue whale 42"}, headers=ORIGIN)
    assert held.status_code == 429 and token not in held.text
    monkeypatch.setattr(module, "HINT_WAIT", 0)
    assert standalone_client.post("/kb/auth/hint", data={"hint": "blue whale 42"}, headers=ORIGIN).status_code == 200
