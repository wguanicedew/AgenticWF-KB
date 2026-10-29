"""service mode: the agent names the user in X-KB-Principal; no authentication."""

from conftest import SECRET, signed, user


def principal(ctx):
    from agenticwf.kb.api.service import encode_principal

    return {"X-KB-Principal": encode_principal(ctx)}


def test_health_and_info_answer_without_a_principal(service_client):
    assert service_client.get("/kb/health").status_code == 200
    info = service_client.get("/kb/api/v1/system/info")
    assert info.status_code == 200 and info.json()["data"]["mode"] == "service"


def test_everything_else_needs_a_principal(service_client):
    response = service_client.get("/kb/api/v1/me")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == 1000 and error["message"].startswith("principal_required")


def test_the_named_user_is_the_principal(service_client):
    ctx = user(subject="alice", org="eic", groups=["atlas"], scopes=("app:access", "knowledge:read", "admin"))
    me = service_client.get("/kb/api/v1/me", headers=principal(ctx)).json()["data"]
    assert me["subject"] == "alice" and me["mode"] == "service"
    assert me["scopes"] == ["knowledge:read"]   # capped: admin and app:access are not forwarded
    assert me["groups"] == ["idp:atlas", "org:eic"]


def test_an_expired_user_token_does_not_stop_background_work(service_client):
    ctx = user(subject="alice", ttl=-3600)   # the agent's task outlives the user's request
    assert service_client.get("/kb/api/v1/me", headers=principal(ctx)).status_code == 200


def test_malformed_principal(service_client):
    for value in ("not base64 !!", "eyJmb28iOiAxfQ"):   # the second is {"foo": 1}: no subject
        response = service_client.get("/kb/api/v1/me", headers={"X-KB-Principal": value})
        assert response.status_code == 400 and "invalid_principal" in response.json()["error"]["message"]


def test_client_signed_contexts_are_ignored(service_client):
    forged = signed(user(subject="mallory"), "GET", "/kb/api/v1/me", SECRET)
    assert service_client.get("/kb/api/v1/me", headers=forged).status_code == 400


def test_no_api_key_routes(service_client):
    assert service_client.get("/kb/api/v1/api-keys", headers=principal(user())).status_code == 404
