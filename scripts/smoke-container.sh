#!/usr/bin/env bash
# Phase-0 acceptance on a built image: the container starts in all three run modes, /kb/health
# answers, and /kb/api/v1/me works with the standalone token, a signed authsvc context and
# X-KB-Principal over the service socket (docs/roadmap.md, Phase 0).
#
#   scripts/smoke-container.sh [image]          default image: agenticwf-kb:latest
#
# Needs docker (or DOCKER=podman), curl and python3 with the agenticwf-auth package (to sign the authsvc
# context; pip install ../AgenticWF-auth/authsvc).
set -euo pipefail

IMAGE=${1:-agenticwf-kb:latest}
DOCKER=${DOCKER:-docker}
PREFIX=kb-smoke-$$
SECRET=$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')
WORK=$(mktemp -d)
trap 'for c in standalone authsvc service; do $DOCKER rm -f "$PREFIX-$c" >/dev/null 2>&1 || true; done;
      $DOCKER volume rm -f "$PREFIX-sock" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $*" >&2; for c in standalone authsvc service; do $DOCKER logs "$PREFIX-$c" 2>&1 | tail -30 || true; done; exit 1; }
wait_healthy() {   # wait_healthy <container> <url>
    for _ in $(seq 1 90); do
        curl -fsS -o /dev/null "$2" 2>/dev/null && return 0
        $DOCKER inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -q true || fail "$1 exited"
        sleep 2
    done
    fail "$1 did not become healthy"
}

echo "== standalone (with the service socket for an agent)"
$DOCKER volume create "$PREFIX-sock" >/dev/null
$DOCKER run -d --name "$PREFIX-standalone" -p 127.0.0.1:18766:8020 \
    -e KB_SERVICE_LISTEN=unix:/run/agenticwf-kb/kb.sock -v "$PREFIX-sock:/run/agenticwf-kb" "$IMAGE" >/dev/null
wait_healthy "$PREFIX-standalone" http://127.0.0.1:18766/kb/health
TOKEN=$($DOCKER exec "$PREFIX-standalone" agenticwf-kb token)
curl -fsS -H "Authorization: Bearer $TOKEN" http://127.0.0.1:18766/kb/api/v1/me | grep -q '"subject":"local:user"' \
    || fail "standalone /me with the token"
code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18766/kb/api/v1/me)
[[ $code == 401 ]] || fail "standalone /me without a key gave $code"
PRINCIPAL=$(python3 -c 'import base64,json; print(base64.urlsafe_b64encode(json.dumps({"subject":"alice","scopes":["knowledge:read"]}).encode()).rstrip(b"=").decode())')
for _ in $(seq 1 30); do
    $DOCKER exec "$PREFIX-standalone" test -S /run/agenticwf-kb/kb.sock && break; sleep 2
done
$DOCKER exec "$PREFIX-standalone" curl -fsS --unix-socket /run/agenticwf-kb/kb.sock \
    -H "X-KB-Principal: $PRINCIPAL" http://kb/kb/api/v1/me | grep -q '"subject":"alice"' \
    || fail "X-KB-Principal over the service socket"
echo "   ok"

echo "== authsvc"
printf '%s' "$SECRET" > "$WORK/secret"; chmod 0444 "$WORK/secret"
$DOCKER run -d --name "$PREFIX-authsvc" -p 127.0.0.1:18767:8020 -e KB_AUTH_MODE=authsvc -e LISTEN_HOST=0.0.0.0 \
    -v "$WORK/secret:/run/secrets/auth_context_secret:ro" "$IMAGE" >/dev/null
wait_healthy "$PREFIX-authsvc" http://127.0.0.1:18767/kb/health
HEADERS=$(SECRET="$SECRET" python3 - <<'EOF'
import os, time
from agenticwf.auth.context import AuthContext, encode_context
ctx = AuthContext(subject="bob", user_id="bob", scopes=frozenset({"app:access", "knowledge:read"}),
                  expires_at=int(time.time()) + 600)
payload, sig = encode_context(ctx, os.environ["SECRET"].encode(), 60, "GET", "/kb/api/v1/me")
print(f"{payload} {sig}")
EOF
)
curl -fsS -H "X-Auth-Context: ${HEADERS% *}" -H "X-Auth-Context-Sig: ${HEADERS#* }" \
    http://127.0.0.1:18767/kb/api/v1/me | grep -q '"subject":"bob"' || fail "authsvc /me with a signed context"
echo "   ok"

echo "== service"
$DOCKER run -d --name "$PREFIX-service" -p 127.0.0.1:18768:8020 -e KB_AUTH_MODE=service -e LISTEN_HOST=0.0.0.0 \
    -e KB_SERVICE_ALLOW_REMOTE=true "$IMAGE" >/dev/null
wait_healthy "$PREFIX-service" http://127.0.0.1:18768/kb/health
curl -fsS -H "X-KB-Principal: $PRINCIPAL" http://127.0.0.1:18768/kb/api/v1/me | grep -q '"subject":"alice"' \
    || fail "service /me with X-KB-Principal"
code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18768/kb/api/v1/me)
[[ $code == 400 ]] || fail "service /me without a principal gave $code"
$DOCKER exec -u kb "$PREFIX-service" python -m agenticwf.kb.api.db check | grep -q pg_search || fail "pg_search missing"
echo "   ok"

echo "phase-0 container smoke test passed"
