"""agenticwf-kb: the access token of standalone mode (docs/run-modes.md, `standalone`).

    agenticwf-kb token                  print the token (for curl, scripts, a personal agent)
    agenticwf-kb token --ensure         create it if missing (or import KB_TOKEN); print the sign-in link
    agenticwf-kb token --rotate         replace it: every browser is signed out, scripts need the new one

In the container: `docker exec agenticwf-kb agenticwf-kb token`. The token is kept in KB_TOKEN_FILE
(default $KB_DATA_ROOT/secrets/token, mode 0600); KB_TOKEN or KB_TOKEN_FILE given to the container
replace a generated one. The running KB picks up a new token without a restart.
"""

from __future__ import annotations

import argparse
import os
import sys

from agenticwf.kb.api.settings import token_path
from agenticwf.kb.api.standalone import ensure_token


def _sign_in_link(token: str) -> str:
    base = os.environ.get("KB_PUBLIC_URL", "http://localhost:8766").rstrip("/")
    return f"{base}{os.environ.get('KB_BASE_PATH', '/kb')}/?token={token}"


def cmd_token(args) -> int:
    path = token_path()
    if args.ensure or args.rotate:
        try:
            # KB_TOKEN_FILE names where the token lives; only KB_TOKEN imports a value.
            token, changed = ensure_token(path, given=os.environ.get("KB_TOKEN") or None, rotate=args.rotate)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        user = os.environ.get("KB_LOCAL_USER") or "user"
        print(f"\n  AgenticWF knowledge base (personal)\n  open:   {_sign_in_link(token)}\n"
              f"  user:   local:{user}; the token: agenticwf-kb token"
              f"{' (new)' if changed else ''}\n", flush=True)
        return 0
    if not path.is_file() or not path.read_text().strip():
        print("error: no token yet; run `agenticwf-kb token --ensure`", file=sys.stderr)
        return 1
    print(path.read_text().strip())
    return 0


def _as_kb_user(argv: list[str] | None) -> None:
    """In the container `docker exec` runs as root, but the token file belongs to the `kb` user:
    run the command again as that user (environment kept)."""
    if os.environ.get("KB_CONTAINER") and hasattr(os, "geteuid") and os.geteuid() == 0:
        args = sys.argv[1:] if argv is None else argv
        os.execvp("runuser", ["runuser", "-u", "kb", "--", sys.argv[0], *args])


def main(argv: list[str] | None = None) -> None:
    _as_kb_user(argv)
    parser = argparse.ArgumentParser(prog="agenticwf-kb", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    token = sub.add_parser("token", help="print, create or rotate the access token")
    group = token.add_mutually_exclusive_group()
    group.add_argument("--ensure", action="store_true", help="create the token if missing; print the sign-in link")
    group.add_argument("--rotate", action="store_true", help="replace the token (signs every browser out)")
    args = parser.parse_args(argv)
    sys.exit(cmd_token(args))


if __name__ == "__main__":
    main()
