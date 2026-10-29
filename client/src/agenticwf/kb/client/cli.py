"""agenticwf kb — the knowledge service from the command line.

    agenticwf kb list                                your knowledge bases
    agenticwf kb search "beam energy" [--kb ID]      hybrid search, best passages first
    agenticwf kb ask "what changed in run 42?"       agentic search: an answer and its sources
    agenticwf kb call GET /knowledge-bases           any KB API call (the path under /api/v1)

Behind Caddy it signs in as `agenticwf token` does (`agenticwf login`, AGENTICWF_TOKEN or
AGENTICWF_CLIENT_SECRET) and finds the KB at AGENTICWF_API_URL + /kb. A standalone KB has its own
token instead:

    export AGENTICWF_KB_URL=http://127.0.0.1:8766/kb
    export AGENTICWF_KB_TOKEN=$(agenticwf-kb token)    # or AGENTICWF_KB_TOKEN_FILE

`--json` prints the API's data as it is (for jq).
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

from agenticwf.kb.client import KBClient, KBClientError


def _list(kb: KBClient, args) -> list:
    return kb.knowledge_bases()


def _search(kb: KBClient, args) -> list:
    options = {"top_k": args.top_k} if args.top_k else {}
    return kb.search(args.query, args.kb_ids, **options)


def _ask(kb: KBClient, args) -> dict:
    body = {"knowledge_base_ids": args.kb_ids, "output": "answer"}
    if args.preset:
        body["preset"] = args.preset
    return kb.agentic_search(args.question, **body)


def _call(kb: KBClient, args):
    return kb.request(args.method.upper(), args.path, json=json.loads(args.data) if args.data else None)


COMMANDS = {"list": _list, "search": _search, "ask": _ask, "call": _call}


def _one_line(text: str, width: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[:width - 1] + "…"


def _show(command: str, data) -> None:
    if command == "list":
        for item in data:
            print(f"{item['id']}  {item.get('name', '')}  ({item.get('type', '')}, {item.get('ref', '')})")
    elif command == "search":
        for n, item in enumerate(data, start=1):
            print(f"[{n}] {item.get('score', 0):.3f}  {item.get('knowledge_title') or ''}  "
                  f"(kb {item.get('knowledge_base_id', '')})\n    {_one_line(item.get('content', ''))}")
    elif command == "ask":
        print(data.get("answer") or "(no answer)")
        if data.get("sources"):
            print("\nSources:")
            for source in data["sources"]:
                print(f"  [{source['n']}] {source.get('citation_text') or source.get('title', '')}"
                      f"{'  ' + source['url'] if source.get('url') else ''}")
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))


def run(args) -> int:
    try:
        with KBClient.from_env(url=args.url, **args.auth) as kb:
            data = COMMANDS[args.kb_command](kb, args)
    except KBClientError as exc:
        print(f"agenticwf kb: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"agenticwf kb: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        _show(args.kb_command, data)
    return 0


def register(sub) -> None:
    """The `agenticwf.commands` entry point: adds `kb` to the `agenticwf` command."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", help="the KB's base URL, ending with /kb (AGENTICWF_KB_URL)")
    common.add_argument("--json", action="store_true", help="print the API's data as JSON")
    scope = argparse.ArgumentParser(add_help=False)
    scope.add_argument("--kb", dest="kb_ids", action="append", metavar="ID",
                       help="a knowledge base id (repeat for several; default: every KB you can read)")

    parser = sub.add_parser("kb", help="the knowledge service", description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.set_defaults(run=run)
    commands = parser.add_subparsers(dest="kb_command", required=True)
    commands.add_parser("list", parents=[common], help="your knowledge bases")
    search = commands.add_parser("search", parents=[common, scope], help="hybrid search")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, help="how many passages (default: the KB's setting)")
    ask = commands.add_parser("ask", parents=[common, scope], help="agentic search, with an answer")
    ask.add_argument("question")
    ask.add_argument("--preset", help="quick | deep | smart | wiki | auto (default: the KB's)")
    call = commands.add_parser("call", parents=[common], help="any KB API call")
    call.add_argument("method")
    call.add_argument("path", help="under /api/v1, e.g. /knowledge-bases")
    call.add_argument("--data", help="JSON request body")
