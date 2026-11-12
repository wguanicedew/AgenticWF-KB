"""
Search eval (docs/roadmap.md, phase 2 acceptance): plain search against agentic search
(`quick`, `deep`) on accuracy, latency and tokens, including multi-document questions.

    python -m agenticwf.kb.api.evaluate eval/phase2.yaml --url http://localhost:8766 --token-file ~/.agenticwf-kb/token \\
        [--kb <id or owner:slug> | --load] [--modes search,quick,deep] [--model-id …] [--json out.json]

The eval file holds a corpus (loaded into a new KB with `--load`) and questions, each naming the
document(s) that answer it. Per question and mode it records which expected documents appear in
the top `k` search results or in the agentic evidence, the wall time, and the model tokens.
It talks to the API only (httpx), so it measures a deployment as configured: point the KB at
real embedding, rerank and chat models for numbers that mean something.

Answers too: a question with `answer` (the reference answer, or the facts it must contain) is
asked with `output: answer` in the agentic modes, and with a judge model (`--judge-url`,
`--judge-model`, `--judge-key-env`; any OpenAI-compatible chat model, ideally a strong one) each
answer is graded `correct` (1 / 0.5 / 0 against the reference) and `faithful` (every claim backed
by the passages the search returned). `expect` may then be left out: real questions on an
existing KB (`--kb`) often name no single document (eval/template.yaml; the agent's admin page
exports badly rated replies in this format).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

MODES = ("search", "quick", "deep")
WIKI_MODES = ("quick", "wiki", "global")
# wiki: the wiki preset; global: the wiki's global search (both need a wiki-enabled KB, --wiki)
ALL_MODES = MODES + ("wiki", "global", "auto")       # auto: route → gather → merge


@dataclass
class Outcome:
    question: str
    mode: str
    expected: list[str]
    found: list[str]
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    stopped: str = ""
    error: str = ""
    kind: str = "single"
    answer: str = ""
    correct: float | None = None       # judged against the reference answer
    faithful: bool | None = None       # judged against the evidence
    judge_reason: str = ""

    @property
    def recall(self) -> float:
        return len(set(self.expected) & set(self.found)) / len(self.expected) if self.expected else 0.0

    @property
    def precision(self) -> float:
        """Share of the documents found that were expected: recall alone rewards returning
        everything (a small corpus is covered by any 30-passage evidence list)."""
        return len(set(self.expected) & set(self.found)) / len(self.found) if self.found else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def complete(self) -> bool:
        return set(self.expected) <= set(self.found)


@dataclass
class EvalSet:
    questions: list[dict]
    corpus: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "EvalSet":
        data = yaml.safe_load(Path(path).read_text()) or {}
        questions = []
        for q in data.get("questions") or []:
            expect = q.get("expect") or []
            questions.append({"q": q["q"], "expect": [expect] if isinstance(expect, str) else list(expect),
                              "answer": str(q.get("answer") or "").strip(),
                              "kind": q.get("kind") or ("multi" if not isinstance(expect, str) and len(expect) > 1
                                                        else "single")})
        return cls(questions=questions, corpus=list(data.get("corpus") or []))


class Evaluator:
    """`client` is any httpx-style client (httpx.Client, or Starlette's TestClient); `headers`
    returns the request headers for a method and path."""

    def __init__(self, client, base: str = "/kb/api/v1", headers=None, k: int = 5, model_id: str | None = None,
                 max_iterations: int | None = None, judge=None):
        self.client, self.base, self.k = client, base.rstrip("/"), k
        self.headers = headers or (lambda method, path: {})
        self.model_id, self.max_iterations = model_id, max_iterations
        self.judge = judge               # (question, reference, answer, passages) -> {correct, faithful, reason}
        self._sources: dict[str, list[str]] = {}

    def _call(self, method: str, path: str, **kw):
        response = self.client.request(method, self.base + path, headers=self.headers(method, self.base + path), **kw)
        body = response.json()
        if response.status_code >= 400 or not body.get("success"):
            raise RuntimeError((body.get("error") or {}).get("message") or f"HTTP {response.status_code}")
        return body

    # -- corpus ---------------------------------------------------------------------------------
    def load_corpus(self, corpus: list[dict], name: str = "eval", wait: float = 300.0, wiki: bool = False,
                    wiki_model_id: str | None = None) -> tuple[str, dict[str, str]]:
        """A new KB with every corpus document (manual items); waits until all are processed (with
        `wiki`, until the wiki is built too). Returns (kb id, knowledge id -> document name)."""
        body: dict = {"name": f"{name}-{int(time.time())}"}
        if wiki:
            body.update(visibility="private", indexing_strategy={"wiki_enabled": True})
            if wiki_model_id or self.model_id:
                body["wiki_config"] = {"synthesis_model_id": wiki_model_id or self.model_id}
        kb = self._call("POST", "/knowledge-bases", json=body)["data"]["id"]
        names = {}
        for doc in corpus:
            item = self._call("POST", f"/knowledge-bases/{kb}/knowledge/manual",
                              json={"title": doc["name"], "content": doc["content"]})["data"]
            names[item["id"]] = doc["name"]
        self.wait(kb, wait)
        if wiki:
            self.wait_wiki(kb, wait)
        return kb, names

    def wait_wiki(self, kb: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        settled = 0
        while time.monotonic() < deadline:
            if self._call("GET", f"/knowledgebase/{kb}/wiki/stats")["data"]["pending_ops"] == 0:
                settled += 1
                if settled >= 3:             # finalize and community pages follow the last batch
                    return
            else:
                settled = 0
            time.sleep(2.0)
        raise RuntimeError("the wiki was not built in time")

    def wiki_quality(self, kb: str) -> dict:
        return self._call("GET", f"/knowledgebase/{kb}/wiki/quality")["data"]

    def _page_sources(self, kb: str, slug: str) -> list[str]:
        if slug not in self._sources:
            try:
                self._sources[slug] = self._call("GET", f"/knowledgebase/{kb}/wiki/pages/{slug}")["data"]["source_ids"]
            except RuntimeError:
                self._sources[slug] = []
        return self._sources[slug]

    def wait(self, kb: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            items = self._call("GET", f"/knowledge-bases/{kb}/knowledge?page_size=100")["data"]
            if all(i["parse_status"] in ("completed", "failed") for i in items):
                return
            time.sleep(1.0)
        raise RuntimeError("the corpus was not processed in time")

    def names(self, kb: str) -> dict[str, str]:
        out, page = {}, 1
        while True:
            body = self._call("GET", f"/knowledge-bases/{kb}/knowledge?page={page}&page_size=100")
            for item in body["data"]:
                out[item["id"]] = item["title"] or item["file_name"]
            if page * 100 >= body["total"]:
                return out
            page += 1

    # -- one question ---------------------------------------------------------------------------
    def ask(self, kb: str, names: dict[str, str], question: dict, mode: str) -> Outcome:
        started = time.monotonic()
        out = Outcome(question["q"], mode, question["expect"], [], 0.0)
        try:
            if mode == "search":
                hits = self._call("POST", "/knowledge-search", json={
                    "query": question["q"], "knowledge_base_ids": [kb], "match_count": self.k})["data"]
                ids = [h["knowledge_id"] for h in hits]
            elif mode == "global":
                data = self._call("POST", f"/knowledgebase/{kb}/wiki/global-search",
                                  json={"query": question["q"], "level": 0})["data"]
                ids = [sid for slug in data["sources"] for sid in self._page_sources(kb, slug)]
                out.stopped = "answered" if data["answer"] else "no_results"
            else:
                graded = bool(question.get("answer"))
                body = {"query": question["q"], "knowledge_base_ids": [kb], "preset": mode,
                        "output": "answer" if graded else "evidence"}
                if self.model_id:
                    body["model_id"] = self.model_id
                if self.max_iterations:
                    body["max_iterations"] = self.max_iterations
                data = self._call("POST", "/agentic-search", json=body)["data"]
                ids = []
                for e in data["evidence"]:                 # a wiki page counts through the documents it cites
                    if e.get("knowledge_id"):
                        ids.append(e["knowledge_id"])
                    elif e.get("kind") == "wiki_page" and e.get("slug"):
                        ids += self._page_sources(kb, e["slug"])
                usage = data.get("usage") or {}
                out.input_tokens, out.output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                out.llm_calls, out.stopped = usage.get("llm_calls", 0), data.get("stopped", "")
                out.answer = data.get("answer") or ""
                if graded and self.judge is not None:
                    verdict = self.judge(question["q"], question["answer"], out.answer,
                                         [e.get("content") or "" for e in data["evidence"] if e.get("content")])
                    out.correct, out.faithful = verdict.get("correct"), verdict.get("faithful")
                    out.judge_reason = verdict.get("reason", "")
            out.found = list(dict.fromkeys(names.get(i, i) for i in ids))[:self.k if mode == "search" else None]
        except Exception as exc:          # one failing question must not end the run
            out.error = str(exc)
        out.seconds = time.monotonic() - started
        return out

    def run(self, kb: str, names: dict[str, str], questions: list[dict], modes=MODES) -> list[Outcome]:
        out = []
        for q in questions:
            for mode in modes:
                outcome = self.ask(kb, names, q, mode)
                outcome.kind = q.get("kind", "single")
                out.append(outcome)
        return out


def summarise(outcomes: list[Outcome]) -> dict[str, dict]:
    """Per mode: recall, complete answers, multi-document recall, latency, tokens, errors."""
    out = {}
    for mode in dict.fromkeys(o.mode for o in outcomes):
        everything = [o for o in outcomes if o.mode == mode]
        judged = [o for o in everything if o.correct is not None]
        rows = [o for o in everything if o.expected]      # retrieval scores need the expected documents
        multi = [o for o in rows if o.kind == "multi"]
        glob = [o for o in rows if o.kind == "global"]
        seconds = sorted(o.seconds for o in everything)
        out[mode] = {
            "questions": len(everything),
            "correct": round(statistics.mean(o.correct for o in judged), 3) if judged else None,
            "faithful": round(sum(bool(o.faithful) for o in judged) / len(judged), 3) if judged else None,
            "recall": round(statistics.mean(o.recall for o in rows), 3) if rows else 0.0,
            "precision": round(statistics.mean(o.precision for o in rows), 3) if rows else 0.0,
            "f1": round(statistics.mean(o.f1 for o in rows), 3) if rows else 0.0,
            "docs_per_answer": round(statistics.mean(len(o.found) for o in rows), 1) if rows else 0.0,
            "complete": sum(o.complete for o in rows),
            "multi_doc_recall": round(statistics.mean(o.recall for o in multi), 3) if multi else None,
            "global_recall": round(statistics.mean(o.recall for o in glob), 3) if glob else None,
            "multi_doc_f1": round(statistics.mean(o.f1 for o in multi), 3) if multi else None,
            "global_f1": round(statistics.mean(o.f1 for o in glob), 3) if glob else None,
            "latency_p50_s": round(statistics.median(seconds), 3) if seconds else 0.0,
            "latency_max_s": round(seconds[-1], 3) if seconds else 0.0,
            "input_tokens": sum(o.input_tokens for o in everything), "output_tokens": sum(o.output_tokens for o in everything),
            "llm_calls": sum(o.llm_calls for o in everything), "errors": sum(1 for o in everything if o.error),
        }
    return out


def table(summary: dict[str, dict]) -> str:
    cols = ("questions", "correct", "faithful", "recall", "precision", "f1", "docs_per_answer", "multi_doc_f1", "global_f1", "latency_p50_s", "latency_max_s", "input_tokens",
            "output_tokens", "llm_calls", "errors")
    lines = ["mode    " + " ".join(f"{c:>15}" for c in cols)]
    for mode, row in summary.items():
        lines.append(f"{mode:<8}" + " ".join(f"{str(row[c]):>15}" for c in cols))
    return "\n".join(lines)


JUDGE_PROMPT = """You grade an answer from a knowledge-base search system.

Question: {question}

Reference answer (what a correct answer must say): {reference}

Passages the system retrieved:
{passages}

System answer:
{answer}

Grade two things:
- correct: 1 if the answer states the reference's facts without contradicting them, 0.5 if it is \
partly right or leaves out an essential fact, 0 if it is wrong, says it found nothing while the \
reference has an answer, or answers another question.
- faithful: true if every factual claim in the answer is backed by the passages above (citations \
and phrasing aside), false if it adds facts the passages do not contain.

Reply with JSON only: {{"correct": <1|0.5|0>, "faithful": <true|false>, "reason": "<one sentence>"}}"""


def openai_judge(client, url: str, model: str, api_key: str = "", timeout: float = 120.0):
    """A judge on an OpenAI-compatible chat model: POST <url>/chat/completions."""
    import re

    def judge(question: str, reference: str, answer: str, passages: list[str]) -> dict:
        text = "\n\n".join(f"[{i + 1}] {' '.join(p.split())[:1500]}" for i, p in enumerate(passages[:15])) or "(none)"
        prompt = JUDGE_PROMPT.format(question=question, reference=reference, passages=text, answer=answer or "(no answer)")
        try:
            r = client.post(url.rstrip("/") + "/chat/completions", timeout=timeout,
                            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                            json={"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            content = re.sub(r"<think>.*?</think>", "", r.json()["choices"][0]["message"]["content"] or "", flags=re.S)
            verdict = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
            return {"correct": float(verdict.get("correct", 0)), "faithful": bool(verdict.get("faithful")),
                    "reason": str(verdict.get("reason", ""))}
        except Exception as exc:          # an unusable verdict is no verdict, not a wrong answer
            return {"correct": None, "faithful": None, "reason": f"judge failed: {exc}"}
    return judge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agenticwf.kb.api.evaluate", description=__doc__.split("\n\n")[0])
    parser.add_argument("eval_file")
    parser.add_argument("--url", default="http://localhost:8766", help="the KB's address (without /kb)")
    parser.add_argument("--base-path", default="/kb")
    parser.add_argument("--token-file", help="a bearer token (standalone: ~/.agenticwf-kb/token)")
    parser.add_argument("--kb", help="an existing KB holding the corpus (titles = document names)")
    parser.add_argument("--load", action="store_true", help="load the corpus into a new KB first")
    parser.add_argument("--wiki", action="store_true", help="with --load: build the KB's wiki and report its quality")
    parser.add_argument("--wait", type=float, default=300.0, help="seconds to wait for processing (and the wiki)")
    parser.add_argument("--modes", help=f"default {','.join(MODES)}; with --wiki {','.join(WIKI_MODES)}")
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per request (slow local models)")
    parser.add_argument("--model-id")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--json", help="write every outcome and the summary here")
    parser.add_argument("--judge-url", help="grade answers with this OpenAI-compatible chat API (e.g. https://aigw.cern.ch)")
    parser.add_argument("--judge-model", help="the judge's model name (a strong one)")
    parser.add_argument("--judge-key-env", help="environment variable holding the judge's API key")
    args = parser.parse_args(argv)

    evalset = EvalSet.load(args.eval_file)
    headers = {}
    if args.token_file:
        headers["Authorization"] = "Bearer " + Path(args.token_file).expanduser().read_text().strip()
    modes = [m for m in (args.modes or ",".join(WIKI_MODES if args.wiki else MODES)).split(",") if m]
    if any(m not in ALL_MODES for m in modes):
        parser.error(f"modes are {', '.join(ALL_MODES)}")
    if bool(args.judge_url) != bool(args.judge_model):
        parser.error("--judge-url and --judge-model go together")
    import os
    judge_client = httpx.Client(timeout=args.timeout) if args.judge_url else None
    judge = openai_judge(judge_client, args.judge_url, args.judge_model,
                         os.environ.get(args.judge_key_env or "", "")) if judge_client else None
    with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
        ev = Evaluator(client, args.base_path + "/api/v1", lambda m, p: headers, args.k, args.model_id,
                       args.max_iterations, judge)
        wiki_quality = None
        if args.load:
            kb, names = ev.load_corpus(evalset.corpus, wait=args.wait, wiki=args.wiki)
            if args.wiki:
                wiki_quality = ev.wiki_quality(kb)
                print("wiki quality:", json.dumps(wiki_quality, indent=1))
        elif args.kb:
            kb, names = args.kb, ev.names(args.kb)
        else:
            parser.error("give --kb or --load")
        outcomes = ev.run(kb, names, evalset.questions, modes)
    summary = summarise(outcomes)
    print(table(summary))
    for o in outcomes:
        if o.error or (o.expected and not o.complete):
            print(f"- [{o.mode}] {o.question!r}: expected {o.expected}, found {o.found[:5]} {o.error}", file=sys.stderr)
        if o.correct is not None and (o.correct < 1 or not o.faithful):
            print(f"- [{o.mode}] {o.question!r}: correct {o.correct}, faithful {o.faithful}: {o.judge_reason}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "wiki_quality": wiki_quality, "outcomes": [
            {**vars(o), "recall": o.recall, "precision": o.precision, "f1": o.f1} for o in outcomes]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
