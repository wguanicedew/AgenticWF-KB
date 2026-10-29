"""
FAQ entry rules and search post-processing (docs/faq.md, Rules and FAQ priority). Pure
functions, unit-tested without a database.

    normalise          NFKC, casefold, punctuation and whitespace removed
    clean_entry        validate one raw entry → (fields, warnings) or Invalid
    entry_content      the chunk text a model reads
    index_texts        the kb_embeddings texts of one entry, per faq_config
    drop_negatives     a query equal to an entry's negative question never returns it
    apply_priority     order FAQ hits by score × boost; mark the first close one `match: exact`
"""

from __future__ import annotations

import re
import unicodedata

from agenticwf.kb.core.common.config import config_get, config_has_option
from agenticwf.kb.core.orm.models.kb import SOURCE_CHUNK, SOURCE_QUESTION
from agenticwf.kb.runtime.errors import Invalid

MAX_QUESTION = 1000
MAX_ANSWER = 10_000
MAX_LIST = 200
INDEX_MODES = ("question_answer", "question_only")
QUESTION_INDEX_MODES = ("combined", "separate")
_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def normalise(text: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text or "").casefold())


def faq_config(kb) -> dict:
    cfg = dict((kb.indexing_strategy or {}).get("faq") or {})
    return {"index_mode": cfg.get("index_mode") or "question_answer",
            "question_index_mode": cfg.get("question_index_mode") or "combined"}


def check_faq_config(value: dict | None) -> dict:
    value = dict(value or {})
    out = {"index_mode": value.get("index_mode") or "question_answer",
           "question_index_mode": value.get("question_index_mode") or "combined"}
    if out["index_mode"] not in INDEX_MODES:
        raise Invalid(f"faq_config.index_mode must be one of {', '.join(INDEX_MODES)}")
    if out["question_index_mode"] not in QUESTION_INDEX_MODES:
        raise Invalid(f"faq_config.question_index_mode must be one of {', '.join(QUESTION_INDEX_MODES)}")
    return out


def _strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split("##")
    out = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out[:MAX_LIST]


def _flag(value, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on", "y")


def clean_entry(raw: dict, base: dict | None = None, merge: bool = False) -> tuple[dict, list[str]]:
    """The entry's fields after the rules. `base` is the stored entry being updated: fields not
    given are kept; with `merge` (append mode) similar and negative questions are merged in."""
    base = base or {}
    warnings: list[str] = []

    def given(name):
        return name in raw and raw[name] is not None

    standard = str(raw["standard_question"]).strip() if given("standard_question") else base.get("standard_question", "")
    if not standard:
        raise Invalid("a standard question is required")
    if len(standard) > MAX_QUESTION:
        raise Invalid(f"the standard question is longer than {MAX_QUESTION} characters")
    answers = _strings(raw["answers"]) if given("answers") else list(base.get("answers") or [])
    if not answers:
        raise Invalid("at least one answer is required")
    if any(len(a) > MAX_ANSWER for a in answers):
        raise Invalid(f"an answer is longer than {MAX_ANSWER} characters")

    def merged(name):
        new = _strings(raw[name]) if given(name) else None
        old = list(base.get(name) or [])
        if new is None:
            return old
        return _strings(old + new) if merge else new

    similar = merged("similar_questions")
    std_key = normalise(standard)
    kept, seen = [], {std_key}
    for question in similar:
        key = normalise(question)
        if not key or key in seen:
            continue
        if len(question) > MAX_QUESTION:
            warnings.append(f"similar question longer than {MAX_QUESTION} characters dropped")
            continue
        seen.add(key)
        kept.append(question)
    negative = []
    for question in merged("negative_questions"):
        if normalise(question) in seen:
            warnings.append(f"negative question {question!r} equals a positive question and was removed")
            continue
        negative.append(question)
    strategy = raw.get("answer_strategy") or base.get("answer_strategy") or "all"
    if strategy not in ("all", "random"):
        raise Invalid("answer_strategy must be all or random")
    return {"standard_question": standard, "similar_questions": kept, "negative_questions": negative,
            "answers": answers, "answer_strategy": strategy,
            "is_enabled": _flag(raw.get("is_enabled"), base.get("is_enabled", True)),
            "is_recommended": _flag(raw.get("is_recommended"), base.get("is_recommended", True))}, warnings


def entry_content(fields: dict) -> str:
    return f"Q: {fields['standard_question']}\nAnswer:\n" + "\n".join(f"- {a}" for a in fields["answers"])


def index_texts(chunk_id: str, fields: dict, cfg: dict) -> list[tuple[str, int, str]]:
    """(source_id, source_type, text) rows for one entry (docs/faq.md, Indexing)."""
    tail = ("\n" + "\n".join(fields["answers"])) if cfg["index_mode"] == "question_answer" else ""
    if cfg["question_index_mode"] == "separate":
        rows = [(chunk_id, SOURCE_CHUNK, fields["standard_question"] + tail)]
        rows += [(f"{chunk_id}:s{i}", SOURCE_QUESTION, q + tail) for i, q in enumerate(fields["similar_questions"])]
        return rows
    return [(chunk_id, SOURCE_CHUNK, "\n".join([fields["standard_question"], *fields["similar_questions"]]) + tail)]


def comparable(fields: dict, tag_id: str) -> tuple:
    return (fields["standard_question"], tuple(fields["similar_questions"]), tuple(fields["negative_questions"]),
            tuple(fields["answers"]), fields["answer_strategy"], fields["is_enabled"], fields["is_recommended"],
            tag_id or "")


# --- search post-processing ---------------------------------------------------------------------------
def priority_settings() -> dict:
    out = {"priority_enabled": True, "score_boost": 1.2, "direct_answer_threshold": 0.9}
    for key, value in out.items():
        if config_has_option("faq", key):
            raw = config_get("faq", key).strip()
            out[key] = raw.lower() in ("1", "true", "yes", "on") if isinstance(value, bool) else float(raw)
    return out


def drop_negatives(query: str, items: list[dict]) -> list[dict]:
    key = normalise(query)
    if not key:
        return items
    return [i for i in items if i.get("chunk_type") != "faq"
            or key not in {normalise(q) for q in (i.get("metadata") or {}).get("negative_questions") or []}]


def apply_priority(items: list[dict], top_k: int, settings: dict | None = None) -> list[dict]:
    """Order by score × boost for FAQ hits (the returned score stays the pre-boost one), cut to
    top_k, and mark the first FAQ hit at or above the threshold `match: exact`."""
    settings = settings or priority_settings()
    if not settings["priority_enabled"] or not any(i.get("chunk_type") == "faq" for i in items):
        return items[:top_k]
    boost = float(settings["score_boost"])

    def key(item):
        score = item["score"] * (boost if item.get("chunk_type") == "faq" else 1.0)
        return (-score, item.get("knowledge_id", ""), item.get("chunk_index", 0))

    ordered = sorted(items, key=key)[:top_k]
    for item in ordered:
        if item.get("chunk_type") == "faq" and item["score"] >= float(settings["direct_answer_threshold"]):
            item["match"] = "exact"
            break
    return ordered


def matched_question(query: str, fields: dict) -> str:
    """The entry question closest to the query: equal after normalising, else the most shared words."""
    questions = [fields.get("standard_question", ""), *(fields.get("similar_questions") or [])]
    key = normalise(query)
    for question in questions:
        if normalise(question) == key:
            return question
    words = set(re.findall(r"\w+", query.casefold()))
    return max(questions, key=lambda q: len(words & set(re.findall(r"\w+", q.casefold())))) if questions else ""
