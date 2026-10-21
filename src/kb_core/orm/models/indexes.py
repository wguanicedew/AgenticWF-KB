"""
Indexes that depend on configuration or data, so `create_all` cannot make them:

* the BM25 index on kb_embeddings (ParadeDB pg_search), with the deployment's tokenizer
  `[retrieval] bm25_tokenizer` (default `icu`). Created by revision 0002; rebuilt by
  `python -m kb_api.db reindex-bm25` when the tokenizer changes;
* one HNSW index per embedding dimension, created the first time a KB indexes with that
  dimension (`ensure_vector_index`), in a short transaction of its own under an advisory lock,
  before any transaction that writes rows (CREATE INDEX waits for open writers).
"""

from __future__ import annotations

import json
import zlib

import sqlalchemy as sa

from kb_core.common.config import config_get, config_has_option

BM25_INDEX = "kb_embeddings_bm25"
# Filter columns are indexed too, so pg_search can apply them during the search.
BM25_FIELDS = ("id", "content", "knowledge_base_id", "knowledge_id", "tag_id", "source_type", "owner_subject",
               "visibility", "is_enabled", "is_deprecated")
TOKENIZERS = ("icu", "default", "raw", "whitespace", "chinese_lindera", "chinese_compatible", "jieba",
              "korean_lindera", "japanese_lindera")


def bm25_tokenizer() -> str:
    value = config_get("retrieval", "bm25_tokenizer").strip() if config_has_option("retrieval", "bm25_tokenizer") \
        else "icu"
    if value not in TOKENIZERS:
        raise ValueError(f"[retrieval] bm25_tokenizer must be one of {', '.join(TOKENIZERS)}, not {value!r}")
    return value


def bm25_index_sql(tokenizer: str | None = None) -> str:
    tokenizer = tokenizer or bm25_tokenizer()
    text_fields = json.dumps({"content": {"tokenizer": {"type": tokenizer}}})
    return (f"CREATE INDEX IF NOT EXISTS {BM25_INDEX} ON kb_embeddings USING bm25 ({', '.join(BM25_FIELDS)}) "
            f"WITH (key_field = 'id', text_fields = '{text_fields}')")


def has_index(conn, name: str) -> bool:
    return conn.execute(sa.text("SELECT 1 FROM pg_indexes WHERE indexname = :n"), {"n": name}).first() is not None


def ensure_bm25_index(conn, tokenizer: str | None = None) -> None:
    conn.execute(sa.text(bm25_index_sql(tokenizer)))


def rebuild_bm25_index(conn, tokenizer: str | None = None) -> None:
    conn.execute(sa.text(f"DROP INDEX IF EXISTS {BM25_INDEX}"))
    ensure_bm25_index(conn, tokenizer)


def hnsw_index_name(dimension: int) -> str:
    return f"kb_embeddings_hnsw_{int(dimension)}"


def ensure_vector_index(engine, dimension: int) -> None:
    """Create the HNSW index for `dimension` if missing. pgvector's halfvec HNSW supports up to
    4000 dimensions."""
    dimension = int(dimension)
    if not 1 <= dimension <= 4000:
        raise ValueError(f"embedding dimension {dimension} is outside 1..4000 (halfvec HNSW)")
    name = hnsw_index_name(dimension)
    with engine.connect() as conn:
        if has_index(conn, name):
            return
    with engine.begin() as conn:
        conn.execute(sa.text("SELECT pg_advisory_xact_lock(:k)"), {"k": zlib.crc32(name.encode())})
        if has_index(conn, name):
            return
        conn.execute(sa.text(
            f"CREATE INDEX IF NOT EXISTS {name} ON kb_embeddings "
            f"USING hnsw ((embedding::halfvec({dimension})) halfvec_cosine_ops) WITH (m = 16, ef_construction = 64) "
            f"WHERE dimension = {dimension}"))


def memory_bm25_index_sql(tokenizer: str | None = None) -> str:
    """BM25 over memory facts (docs/data-model.md, Temporal memory), same tokenizer as kb_embeddings."""
    tokenizer = tokenizer or bm25_tokenizer()
    text_fields = json.dumps({"fact": {"tokenizer": {"type": tokenizer}}})
    return ("CREATE INDEX IF NOT EXISTS kb_memory_facts_bm25 ON kb_memory_facts USING bm25 "
            "(seq, fact, owner_subject, group_id) "
            f"WITH (key_field = 'seq', text_fields = '{text_fields}')")
