"""Create or upgrade the KB's database schema.

    python -m agenticwf.kb.api.db init       check the extensions, create missing tables, Alembic to head
    python -m agenticwf.kb.api.db check      only check that vector, pg_trgm and pg_search exist
    python -m agenticwf.kb.api.db reindex-bm25   rebuild the BM25 index ([retrieval] bm25_tokenizer changed)
    python -m agenticwf.kb.api.db reassign-owner --from local:wguan --to <logto subject>
                                   move everything one owner has to another (a personal KB moving
                                   behind authsvc); refuses when a KB slug would collide

The database URL is ``[database] default`` of kb.cfg (AGENTICWF_KB_CONFIG). Alembic uses
agenticwf.kb.core.orm.base.alembic with that same URL, so there is no separate alembic.ini.
"""

from __future__ import annotations

import argparse
import sys

from agenticwf.kb.core.common.exceptions import DatabaseException
from agenticwf.kb.core.common.logging import get_logger
from agenticwf.kb.core.orm.base import session as db_session
from agenticwf.kb.core.orm.base import utils as db_utils

import agenticwf.kb.core.orm.models.kb  # noqa: F401  (registers the tables on BASE.metadata)

LOG = get_logger(__name__)


def init_db() -> None:
    engine = db_utils.build_database()   # extensions, missing tables, then Alembic to head
    LOG.info("database schema ready (%s)", engine.url.render_as_string(hide_password=True))


def check_db() -> dict[str, str]:
    return db_utils.check_extensions(db_session.get_engine())


def reindex_bm25() -> None:
    from agenticwf.kb.core.orm.models.indexes import bm25_tokenizer, rebuild_bm25_index

    tokenizer = bm25_tokenizer()
    with db_session.get_engine().begin() as conn:
        rebuild_bm25_index(conn, tokenizer)
    LOG.info("BM25 index rebuilt with tokenizer %s", tokenizer)


OWNED_TABLES = ("kb_knowledges", "kb_embeddings", "kb_models", "kb_tasks", "kb_events")


def reassign_owner(old: str, new: str) -> dict[str, int]:
    """Change the owner subject `old` to `new` everywhere, in one transaction."""
    import sqlalchemy as sa

    if not old or not new or old == new:
        raise ValueError("--from and --to must be two different subjects")
    with db_session.get_engine().begin() as conn:
        clash = conn.execute(sa.text(
            "SELECT a.slug FROM kb_knowledge_bases a JOIN kb_knowledge_bases b ON a.slug = b.slug "
            "WHERE a.owner_subject = :old AND b.owner_subject = :new "
            "AND a.deleted_at IS NULL AND b.deleted_at IS NULL"), {"old": old, "new": new}).scalars().all()
        if clash:
            raise ValueError(f"{new} already has knowledge bases named {', '.join(clash)}; rename one first")
        counts = {"kb_knowledge_bases": conn.execute(sa.text(
            "UPDATE kb_knowledge_bases SET owner_subject = :new WHERE owner_subject = :old"),
            {"old": old, "new": new}).rowcount}
        for table in OWNED_TABLES:
            counts[table] = conn.execute(sa.text(f"UPDATE {table} SET owner_subject = :new WHERE owner_subject = :old"),
                                         {"old": old, "new": new}).rowcount
    LOG.info("reassigned %s to %s: %s", old, new, counts)
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m agenticwf.kb.api.db", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["init", "check", "reindex-bm25", "reassign-owner"])
    parser.add_argument("--from", dest="old", help="reassign-owner: the current owner subject")
    parser.add_argument("--to", dest="new", help="reassign-owner: the new owner subject")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            init_db()
        elif args.command == "reindex-bm25":
            reindex_bm25()
        elif args.command == "reassign-owner":
            for table, count in reassign_owner(args.old, args.new).items():
                print(f"{table} {count}")
        else:
            for name, version in sorted(check_db().items()):
                print(f"{name} {version}")
    except (DatabaseException, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
