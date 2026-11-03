"""Unique owner names: kb_owner_names.

Revision ID: 0012
Revises: 0011
Create Date: 2026-11-03

KB refs name the owner in a namespace: `u:<username>:<slug>` for a user, `g:<group>:<slug>` for
a group, and each name belongs to one owner (kb_owner_names). `create_all` makes the table on new
databases; this revision adds it to existing ones and fills it from kb_knowledge_bases.owner_name:
where several owners had the same name, the one with the oldest KB keeps it and the others' KBs
lose their owner name (their refs use the subject until they claim a name of their own).
"""
from alembic import op

revision = '0012'
down_revision = '0011'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS kb_owner_names (
            kind varchar(1) NOT NULL,
            name varchar(320) NOT NULL,
            requested varchar(320) NOT NULL,
            owner_subject varchar(255) NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (kind, name))""")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_kb_owner_names_owner ON kb_owner_names (owner_subject)")
    op.execute("""
        INSERT INTO kb_owner_names (kind, name, requested, owner_subject)
        SELECT kind, owner_name, owner_name, owner_subject FROM (
            SELECT CASE WHEN owner_subject LIKE 'org:%' THEN 'g' ELSE 'u' END AS kind,
                   owner_name, owner_subject, min(created_at) AS first
            FROM kb_knowledge_bases WHERE owner_name IS NOT NULL
            GROUP BY 1, 2, 3) owners
        ORDER BY first
        ON CONFLICT DO NOTHING""")
    op.execute("""
        UPDATE kb_knowledge_bases kb SET owner_name = NULL
        WHERE owner_name IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM kb_owner_names n WHERE n.owner_subject = kb.owner_subject AND n.name = kb.owner_name)""")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS kb_owner_names")
