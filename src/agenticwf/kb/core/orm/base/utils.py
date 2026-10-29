"""
Utils to create the database or destroy the database
"""

import logging
import traceback

import sqlalchemy
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql.base import PGInspector
from sqlalchemy.schema import CreateSchema, MetaData, Table, DropTable, ForeignKeyConstraint, DropConstraint
from sqlalchemy.sql.ddl import DropSchema

from agenticwf.kb.core.common.config import config_get, config_has_option
from agenticwf.kb.core.common.exceptions import DatabaseException
from agenticwf.kb.core.orm.base import migrate, models, session

LOG = logging.getLogger(__name__)

# The KB needs pgvector (vectors), ParadeDB pg_search (BM25) and pg_trgm (grep). The container's
# entrypoint creates them as the superuser; the `kb` role cannot. Tests against a server without
# one of them may narrow the list with [database] required_extensions.
REQUIRED_EXTENSIONS = ('vector', 'pg_trgm', 'pg_search')


def required_extensions() -> tuple[str, ...]:
    if config_has_option(session.DATABASE_SECTION, 'required_extensions'):
        return tuple(config_get(session.DATABASE_SECTION, 'required_extensions').replace(',', ' ').split())
    return REQUIRED_EXTENSIONS


def installed_extensions(conn) -> dict[str, str]:
    """ name -> version of the extensions created in the current database. """
    rows = conn.execute(sqlalchemy.text('SELECT extname, extversion FROM pg_extension'))
    return {name: version for name, version in rows}


def check_extensions(engine) -> dict[str, str]:
    """ Fail with a clear message unless every required extension exists in the database. """
    with engine.connect() as conn:
        found = installed_extensions(conn)
    missing = [name for name in required_extensions() if name not in found]
    if missing:
        statements = ' '.join(f'CREATE EXTENSION IF NOT EXISTS {name};' for name in missing)
        raise DatabaseException(
            f'missing PostgreSQL extensions: {", ".join(missing)}. The container entrypoint creates them; '
            f'on another server run, as a superuser in this database: {statements}')
    return found


def build_database(echo=True, tests=False):
    """Build the database: check the extensions, the configured schema, missing tables, then
    Alembic to head. """
    engine = session.get_engine(echo=echo)
    check_extensions(engine)

    schema = session.DEFAULT_SCHEMA_NAME
    if schema:
        try:
            with engine.begin() as conn:
                conn.execute(CreateSchema(schema, if_not_exists=True))
        except Exception as e:
            LOG.warning('Cannot create schema %s, please validate manually if schema creation is needed, '
                        'continuing: %s', schema, e)

    models.register_models(engine)   # CREATE TABLE IF NOT EXISTS for every model

    # Revisions are idempotent (they check before they alter), so a database without a recorded
    # revision runs all of them: tables that existed before create_all get their new columns.
    migrate.upgrade("head")
    return engine


def destroy_database(echo=True):
    """Destroy the database"""
    engine = session.get_engine(echo=echo)

    try:
        models.unregister_models(engine)
    except Exception as e:
        print('Cannot destroy schema -- assuming already gone, continuing:', e)
        print(traceback.format_exc())


def destroy_everything(echo=True):
    """ Using metadata.reflect() to get all constraints and tables.
        metadata.drop_all() as it handles cyclical constraints between tables.
        Ref. http://www.sqlalchemy.org/trac/wiki/UsageRecipes/DropEverything
    """
    engine = session.get_engine(echo=echo)

    try:
        # the transaction only applies if the DB supports
        # transactional DDL, i.e. Postgresql, MS SQL Server
        with engine.begin() as conn:

            inspector = inspect(conn)

            for tname, fkcs in reversed(
                    inspector.get_sorted_table_and_fkc_names(schema='*')):
                if tname:
                    drop_table_stmt = DropTable(Table(tname, MetaData(), schema='*'))
                    conn.execute(drop_table_stmt)
                elif fkcs:
                    if not engine.dialect.supports_alter:
                        continue
                    for tname, fkc in fkcs:
                        fk_constraint = ForeignKeyConstraint((), (), name=fkc)
                        Table(tname, MetaData(), fk_constraint)
                        drop_constraint_stmt = DropConstraint(fk_constraint)
                        conn.execute(drop_constraint_stmt)

            if session.DEFAULT_SCHEMA_NAME:
                conn.execute(DropSchema(session.DEFAULT_SCHEMA_NAME, cascade=True))

            if engine.dialect.name == 'postgresql':
                assert isinstance(inspector, PGInspector), 'expected a PGInspector'
                for enum in inspector.get_enums(schema='*'):
                    sqlalchemy.Enum(**enum).drop(bind=conn)

    except Exception as e:
        print('Cannot destroy db:', e)
        print(traceback.format_exc())


def dump_schema():
    """ Creates a schema dump to a specific database. """
    engine = session.get_dump_engine()
    models.unregister_models(engine)
    print("\n")
    models.register_models(engine)


def row2dict(row):
    """ Convert a result row to dict. """
    return {str(col): value for col, value in row._mapping.items()}


def rows2dict(rows):
    """ Convert result rows to dicts. """
    return [row2dict(row) for row in rows]
