"""Programmatic Alembic access (used by `python -m api.db init` through build_database()).

The database URL comes from the ``[database] default`` option of the core configuration, so there is
no separate alembic.ini to keep in step.
"""

from alembic import command
from alembic.config import Config

from kb_core.common.config import config_get
from kb_core.orm.base.session import DATABASE_SECTION, DEFAULT_SCHEMA_NAME

SCRIPT_LOCATION = "kb_core.orm.base:alembic"


def alembic_config(url: str | None = None, **kwargs) -> Config:
    cfg = Config(**kwargs)
    cfg.set_main_option("script_location", SCRIPT_LOCATION)
    # ConfigParser interpolation: a literal % (URL-encoded password) must be doubled.
    url = url or config_get(DATABASE_SECTION, "default")
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    if DEFAULT_SCHEMA_NAME:
        cfg.set_main_option("version_table_schema", DEFAULT_SCHEMA_NAME)
    return cfg


def upgrade(revision: str = "head", url: str | None = None) -> None:
    command.upgrade(alembic_config(url), revision)


def stamp(revision: str = "head", url: str | None = None) -> None:
    command.stamp(alembic_config(url), revision)
