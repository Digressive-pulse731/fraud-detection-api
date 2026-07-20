"""PostgreSQL connection via SQLAlchemy.

Engine/session creation is lazy so importing this module never requires
a reachable database (keeps tests and the training script DB-free).
All connection settings come from the environment — never hardcoded.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def database_url() -> str:
    return (
        "postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}".format(
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            host=os.environ.get("DB_HOST", "localhost"),
            port=os.environ.get("DB_PORT", "5432"),
            name=os.environ["DB_NAME"],
        )
    )


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(database_url(), pool_pre_ping=True)
    return _engine


def get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine())
    return _session_factory
