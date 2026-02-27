import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config, create_async_engine

from alembic import context

# --- CUSTOM IMPORTS ---
# Import your config and models so Alembic knows the schema and the URL
from src.core.config import settings
from src.database.models import Base
# ----------------------

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the metadata for 'autogenerate' support
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    # Pull URL from our Pydantic settings instead of alembic.ini
    url = settings.DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()

# alembic/env.py inside run_async_migrations()

async def run_async_migrations() -> None:
    # Force the driver to be asyncpg
    url = settings.DATABASE_URL.replace("postgres://", "postgresql://")
    if "postgresql+asyncpg://" not in url:
        database_url = url.replace("postgresql://", "postgresql+asyncpg://")
    else:
        database_url = url

    connectable = create_async_engine(
        database_url,
        poolclass=pool.NullPool,
        connect_args={"ssl": True}
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Use the current event loop if it exists, otherwise use asyncio.run
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # This handles cases where we might be running inside an existing loop
        loop.create_task(run_async_migrations())
    else:
        asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()