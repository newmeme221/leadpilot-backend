from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import create_async_engine
from alembic import context
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import SQLModel
from database import DATABASE_URL

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True, compare_type=True
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    import asyncio
    async def do_run_migrations():
        connectable = create_async_engine(DATABASE_URL)
        async with connectable.connect() as connection:
            await connection.run_sync(
                lambda sync_conn: context.configure(
                    connection=sync_conn, target_metadata=target_metadata, compare_type=True
                )
            )
            async with connection.begin():
                await connection.run_sync(lambda _: context.run_migrations())
    asyncio.run(do_run_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
