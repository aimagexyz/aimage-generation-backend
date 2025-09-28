from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    """
    Data migration placeholder.
    The actual data migration is handled by the manual_rpd_migration.py script.
    """
    return """
        -- This migration is handled by manual_rpd_migration.py
        -- Run that script after applying the schema migration
        SELECT 1;
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    """
    Downgrade migration placeholder.
    """
    return """
        -- Downgrade is handled manually
        SELECT 1;
    """ 