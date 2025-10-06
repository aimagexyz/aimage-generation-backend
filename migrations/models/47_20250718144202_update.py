from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE INDEX IF NOT EXISTS "idx_tasks_project_32b4a7" ON "tasks" ("project_id", "status_id", "priority_id", "assignee_id", "created_at");"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_tasks_project_32b4a7";"""
