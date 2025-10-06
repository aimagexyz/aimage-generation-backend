from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- Add index on project_id for filtering by project (very common query)
        CREATE INDEX IF NOT EXISTS "idx_items_project_id" ON "items" ("project_id");
        
        -- Add index on uploaded_by_id for filtering by user
        CREATE INDEX IF NOT EXISTS "idx_items_uploaded_by_id" ON "items" ("uploaded_by_id");
        
        -- Add composite index for common query pattern: project + created_at
        CREATE INDEX IF NOT EXISTS "idx_items_project_created" ON "items" ("project_id", "created_at" DESC);
        
        -- Add composite index for user + created_at
        CREATE INDEX IF NOT EXISTS "idx_items_user_created" ON "items" ("uploaded_by_id", "created_at" DESC);
        
        -- Add index on created_at for sorting (most common sort field)
        CREATE INDEX IF NOT EXISTS "idx_items_created_at" ON "items" ("created_at" DESC);
        
        -- Add GIN index on tags for JSON contains queries
        CREATE INDEX IF NOT EXISTS "idx_items_tags_gin" ON "items" USING GIN ("tags");
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_items_project_id";
        DROP INDEX IF EXISTS "idx_items_uploaded_by_id";
        DROP INDEX IF EXISTS "idx_items_project_created";
        DROP INDEX IF EXISTS "idx_items_user_created";
        DROP INDEX IF EXISTS "idx_items_created_at";
        DROP INDEX IF EXISTS "idx_items_tags_gin";
    """