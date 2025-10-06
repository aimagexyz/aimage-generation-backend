from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "generated_references" (
            "id" UUID NOT NULL PRIMARY KEY,
            "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "base_prompt" TEXT NOT NULL,
            "enhanced_prompt" TEXT NOT NULL,
            "tags" JSONB NOT NULL DEFAULT '{}',
            "image_path" VARCHAR(1024) NOT NULL,
            "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE,
            "created_by_id" UUID NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
        );
        CREATE INDEX "idx_generated_references_project_created" ON "generated_references" ("project_id", "created_at");
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_generated_references_project_created";
        DROP TABLE IF EXISTS "generated_references";
    """ 