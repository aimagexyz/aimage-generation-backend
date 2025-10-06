from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "generated_references" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "base_prompt" TEXT NOT NULL,
    "enhanced_prompt" TEXT NOT NULL,
    "image_path" VARCHAR(1024) NOT NULL,
    "tags" JSONB NOT NULL,
    "created_by_id" UUID NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE
);
COMMENT ON TABLE "generated_references" IS 'AI-generated reference images - MVP version';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "generated_references";"""
