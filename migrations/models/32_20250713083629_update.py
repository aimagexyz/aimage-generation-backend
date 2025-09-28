from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "user_preferences" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "liked_images" JSONB NOT NULL,
    "settings" JSONB NOT NULL,
    "user_id" UUID NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
COMMENT ON COLUMN "user_preferences"."liked_images" IS 'List of liked image URLs or S3 paths';
COMMENT ON COLUMN "user_preferences"."settings" IS 'Additional user settings';
COMMENT ON TABLE "user_preferences" IS 'User preferences including liked images';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "user_preferences";"""
