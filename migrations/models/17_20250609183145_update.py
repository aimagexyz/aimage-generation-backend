from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "illustration_docs" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "task_id" UUID NOT NULL UNIQUE REFERENCES "tasks" ("id") ON DELETE CASCADE
);
        CREATE TABLE IF NOT EXISTS "illustration_doc_contents" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "prompt" TEXT NOT NULL,
    "tags" JSONB,
    "image_s3_path" VARCHAR(1024) NOT NULL,
    "doc_id" UUID NOT NULL REFERENCES "illustration_docs" ("id") ON DELETE CASCADE
);
COMMENT ON COLUMN "illustration_doc_contents"."tags" IS 'Tags for the illustration';
COMMENT ON COLUMN "illustration_doc_contents"."image_s3_path" IS 'S3 path for the generated image';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "illustration_doc_contents";
        DROP TABLE IF EXISTS "illustration_docs";"""
