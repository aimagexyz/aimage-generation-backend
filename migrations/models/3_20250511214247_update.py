from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "documents" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "file_path" VARCHAR(1024) NOT NULL,
    "file_name" VARCHAR(255) NOT NULL,
    "file_size" INT NOT NULL,
    "file_type" VARCHAR(255),
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_documents_file_pa_4ed879" ON "documents" ("file_path");
CREATE INDEX IF NOT EXISTS "idx_documents_file_na_30f77d" ON "documents" ("file_name");
        CREATE TABLE IF NOT EXISTS "reference_cards" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "title" VARCHAR(255) NOT NULL,
    "description" TEXT NOT NULL,
    "source" VARCHAR(9) NOT NULL DEFAULT 'anime',
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_reference_c_title_ff71f8" ON "reference_cards" ("title");
COMMENT ON COLUMN "reference_cards"."source" IS 'ANIME: anime\nHISTORY: history\nCHARACTER: character';
        ALTER TABLE "subtasks" ADD "ai_review_output" JSONB;
        ALTER TABLE "subtasks" ALTER COLUMN "content" TYPE JSONB USING "content"::JSONB;
        CREATE INDEX IF NOT EXISTS "idx_tasks_s3_path_1fe1f8" ON "tasks" ("s3_path");"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_tasks_s3_path_1fe1f8";
        ALTER TABLE "subtasks" DROP COLUMN "ai_review_output";
        ALTER TABLE "subtasks" ALTER COLUMN "content" TYPE JSONB USING "content"::JSONB;
        DROP TABLE IF EXISTS "documents";
        DROP TABLE IF EXISTS "reference_cards";"""
