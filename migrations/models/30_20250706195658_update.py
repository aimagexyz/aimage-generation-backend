from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "review_sets" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" VARCHAR(255) NOT NULL,
    "description" TEXT,
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_review_sets_name_c33e38" UNIQUE ("name", "project_id")
);
        CREATE TABLE IF NOT EXISTS "task_tags" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" VARCHAR(255) NOT NULL,
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_task_tags_name_7eada3" UNIQUE ("name", "project_id")
);
CREATE INDEX IF NOT EXISTS "idx_task_tags_name_5b0274" ON "task_tags" ("name");
        CREATE TABLE "review_sets_characters" (
    "character_id" UUID NOT NULL REFERENCES "characters" ("id") ON DELETE CASCADE,
    "review_sets_id" UUID NOT NULL REFERENCES "review_sets" ("id") ON DELETE CASCADE
);
        CREATE TABLE "review_sets_rpds" (
    "reviewpointdefinition_id" UUID NOT NULL REFERENCES "review_point_definitions" ("id") ON DELETE CASCADE,
    "review_sets_id" UUID NOT NULL REFERENCES "review_sets" ("id") ON DELETE CASCADE
);
        CREATE TABLE "tasks_tags" (
    "tasks_id" UUID NOT NULL REFERENCES "tasks" ("id") ON DELETE CASCADE,
    "tasktag_id" UUID NOT NULL REFERENCES "task_tags" ("id") ON DELETE CASCADE
);
        CREATE TABLE "review_sets_task_tags" (
    "tasktag_id" UUID NOT NULL REFERENCES "task_tags" ("id") ON DELETE CASCADE,
    "review_sets_id" UUID NOT NULL REFERENCES "review_sets" ("id") ON DELETE CASCADE
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "review_sets_rpds";
        DROP TABLE IF EXISTS "review_sets_characters";
        DROP TABLE IF EXISTS "tasks_tags";
        DROP TABLE IF EXISTS "review_sets";
        DROP TABLE IF EXISTS "task_tags";"""
