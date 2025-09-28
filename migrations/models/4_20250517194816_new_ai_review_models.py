from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "subtasks_reference_cards";

        CREATE TABLE IF NOT EXISTS "review_point_definitions" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "key" VARCHAR(255) NOT NULL UNIQUE,
    "is_active" BOOL NOT NULL DEFAULT True
);
COMMENT ON COLUMN "review_point_definitions"."key" IS 'e.g., "general_ng_review", "visual_review"';

        CREATE TABLE IF NOT EXISTS "review_point_definition_versions" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "version_number" INT NOT NULL,
    "title" VARCHAR(255) NOT NULL,
    "description_for_ai" TEXT NOT NULL,
    "is_active_version" BOOL NOT NULL DEFAULT True,
    "created_by" VARCHAR(255),
    "review_point_definition_id" UUID NOT NULL REFERENCES "review_point_definitions" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_review_poin_review__e4d6ed" UNIQUE ("review_point_definition_id", "version_number")
);

        CREATE TABLE IF NOT EXISTS "ai_reviews" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "version" INT NOT NULL,
    "is_latest" BOOL NOT NULL DEFAULT True,
    "ai_review_output_json" JSONB,
    "review_timestamp" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "initiated_by_user_id" UUID REFERENCES "users" ("id") ON DELETE SET NULL,
    "last_modified_by_user_id" UUID REFERENCES "users" ("id") ON DELETE SET NULL,
    "subtask_id" UUID NOT NULL REFERENCES "subtasks" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_ai_reviews_subtask_946edb" UNIQUE ("subtask_id", "version")
);
        CREATE TABLE IF NOT EXISTS "ai_review_finding_entries" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "description" TEXT NOT NULL,
    "severity" VARCHAR(50) NOT NULL,
    "suggestion" TEXT,
    "area_x" INT,
    "area_y" INT,
    "area_width" INT,
    "area_height" INT,
    "citation_reference_images_json" JSONB,
    "citation_reference_source" TEXT,
    "is_ai_generated" BOOL NOT NULL,
    "status" VARCHAR(100) NOT NULL,
    "ai_review_id" UUID NOT NULL REFERENCES "ai_reviews" ("id") ON DELETE CASCADE,
    "original_ai_finding_id" UUID REFERENCES "ai_review_finding_entries" ("id") ON DELETE SET NULL,
    "review_point_definition_version_id" UUID NOT NULL REFERENCES "review_point_definition_versions" ("id") ON DELETE RESTRICT
);
        CREATE TABLE IF NOT EXISTS "promoted_findings" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "subtask_id_context" UUID NOT NULL,
    "promotion_timestamp" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "notes" TEXT,
    "tags" JSONB,
    "sharing_scope" VARCHAR(100),
    "promoted_by_user_id" UUID REFERENCES "users" ("id") ON DELETE SET NULL,
    "finding_entry_id" UUID NOT NULL UNIQUE REFERENCES "ai_review_finding_entries" ("id") ON DELETE CASCADE
);
COMMENT ON COLUMN "promoted_findings"."subtask_id_context" IS 'Stores the UUID of the Subtask for context, not a direct FK to allow Subtask deletion without affecting KB.';

        CREATE TABLE IF NOT EXISTS "prompts" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" VARCHAR(255) NOT NULL,
    "description" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_prompts_name_721503" ON "prompts" ("name");

        ALTER TABLE "subtasks" DROP COLUMN "ai_review_output";
        DROP TABLE IF EXISTS "reference_cards";"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "subtasks" ADD "ai_review_output" JSONB;
        DROP TABLE IF EXISTS "ai_reviews";
        DROP TABLE IF EXISTS "promoted_findings";
        DROP TABLE IF EXISTS "review_point_definitions";
        DROP TABLE IF EXISTS "review_point_definition_versions";
        DROP TABLE IF EXISTS "prompts";
        DROP TABLE IF EXISTS "ai_review_finding_entries";
        CREATE TABLE "subtasks_reference_cards" (
    "subtask_id" UUID NOT NULL REFERENCES "subtasks" ("id") ON DELETE CASCADE,
    "reference_cards_id" UUID NOT NULL REFERENCES "reference_cards" ("id") ON DELETE CASCADE
);"""
