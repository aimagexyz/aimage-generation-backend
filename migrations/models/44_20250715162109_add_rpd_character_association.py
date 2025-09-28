from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_tasks_is_dele_66c232";
        DROP INDEX IF EXISTS "idx_review_poin_is_dele_f74d9c";
        CREATE TABLE IF NOT EXISTS "rpd_character_associations" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "weight" DOUBLE PRECISION NOT NULL DEFAULT 1,
    "association_type" VARCHAR(50) NOT NULL DEFAULT 'general',
    "character_id" UUID NOT NULL REFERENCES "characters" ("id") ON DELETE CASCADE,
    "rpd_id" UUID NOT NULL REFERENCES "review_point_definitions" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_rpd_charact_rpd_id_4cb14e" UNIQUE ("rpd_id", "character_id")
);
COMMENT ON COLUMN "rpd_character_associations"."weight" IS '关联权重，用于AI审核时的重要性';
COMMENT ON COLUMN "rpd_character_associations"."association_type" IS '关联类型，如：primary, secondary, reference';
COMMENT ON TABLE "rpd_character_associations" IS 'RPD和Character的多对多关联表';
        ALTER TABLE "review_point_definitions" DROP COLUMN "is_deleted";
        ALTER TABLE "tasks" DROP COLUMN "is_deleted";"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "tasks" ADD "is_deleted" BOOL NOT NULL DEFAULT False;
        ALTER TABLE "review_point_definitions" ADD "is_deleted" BOOL NOT NULL DEFAULT False;
        DROP TABLE IF EXISTS "rpd_character_associations";
        CREATE INDEX IF NOT EXISTS "idx_tasks_is_dele_66c232" ON "tasks" ("is_deleted");
        CREATE INDEX IF NOT EXISTS "idx_review_poin_is_dele_f74d9c" ON "review_point_definitions" ("is_deleted");"""
