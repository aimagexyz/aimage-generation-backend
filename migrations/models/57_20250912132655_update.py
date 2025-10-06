from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "review_point_definition_versions" ADD "constraints" TEXT;
        ALTER TABLE "review_point_definition_versions" ADD "is_ready_for_ai_review" BOOL NOT NULL DEFAULT True;
        ALTER TABLE "review_point_definition_versions" ADD "assessor" TEXT;
        ALTER TABLE "review_point_definition_versions" ADD "detector" TEXT;
        ALTER TABLE "review_point_definition_versions" ADD "rpd_type" VARCHAR(100);
        ALTER TABLE "review_point_definition_versions" ADD "guidelines" TEXT;
        CREATE INDEX IF NOT EXISTS "idx_review_poin_is_read_ac5003" ON "review_point_definition_versions" ("is_ready_for_ai_review");"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_review_poin_is_read_ac5003";
        ALTER TABLE "review_point_definition_versions" DROP COLUMN "constraints";
        ALTER TABLE "review_point_definition_versions" DROP COLUMN "is_ready_for_ai_review";
        ALTER TABLE "review_point_definition_versions" DROP COLUMN "assessor";
        ALTER TABLE "review_point_definition_versions" DROP COLUMN "detector";
        ALTER TABLE "review_point_definition_versions" DROP COLUMN "rpd_type";
        ALTER TABLE "review_point_definition_versions" DROP COLUMN "guidelines";"""
