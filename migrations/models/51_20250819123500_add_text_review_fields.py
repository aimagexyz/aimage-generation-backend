from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "review_point_definition_versions" ADD "reference_files" JSONB;
        ALTER TABLE "review_point_definition_versions" ADD "special_rules" TEXT;
        COMMENT ON COLUMN "review_point_definition_versions"."reference_files" IS 'List of S3 URLs for reference files (e.g., appellation table for text_review)';
        COMMENT ON COLUMN "review_point_definition_versions"."special_rules" IS 'Special rules and guidelines for text_review RPD type';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "review_point_definition_versions" DROP COLUMN "reference_files";
        ALTER TABLE "review_point_definition_versions" DROP COLUMN "special_rules";"""
