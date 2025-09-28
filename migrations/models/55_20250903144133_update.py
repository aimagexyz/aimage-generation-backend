from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "review_point_definition_versions" ADD "user_instruction" TEXT;
        ALTER TABLE "review_point_definition_versions" DROP COLUMN "eng_description_for_ai";
        COMMENT ON COLUMN "review_point_definition_versions"."description_for_ai" IS 'generated prompt for AI review';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "review_point_definition_versions" ADD "eng_description_for_ai" TEXT;
        ALTER TABLE "review_point_definition_versions" DROP COLUMN "user_instruction";
        COMMENT ON COLUMN "review_point_definition_versions"."description_for_ai" IS NULL;"""
