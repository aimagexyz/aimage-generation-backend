from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_review_finding_entries" ADD "reference_source" TEXT;
        ALTER TABLE "ai_review_finding_entries" ADD "reference_images" JSONB;
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "citation";"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_review_finding_entries" ADD "citation" JSONB;
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "reference_source";
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "reference_images";"""
