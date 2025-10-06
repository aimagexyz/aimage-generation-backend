from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_review_finding_entries" ADD "citation" JSONB;
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "citation_reference_images_json";
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "citation_reference_source";"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_review_finding_entries" ADD "citation_reference_images_json" JSONB;
        ALTER TABLE "ai_review_finding_entries" ADD "citation_reference_source" TEXT;
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "citation";"""
