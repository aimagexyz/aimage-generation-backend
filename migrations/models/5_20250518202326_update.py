from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_review_finding_entries" ADD "area" JSONB;
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "area_width";
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "area_y";
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "area_height";
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "area_x";"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_review_finding_entries" ADD "area_width" INT;
        ALTER TABLE "ai_review_finding_entries" ADD "area_y" INT;
        ALTER TABLE "ai_review_finding_entries" ADD "area_height" INT;
        ALTER TABLE "ai_review_finding_entries" ADD "area_x" INT;
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "area";"""
