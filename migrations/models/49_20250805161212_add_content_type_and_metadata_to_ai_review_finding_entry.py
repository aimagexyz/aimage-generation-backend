from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_review_finding_entries" ADD "content_type" VARCHAR(7) NOT NULL DEFAULT 'picture';
        ALTER TABLE "ai_review_finding_entries" ADD "content_metadata" JSONB;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "content_type";
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "content_metadata";"""
