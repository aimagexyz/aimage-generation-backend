from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_reviews" ADD "processing_completed_at" TIMESTAMPTZ;
        ALTER TABLE "ai_reviews" ADD "processing_started_at" TIMESTAMPTZ;
        ALTER TABLE "ai_reviews" ADD "error_message" TEXT;
        ALTER TABLE "ai_reviews" ADD "processing_status" VARCHAR(10) NOT NULL DEFAULT 'pending';
        DROP TABLE IF EXISTS "user_preferences";
        DROP TABLE IF EXISTS "generated_references";"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_reviews" DROP COLUMN "processing_completed_at";
        ALTER TABLE "ai_reviews" DROP COLUMN "processing_started_at";
        ALTER TABLE "ai_reviews" DROP COLUMN "error_message";
        ALTER TABLE "ai_reviews" DROP COLUMN "processing_status";"""
