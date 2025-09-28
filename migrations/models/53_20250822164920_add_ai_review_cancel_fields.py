from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_reviews" ADD "should_cancel" BOOL NOT NULL DEFAULT False;
        COMMENT ON COLUMN "ai_reviews"."processing_status" IS 'PENDING: pending
PROCESSING: processing
COMPLETED: completed
FAILED: failed
CANCELLED: cancelled';
"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_reviews" DROP COLUMN "should_cancel";
        COMMENT ON COLUMN "ai_reviews"."processing_status" IS 'PENDING: pending
PROCESSING: processing
COMPLETED: completed
FAILED: failed';"""
