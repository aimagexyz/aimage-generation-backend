from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "ai_review_finding_entries" ADD "is_fixed" BOOL NOT NULL DEFAULT FALSE;
        CREATE INDEX IF NOT EXISTS "idx_ai_review_finding_entries_is_fixed" ON "ai_review_finding_entries" ("is_fixed");"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_ai_review_finding_entries_is_fixed";
        ALTER TABLE "ai_review_finding_entries" DROP COLUMN "is_fixed";"""
