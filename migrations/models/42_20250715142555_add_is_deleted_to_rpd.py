from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "review_point_definitions" ADD "is_deleted" BOOL NOT NULL DEFAULT False;
        CREATE INDEX IF NOT EXISTS "idx_review_poin_is_dele_f74d9c" ON "review_point_definitions" ("is_deleted");"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_review_poin_is_dele_f74d9c";
        ALTER TABLE "review_point_definitions" DROP COLUMN "is_deleted";"""
