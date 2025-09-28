from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "review_point_definition_versions" ALTER COLUMN "guidelines" TYPE JSONB USING "guidelines"::JSONB;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "review_point_definition_versions" ALTER COLUMN "guidelines" TYPE TEXT USING "guidelines"::TEXT;"""
