from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "review_point_definition_versions" ALTER COLUMN "constraints" TYPE JSONB USING "constraints"::JSONB;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "review_point_definition_versions" ALTER COLUMN "constraints" TYPE TEXT USING "constraints"::TEXT;"""
