from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- Remove the unique constraint on (project_id, key) to allow multiple RPDs with same key
        ALTER TABLE "review_point_definitions" DROP CONSTRAINT IF EXISTS "uid_review_poi_project_77a60f";
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- Restore the unique constraint on (project_id, key)
        -- Note: This will fail if there are duplicate (project_id, key) combinations
        ALTER TABLE "review_point_definitions" ADD CONSTRAINT "uid_review_poi_project_77a60f" UNIQUE ("project_id", "key");
    """ 