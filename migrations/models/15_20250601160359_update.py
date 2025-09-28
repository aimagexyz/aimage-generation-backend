from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        COMMENT ON COLUMN "review_point_definition_versions"."reference_images" IS 'List of S3 paths for reference images that provide visual context and guidance for AI reviews';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        COMMENT ON COLUMN "review_point_definition_versions"."reference_images" IS 'List of S3 paths for reference images used in copyright detection when this is a copyright_review RPD';"""
