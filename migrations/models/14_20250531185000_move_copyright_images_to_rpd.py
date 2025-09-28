from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- Add reference_images field to review_point_definition_versions table
        ALTER TABLE "review_point_definition_versions" ADD "reference_images" JSONB;
        
        -- Remove copyright_reference_images field from ips table  
        ALTER TABLE "ips" DROP COLUMN IF EXISTS "copyright_reference_images";
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- Add back copyright_reference_images field to ips table
        ALTER TABLE "ips" ADD "copyright_reference_images" JSONB;
        
        -- Remove reference_images field from review_point_definition_versions table
        ALTER TABLE "review_point_definition_versions" DROP COLUMN IF EXISTS "reference_images";
    """ 