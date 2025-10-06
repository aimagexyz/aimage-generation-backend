from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- Add copyright_reference_images field to ips table
        ALTER TABLE "ips" ADD "copyright_reference_images" JSONB;
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- Remove copyright_reference_images field from ips table
        ALTER TABLE "ips" DROP COLUMN IF EXISTS "copyright_reference_images";
    """ 