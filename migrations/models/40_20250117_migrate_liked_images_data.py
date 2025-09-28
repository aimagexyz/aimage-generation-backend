from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    """
    Data migration for transferring UserPreferences.liked_images to LikedImage model.
    The actual data migration is handled by the migrate_liked_images_data.py script.
    """
    return """
        -- This migration is handled by migrate_liked_images_data.py
        -- Run that script after applying the schema migration  
        -- It will transfer existing UserPreferences.liked_images data to the new LikedImage model
        SELECT 1;
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    """
    Downgrade migration placeholder.
    Note: Downgrading will lose source tracking information from LikedImage records.
    """
    return """
        -- Downgrade is handled manually
        -- WARNING: Source tracking information will be lost
        SELECT 1;
    """ 