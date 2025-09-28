from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    """
    Remove the deprecated liked_images field from UserPreferences model.
    This field has been replaced by the LikedImage model for better data integrity.
    
    IMPORTANT: Run the data migration script (migrate_liked_images_data.py) BEFORE applying this migration
    to ensure no data is lost during the transition.
    """
    return """
        -- Remove the deprecated liked_images column from user_preferences table
        -- The data should have been migrated to the liked_images table before this migration
        ALTER TABLE "user_preferences" DROP COLUMN IF EXISTS "liked_images";
        
        -- Update table comment to reflect the change
        COMMENT ON TABLE "user_preferences" IS 'User preferences and settings (liked images moved to dedicated table)';
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    """
    Restore the liked_images field to UserPreferences model.
    
    WARNING: This downgrade will restore the column but will not restore the data.
    The liked images data will remain in the LikedImage model and would need to be
    manually migrated back if needed.
    """
    return """
        -- Restore the liked_images column (but data will be empty)
        ALTER TABLE "user_preferences" ADD COLUMN IF NOT EXISTS "liked_images" JSONB DEFAULT '[]'::jsonb;
        
        -- Restore column comment
        COMMENT ON COLUMN "user_preferences"."liked_images" IS 'List of liked image URLs or S3 paths (deprecated - use LikedImage model)';
        
        -- Restore table comment
        COMMENT ON TABLE "user_preferences" IS 'User preferences including liked images';
    """ 