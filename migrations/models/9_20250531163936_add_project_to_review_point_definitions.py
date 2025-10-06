from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- Add project_id column as nullable first
        ALTER TABLE "review_point_definitions" ADD "project_id" UUID;
        
        -- Add foreign key constraint
        ALTER TABLE "review_point_definitions" ADD CONSTRAINT "fk_review_p_project_5e0b4d42" FOREIGN KEY ("project_id") REFERENCES "projects" ("id") ON DELETE CASCADE;
        
        -- Update unique constraint to remove the old one first
        ALTER TABLE "review_point_definitions" DROP CONSTRAINT IF EXISTS "uid_review_poi_key_53b1fc";
        
        -- Add new composite unique constraint (will be enforced after data migration)
        -- We'll add this in the data migration after populating project_id values
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- Drop the foreign key constraint
        ALTER TABLE "review_point_definitions" DROP CONSTRAINT IF EXISTS "fk_review_p_project_5e0b4d42";
        
        -- Drop the project_id column
        ALTER TABLE "review_point_definitions" DROP COLUMN IF EXISTS "project_id";
        
        -- Restore original unique constraint
        ALTER TABLE "review_point_definitions" ADD CONSTRAINT "uid_review_poi_key_53b1fc" UNIQUE ("key");
    """
