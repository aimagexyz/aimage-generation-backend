from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "liked_images" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "image_path" VARCHAR(1024) NOT NULL,
    "source_type" VARCHAR(50) NOT NULL,
    "source_id" UUID NOT NULL,
    "display_name" VARCHAR(255),
    "tags" JSONB,
    "user_id" UUID NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_liked_images_image_path" ON "liked_images" ("image_path");
CREATE INDEX IF NOT EXISTS "idx_liked_images_source_type" ON "liked_images" ("source_type");
CREATE INDEX IF NOT EXISTS "idx_liked_images_source_id" ON "liked_images" ("source_id");
CREATE INDEX IF NOT EXISTS "idx_liked_images_user_id" ON "liked_images" ("user_id");
CREATE UNIQUE INDEX IF NOT EXISTS "idx_liked_images_user_source_path" ON "liked_images" ("user_id", "source_type", "source_id", "image_path");
COMMENT ON COLUMN "liked_images"."image_path" IS 'S3 path/key for the liked image';
COMMENT ON COLUMN "liked_images"."source_type" IS 'Type of source object (character, item, generated_reference, etc.)';
COMMENT ON COLUMN "liked_images"."source_id" IS 'UUID of the source object that contains this image';
COMMENT ON COLUMN "liked_images"."display_name" IS 'Optional display name for the liked image';
COMMENT ON COLUMN "liked_images"."tags" IS 'Optional tags for categorizing liked images';
COMMENT ON TABLE "liked_images" IS 'Liked images with proper S3 path storage and polymorphic source tracking';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "liked_images";""" 