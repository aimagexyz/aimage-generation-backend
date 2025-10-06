from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "items" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "filename" VARCHAR(255) NOT NULL,
    "s3_path" VARCHAR(1024) NOT NULL,
    "s3_url" VARCHAR(2048),
    "content_type" VARCHAR(100),
    "file_size" INT,
    "tags" JSONB,
    "description" TEXT,
    "project_id" UUID REFERENCES "projects" ("id") ON DELETE SET NULL,
    "uploaded_by_id" UUID NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_items_filenam_1a1992" ON "items" ("filename");
CREATE INDEX IF NOT EXISTS "idx_items_s3_path_3422eb" ON "items" ("s3_path");
COMMENT ON COLUMN "items"."filename" IS 'Original filename of the uploaded image';
COMMENT ON COLUMN "items"."s3_path" IS 'S3 path where the image is stored';
COMMENT ON COLUMN "items"."s3_url" IS 'Pre-signed S3 URL for accessing the image';
COMMENT ON COLUMN "items"."content_type" IS 'MIME type of the uploaded file';
COMMENT ON COLUMN "items"."file_size" IS 'File size in bytes';
COMMENT ON COLUMN "items"."tags" IS 'Tags associated with the item';
COMMENT ON COLUMN "items"."description" IS 'Description of the item';
COMMENT ON COLUMN "items"."project_id" IS 'Project this item belongs to';
COMMENT ON COLUMN "items"."uploaded_by_id" IS 'User who uploaded this item';
COMMENT ON TABLE "items" IS 'Item model for storing uploaded images and their metadata';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "items";"""
