from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "generated_references" (
    "id" UUID NOT NULL  PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "base_prompt" TEXT NOT NULL,
    "enhanced_prompt" TEXT NOT NULL,
    "image_path" VARCHAR(1024) NOT NULL,
    "tags" JSONB NOT NULL,
    "project_id" UUID NOT NULL,
    "created_by_user_id" UUID NOT NULL
);
CREATE INDEX IF NOT EXISTS "idx_generated_r_project_81e2b3" ON "generated_references" ("project_id");
CREATE INDEX IF NOT EXISTS "idx_generated_r_created_26d865" ON "generated_references" ("created_by_user_id");
COMMENT ON TABLE "generated_references" IS 'AI-generated reference images - MVP version';
CREATE TABLE IF NOT EXISTS "aerich" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "version" VARCHAR(255) NOT NULL,
    "app" VARCHAR(100) NOT NULL,
    "content" JSONB NOT NULL
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
