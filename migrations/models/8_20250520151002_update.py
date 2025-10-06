from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "characters" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" VARCHAR(255) NOT NULL,
    "alias" VARCHAR(255),
    "description" TEXT,
    "features" TEXT,
    "image_path" VARCHAR(1024),
    "ip_id" UUID REFERENCES "ips" ("id") ON DELETE SET NULL,
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_characters_name_40b63e" ON "characters" ("name");
COMMENT ON TABLE "characters" IS '角色模型，用于存储角色信息';
        CREATE TABLE IF NOT EXISTS "ips" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" VARCHAR(255) NOT NULL,
    "description" TEXT
);
CREATE INDEX IF NOT EXISTS "idx_ips_name_b01047" ON "ips" ("name");
COMMENT ON TABLE "ips" IS '知识产权模型，如动漫、游戏等';
        COMMENT ON COLUMN "subtasks"."task_type" IS 'PICTURE: picture
VIDEO: video
TEXT: text
AUDIO: audio
WORD: word
EXCEL: excel';
        CREATE TABLE IF NOT EXISTS "ip_projects" (
    "ips_id" UUID NOT NULL REFERENCES "ips" ("id") ON DELETE CASCADE,
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "ip_projects";
        COMMENT ON COLUMN "subtasks"."task_type" IS 'PICTURE: picture
VIDEO: video
TEXT: text
AUDIO: audio';
        DROP TABLE IF EXISTS "ips";
        DROP TABLE IF EXISTS "characters";"""
