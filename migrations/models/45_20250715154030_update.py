from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "models.RPDCharacterAssociation";
        ALTER TABLE "review_point_definitions" ADD "is_deleted" BOOL NOT NULL DEFAULT False;
        ALTER TABLE "tasks" ADD "is_deleted" BOOL NOT NULL DEFAULT False;
        DROP TABLE IF EXISTS "rpd_character_associations";
        CREATE INDEX IF NOT EXISTS "idx_review_poin_is_dele_f74d9c" ON "review_point_definitions" ("is_deleted");
        CREATE INDEX IF NOT EXISTS "idx_tasks_is_dele_66c232" ON "tasks" ("is_deleted");"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_review_poin_is_dele_f74d9c";
        DROP INDEX IF EXISTS "idx_tasks_is_dele_66c232";
        ALTER TABLE "tasks" DROP COLUMN "is_deleted";
        ALTER TABLE "review_point_definitions" DROP COLUMN "is_deleted";
        CREATE TABLE "models.RPDCharacterAssociation" (
    "character_id" UUID NOT NULL REFERENCES "characters" ("id") ON DELETE CASCADE,
    "review_point_definitions_id" UUID NOT NULL REFERENCES "review_point_definitions" ("id") ON DELETE CASCADE
);"""
