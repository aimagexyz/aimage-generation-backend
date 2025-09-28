from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "RPDCharacterAssociation";
        DROP TABLE IF EXISTS "rpd_character_associations";
        CREATE TABLE "rpd_characters" (
    "character_id" UUID NOT NULL REFERENCES "characters" ("id") ON DELETE CASCADE,
    "review_point_definitions_id" UUID NOT NULL REFERENCES "review_point_definitions" ("id") ON DELETE CASCADE
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "rpd_characters";
        CREATE TABLE "RPDCharacterAssociation" (
    "character_id" UUID NOT NULL REFERENCES "characters" ("id") ON DELETE CASCADE,
    "review_point_definitions_id" UUID NOT NULL REFERENCES "review_point_definitions" ("id") ON DELETE CASCADE
);"""
