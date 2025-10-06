from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "subtasks" ADD COLUMN "user_selected_character_ids" JSON;
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "subtasks" DROP COLUMN "user_selected_character_ids";
    """
