from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "review_point_definition_versions" ALTER COLUMN "special_rules" TYPE JSONB USING 
            CASE 
                WHEN "special_rules" IS NULL THEN NULL
                WHEN "special_rules" = '' THEN NULL
                ELSE '[]'::jsonb
            END;
        COMMENT ON COLUMN "review_point_definition_versions"."special_rules" IS 'Special rules in JSON format: [{"speaker": "角色A", "target": "角色B", "alias": "特殊称呼", "conditions": ["条件1", "条件2"]}]';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "review_point_definition_versions" ALTER COLUMN "special_rules" TYPE TEXT USING 
            CASE 
                WHEN "special_rules" IS NULL THEN NULL
                ELSE "special_rules"::text
            END;
        COMMENT ON COLUMN "review_point_definition_versions"."special_rules" IS 'Special rules and guidelines for text_review RPD type';"""
