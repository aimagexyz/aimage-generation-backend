from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "batch_process_jobs" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "batch_id" VARCHAR(255) NOT NULL,
    "job_name" VARCHAR(255) NOT NULL,
    "job_type" VARCHAR(100) NOT NULL,
    "status" VARCHAR(9) NOT NULL DEFAULT 'pending',
    "total_items" INT NOT NULL DEFAULT 0,
    "processed_items" INT NOT NULL DEFAULT 0,
    "successful_items" INT NOT NULL DEFAULT 0,
    "failed_items" INT NOT NULL DEFAULT 0,
    "started_at" TIMESTAMPTZ,
    "completed_at" TIMESTAMPTZ,
    "max_concurrent" INT NOT NULL DEFAULT 5,
    "parameters" JSONB,
    "results" JSONB,
    "error_message" TEXT,
    "created_by_id" UUID REFERENCES "users" ("id") ON DELETE SET NULL,
    "project_id" UUID REFERENCES "projects" ("id") ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS "idx_batch_proce_batch_i_f7512b" ON "batch_process_jobs" ("batch_id");
CREATE INDEX IF NOT EXISTS "idx_batch_proce_job_nam_97e0bd" ON "batch_process_jobs" ("job_name");
CREATE INDEX IF NOT EXISTS "idx_batch_proce_job_typ_b89f42" ON "batch_process_jobs" ("job_type");
COMMENT ON COLUMN "batch_process_jobs"."batch_id" IS '批次标识符，用于关联相关的处理任务';
COMMENT ON COLUMN "batch_process_jobs"."job_name" IS '任务名称';
COMMENT ON COLUMN "batch_process_jobs"."job_type" IS '任务类型，如ai_review_cr_check';
COMMENT ON COLUMN "batch_process_jobs"."status" IS '批处理任务状态';
COMMENT ON COLUMN "batch_process_jobs"."total_items" IS '总任务数量';
COMMENT ON COLUMN "batch_process_jobs"."processed_items" IS '已处理任务数量';
COMMENT ON COLUMN "batch_process_jobs"."successful_items" IS '成功处理的任务数量';
COMMENT ON COLUMN "batch_process_jobs"."failed_items" IS '失败的任务数量';
COMMENT ON COLUMN "batch_process_jobs"."started_at" IS '任务开始时间';
COMMENT ON COLUMN "batch_process_jobs"."completed_at" IS '任务完成时间';
COMMENT ON COLUMN "batch_process_jobs"."max_concurrent" IS '最大并发数';
COMMENT ON COLUMN "batch_process_jobs"."parameters" IS '任务参数，以JSON格式存储';
COMMENT ON COLUMN "batch_process_jobs"."results" IS '任务结果，以JSON格式存储';
COMMENT ON COLUMN "batch_process_jobs"."error_message" IS '错误信息';
COMMENT ON COLUMN "batch_process_jobs"."created_by_id" IS '创建该批处理任务的用户';
COMMENT ON COLUMN "batch_process_jobs"."project_id" IS '关联的项目';
COMMENT ON TABLE "batch_process_jobs" IS '批处理任务模型，用于记录和跟踪批量处理任务的状态';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "batch_process_jobs";"""
