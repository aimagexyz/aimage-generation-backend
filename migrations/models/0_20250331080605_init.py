from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "organizations" (
    "id" UUID NOT NULL  PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "name" VARCHAR(255) NOT NULL UNIQUE,
    "description" TEXT NOT NULL,
    "domain" VARCHAR(255)
);
CREATE INDEX IF NOT EXISTS "idx_organizatio_name_75f36f" ON "organizations" ("name");
CREATE INDEX IF NOT EXISTS "idx_organizatio_domain_f6b307" ON "organizations" ("domain");
CREATE TABLE IF NOT EXISTS "projects" (
    "id" UUID NOT NULL  PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "name" VARCHAR(255) NOT NULL,
    "description" TEXT NOT NULL,
    "owner_org_id" UUID NOT NULL REFERENCES "organizations" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_projects_name_7b5b92" ON "projects" ("name");
CREATE TABLE IF NOT EXISTS "documents" (
    "id" UUID NOT NULL  PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "file_path" VARCHAR(1024) NOT NULL,
    "file_name" VARCHAR(255) NOT NULL,
    "file_size" INT NOT NULL,
    "file_type" VARCHAR(255),
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_documents_file_pa_4ed879" ON "documents" ("file_path");
CREATE INDEX IF NOT EXISTS "idx_documents_file_na_30f77d" ON "documents" ("file_name");
CREATE TABLE IF NOT EXISTS "reference_cards" (
    "id" UUID NOT NULL  PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "title" VARCHAR(255) NOT NULL,
    "description" TEXT NOT NULL,
    "source" VARCHAR(9) NOT NULL  DEFAULT 'anime',
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_reference_c_title_ff71f8" ON "reference_cards" ("title");
COMMENT ON COLUMN "reference_cards"."source" IS 'ANIME: anime\nHISTORY: history\nCHARACTER: character';
CREATE TABLE IF NOT EXISTS "task_priority_candidates" (
    "id" UUID NOT NULL  PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "oid" INT NOT NULL,
    "name" VARCHAR(100) NOT NULL,
    "owner_org_id" UUID NOT NULL REFERENCES "organizations" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_task_priori_oid_662be2" ON "task_priority_candidates" ("oid");
CREATE INDEX IF NOT EXISTS "idx_task_priori_name_fe4521" ON "task_priority_candidates" ("name");
CREATE TABLE IF NOT EXISTS "task_status_candidates" (
    "id" UUID NOT NULL  PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "oid" INT NOT NULL,
    "name" VARCHAR(100) NOT NULL,
    "owner_org_id" UUID NOT NULL REFERENCES "organizations" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_task_status_oid_3c4f77" ON "task_status_candidates" ("oid");
CREATE INDEX IF NOT EXISTS "idx_task_status_name_a0a852" ON "task_status_candidates" ("name");
CREATE TABLE IF NOT EXISTS "task_kanban_order" (
    "id" UUID NOT NULL  PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "task_order" JSONB,
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE,
    "status_id" UUID NOT NULL REFERENCES "task_status_candidates" ("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "users" (
    "id" UUID NOT NULL  PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "email" VARCHAR(255) NOT NULL UNIQUE,
    "display_name" VARCHAR(255) NOT NULL,
    "role" VARCHAR(16) NOT NULL  DEFAULT 'user'
);
COMMENT ON COLUMN "users"."role" IS 'USER: user\nADMIN: admin';
CREATE TABLE IF NOT EXISTS "assets" (
    "id" UUID NOT NULL  PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "drive_file_id" VARCHAR(255) NOT NULL,
    "file_name" VARCHAR(255) NOT NULL,
    "s3_path" VARCHAR(1024) NOT NULL,
    "mime_type" VARCHAR(255),
    "status" VARCHAR(10) NOT NULL  DEFAULT 'uploading',
    "author_id" UUID NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_assets_drive_f_53c25d" ON "assets" ("drive_file_id");
CREATE INDEX IF NOT EXISTS "idx_assets_file_na_fa94b5" ON "assets" ("file_name");
CREATE INDEX IF NOT EXISTS "idx_assets_s3_path_78ea6b" ON "assets" ("s3_path");
COMMENT ON COLUMN "assets"."status" IS 'UPLOADING: uploading\nPENDING: pending\nPROCESSING: processing\nDONE: done\nFAILED: failed';
CREATE TABLE IF NOT EXISTS "tasks" (
    "id" UUID NOT NULL  PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "tid" VARCHAR(255) NOT NULL,
    "name" VARCHAR(255) NOT NULL,
    "description" TEXT NOT NULL,
    "s3_path" VARCHAR(1024) NOT NULL,
    "assignee_id" UUID NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE,
    "priority_id" UUID NOT NULL REFERENCES "task_priority_candidates" ("id") ON DELETE CASCADE,
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE,
    "status_id" UUID NOT NULL REFERENCES "task_status_candidates" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_tasks_tid_2c7233" ON "tasks" ("tid");
CREATE INDEX IF NOT EXISTS "idx_tasks_name_1ac525" ON "tasks" ("name");
CREATE INDEX IF NOT EXISTS "idx_tasks_s3_path_1fe1f8" ON "tasks" ("s3_path");
CREATE TABLE IF NOT EXISTS "subtasks" (
    "id" UUID NOT NULL  PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "oid" INT NOT NULL,
    "name" VARCHAR(255) NOT NULL,
    "task_type" VARCHAR(7) NOT NULL  DEFAULT 'picture',
    "description" TEXT,
    "slide_page_number" INT,
    "content" JSONB,
    "history" JSONB,
    "annotations" JSONB,
    "status_history" JSONB,
    "status" VARCHAR(8) NOT NULL  DEFAULT 'pending',
    "ai_detection" JSONB,
    "task_id" UUID NOT NULL REFERENCES "tasks" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_subtasks_oid_77f82d" ON "subtasks" ("oid");
CREATE INDEX IF NOT EXISTS "idx_subtasks_name_15b24e" ON "subtasks" ("name");
COMMENT ON COLUMN "subtasks"."task_type" IS 'PICTURE: picture\nVIDEO: video\nTEXT: text\nAUDIO: audio';
COMMENT ON COLUMN "subtasks"."status" IS 'PENDING: pending\nDENIED: denied\nACCEPTED: accepted';
COMMENT ON COLUMN "subtasks"."ai_detection" IS 'AI detection result';
CREATE TABLE IF NOT EXISTS "aerich" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "version" VARCHAR(255) NOT NULL,
    "app" VARCHAR(100) NOT NULL,
    "content" JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS "organizations_members" (
    "organizations_id" UUID NOT NULL REFERENCES "organizations" ("id") ON DELETE CASCADE,
    "user_id" UUID NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "uidx_organizatio_organiz_505bf8" ON "organizations_members" ("organizations_id", "user_id");
CREATE TABLE IF NOT EXISTS "organizations_admins" (
    "organizations_id" UUID NOT NULL REFERENCES "organizations" ("id") ON DELETE CASCADE,
    "user_id" UUID NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "uidx_organizatio_organiz_7ed3f2" ON "organizations_admins" ("organizations_id", "user_id");
CREATE TABLE IF NOT EXISTS "projects_coop_members" (
    "projects_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE,
    "user_id" UUID NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "uidx_projects_co_project_5694aa" ON "projects_coop_members" ("projects_id", "user_id");
CREATE TABLE IF NOT EXISTS "projects_priority_candidates" (
    "projects_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE,
    "taskpriority_id" UUID NOT NULL REFERENCES "task_priority_candidates" ("id") ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "uidx_projects_pr_project_a8f618" ON "projects_priority_candidates" ("projects_id", "taskpriority_id");
CREATE TABLE IF NOT EXISTS "projects_coop_orgs" (
    "projects_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE,
    "organization_id" UUID NOT NULL REFERENCES "organizations" ("id") ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "uidx_projects_co_project_c29513" ON "projects_coop_orgs" ("projects_id", "organization_id");
CREATE TABLE IF NOT EXISTS "projects_status_candidates" (
    "projects_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE,
    "taskstatus_id" UUID NOT NULL REFERENCES "task_status_candidates" ("id") ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "uidx_projects_st_project_897134" ON "projects_status_candidates" ("projects_id", "taskstatus_id");
CREATE TABLE IF NOT EXISTS "subtasks_reference_cards" (
    "reference_cards_id" UUID NOT NULL REFERENCES "reference_cards" ("id") ON DELETE CASCADE,
    "subtask_id" UUID NOT NULL REFERENCES "subtasks" ("id") ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "uidx_subtasks_re_referen_e359c7" ON "subtasks_reference_cards" ("reference_cards_id", "subtask_id");"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
