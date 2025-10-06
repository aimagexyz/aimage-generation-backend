from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "items" ADD "source_type" VARCHAR(20) NOT NULL DEFAULT 'direct_upload';
        ALTER TABLE "items" ADD "pdf_page_number" INT;
        ALTER TABLE "items" ADD "pdf_image_index" INT;
        ALTER TABLE "items" ADD "source_pdf_id" UUID;
        CREATE TABLE IF NOT EXISTS "pdfs" (
    "id" UUID NOT NULL PRIMARY KEY,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "filename" VARCHAR(255) NOT NULL,
    "s3_path" VARCHAR(1024) NOT NULL,
    "file_size" BIGINT NOT NULL,
    "total_pages" INT NOT NULL,
    "extraction_session_id" VARCHAR(255),
    "extracted_at" TIMESTAMPTZ NOT NULL,
    "extraction_method" VARCHAR(50) NOT NULL DEFAULT 'pymupdf',
    "extraction_stats" JSONB,
    "project_id" UUID NOT NULL REFERENCES "projects" ("id") ON DELETE CASCADE,
    "uploaded_by_id" UUID NOT NULL REFERENCES "users" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_pdfs_s3_path_b19533" ON "pdfs" ("s3_path");
COMMENT ON COLUMN "items"."source_type" IS 'Source type: direct_upload | pdf_extracted';
COMMENT ON COLUMN "items"."pdf_page_number" IS 'PDF page number (1-based) if extracted from PDF';
COMMENT ON COLUMN "items"."pdf_image_index" IS 'Image index within the PDF page (0-based) if extracted from PDF';
COMMENT ON COLUMN "items"."source_pdf_id" IS 'Source PDF if this item was extracted from a PDF';
COMMENT ON COLUMN "pdfs"."filename" IS 'Original PDF filename';
COMMENT ON COLUMN "pdfs"."s3_path" IS 'S3 path where the PDF is stored';
COMMENT ON COLUMN "pdfs"."file_size" IS 'PDF file size in bytes';
COMMENT ON COLUMN "pdfs"."total_pages" IS 'Total number of pages in PDF';
COMMENT ON COLUMN "pdfs"."extraction_session_id" IS 'Session ID used during extraction';
COMMENT ON COLUMN "pdfs"."extracted_at" IS 'When the PDF was processed for extraction';
COMMENT ON COLUMN "pdfs"."extraction_method" IS 'Method used for extraction';
COMMENT ON COLUMN "pdfs"."extraction_stats" IS 'Statistics from the extraction process';
COMMENT ON COLUMN "pdfs"."project_id" IS 'Project this PDF belongs to';
COMMENT ON COLUMN "pdfs"."uploaded_by_id" IS 'User who uploaded this PDF';
COMMENT ON TABLE "pdfs" IS 'PDF model for storing uploaded PDF files and their extraction metadata';
        ALTER TABLE "items" ADD CONSTRAINT "fk_items_pdfs_09010999" FOREIGN KEY ("source_pdf_id") REFERENCES "pdfs" ("id") ON DELETE SET NULL;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "items" DROP CONSTRAINT IF EXISTS "fk_items_pdfs_09010999";
        ALTER TABLE "items" DROP COLUMN "source_type";
        ALTER TABLE "items" DROP COLUMN "pdf_page_number";
        ALTER TABLE "items" DROP COLUMN "pdf_image_index";
        ALTER TABLE "items" DROP COLUMN "source_pdf_id";
        DROP TABLE IF EXISTS "pdfs";"""
