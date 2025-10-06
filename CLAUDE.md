# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Mage Supervision Backend - A FastAPI-based REST API for AI-powered image supervision and review management system. The backend handles image generation, multimodal embeddings, AI review processes, and integrates with various cloud services (AWS S3, Google Cloud, Vertex AI).

## Key Commands

### Development Server
```bash
# Activate virtual environment first
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Run development server
uvicorn aimage_supervision.app:app --reload
# or
python main.py
```

### Database Operations (Tortoise ORM with Aerich)
```bash
# Make new migrations after model changes
aerich migrate

# Apply migrations to database
aerich upgrade
```

### Code Quality & Type Checking
```bash
# Run all pre-commit hooks (autopep8, mypy, pyright)
pre-commit run --all-files

# Individual tools
autopep8 --in-place --recursive .
mypy aimage_supervision/
pyright
```

### Dependency Management
```bash
pip install -r requirements.txt
```

## Architecture Overview

### Core Structure
- **FastAPI Application**: Main app defined in `aimage_supervision/app.py`, configured with CORS, Sentry monitoring, and Tortoise ORM
- **API Versioning**: All endpoints under `/api/v1` prefix via `routers/v1.py`
- **Database**: PostgreSQL with Tortoise ORM for async operations, Aerich for migrations
- **Authentication**: JWT-based auth with Google OAuth integration

### Key Components

1. **Models** (`models.py`): Domain models with Tortoise ORM
   - User system with roles and preferences
   - Project/Task management (RPD - Review Point Definitions)
   - AI Review system with findings and processing states
   - Asset management with S3 integration
   - Character and Item catalogs

2. **Endpoints** (`endpoints/`): RESTful API endpoints
   - Authentication (`auth.py`)
   - Project management (`projects.py`, `rpd.py`)
   - AI review pipeline (`ai_review.py`)
   - Asset handling (`assets.py`)
   - Reference generation (`reference_generation.py`)
   - Video processing (`video.py`)
   - Batch operations (`batch.py`)

3. **Services** (`services/`): Business logic layer
   - AI processing pipeline with Gemini/Vertex AI
   - Image generation and embedding services
   - Review set management
   - Export functionality (PPTX)
   - Google Drive integration

4. **Clients** (`clients/`): External service integrations
   - AWS S3 for file storage
   - Google Cloud services (Gemini, Vertex AI)
   - Vector database (vecs/Qdrant) for embeddings
   - Multimodal embedding generation

5. **Middleware** (`middlewares/`): Cross-cutting concerns
   - Authentication middleware
   - Pagination support

### External Integrations
- **AWS S3**: Primary file storage
- **Google Cloud**: Gemini API for AI processing, Vertex AI for embeddings
- **Qdrant**: Vector database for similarity search
- **Sentry**: Error tracking and performance monitoring
- **Deepgram/Groq**: Additional AI services

### Database Schema Patterns
- UUID primary keys with timestamp mixins
- Enum-based status tracking
- JSON fields for flexible metadata storage
- Many-to-many relationships for project collaboration
- Soft deletes with `is_deleted` flags

### Key Workflows
1. **AI Review Pipeline**: Upload assets → Generate embeddings → Run AI analysis → Store findings → Export results
2. **Reference Generation**: Process character/item data → Generate AI references → Store in vector DB
3. **Batch Processing**: Queue multiple tasks → Process asynchronously → Track job status
4. **Export System**: Compile review findings → Generate PPTX reports → Upload to cloud storage

## Environment Configuration

Required environment variables (in `.env`):
- Database: `DATABASE_URL`
- AWS: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_BUCKET_NAME`, `AWS_REGION`
- Google Cloud: `GEMINI_API_KEY`, `VERTEX_AI_PROJECT`, `VERTEX_AI_LOCATION`, `GOOGLE_CREDS`
- Auth: `JWT_SECRET`, `GOOGLE_CLIENT_ID`
- Frontend: `FRONTEND_ORIGINS` (comma-separated CORS origins)
- Vector DB: `QDRANT_URL`, `QDRANT_API_KEY`

## Testing Approach

Currently no automated tests configured. Manual testing via:
- FastAPI's automatic `/docs` endpoint for API documentation and testing
- Direct API calls during development

## Important Notes

- Always use virtual environment (`venv`) for development
- Follow PEP8 conventions (enforced by autopep8)
- Type hints are required (checked by mypy and pyright)
- Database migrations must be created with `aerich migrate` before schema changes are applied
- Pre-commit hooks must pass before committing code
- Async/await patterns used throughout for database and external API calls
- File paths in database are S3 keys, not local paths