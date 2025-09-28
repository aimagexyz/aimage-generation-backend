import logging
from os import getenv as ENV
from pathlib import Path

import aioboto3
from dotenv import load_dotenv

from aimage_supervision.clients.vecs import create_vecs_client

# Load environment variables
load_dotenv()

# FastAPI settings
# 不再使用通配符*，而是读取环境变量中的前端域名列表
# 格式：FRONTEND_ORIGINS=https://example.com,https://app.example.com
FRONTEND_ORIGINS = ENV('FRONTEND_ORIGINS', 'http://localhost:5173')
ALLOW_ORIGINS = FRONTEND_ORIGINS.split(',')

# Debug mode
DEBUG = (ENV('DEBUG', 'True').lower() != 'false')


# Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('aimage-supervision-backend')

# Sentry
SENTRY_DSN = ENV('SENTRY_DSN', '')
SENTRY_ENVIRONMENT = ENV('SENTRY_ENVIRONMENT', 'development')

# Base directory
BASE_DIR = Path(__file__).resolve().parent


# TortoiseORM settings
DATABASE_URL = ENV('DATABASE_URL', '')
TORTOISE_ORM = {
    'connections': {
        'default': DATABASE_URL,
    },
    'apps': {
        'models': {
            'models': ['aimage_supervision.models', 'aerich.models'],
            'default_connection': 'default',
        },
    },
}

# Batch API
SUPERVISION_BATCH_API_URL = ENV('SUPERVISION_BATCH_API_URL', '')
SUPERVISION_BATCH_API_USER = ENV('SUPERVISION_BATCH_API_USER', '')
SUPERVISION_BATCH_API_PASSWORD = ENV('SUPERVISION_BATCH_API_PASSWORD', '')

# AWS S3
AWS_ACCESS_KEY_ID = ENV('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = ENV('AWS_SECRET_ACCESS_KEY', '')
AWS_REGION = ENV('AWS_REGION', '')
AWS_BUCKET_NAME = ENV('AWS_BUCKET_NAME', '')

boto3_session = aioboto3.Session(
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Vecs
vecs_client = create_vecs_client(DATABASE_URL)


# JWT Settings
JWT_SECRET = ENV('JWT_SECRET', 'your-default-secret-key-for-development')

# Google OAuth
GOOGLE_CLIENT_ID = ENV('GOOGLE_CLIENT_ID', '')

# Cookie Settings
USE_SECURE_COOKIES = SENTRY_ENVIRONMENT == 'production'


def get_jwt_secret():
    return JWT_SECRET


def get_google_client_id():
    return GOOGLE_CLIENT_ID


# Google Cloud
GEMINI_API_KEY = ENV('GEMINI_API_KEY', '')
TEXT_EMBEDDING_MODEL = ENV('TEXT_EMBEDDING_MODEL',
                           'text-multilingual-embedding-002')
MULTIMODAL_EMBEDDING_MODEL = "multimodalembedding"
QDRANT_API_KEY = ENV('QDRANT_API_KEY', '')
QDRANT_URL = ENV('QDRANT_URL', '')
GOOGLE_CREDS = ENV('GOOGLE_CREDS', '')
EMBEDDING_BATCH_SIZE = 100
IMAGE_EMBEDDING_BATCH_SIZE = 100
IMAGE_EMBEDDING_WORKERS = 10
IMAGE_EMBEDDING_DIM = 1536
VERTEX_AI_PROJECT = ENV('VERTEX_AI_PROJECT', '')
VERTEX_AI_LOCATION = ENV('VERTEX_AI_LOCATION', '')

# Concurrency Settings
MAX_CONCURRENT = int(ENV('MAX_CONCURRENT', '4'))

# AGI Server
AGI_SERVER_URL = ENV('AGI_SERVER_URL', '')
AGI_API_KEY = ENV('AGI_API_KEY', '')
