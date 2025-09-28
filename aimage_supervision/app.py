import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination
from tortoise.contrib.fastapi import register_tortoise

from aimage_supervision.routers.v1 import router as routers_v1
from aimage_supervision.settings import (ALLOW_ORIGINS, SENTRY_DSN,
                                         SENTRY_ENVIRONMENT, TORTOISE_ORM)

if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
        environment=SENTRY_ENVIRONMENT,
    )


# FastAPI app
app = FastAPI(
    title='AI Mage Supervision API',
    description='Supervision API for AI Mage project',
    version='0.6.0',
)

app.include_router(routers_v1)

add_pagination(app)

# Register Tortoise ORM
register_tortoise(
    app,
    config=TORTOISE_ORM,
    generate_schemas=True,
)

# CORSMiddleware should be the last middleware
# Error are not reported using CORS middleware
# https://github.com/encode/starlette/issues/1116
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
