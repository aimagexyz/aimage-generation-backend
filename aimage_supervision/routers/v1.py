from fastapi import APIRouter

from aimage_supervision.endpoints.auth import router as auth_router
from aimage_supervision.endpoints.health import router as health_router
from aimage_supervision.endpoints.projects import router as projects_router
from aimage_supervision.endpoints.reference_generation import \
    router as reference_generation_router
from aimage_supervision.endpoints.users import router as users_router

prefix = '/api/v1'

router = APIRouter(prefix=prefix)

router.include_router(health_router)
router.include_router(users_router)
router.include_router(projects_router)
router.include_router(auth_router)
router.include_router(reference_generation_router)
