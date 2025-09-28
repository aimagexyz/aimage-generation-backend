from fastapi import APIRouter

from aimage_supervision.endpoints.admin_dashboard import \
    router as admin_dashboard_router
from aimage_supervision.endpoints.ai_review import \
    router as ai_review_process_router
from aimage_supervision.endpoints.assets import router as assets_router
from aimage_supervision.endpoints.auth import router as auth_router
from aimage_supervision.endpoints.batch import router as batch_router
from aimage_supervision.endpoints.characters import ip_router
from aimage_supervision.endpoints.characters import router as characters_router
from aimage_supervision.endpoints.drawing import router as drawing_router
from aimage_supervision.endpoints.health import router as health_router
from aimage_supervision.endpoints.items import router as items_router
from aimage_supervision.endpoints.pdf import router as pdf_router
from aimage_supervision.endpoints.projects import router as projects_router
from aimage_supervision.endpoints.reference_generation import \
    router as reference_generation_router
from aimage_supervision.endpoints.review_sets import \
    router as review_sets_router
from aimage_supervision.endpoints.rpd import router as rpd_router
from aimage_supervision.endpoints.rpdset_character_associations import \
    router as review_set_character_associations_router
from aimage_supervision.endpoints.task_tags import router as task_tags_router
from aimage_supervision.endpoints.tasks import router as tasks_router
from aimage_supervision.endpoints.user_preferences import \
    router as user_preferences_router
from aimage_supervision.endpoints.users import router as users_router
from aimage_supervision.endpoints.video import router as video_router

prefix = '/api/v1'

router = APIRouter(prefix=prefix)

router.include_router(health_router)
router.include_router(admin_dashboard_router)
router.include_router(users_router)
router.include_router(user_preferences_router)
router.include_router(projects_router)
router.include_router(tasks_router)
router.include_router(assets_router)
router.include_router(auth_router)
router.include_router(rpd_router)
router.include_router(ai_review_process_router)
router.include_router(characters_router)
router.include_router(ip_router)
router.include_router(items_router)
router.include_router(reference_generation_router)
router.include_router(batch_router)
router.include_router(task_tags_router)
router.include_router(review_sets_router)
router.include_router(review_set_character_associations_router)
router.include_router(video_router)
router.include_router(pdf_router)
router.include_router(drawing_router)
