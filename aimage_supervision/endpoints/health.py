from fastapi import APIRouter

router = APIRouter(prefix='', tags=['Health Check'])


@router.get('/health-check')
async def health_check() -> dict[str, str]:
    return {'status': 'ok'}
