import uuid
from typing import Annotated, List

from fastapi import (APIRouter, File, Form, HTTPException, Security,
                     UploadFile, status)
from fastapi.background import BackgroundTasks
from fastapi.exceptions import HTTPException
from fastapi.params import Query, Security
from pydantic import BaseModel

from aimage_supervision.clients.aws_s3 import (cleanup_s3_prefix,
                                               get_s3_url_from_path,
                                               move_s3_files,
                                               upload_file_to_s3,
                                               upload_file_to_s3_with_ttl)
from aimage_supervision.enums import AssetStatus
from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.middlewares.tortoise_paginate import (
    Page, tortoise_paginate)
from aimage_supervision.models import Asset, AssetIn, AssetOut, Project, User
from aimage_supervision.services.google_drive import \
    upload_file_from_google_drive_task
from aimage_supervision.utils.extract_pptx import PPTXProcessor

router = APIRouter(prefix='', tags=['Assets'])


class AssetResponse(BaseModel):
    url: str


class ProcessPPTXResponse(BaseModel):
    asset_id: str | uuid.UUID
    status: str


S3_URL_PREFIX = 's3://ai-mage-supervision/'
S3_URL_PREFIX_LEN = len(S3_URL_PREFIX)


@router.get('/assets')
async def get_asset(
    s3_path: str,
) -> AssetResponse:
    # Remove s3://ai-mage-supervision/ prefix
    if s3_path.startswith(S3_URL_PREFIX):
        s3_path = s3_path[S3_URL_PREFIX_LEN:]
    url = await get_s3_url_from_path(s3_path)
    return AssetResponse(
        url=url,
    )


@router.post('/assets/images')
async def upload_image(
    user: Annotated[User, Security(get_current_user)],
    file: UploadFile,
) -> AssetResponse:
    '''Upload an image file to S3 and return its URL.'''
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='File must be an image',
        )

    # Generate a unique path for the image
    file_extension = 'jpg'  # default extension
    if file.filename:
        parts = file.filename.split('.')
        if len(parts) > 1:
            file_extension = parts[-1]

    # Generate path in images directory
    s3_path = f'images/{uuid.uuid4()}.{file_extension}'

    # Upload to S3
    await upload_file_to_s3(file.file, s3_path)

    # Return the s3:// format URL
    return AssetResponse(
        url=f'{S3_URL_PREFIX}{s3_path}',
    )


@router.get('/projects/{project_id}/assets')
async def list_assets(
    project_id: str,
    user: Annotated[User, Security(get_current_user)],
) -> Page[AssetOut]:
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found',
        )
    assets_queryset = Asset.filter(project=project)

    response = await tortoise_paginate(
        assets_queryset,
        model=AssetOut,  # type: ignore
    )
    return response


@router.post('/projects/{project_id}/assets')
async def upload_asset_from_google_drive(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    payload: AssetIn,
    google_access_token: Annotated[str, Query(
        description='Google access token',
    )],
    background_tasks: BackgroundTasks,
) -> AssetOut:
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found',
        )
    s3_path = f'upload_assets/{project.id}/{payload.drive_file_id}_{payload.file_name}'
    if not (asset := await Asset.filter(drive_file_id=payload.drive_file_id).first()):
        asset = await Asset.create(
            drive_file_id=payload.drive_file_id,
            file_name=payload.file_name,
            s3_path=s3_path,
            mime_type=payload.mime_type,
            author=user,
            project=project,
        )
    background_tasks.add_task(
        upload_file_from_google_drive_task,
        asset.drive_file_id,
        asset.file_name,
        google_access_token,
    )
    response = await AssetOut.from_tortoise_orm(asset)
    return response


@router.post('/projects/{project_id}/assets/process-pptx')
async def process_pptx_assets(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    asset_ids: List[str],
    background_tasks: BackgroundTasks,
) -> List[ProcessPPTXResponse]:
    """
    处理项目中的PPT文件资源。
    立即返回assets的processing状态，然后在后台处理PPT文件。

    Args:
        project_id: 项目ID
        asset_ids: 要处理的Asset ID列表

    Returns:
        List[ProcessPPTXResponse]: 包含每个asset的ID和初始状态
    """
    # 验证项目权限
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found',
        )

    # 获取assets
    assets = await Asset.filter(
        id__in=asset_ids,
        project=project,
    ).prefetch_related('project')

    if not assets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='No assets found',
        )

    # 更新所有assets状态为PROCESSING
    responses = []
    for asset in assets:
        asset.status = AssetStatus.PROCESSING
        await asset.save()
        responses.append(ProcessPPTXResponse(
            asset_id=asset.id,
            status=asset.status,
        ))

    # 添加后台任务
    async def process_assets_background():
        processor = PPTXProcessor()
        await processor.init_db()
        try:
            await processor.process_assets(assets)
        finally:
            await processor.close_db()

    background_tasks.add_task(process_assets_background)

    return responses


# === 临时存储相关接口 ===


class CleanupSessionResponse(BaseModel):
    deleted_count: int
    deleted_files: List[str]


class PromoteTempImagesRequest(BaseModel):
    project_id: str


class PromoteTempImagesResponse(BaseModel):
    moved_files: List[str]
    errors: List[str]


@router.post('/assets/temp-images')
async def upload_temp_image(
    user: Annotated[User, Security(get_current_user)],
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> AssetResponse:
    """上传临时图片，设置2小时TTL"""

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='File must be an image',
        )

    # 验证会话ID格式
    if not session_id or not session_id.startswith('rpd-'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid session ID format',
        )

    try:
        # 生成临时文件路径
        file_extension = 'jpg'
        if file.filename:
            parts = file.filename.split('.')
            if len(parts) > 1:
                file_extension = parts[-1]

        s3_path = f'temp/rpd-sessions/{session_id}/{uuid.uuid4()}.{file_extension}'

        # 上传到S3并设置TTL
        await upload_file_to_s3_with_ttl(
            file.file,
            s3_path,
            ttl_hours=2,
            content_type=file.content_type
        )

        # 返回S3 URL
        return AssetResponse(
            url=f'{S3_URL_PREFIX}{s3_path}',
        )

    except Exception as e:
        print(f"Failed to upload temp image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to upload temporary image: {str(e)}',
        )


@router.delete('/assets/temp-sessions/{session_id}')
async def cleanup_temp_session(
    session_id: str,
) -> CleanupSessionResponse:
    """清理指定会话的所有临时文件"""

    # 验证会话ID格式
    if not session_id or not session_id.startswith('rpd-'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid session ID format',
        )

    try:
        prefix = f'temp/rpd-sessions/{session_id}/'
        deleted_count, deleted_files = await cleanup_s3_prefix(prefix)

        return CleanupSessionResponse(
            deleted_count=deleted_count,
            deleted_files=deleted_files,
        )

    except Exception as e:
        print(f"Failed to cleanup temp session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to cleanup temporary session: {str(e)}',
        )


@router.post('/assets/temp-sessions/{session_id}/promote')
async def promote_temp_images(
    user: Annotated[User, Security(get_current_user)],
    session_id: str,
    request: PromoteTempImagesRequest,
) -> PromoteTempImagesResponse:
    """将临时图片提升为正式图片（移动到正式存储路径）"""

    # 验证会话ID格式
    if not session_id or not session_id.startswith('rpd-'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid session ID format',
        )

    # 验证项目权限
    project = await Project.of_user(user).get_or_none(id=request.project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found',
        )

    try:
        from_prefix = f'temp/rpd-sessions/{session_id}/'
        to_prefix = f'projects/{request.project_id}/rpd-images/'

        moved_files, failed_files = await move_s3_files(from_prefix, to_prefix)

        return PromoteTempImagesResponse(
            moved_files=moved_files,
            errors=failed_files,
        )

    except Exception as e:
        print(f"Failed to promote temp images for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to promote temporary images: {str(e)}',
        )
