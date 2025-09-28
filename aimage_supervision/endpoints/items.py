import asyncio
import json
import time
import uuid

from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

from fastapi import (APIRouter, BackgroundTasks, Form, HTTPException, Security,
                     UploadFile, status)
from fastapi.params import Query
from fastapi.responses import JSONResponse
from fastapi_pagination import Page as FastAPIPage
from pydantic import BaseModel
from tortoise.expressions import Q

from aimage_supervision.clients.aws_s3 import (get_s3_url_from_path,
                                               upload_file_to_s3)
from aimage_supervision.enums import BatchJobStatus
from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.middlewares.tortoise_paginate import (
    Page, tortoise_paginate)
from aimage_supervision.models import (BatchProcessJob, Item, Project, Subtask,
                                       User)
from aimage_supervision.schemas import (ItemBatchUploadResponse, ItemCreate,
                                        ItemResponse, ItemUpdate)
from aimage_supervision.services.items_service import (
    list_up_items_in_image, process_project_items_embeddings,
    search_project_items_by_image)
from aimage_supervision.settings import MAX_CONCURRENT, logger
from aimage_supervision.utils.file_validation import \
    is_valid_image_or_design_file
from aimage_supervision.utils.image_compression import compress_image_async

router = APIRouter(prefix='/items', tags=['Items'])


async def _process_single_file_from_data(
    filename: str,
    content_type: str,
    file_content: bytes,
    project_id: Optional[str],
    parsed_tags: List,
    description: Optional[str],
    project: Optional[Project],
    user: User,
    # PDF相关参数
    source_type: str = 'direct_upload',
    source_pdf_id: Optional[str] = None,
    pdf_page_number: Optional[int] = None,
    pdf_image_index: Optional[int] = None
) -> dict:
    """
    从文件数据处理单个文件的函数，用于后台任务

    Returns:
        dict: 包含成功的ItemResponse或失败信息
    """
    try:
        # Validate file type - allow images, PSD, and AI files
        if not is_valid_image_or_design_file(filename, content_type):
            return {
                'success': False,
                'filename': filename,
                'error': 'ファイルは画像、PSD、またはAIファイルである必要があります'
            }

        # Compress image before upload
        try:
            from io import BytesIO
            file_stream = BytesIO(file_content)
            compressed_file, output_format, compressed_size = await compress_image_async(file_stream)

            # Determine file extension based on compression output
            if output_format == 'JPEG':
                file_extension = 'jpg'
            elif output_format == 'PNG':
                file_extension = 'png'
            elif output_format == 'SVG':
                file_extension = 'svg'
            elif output_format == 'PSD':
                file_extension = 'psd'
            elif output_format == 'AI':
                file_extension = 'ai'
            else:
                file_extension = output_format.lower()

            # Generate unique S3 path with proper extension
            s3_path = f'projects/{project_id}/items/{uuid.uuid4()}.{file_extension}'

            # Upload compressed image to S3
            await upload_file_to_s3(compressed_file, s3_path)

            # Update content type based on compression result
            compressed_content_type = f'image/{file_extension}'

        except ValueError as compression_error:
            return {
                'success': False,
                'filename': filename,
                'error': f'画像圧縮エラー: {str(compression_error)}'
            }

        # Create Item record with compressed file info
        item = await Item.create(
            filename=filename or f'upload.{file_extension}',
            s3_path=s3_path,
            content_type=compressed_content_type,
            file_size=compressed_size,  # Use compressed size
            tags=parsed_tags if parsed_tags else None,
            description=description,
            project=project,
            uploaded_by=user,
            # PDF相关字段
            source_type=source_type,
            source_pdf_id=source_pdf_id,
            pdf_page_number=pdf_page_number,
            pdf_image_index=pdf_image_index
        )

        # Fetch image URL
        await item.fetch_image_url()

        # Convert to response model
        item_response = ItemResponse(
            id=item.id,
            filename=item.filename,
            s3_path=item.s3_path,
            s3_url=item.s3_url,
            image_url=item.image_url(),
            content_type=item.content_type,
            file_size=item.file_size,
            tags=item.tags,
            description=item.description,
            project_id=item.project_id,
            uploaded_by=item.uploaded_by_id,
            created_at=item.created_at,
            updated_at=item.updated_at,
            # PDF相关字段
            source_type=item.source_type,
            source_pdf_id=item.source_pdf_id,
            pdf_page_number=item.pdf_page_number,
            pdf_image_index=item.pdf_image_index
        )

        return {
            'success': True,
            'item': item_response
        }

    except Exception as e:
        return {
            'success': False,
            'filename': filename,
            'error': str(e)
        }


async def _process_single_file(
    file: UploadFile,
    project_id: Optional[str],
    parsed_tags: List,
    description: Optional[str],
    project: Optional[Project],
    user: User,
    # PDF相关参数
    source_type: str = 'direct_upload',
    source_pdf_id: Optional[str] = None,
    pdf_page_number: Optional[int] = None,
    pdf_image_index: Optional[int] = None
) -> dict:
    """
    单个文件的处理函数，支持并发调用

    Returns:
        dict: 包含成功的ItemResponse或失败信息
    """
    try:
        # Validate file type - allow images, PSD, and AI files
        if not is_valid_image_or_design_file(file.filename, file.content_type):
            return {
                'success': False,
                'filename': file.filename,
                'error': 'ファイルは画像、PSD、またはAIファイルである必要があります'
            }

        # Compress image before upload
        try:
            compressed_file, output_format, compressed_size = await compress_image_async(file.file)

            # Determine file extension based on compression output
            if output_format == 'JPEG':
                file_extension = 'jpg'
            elif output_format == 'PNG':
                file_extension = 'png'
            elif output_format == 'SVG':
                file_extension = 'svg'
            elif output_format == 'PSD':
                file_extension = 'psd'
            elif output_format == 'AI':
                file_extension = 'ai'
            else:
                file_extension = output_format.lower()

            # Generate unique S3 path with proper extension
            s3_path = f'projects/{project_id}/items/{uuid.uuid4()}.{file_extension}'

            # Upload compressed image to S3
            await upload_file_to_s3(compressed_file, s3_path)

            # Update content type based on compression result
            compressed_content_type = f'image/{file_extension}'

        except ValueError as compression_error:
            return {
                'success': False,
                'filename': file.filename,
                'error': f'画像圧縮エラー: {str(compression_error)}'
            }

        # Create Item record with compressed file info
        item = await Item.create(
            filename=file.filename or f'upload.{file_extension}',
            s3_path=s3_path,
            content_type=compressed_content_type,
            file_size=compressed_size,  # Use compressed size
            tags=parsed_tags if parsed_tags else None,
            description=description,
            project=project,
            uploaded_by=user,
            # PDF相关字段
            source_type=source_type,
            source_pdf_id=source_pdf_id,
            pdf_page_number=pdf_page_number,
            pdf_image_index=pdf_image_index
        )

        # Fetch image URL
        await item.fetch_image_url()

        # Convert to response model
        item_response = ItemResponse(
            id=item.id,
            filename=item.filename,
            s3_path=item.s3_path,
            s3_url=item.s3_url,
            image_url=item.image_url(),
            content_type=item.content_type,
            file_size=item.file_size,
            tags=item.tags,
            description=item.description,
            project_id=item.project_id,
            uploaded_by=item.uploaded_by_id,
            created_at=item.created_at,
            updated_at=item.updated_at,
            # PDF相关字段
            source_type=item.source_type,
            source_pdf_id=item.source_pdf_id,
            pdf_page_number=item.pdf_page_number,
            pdf_image_index=item.pdf_image_index
        )

        return {
            'success': True,
            'item': item_response
        }

    except Exception as e:
        return {
            'success': False,
            'filename': file.filename,
            'error': str(e)
        }


@router.post('/batch-upload')
async def batch_upload_items(
    user: Annotated[User, Security(get_current_user)],
    files: List[UploadFile],
    background_tasks: BackgroundTasks,
    project_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # JSON string of tags
    description: Optional[str] = Form(None),
) -> JSONResponse:
    """
    批量上传图片到S3并创建Items记录 - 使用后台任务处理以避免超时

    Args:
        files: 上传的图片文件列表
        project_id: 可选的项目ID
        tags: 可选的标签列表(JSON字符串格式)
        description: 可选的描述

    Returns:
        JSONResponse: 包含批次ID和状态的响应，实际处理在后台进行
    """
    # Validate project if provided
    project = None
    if project_id:
        project = await Project.of_user(user).get_or_none(id=project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Project not found'
            )

    # Parse tags if provided
    parsed_tags = []
    if tags:
        try:
            parsed_tags = json.loads(tags)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Invalid tags format. Must be valid JSON array.'
            )

    # Generate batch ID for tracking
    batch_id = str(uuid.uuid4())[:8]

    # Create database record for batch job
    await BatchProcessJob.create(
        batch_id=batch_id,
        job_name=f"Items_Upload_{batch_id}_{len(files)}files",
        job_type="items_batch_upload",
        status=BatchJobStatus.RUNNING,
        created_by=user,
        project=project,
        total_items=len(files),
        processed_items=0,
        successful_items=0,
        failed_items=0,
        started_at=datetime.now(timezone.utc),
        max_concurrent=MAX_CONCURRENT,
        parameters={
            "project_id": project_id,
            "tags": parsed_tags,
            "description": description,
            "file_count": len(files),
            "user_id": str(user.id)
        }
    )

    # Prepare file data for background task
    file_data_list = []
    for file in files:
        # Read file content into memory
        content = await file.read()
        file_data_list.append({
            'filename': file.filename,
            'content_type': file.content_type,
            'content': content
        })
        # Reset file pointer for potential reuse
        await file.seek(0)

    # Start background task
    background_tasks.add_task(
        _batch_upload_background_task,
        batch_id=batch_id,
        file_data_list=file_data_list,
        project_id=project_id,
        parsed_tags=parsed_tags,
        description=description,
        project=project,
        user_id=str(user.id)
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            'message': f'Batch upload initiated with {len(files)} files',
            'batch_id': batch_id,
            'status': 'processing',
            'total_files': len(files),
            'project_id': project_id
        }
    )


@router.get('/batch-upload/{batch_id}/status')
async def get_batch_upload_status(
    batch_id: str,
    user: Annotated[User, Security(get_current_user)],
) -> dict:
    """
    获取批量上传任务的状态

    Args:
        batch_id: 批次ID

    Returns:
        dict: 批次状态信息
    """
    batch_job = await BatchProcessJob.filter(
        batch_id=batch_id,
        created_by=user
    ).first()

    if not batch_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Batch job not found'
        )

    # Calculate duration if completed
    duration_seconds = None
    if batch_job.completed_at and batch_job.started_at:
        duration_seconds = (batch_job.completed_at -
                            batch_job.started_at).total_seconds()

    return {
        'batch_id': batch_job.batch_id,
        'status': batch_job.status.value,
        'total_items': batch_job.total_items,
        'processed_items': batch_job.processed_items,
        'successful_items': batch_job.successful_items,
        'failed_items': batch_job.failed_items,
        'started_at': batch_job.started_at.isoformat() if batch_job.started_at else None,
        'completed_at': batch_job.completed_at.isoformat() if batch_job.completed_at else None,
        'duration_seconds': duration_seconds,
        'parameters': batch_job.parameters
    }


@router.get('/projects/{project_id}')
async def list_items(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    tags: Annotated[Optional[str], Query()] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=1000)] = 300,
) -> Page[ItemResponse]:
    """
    获取用户的Items列表

    Args:
        project_id: 可选的项目ID过滤
        tags: 可选的标签过滤(逗号分隔)
        page: 页码 (从1开始)
        size: 每页数量 (1-1000)

    Returns:
        Page[ItemResponse]: 分页的Items列表
    """
    # Base query - user can see their own items and items from projects they have access to
    query = Item.filter(uploaded_by=user)

    # If user has access to projects, they can also see items from those projects
    accessible_projects = await Project.of_user(user).all()
    if accessible_projects:
        project_ids = [p.id for p in accessible_projects]
        query = Item.filter(
            Q(uploaded_by=user) | Q(project__id__in=project_ids)
        ).distinct()

    # Filter by project if specified
    if project_id:
        project = await Project.of_user(user).get_or_none(id=project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Project not found'
            )
        query = query.filter(project=project)

    # Filter by tags if specified
    if tags:
        tag_list = [tag.strip() for tag in tags.split(',')]
        query = query.filter(tags__contains=tag_list)

    # Include relations and order by created_at desc for consistent pagination
    query = query.prefetch_related(
        'uploaded_by', 'project').order_by('-created_at')

    # Calculate pagination offset
    offset = (page - 1) * size

    # Get total count for pagination
    total_count = await query.count()

    # Get paginated items with limit and offset
    items = await query.offset(offset).limit(size).all()

    # Batch fetch S3 URLs for all items at once
    s3_paths = [item.s3_path for item in items if item.s3_path]

    # Batch fetch S3 URLs concurrently
    s3_url_tasks = [get_s3_url_from_path(path) for path in s3_paths]
    s3_urls = await asyncio.gather(*s3_url_tasks)

    # Create a mapping from s3_path to s3_url
    s3_url_map = dict(zip(s3_paths, s3_urls))

    # Map items to response models with pre-fetched S3 URLs
    mapped_items = []
    for item in items:
        # Set the image URL from our batch-fetched results
        if item.s3_path:
            setattr(item, '_image_url', s3_url_map.get(item.s3_path, ''))
        else:
            setattr(item, '_image_url', '')

        mapped_item = ItemResponse(
            id=item.id,
            filename=item.filename,
            s3_path=item.s3_path,
            s3_url=item.s3_url,
            image_url=item.image_url(),
            content_type=item.content_type,
            file_size=item.file_size,
            tags=item.tags,
            description=item.description,
            project_id=item.project_id,
            uploaded_by=item.uploaded_by_id,
            created_at=item.created_at,
            updated_at=item.updated_at,
            # PDF相关字段
            source_type=item.source_type,
            source_pdf_id=item.source_pdf_id,
            pdf_page_number=item.pdf_page_number,
            pdf_image_index=item.pdf_image_index
        )
        mapped_items.append(mapped_item)

    # Create page response with proper pagination metadata
    return FastAPIPage(
        items=mapped_items,
        total=total_count,
        page=page,
        size=size,
        pages=(total_count + size - 1) // size if total_count > 0 else 0
    )


@router.post('/projects/{project_id}/generate-embeddings')
async def generate_project_embeddings(
    project_id: str,
    user: Annotated[User, Security(get_current_user)],
    item_ids: Optional[List[str]] = None,
) -> dict:
    """
    为项目的items生成vector embeddings

    Args:
        project_id: 项目ID
        item_ids: 可选的item ID列表，如果为None则处理所有items

    Returns:
        dict: 处理结果统计
    """
    # 验证项目是否存在且用户有权限
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found'
        )

    try:
        result = await process_project_items_embeddings(project_id, item_ids)
        return {
            'message': 'Embeddings generation completed',
            'project_id': project_id,
            'result': result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Error generating embeddings: {str(e)}'
        )


@router.get('/{item_id}')
async def get_item(
    item_id: str,
    user: Annotated[User, Security(get_current_user)],
) -> ItemResponse:
    """
    获取特定Item的详细信息

    Args:
        item_id: Item ID

    Returns:
        ItemResponse: Item详细信息
    """
    # User can access their own items or items from projects they have access to
    accessible_projects = await Project.of_user(user).all()
    project_ids = [
        p.id for p in accessible_projects] if accessible_projects else []

    item = await Item.filter(
        Q(id=item_id) & (
            Q(uploaded_by=user) | Q(project__id__in=project_ids)
        )
    ).prefetch_related('uploaded_by', 'project').first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Item not found'
        )

    await item.fetch_image_url()

    return ItemResponse(
        id=item.id,
        filename=item.filename,
        s3_path=item.s3_path,
        s3_url=item.s3_url,
        image_url=item.image_url(),
        content_type=item.content_type,
        file_size=item.file_size,
        tags=item.tags,
        description=item.description,
        project_id=item.project_id,
        uploaded_by=item.uploaded_by_id,
        created_at=item.created_at,
        updated_at=item.updated_at,
        # PDF相关字段
        source_type=item.source_type,
        source_pdf_id=item.source_pdf_id,
        pdf_page_number=item.pdf_page_number,
        pdf_image_index=item.pdf_image_index
    )


@router.put('/{item_id}')
async def update_item(
    item_id: str,
    user: Annotated[User, Security(get_current_user)],
    item_update: ItemUpdate,
) -> ItemResponse:
    """
    更新Item信息

    Args:
        item_id: Item ID
        item_update: 更新数据

    Returns:
        ItemResponse: 更新后的Item信息
    """
    # Only the uploader can update the item
    item = await Item.filter(id=item_id, uploaded_by=user).prefetch_related('uploaded_by', 'project').first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Item not found or you do not have permission to update it'
        )

    # Update fields
    if item_update.filename is not None:
        item.filename = item_update.filename
    if item_update.tags is not None:
        item.tags = item_update.tags
    if item_update.description is not None:
        item.description = item_update.description
    if item_update.project_id is not None:
        project = await Project.of_user(user).get_or_none(id=item_update.project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='Project not found'
            )
        item.project = project

    await item.save()
    await item.fetch_image_url()

    return ItemResponse(
        id=item.id,
        filename=item.filename,
        s3_path=item.s3_path,
        s3_url=item.s3_url,
        image_url=item.image_url(),
        content_type=item.content_type,
        file_size=item.file_size,
        tags=item.tags,
        description=item.description,
        project_id=item.project_id,
        uploaded_by=item.uploaded_by_id,
        created_at=item.created_at,
        updated_at=item.updated_at
    )


@router.delete('/{item_id}')
async def delete_item(
    item_id: str,
    user: Annotated[User, Security(get_current_user)],
) -> dict:
    """
    删除Item

    Args:
        item_id: Item ID

    Returns:
        dict: 删除确认消息
    """
    # Only the uploader can delete the item
    item = await Item.filter(id=item_id, uploaded_by=user).first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Item not found or you do not have permission to delete it'
        )

    await item.delete()

    return {'message': 'Item deleted successfully'}


@router.get('/')
async def list_project_items(
    project_id: str,
    user: Annotated[User, Security(get_current_user)],
    tags: Annotated[Optional[str], Query()] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=1000)] = 300,
) -> Page[ItemResponse]:
    """
    获取特定项目的Items列表

    Args:
        project_id: 项目ID
        tags: 可选的标签过滤(逗号分隔)
        page: 页码 (从1开始)
        size: 每页数量 (1-1000)

    Returns:
        Page[ItemResponse]: 项目Items列表
    """
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found'
        )

    query = Item.filter(project=project)

    # Filter by tags if specified
    if tags:
        tag_list = [tag.strip() for tag in tags.split(',')]
        query = query.filter(tags__contains=tag_list)

    # Include relations and order by created_at desc for consistent pagination
    query = query.prefetch_related(
        'uploaded_by', 'project').order_by('-created_at')

    # Calculate pagination offset
    offset = (page - 1) * size

    # Get total count for pagination
    total_count = await query.count()

    # Get paginated items with limit and offset
    items = await query.offset(offset).limit(size).all()

    # Batch fetch S3 URLs for all items at once
    s3_paths = [item.s3_path for item in items if item.s3_path]

    # Batch fetch S3 URLs concurrently
    s3_url_tasks = [get_s3_url_from_path(path) for path in s3_paths]
    s3_urls = await asyncio.gather(*s3_url_tasks)

    # Create a mapping from s3_path to s3_url
    s3_url_map = dict(zip(s3_paths, s3_urls))

    # Map items to response models with pre-fetched S3 URLs
    mapped_items = []
    for item in items:
        # Set the image URL from our batch-fetched results
        if item.s3_path:
            setattr(item, '_image_url', s3_url_map.get(item.s3_path, ''))
        else:
            setattr(item, '_image_url', '')

        mapped_item = ItemResponse(
            id=item.id,
            filename=item.filename,
            s3_path=item.s3_path,
            s3_url=item.s3_url,
            image_url=item.image_url(),
            content_type=item.content_type,
            file_size=item.file_size,
            tags=item.tags,
            description=item.description,
            project_id=item.project_id,
            uploaded_by=item.uploaded_by_id,
            created_at=item.created_at,
            updated_at=item.updated_at
        )
        mapped_items.append(mapped_item)

    # Create page response with proper pagination metadata
    return FastAPIPage(
        items=mapped_items,
        total=total_count,
        page=page,
        size=size,
        pages=(total_count + size - 1) // size if total_count > 0 else 0
    )


class CropInfo(BaseModel):
    x: float  # 相对于原图的x坐标 (0-1)
    y: float  # 相对于原图的y坐标 (0-1)
    width: float  # 相对于原图的宽度 (0-1)
    height: float  # 相对于原图的高度 (0-1)


class ImageSearchRequest(BaseModel):
    image_url: str
    limit: int = 20
    crop: Optional[CropInfo] = None  # 可选的裁剪区域


@router.post('/projects/{project_id}/search-by-image')
async def search_items_by_image(
    project_id: str,
    request: ImageSearchRequest,
    user: Annotated[User, Security(get_current_user)],
) -> dict:
    """
    根据图片URL在项目中搜索相似的items

    Args:
        project_id: 项目ID
        request: 包含图片URL和结果数量限制的请求体

    Returns:
        dict: 搜索结果
    """
    # 验证项目访问权限
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found'
        )

    try:
        # 传递crop信息给服务函数
        crop_dict = None
        if request.crop:
            crop_dict = {
                'x': request.crop.x,
                'y': request.crop.y,
                'width': request.crop.width,
                'height': request.crop.height
            }

        results = await search_project_items_by_image(
            project_id,
            request.image_url,
            request.limit,
            crop=crop_dict
        )

        return {
            'results': results,
            'total': len(results)
        }

    except ValueError as e:
        # 处理下载错误（S3或HTTP）
        error_message = str(e)

        # HTTP相关错误
        if "HTTP error downloading image" in error_message:
            if "404" in error_message:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail='Image not found at the provided URL. Please check the image URL.'
                )
            elif "403" in error_message:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail='Access denied to the image URL. The URL may have expired.'
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Error downloading image from URL: {error_message}'
                )

        # HTTP超时错误
        elif "Timeout downloading image" in error_message:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail='Timeout downloading image. Please try again or check your network connection.'
            )

        # S3相关错误
        elif "Error downloading file from S3" in error_message:
            if "Not Found" in error_message or "404" in error_message:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail='Image not found in storage. Please check the image path.'
                )
            elif "NoCredentialsError" in error_message or "credentials" in error_message.lower():
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail='Storage service configuration error'
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Error accessing image from storage: {error_message}'
                )

        # 其他ValueError
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid request: {error_message}'
            )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Image file not found'
        )

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Access denied to image resource'
        )

    except Exception as e:
        # 记录详细错误信息用于调试
        logger.error(
            f"Unexpected error in search_items_by_image: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred while searching items'
        )


class BoundingBoxDetectionRequest(BaseModel):
    image_url: str
    subtask_id: str
    limit: int = 10


@router.post('/projects/{project_id}/detect-objects')
async def detect_objects_in_image(
    project_id: str,
    request: BoundingBoxDetectionRequest,
    background_tasks: BackgroundTasks,
    user: Annotated[User, Security(get_current_user)],
) -> JSONResponse:
    """
    在图片中检测对象并返回边界框信息，并保存到subtask的ai_detection字段

    使用background task来执行，避免阻塞UI

    Args:
        project_id: 项目ID
        request: 包含图片URL、subtask_id和限制数量的请求体

    Returns:
        dict: 响应消息，表示任务已开始
    """
    # 验证项目访问权限
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found'
        )

    # 验证subtask是否存在且属于该项目
    subtask = await Subtask.filter(
        id=request.subtask_id,
        task__project=project
    ).select_related('task').first()

    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Subtask not found or does not belong to this project'
        )

    try:
        # 添加background task来执行对象检测
        background_tasks.add_task(
            _detect_objects_background_task,
            project_id=project_id,
            subtask_id=request.subtask_id,
            image_url=request.image_url,
            limit=request.limit,
            user_id=str(user.id)
        )

        # 设置初始处理状态
        subtask.ai_detection = {
            'status': 'processing',
            'started_at': datetime.now().isoformat(),
            'detection_version': 'gemini-2.5-pro'
        }
        await subtask.save(update_fields=['ai_detection', 'updated_at'])

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                'message': f'Object detection initiated for subtask {request.subtask_id}',
                'status': 'processing',
                'project_id': project_id,
                'subtask_id': request.subtask_id,
                'image_url': request.image_url
            }
        )

    except Exception as e:
        logger.error(
            f"Unexpected error in detect_objects_in_image: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred while initiating object detection'
        )


# Background task functions
async def _batch_upload_background_task(
    batch_id: str,
    file_data_list: List[Dict[str, Any]],
    project_id: Optional[str],
    parsed_tags: List,
    description: Optional[str],
    project: Optional[Project],
    user_id: str,
    max_concurrent: Optional[int] = None
) -> None:
    """
    Background task for batch uploading files.

    Args:
        batch_id: Unique batch identifier
        file_data_list: List of file data dictionaries
        project_id: Optional project ID
        parsed_tags: Parsed tags list
        description: Optional description
        project: Project object (can be None)
        user_id: User ID string
        max_concurrent: Maximum concurrent uploads, defaults from settings
    """
    batch_start_time = time.time()

    try:
        # Get batch job record
        batch_job = await BatchProcessJob.get_or_none(batch_id=batch_id)
        if not batch_job:
            logger.error(f"Batch job {batch_id} not found")
            return

        # Get user object
        user = await User.get_or_none(id=user_id)
        if not user:
            logger.error(f"User {user_id} not found")
            return

        logger.info(f"🚀 [批次 {batch_id}] 开始处理 {len(file_data_list)} 个文件")

        # Create semaphore for concurrent processing
        if max_concurrent is None:
            max_concurrent = MAX_CONCURRENT
        semaphore = asyncio.Semaphore(max_concurrent)
        uploaded_items = []
        failed_uploads = []

        async def process_single_file_with_limit(file_data: Dict[str, Any], index: int) -> Dict[str, Any]:
            async with semaphore:
                try:
                    logger.info(
                        f"📄 [批次 {batch_id}] 处理文件 {index + 1}/{len(file_data_list)}: {file_data['filename']}")

                    # Use the data-based single file processing function
                    result = await _process_single_file_from_data(
                        filename=file_data['filename'],
                        content_type=file_data['content_type'],
                        file_content=file_data['content'],
                        project_id=project_id,
                        parsed_tags=parsed_tags,
                        description=description,
                        project=project,
                        user=user
                    )

                    # Update progress - ensure batch_job is not None
                    if batch_job:
                        batch_job.processed_items += 1
                        if result.get('success'):
                            batch_job.successful_items += 1
                            logger.info(
                                f"✅ [批次 {batch_id}] 文件处理成功: {file_data['filename']}")
                        else:
                            batch_job.failed_items += 1
                            logger.warning(
                                f"❌ [批次 {batch_id}] 文件处理失败: {file_data['filename']} - {result.get('error', 'Unknown error')}")

                        await batch_job.save(update_fields=['processed_items', 'successful_items', 'failed_items', 'updated_at'])

                    return result

                except Exception as e:
                    logger.error(
                        f"❌ [批次 {batch_id}] 文件处理异常: {file_data['filename']} - {str(e)}")
                    if batch_job:
                        batch_job.failed_items += 1
                        batch_job.processed_items += 1
                        await batch_job.save(update_fields=['processed_items', 'failed_items', 'updated_at'])

                    return {
                        'success': False,
                        'filename': file_data['filename'],
                        'error': f'处理异常: {str(e)}'
                    }

        # Process all files concurrently
        tasks = [
            process_single_file_with_limit(file_data, index)
            for index, file_data in enumerate(file_data_list)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                failed_uploads.append({
                    'filename': 'unknown',
                    'error': f'予期しないエラー: {str(result)}'
                })
            elif isinstance(result, dict) and result.get('success'):
                uploaded_items.append(result['item'])
            elif isinstance(result, dict):
                failed_uploads.append({
                    'filename': result['filename'],
                    'error': result['error']
                })

        # Update final status
        batch_job.status = BatchJobStatus.COMPLETED
        batch_job.completed_at = datetime.now(timezone.utc)
        await batch_job.save(update_fields=['status', 'completed_at', 'updated_at'])

        batch_end_time = time.time()
        duration = batch_end_time - batch_start_time

        logger.info(
            f"✅ [批次 {batch_id}] 批量上传完成: {len(uploaded_items)} 成功, {len(failed_uploads)} 失败, 耗时 {duration:.2f}秒")

        # Generate embeddings for successful uploads if project_id exists
        if uploaded_items and project_id:
            logger.info(f"🔄 [批次 {batch_id}] 开始生成 embeddings...")
            try:
                uploaded_item_ids = [str(item.id) for item in uploaded_items]
                await process_project_items_embeddings(project_id, uploaded_item_ids)
                logger.info(f"✅ [批次 {batch_id}] Embeddings 生成完成")
            except Exception as e:
                logger.error(f"❌ [批次 {batch_id}] Embeddings 生成失败: {str(e)}")

    except Exception as e:
        logger.error(f"❌ [批次 {batch_id}] 批量上传任务异常: {str(e)}")

        # Update batch job status to failed
        try:
            batch_job = await BatchProcessJob.get_or_none(batch_id=batch_id)
            if batch_job:
                batch_job.status = BatchJobStatus.FAILED
                batch_job.completed_at = datetime.now(timezone.utc)
                await batch_job.save(update_fields=['status', 'completed_at', 'updated_at'])
        except Exception as save_error:
            logger.error(f"Failed to update batch job status: {save_error}")


async def _detect_objects_background_task(
    project_id: str,
    subtask_id: str,
    image_url: str,
    limit: int,
    user_id: str
) -> None:
    """
    Background task for object detection.
    """
    try:
        # 获取subtask以便保存结果
        subtask = await Subtask.get_or_none(id=subtask_id)
        if not subtask:
            print(f"Subtask {subtask_id} not found in background task")
            return

        # 执行对象检测
        results = await list_up_items_in_image(
            project_id,
            image_url,
            limit
        )

        # 准备保存到ai_detection字段的数据
        detection_data = {
            'bounding_boxes': results,
            'total': len(results),
            'detected_at': datetime.now().isoformat(),
            'detection_version': 'gemini-2.5-pro',
            'status': 'completed'
        }

        # 更新subtask的ai_detection字段
        subtask.ai_detection = detection_data
        await subtask.save(update_fields=['ai_detection', 'updated_at'])

        print(f"Object detection completed for subtask {subtask_id}")

    except ValueError as e:
        # 处理下载错误（S3或HTTP）
        error_message = str(e)

        # 保存错误信息到subtask的ai_detection字段
        try:
            subtask = await Subtask.get_or_none(id=subtask_id)
            if subtask:
                error_detection_data = {
                    'error': error_message,
                    'failed_at': datetime.now().isoformat(),
                    'detection_version': 'gemini-2.5-pro',
                    'status': 'failed'
                }
                subtask.ai_detection = error_detection_data
                await subtask.save(update_fields=['ai_detection', 'updated_at'])
        except Exception as save_error:
            print(
                f"Failed to save error to subtask ai_detection: {save_error}")

        print(
            f"Error in object detection background task for subtask {subtask_id}: {error_message}")

    except Exception as e:
        # 保存通用错误信息到subtask的ai_detection字段
        try:
            subtask = await Subtask.get_or_none(id=subtask_id)
            if subtask:
                error_detection_data = {
                    'error': f'Unexpected error: {str(e)}',
                    'failed_at': datetime.now().isoformat(),
                    'detection_version': 'gemini-2.5-pro',
                    'status': 'failed'
                }
                subtask.ai_detection = error_detection_data
                await subtask.save(update_fields=['ai_detection', 'updated_at'])
        except Exception as save_error:
            print(
                f"Failed to save error to subtask ai_detection: {save_error}")

        print(
            f"Unexpected error in object detection background task for subtask {subtask_id}: {e}")
        import traceback
        traceback.print_exc()
