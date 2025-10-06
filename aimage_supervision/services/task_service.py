import datetime
import io
from aimage_supervision.settings import logger
import os
import re
import uuid
from typing import BinaryIO, List, Optional
from uuid import UUID

from fastapi import HTTPException, UploadFile, status
from tortoise.exceptions import DoesNotExist
from tortoise.expressions import Q

from aimage_supervision.clients.aws_s3 import (AWS_BUCKET_NAME,
                                               upload_file_to_s3)
# Renamed to avoid conflict with model name
from aimage_supervision.enums import SubtaskStatus as SubtaskStatusEnum
from aimage_supervision.enums import SubtaskType
# Assuming TaskOut is the Pydantic model for Task response
from aimage_supervision.models import (Project, ReviewSet, Subtask, Task,
                                       TaskOut, TaskPriority, TaskStatus,
                                       TaskTag, User)
from aimage_supervision.schemas import ReviewSetOut, SubtaskContent
from aimage_supervision.utils.image_compression import compress_image_async

# Helper function to clean filename (can be moved to a utils module if preferred)


def clean_filename(filename: str) -> str:

    # Remove extension first
    name_part, ext_part = os.path.splitext(filename)
    # Replace special characters with underscore
    cleaned_name = re.sub(r'[^\w.-]', '_', name_part)
    # Limit length of the name part
    cleaned_name = cleaned_name[:100]  # Limit name part to 100 chars
    return cleaned_name + ext_part


async def create_task_from_image(
    project_id: UUID,
    image_file: UploadFile,
    current_user: User,
    tag_ids: Optional[List[UUID]] = None,
) -> TaskOut:
    """
    Creates a new task and its first subtask from an uploaded image file.
    """
    # 1. Verify project existence and user access
    project = await Project.get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="プロジェクトが見つかりません",
        )
    if not await project.is_user_can_access(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="このプロジェクトでタスクを作成する権限がありません",
        )

    # 2. Upload original and compressed image to S3
    file_content = await image_file.read()
    original_filename = image_file.filename or "untitled_image"
    base_filename = f"{uuid.uuid4()}_{original_filename}"

    # Upload original
    original_s3_path = f"images/projects/{project_id}/tasks/{base_filename}"
    await upload_file_to_s3(io.BytesIO(file_content), original_s3_path)

    # Compress and upload
    compressed_s3_path = None
    try:
        with io.BytesIO(file_content) as f:
            compressed_file, output_format, _ = await compress_image_async(f)

        # Handle different formats including SVG, PSD, AI
        if output_format == 'JPEG':
            ext = 'jpg'
        elif output_format == 'PNG':
            ext = 'png'
        elif output_format == 'SVG':
            ext = 'svg'
        elif output_format == 'PSD':
            ext = 'psd'
        elif output_format == 'AI':
            ext = 'ai'
        else:
            ext = output_format.lower()
        
        base_fn_without_ext = os.path.splitext(base_filename)[0]
        compressed_filename = f"{base_fn_without_ext}_compressed.{ext}"

        compressed_s3_path = f"images/projects/{project_id}/tasks/{compressed_filename}"

        # Check if compressed_file is the same as original (no compression benefit)
        if compressed_file.closed or hasattr(compressed_file, 'name'):
            # If original file was returned (no compression benefit), create a new BytesIO
            compressed_file = io.BytesIO(file_content)

        await upload_file_to_s3(compressed_file, compressed_s3_path)
    except Exception as e:
        logger.error(
            f"Image compression failed for new task from image: {e}", exc_info=True)

    # 3. Create Task
    timestamp_tid = datetime.datetime.now().strftime("T%Y%m%d%H%M%S%f")
    task_name = os.path.splitext(original_filename)[0]
    task_description = f"画像「{original_filename}」からのタスク"

    default_status = await TaskStatus.all().first()
    default_priority = await TaskPriority.all().first()

    if not default_status or not default_priority:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="プロジェクトの初期ステータスまたは優先度設定が見つかりません",
        )

    new_task = await Task.create(
        tid=timestamp_tid,
        name=task_name,
        description=task_description,
        s3_path=original_s3_path,  # Main task S3 path can point to the original image
        project=project,
        assignee=current_user,
        status=default_status,
        priority=default_priority,
    )

    # Add tags if provided (m2m relationship must be set after object creation)
    if tag_ids:
        tags = await TaskTag.filter(id__in=tag_ids, project_id=project.id)
        await new_task.tags.add(*tags)

    # 4. Create Subtask
    subtask_content_data = {
        'title': task_name,
        'description': task_description,
        'task_type': SubtaskType.PICTURE.value,
        's3_path': original_s3_path,
        'compressed_s3_path': compressed_s3_path,
        'author': str(current_user.id),
        'created_at': datetime.datetime.now().isoformat(),
    }

    await Subtask.create(
        oid=1,
        name=task_name,
        task_type=SubtaskType.PICTURE,
        description=task_description,
        content=subtask_content_data,
        task=new_task,
        status=SubtaskStatusEnum.PENDING,
    )

    # 5. Return created task
    # We need to fetch the relations to serialize them correctly in TaskOut
    created_task = await Task.get(id=new_task.id).select_related('status', 'priority', 'assignee')
    return await TaskOut.from_tortoise_orm(created_task)


async def create_task_from_video(
    project_id: UUID,
    video_file: UploadFile,
    current_user: User,
    tag_ids: Optional[List[UUID]] = None,
) -> TaskOut:

    project = await Project.get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='プロジェクトが存在しません'  # Project not found
        )

    if not await project.is_user_can_access(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='このプロジェクトにアクセスする権限がありません'  # No permission to access this project
        )

    # 1. Prepare filenames and S3 path
    original_filename = video_file.filename if video_file.filename else "uploaded_video"
    cleaned_original_filename = clean_filename(original_filename)

    filename_stem, filename_ext = os.path.splitext(cleaned_original_filename)
    timestamp = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]
    s3_filename = f"{filename_stem}_{timestamp}{filename_ext}"

    s3_key = f"projects/{str(project_id)}/videos/{s3_filename}"

    # 2. Upload to S3
    try:
        await upload_file_to_s3(
            file_data=video_file.file,  # type: ignore
            cloud_file_path=s3_key,
            bucket_name=AWS_BUCKET_NAME
        )
    except Exception as e:
        # Log the exception e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'S3への動画のアップロードに失敗しました: {str(e)}'
        )

    # 3. Fetch default TaskStatus and TaskPriority
    default_task_status = await TaskStatus.filter(name='TODO').first()
    if not default_task_status:
        default_task_status = await project.status_candidates.all().first()
        if not default_task_status:
            default_task_status = await TaskStatus.all().first()

    default_task_priority = await TaskPriority.filter(name='LOW').first()
    if not default_task_priority:
        default_task_priority = await project.priority_candidates.all().first()
        if not default_task_priority:
            default_task_priority = await TaskPriority.all().first()

    if not default_task_status or not default_task_priority:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='プロジェクトのデフォルトのステータスまたは優先度が見つかりません'
        )

    # 4. Create Task record
    task_name = filename_stem[:250]
    task_tid = datetime.datetime.now(datetime.timezone.utc).strftime(
        "T%Y%m%d%H%M%S")

    new_task = await Task.create(
        tid=task_tid,
        name=task_name,
        description=f'動画「{original_filename}」から作成されたタスクです。',
        s3_path="",  # Main task s3_path might not be relevant here, or could be folder path,
        project_id=project_id,
        assignee_id=current_user.id,
        status_id=default_task_status.id,
        priority_id=default_task_priority.id,
    )

    # Add tags if provided
    if tag_ids:
        tags = await TaskTag.filter(id__in=tag_ids, project_id=project_id)
        await new_task.tags.add(*tags)

    # 5. Create Subtask record
    last_subtask = await Subtask.filter(task_id=new_task.id).order_by('-oid').first()
    next_oid = (last_subtask.oid + 1) if last_subtask else 0

    subtask_content_obj = SubtaskContent(
        title=filename_stem[:250],
        s3_path=s3_key,
        description="",
        task_type=SubtaskType.VIDEO.value,
        author=str(current_user.id),
        created_at=datetime.datetime.now(
            datetime.timezone.utc).isoformat() + "Z",
        slide_page_number=None
    )

    await Subtask.create(
        oid=next_oid,
        name="アップロードされた動画",
        task_type=SubtaskType.VIDEO,
        description=f'タスク「{task_name}」の初期動画です。',
        content=subtask_content_obj.model_dump(),
        task_id=new_task.id,
        status=SubtaskStatusEnum.PENDING,
        assignee_id=current_user.id,
        slide_page_number=None
    )

    # 6. Return the created task
    created_task_with_relations = await Task.get(id=new_task.id).prefetch_related('assignee', 'status', 'priority', 'project')
    return await TaskOut.from_tortoise_orm(created_task_with_relations)


async def get_suggested_review_sets(task_id: UUID) -> List[ReviewSetOut]:
    """
    Suggests review sets for a given task based on its associated characters and tags.
    """
    try:
        task = await Task.get(id=task_id).prefetch_related("tags")
    except DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    # 1. Get all character IDs from all subtasks of the task
    subtasks = await Subtask.filter(task_id=task_id)
    character_ids = set()
    for subtask in subtasks:
        if subtask.character_ids:
            for char_id in subtask.character_ids:
                character_ids.add(UUID(char_id))

    # 2. Get all tag IDs from the task
    tag_ids = {tag.id for tag in task.tags}

    # 3. Find all review sets linked to these characters OR tags
    # Use Q objects for OR condition
    review_sets = await ReviewSet.filter(
        Q(characters__id__in=list(character_ids)) | Q(
            task_tags__id__in=list(tag_ids)),
        project_id=task.project_id
    ).distinct().prefetch_related("rpds", "characters", "task_tags")

    # 4. Format the output
    results = [await ReviewSetOut.from_tortoise_orm(rs) for rs in review_sets]
    return results


async def soft_delete_task(task_id: UUID) -> bool:
    """
    Soft-deletes a task by setting its `is_deleted` flag to True.
    Returns True if soft-deletion was successful, False if task was not found.
    """
    try:
        task = await Task.get(id=task_id, is_deleted=False)
        task.is_deleted = True
        await task.save()
        return True
    except DoesNotExist:
        return False
