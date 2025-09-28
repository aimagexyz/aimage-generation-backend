import asyncio
import datetime
import io
from aimage_supervision.settings import logger
import os
import urllib.parse
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional

import httpx
from fastapi import (APIRouter, BackgroundTasks, Depends, File, Form,
                     HTTPException, Response, Security, UploadFile, status)
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi_pagination import paginate
from pydantic import BaseModel
from tortoise.exceptions import DoesNotExist

from aimage_supervision.clients.aws_s3 import upload_file_to_s3
from aimage_supervision.enums import SubtaskStatus, SubtaskType
from aimage_supervision.exceptions import (ContentNotFoundError,
                                           FileTooLargeError, InvalidPathError)
from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.middlewares.tortoise_paginate import (
    Page, tortoise_paginate)
from aimage_supervision.models import (Project, Subtask, SubtaskContent,
                                       SubtaskDetail, SubtaskOut, Task, TaskIn,
                                       TaskKanbanOrder, TaskKanbanOrderIn,
                                       TaskKanbanOrderOut, TaskOut,
                                       TaskPriority, TaskPriorityOut,
                                       TaskStatus, TaskStatusOut, TaskTag,
                                       User)
from aimage_supervision.schemas import (ReviewSetOut, SubtaskAnnotation,
                                        SubtaskAnnotationUpdate,
                                        SubtaskCharactersUpdate,
                                        SubtaskContent, SubtaskUpdate,
                                        TaskNavigationItem,
                                        TaskNavigationResponse, TaskSimpleOut,
                                        TaskTagOut, TaskThumbnail,
                                        TaskThumbnailsResponse, TaskUpdate)
from aimage_supervision.services.ai_processing_service import \
    process_subtask_ai_classification
from aimage_supervision.services.export_service import PptxExportService
from aimage_supervision.services.task_service import (
    create_task_from_image, create_task_from_video, get_suggested_review_sets,
    soft_delete_task)
from aimage_supervision.settings import (SUPERVISION_BATCH_API_PASSWORD,
                                         SUPERVISION_BATCH_API_URL,
                                         SUPERVISION_BATCH_API_USER)
from aimage_supervision.utils.get_screenshoot import get_image
from aimage_supervision.utils.image_compression import compress_image_async
from aimage_supervision.utils.file_validation import is_valid_image_or_design_file
from aimage_supervision.utils.upload_pptx import UploadPPTXProcessor


# Constants for file upload
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_VIDEO_SIZE_BYTES = 1000 * 1024 * 1024  # 1000MB
ALLOWED_IMAGE_CONTENT_TYPE_PREFIX = "image/"
ALLOWED_VIDEO_CONTENT_TYPE_PREFIX = "video/"

# Additional allowed file types for design files
ALLOWED_DESIGN_CONTENT_TYPES = [
    "application/postscript",  # AI files (older)
    "application/pdf",  # AI files (newer, PDF-based)
    "application/x-photoshop",  # PSD files
    "application/psd",  # PSD files
    "image/vnd.adobe.photoshop",  # PSD files
    "application/illustrator",  # AI files
    "application/x-illustrator",  # AI files
]

router = APIRouter(prefix='', tags=['Tasks'])

# 用于存储文件处理状态的内存字典
# 键为批次ID，值为包含每个文件处理状态的字典
file_processing_status: Dict[str, Dict[str, Any]] = {}


class FileProcessingStatus(BaseModel):
    """文件处理状态模型"""
    batch_id: str
    total_files: int
    processed_files: int
    successful_files: int
    failed_files: int
    files_status: Dict[str, Dict[str, Any]]


# Helper functions for annotation operations
def check_annotation_permission(annotation: dict, user: User, operation: str) -> bool:
    """检查用户是否有权限操作注释"""
    if annotation.get('author') != str(user.id):
        return False

    # AI生成的注释不能被修改或删除
    if annotation.get('type') in ['ai-annotation', 'ai-comment']:
        return False

    return True


def validate_annotation_update(annotation: dict, new_text: str) -> bool:
    """验证注释更新的合法性"""
    # 检查文本长度
    if not new_text or len(new_text.strip()) == 0:
        return False
    if len(new_text) > 5000:
        return False

    # 检查注释状态（如果有deleted字段）
    if annotation.get('deleted'):
        return False

    return True


@router.get('/tasks/priorities')
async def list_task_priorities(
    user: Annotated[User, Security(get_current_user)],
) -> Page[TaskPriorityOut]:
    # TODO: Permission check
    priorities = TaskPriority.all()
    response = await tortoise_paginate(
        priorities,
        model=TaskPriorityOut,  # type: ignore
    )
    return response


@router.get('/tasks/statuses')
async def list_task_statuses(
    user: Annotated[User, Security(get_current_user)],
) -> Page[TaskStatusOut]:
    # TODO: Permission check
    statuses = TaskStatus.all()
    response = await tortoise_paginate(
        statuses,
        model=TaskStatusOut,  # type: ignore
    )
    return response


@router.get('/projects/{project_id}/tasks')
async def list_project_tasks(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    page: int = 1,
    page_size: int = 20,
    sort_by: Literal["created_at", "priority"] = "created_at",
    order: Literal["asc", "desc"] = "desc",
    status_id: Optional[str] = None,
    priority_id: Optional[str] = None,
    assignee_id: Optional[str] = None
) -> Dict[str, Any]:
    project = await Project.get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    if not await project.is_user_can_access(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="You don't have permission to access tasks in this project")

    tasks_queryset = Task.filter(project_id=project_id, is_deleted=False)

    # Apply status filter if provided
    if status_id is not None:
        tasks_queryset = tasks_queryset.filter(status_id=status_id)

    # Apply priority filter if provided
    if priority_id is not None:
        tasks_queryset = tasks_queryset.filter(priority_id=priority_id)

    # Apply assignee filter if provided
    if assignee_id is not None:
        tasks_queryset = tasks_queryset.filter(assignee_id=assignee_id)

    # 排序表达式
    order_by_field = f"-{sort_by}" if order == "desc" else sort_by

    # 获取总数（单独 COUNT 查询）
    total = await tasks_queryset.count()

    # 构造分页 QuerySet，并一次性 select_related 联表获取关联字段，避免 1+N
    paginated_qs = (
        tasks_queryset
        .order_by(order_by_field)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .select_related('status', 'priority', 'assignee')
    )

    # 直接使用 Pydantic creator 的 from_queryset 批量序列化，提升效率
    # type: ignore[arg-type]
    tasks_out = await TaskOut.from_queryset(paginated_qs)

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "data": tasks_out
    }


@router.patch('/tasks/{task_id}/status')
async def update_task_status(
    user: Annotated[User, Security(get_current_user)],
    task_id: str,
    status_id: str,
) -> TaskOut:
    # TODO: Permission check
    task = await Task.get_or_none(id=task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    task.status_id = status_id
    await task.save()
    response = await TaskOut.from_tortoise_orm(task)
    return response


@router.patch('/tasks/{task_id}/priority')
async def update_task_priority(
    user: Annotated[User, Security(get_current_user)],
    task_id: str,
    priority_id: str,
) -> TaskOut:
    # TODO: Permission check
    task = await Task.get_or_none(id=task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    task.priority_id = priority_id
    await task.save()
    response = await TaskOut.from_tortoise_orm(task)
    return response


@router.get('/projects/{project_id}/tasks-kanban-order')
async def list_project_tasks_kanban_order(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
) -> Page[TaskKanbanOrderOut]:
    # TODO: Permission check
    queryset = TaskKanbanOrder.filter(project_id=project_id)
    response = await tortoise_paginate(
        queryset,
        model=TaskKanbanOrderOut,  # type: ignore
    )
    return response


@router.post('/projects/{project_id}/tasks-kanban-order/statuses/{status_id}', status_code=status.HTTP_201_CREATED)
async def create_or_update_project_tasks_kanban_order(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    status_id: str,
    payload: TaskKanbanOrderIn,
):
    # TODO: Permission check
    try:
        count = await TaskKanbanOrder.filter(project_id=project_id, status_id=status_id).update(
            task_order=payload.task_order,
        )
        if not count:
            raise DoesNotExist(TaskKanbanOrder)
        return Response(status_code=status.HTTP_200_OK)
    except DoesNotExist:
        await TaskKanbanOrder.create(project_id=project_id, status_id=status_id, task_order=payload.task_order)
        return Response(status_code=status.HTTP_201_CREATED)


@router.get('/tasks/{task_id}')
async def get_task(
    user: Annotated[User, Security(get_current_user)],
    task_id: str,
) -> TaskOut:
    # TODO: Permission check
    task = await Task.get_or_none(id=task_id, is_deleted=False)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    response = await TaskOut.from_tortoise_orm(task)
    return response


@router.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: uuid.UUID,
    current_user: User = Security(get_current_user),
):
    """
    Soft-deletes a task.

    This operation marks a task as deleted but does not permanently remove it from the database.
    Associated subtasks and other data are preserved.
    """
    # 1. Check if task exists and is not already deleted.
    task = await Task.get_or_none(id=task_id, is_deleted=False)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    # 2. Check user permissions.
    project = await task.project
    if not await project.is_user_can_access(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this task",
        )

    # 3. Call the service to perform the soft delete.
    success = await soft_delete_task(task_id)  # type: ignore
    if not success:
        # This case should ideally not be hit if the initial check passed, but it's a safeguard.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found during deletion process",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put('/tasks/{task_id}', response_model=TaskOut)
async def update_task_details(
    user: Annotated[User, Security(get_current_user)],
    task_id: str,
    task_update_payload: TaskUpdate,
) -> TaskOut:
    # TODO: Permission check

    task = await Task.get_or_none(id=task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    update_data = task_update_payload.model_dump(exclude_unset=True)
    if "title" in update_data and update_data["title"] is not None:
        task.name = update_data["title"]
    if "description" in update_data and update_data["description"] is not None:
        task.description = update_data["description"]
    if "status_id" in update_data and update_data["status_id"] is not None:
        task.status_id = update_data["status_id"]
    if "priority_id" in update_data and update_data["priority_id"] is not None:
        task.priority_id = update_data["priority_id"]
    if "assignee_id" in update_data and update_data["assignee_id"] is not None:
        task.assignee_id = update_data["assignee_id"]
    if "due_date" in update_data:
        task.due_date = update_data["due_date"]
    await task.save()

    response = await TaskOut.from_tortoise_orm(task)
    return response


@router.post('/tasks/{task_id}/tags/{tag_id}', status_code=status.HTTP_201_CREATED)
async def add_tag_to_task(
    user: Annotated[User, Security(get_current_user)],
    task_id: str,
    tag_id: str,
) -> Response:
    """Add a tag to a task"""
    # Get task and check permissions
    task = await Task.get_or_none(id=task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    # Check project permission
    project = await task.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to modify this task"
        )

    # Get tag and verify it belongs to the same project
    tag = await TaskTag.get_or_none(id=tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )

    if tag.project_id != project.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tag does not belong to the same project as the task"
        )

    # Add tag to task
    await task.tags.add(tag)

    return Response(status_code=status.HTTP_201_CREATED)


@router.delete('/tasks/{task_id}/tags/{tag_id}', status_code=status.HTTP_204_NO_CONTENT)
async def remove_tag_from_task(
    user: Annotated[User, Security(get_current_user)],
    task_id: str,
    tag_id: str,
) -> Response:
    """Remove a tag from a task"""
    # Get task and check permissions
    task = await Task.get_or_none(id=task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    # Check project permission
    project = await task.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to modify this task"
        )

    # Get tag
    tag = await TaskTag.get_or_none(id=tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found"
        )

    # Remove tag from task
    await task.tags.remove(tag)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get('/tasks/{task_id}/tags', response_model=List[TaskTagOut])
async def get_task_tags(
    user: Annotated[User, Security(get_current_user)],
    task_id: str,
) -> List[TaskTagOut]:
    """Get all tags for a task"""
    # Get task and check permissions
    task = await Task.get_or_none(id=task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    # Check project permission
    project = await task.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this task"
        )

    # Get all tags for the task
    tags = await task.tags.all()

    # Convert to TaskTagOut
    return [TaskTagOut.model_validate(tag) for tag in tags]


@router.get('/tasks/{task_id}/subtasks')
async def list_task_subtasks(
    user: Annotated[User, Security(get_current_user)],
    task_id: str,
) -> Page[SubtaskOut]:
    # TODO: Permission check
    subtasks = Subtask.filter(task_id=task_id)
    response = await tortoise_paginate(
        subtasks,
        model=SubtaskOut,  # type: ignore
    )
    return response


@router.get('/subtasks/{subtask_id}')
async def get_subtask(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
) -> SubtaskDetail:
    # TODO: Permission check
    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    # 优先显示用户手动选择的角色，如果没有则显示AI预测的角色
    effective_character_ids = subtask.get_effective_character_ids()

    # 获取SubtaskDetail对象
    response = await SubtaskDetail.from_tortoise_orm(subtask)

    # 直接设置有效的角色ID，避免重新验证带来的额外字段问题
    response.character_ids = effective_character_ids

    return response


@router.put('/subtasks/{subtask_id}', response_model=SubtaskDetail)
async def update_subtask_details(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
    subtask_update_payload: SubtaskUpdate,
) -> SubtaskDetail:
    """Update subtask name and description"""
    # TODO: Permission check

    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subtask not found"
        )

    update_data = subtask_update_payload.model_dump(exclude_unset=True)
    if "name" in update_data and update_data["name"] is not None:
        subtask.name = update_data["name"]
    if "description" in update_data and update_data["description"] is not None:
        subtask.description = update_data["description"]

    await subtask.save()
    response = await SubtaskDetail.from_tortoise_orm(subtask)
    return response


@router.patch('/subtasks/{subtask_id}/characters', response_model=SubtaskDetail)
async def update_subtask_characters(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
    characters_update: SubtaskCharactersUpdate,
) -> SubtaskDetail:
    """Update subtask associated characters (user manual selection)"""
    # TODO: Permission check

    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subtask not found"
        )

    # 保存用户手动选择到专用字段，不覆盖AI预测结果
    subtask.user_selected_character_ids = characters_update.character_ids
    await subtask.save()

    response = await SubtaskDetail.from_tortoise_orm(subtask)
    return response


@router.get('/subtasks/{subtask_id}/annotations')
async def list_subtask_annotations(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
) -> Page[SubtaskAnnotation]:
    '''List all annotations for a subtask.'''
    # TODO: Permission check
    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Subtask not found',
        )

    annotations = subtask.annotations or []
    response = paginate(annotations)
    return response


@router.post('/subtasks/{subtask_id}/annotations', status_code=status.HTTP_201_CREATED)
async def create_subtask_annotation(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
    payload: SubtaskAnnotation,
):
    '''Create a new annotation for a subtask.'''
    # TODO: Permission check
    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Subtask not found',
        )
    if not subtask.annotations:
        subtask.annotations = []

    payload.author = str(user.id)
    # get current UTC time
    payload.timestamp = datetime.datetime.now().isoformat()
    # Use version from payload if provided, otherwise use current version
    if payload.version is None:
        payload.version = subtask.version()

    subtask.annotations.append(payload.model_dump())  # type: ignore
    await subtask.save()

    if payload.type in ['ai-annotation', 'ai-comment']:
        file = []
        if payload.type == 'ai-annotation':
            path = await get_image(str(user.id), subtask.content.s3_path, payload)
            file.append({
                "type": "image",
                "transfer_method": "remote_url",
                "url": path
            })

        ai_response = 'AI response: this is a test response....'

        # 创建一个新的 AI 响应注释，基于原始 payload
        ai_response_annotation = SubtaskAnnotation(
            id=str(uuid.uuid4()),
            text=ai_response,
            type=payload.type,
            to=payload.id,
            version=payload.version,  # 使用与原始注释相同的版本
            timestamp=datetime.datetime.now().isoformat(),
            rect=payload.rect,
            color=payload.color,
            tool=payload.tool,
            start_at=payload.start_at,
            end_at=payload.end_at,
            solved=payload.solved,
            attachment_image_url=payload.attachment_image_url
        )

        # 添加到子任务的注释列表中
        subtask.annotations.append(
            ai_response_annotation.model_dump())  # type: ignore
        await subtask.save()

    return Response(status_code=status.HTTP_201_CREATED)


@router.patch('/subtasks/{subtask_id}/annotations/{annotation_id}/solved')
async def update_subtask_annotation_solved(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
    annotation_id: str,
    solved: bool = True,
) -> SubtaskDetail:
    '''Mark annotation as solved.'''
    # TODO: Permission check
    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Subtask not found',
        )

    if not subtask.annotations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Annotation not found',
        )

    # find annotation by id
    annotation: dict | None = next((a for a in subtask.annotations if a.get(  # type: ignore
        'id') == annotation_id), None)
    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Annotation not found',
        )

    if annotation.get('solved') is not solved:
        annotation['solved'] = solved
        await subtask.save()

    response = await SubtaskDetail.from_tortoise_orm(subtask)
    return response


@router.patch('/subtasks/{subtask_id}/annotations/{annotation_id}', response_model=SubtaskDetail)
async def update_subtask_annotation(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
    annotation_id: str,
    payload: SubtaskAnnotationUpdate,
) -> SubtaskDetail:
    """修改子任务注释内容"""
    # 1. 验证子任务存在性
    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Subtask not found',
        )

    # 2. 验证项目权限
    await subtask.fetch_related('task__project')
    task = subtask.task
    project = task.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You don\'t have permission to access this subtask',
        )

    # 3. 查找注释
    if not subtask.annotations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Annotation not found',
        )

    annotation = next(
        (a for a in subtask.annotations if a.get('id') == annotation_id), None)
    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Annotation not found',
        )

    # 4. 权限检查
    if not check_annotation_permission(annotation, user, 'update'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You don\'t have permission to update this annotation',
        )

    # 5. 验证更新内容
    if payload.text and not validate_annotation_update(annotation, payload.text):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid annotation update',
        )

    # 6. 更新注释
    if payload.text is not None:
        annotation['text'] = payload.text.strip()
    if payload.rect is not None:
        annotation['rect'] = payload.rect.model_dump()

    annotation['updated_at'] = datetime.datetime.now().isoformat()
    annotation['updated_by'] = str(user.id)

    await subtask.save()

    response = await SubtaskDetail.from_tortoise_orm(subtask)
    return response


@router.delete('/subtasks/{subtask_id}/annotations/{annotation_id}', response_model=SubtaskDetail)
async def delete_subtask_annotation(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
    annotation_id: str,
) -> SubtaskDetail:
    """删除子任务注释"""
    # 1. 验证子任务存在性
    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Subtask not found',
        )

    # 2. 验证项目权限
    await subtask.fetch_related('task__project')
    task = subtask.task
    project = task.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You don\'t have permission to access this subtask',
        )

    # 3. 查找注释
    if not subtask.annotations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Annotation not found',
        )

    annotation = next(
        (a for a in subtask.annotations if a.get('id') == annotation_id), None)
    if not annotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Annotation not found',
        )

    # 4. 权限检查
    if not check_annotation_permission(annotation, user, 'delete'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You don\'t have permission to delete this annotation',
        )

    # 5. 删除注释及其相关回复
    # 先删除所有回复到此注释的子注释
    subtask.annotations = [
        a for a in subtask.annotations
        if a.get('id') != annotation_id and a.get('to') != annotation_id
    ]

    await subtask.save()

    response = await SubtaskDetail.from_tortoise_orm(subtask)
    return response


@router.patch('/subtasks/{subtask_id}/status')
async def update_subtask_status(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
    status_name: SubtaskStatus = SubtaskStatus.PENDING,
) -> SubtaskDetail:
    '''Update subtask status to either accepted or denied.'''
    # TODO: Permission check
    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Subtask not found',
        )

    # Update the status
    if status_name == SubtaskStatus.ACCEPTED:
        await subtask.accept(updated_by=user)
    elif status_name == SubtaskStatus.DENIED:
        await subtask.deny(updated_by=user)
    elif status_name == SubtaskStatus.PENDING:
        await subtask.reset(updated_by=user)

    # Return updated subtask
    response = await SubtaskDetail.from_tortoise_orm(subtask)
    return response


@router.get('/tasks/{task_id}/export-pdf')
async def export_task_pdf(
    task_id: str,
    export_service: PptxExportService = Depends(PptxExportService),
) -> Response:
    """Export task and its subtasks as a PDF file."""

    # 1. Get task with subtasks
    task = await Task.get_or_none(id=task_id, is_deleted=False).prefetch_related('subtasks')
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Task not found'
        )

    # 2. Use service to export
    try:
        pdf_buffer = await export_service.export_task_to_pdf(task)
        logger.info(f"Successfully exported PDF for task {task_id}")
    except FileTooLargeError as e:
        # Handle file too large errors (413 Payload Too Large)
        logger.warning(f"File too large for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e)
        )
    except (ContentNotFoundError, InvalidPathError) as e:
        # Handle content not found errors (404 Not Found)
        logger.warning(f"Content error for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        # Handle all other export errors
        logger.error(
            f"PDF export failed for task {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to export PDF'
        )

    # 3. Prepare filename and headers for the response
    base_filename = task.name or f'task_{task_id}'
    pdf_filename = f"{base_filename}_export.pdf"

    # Create ASCII fallback for filename parameter for max compatibility
    ascii_filename = pdf_filename.encode('ascii', 'ignore').decode('ascii')

    # URL-encode the UTF-8 filename for the 'filename*' parameter
    utf8_filename = urllib.parse.quote(pdf_filename)

    headers = {
        'Content-Disposition': f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{utf8_filename}"
    }

    return Response(
        content=pdf_buffer.getvalue(),
        media_type='application/pdf',
        headers=headers
    )


@router.patch('/subtasks/{subtask_id}/content', status_code=status.HTTP_200_OK)
async def update_subtask_content(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
    s3_path: str,
):
    '''Update subtask content and maintain version history.'''
    subtask = await Subtask.get_or_none(id=subtask_id)
    if not subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Subtask not found',
        )

    # Copy current content to history if it exists
    if subtask.content:
        if subtask.history is None:
            subtask.history = []
        content_history = SubtaskContent.model_validate(subtask.content)
        subtask.history.append(content_history.model_dump())  # type: ignore

    # Update current content with new values
    current_content = dict(subtask.content) if subtask.content else {
        'title': subtask.name,
        'description': subtask.description or '',
        'task_type': subtask.task_type,
    }
    current_content.update({
        's3_path': s3_path,
        'author': str(user.id),
        'created_at': datetime.datetime.now().isoformat(),
    })
    content_model = SubtaskContent(**current_content)
    subtask.content = content_model.model_dump()  # type: ignore

    await subtask.save()
    return Response(status_code=status.HTTP_200_OK)


@router.post('/projects/{project_id}/tasks', status_code=status.HTTP_201_CREATED)
async def create_task(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    payload: TaskIn,
) -> TaskSimpleOut:
    # 1. 验证项目存在性并检查用户权限
    project = await Project.get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found',
        )

    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You don\'t have permission to create tasks in this project',
        )

    # 2. 验证状态和优先级的有效性
    task_status = await TaskStatus.get_or_none(id=payload.status_id)  # type: ignore # noqa
    task_priority = await TaskPriority.get_or_none(id=payload.priority_id)  # type: ignore # noqa

    # 3. 创建任务
    created_task = await Task.create(
        tid=payload.tid,
        name=payload.name,
        description=payload.description or "",
        s3_path="",  # 添加默认空字符串值
        project=project,
        assignee_id=payload.assignee_id,  # type: ignore
        status=task_status,
        priority=task_priority,
    )

    # 4. 返回简单的任务信息
    return TaskSimpleOut(id=created_task.id, name=created_task.name)


@router.post('/tasks/{task_id}/subtasks', status_code=status.HTTP_201_CREATED)
async def create_subtask(
    user: Annotated[User, Security(get_current_user)],
    task_id: str,
    name: str = Form(...),
    description: str = Form(None),
    task_type_str: str = Form(SubtaskType.PICTURE.value),
    file: UploadFile = File(None),
) -> Response:
    """创建子任务，可选地通过上传文件（图片或视频）"""
    # 1. 验证任务存在性并检查用户权限
    task = await Task.get_or_none(id=task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Task not found',  # Consider changing to Japanese: 'タスクが見つかりません'
        )

    project = await task.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            # Consider changing to Japanese: 'このプロジェクトでサブタスクを作成する権限がありません'
            detail='You don\'t have permission to create subtasks in this project',
        )

    # 2. 获取当前最大的oid
    last_subtask = await Subtask.filter(task_id=task_id).order_by('-oid').first()
    max_oid = last_subtask.oid if last_subtask else 0

    uploaded_s3_path: str | None = None
    compressed_s3_path: str | None = None
    # Determine the final task_type for the Subtask model
    # If a file is uploaded, determine type based on file type. Otherwise, use the provided task_type_str.
    final_model_task_type: SubtaskType

    try:
        # Attempt to convert task_type_str to SubtaskType enum
        # This allows frontend to send 'picture', 'video', 'text', etc.
        final_model_task_type = SubtaskType(task_type_str)
    except ValueError:
        # If conversion fails (e.g., invalid string), raise error
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"無効なサブタスクタイプが指定されました: {task_type_str}",
        )

        # Handle file upload based on task_type_str
    if file:
        file_contents = await file.read()
        project_id_for_path = str(project.id)
        sanitized_filename = file.filename if file.filename else "untitled"

        if final_model_task_type == SubtaskType.PICTURE:
            # Validate image file content type (including PSD and AI files)
            if not is_valid_image_or_design_file(file.filename, file.content_type):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="無効なファイルタイプです。画像、PSD、またはAIファイルをアップロードしてください。",
                )

            if len(file_contents) > MAX_IMAGE_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"画像ファイルサイズが大きすぎます（最大{MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB）。",
                )

            # S3 Upload logic for Picture
            base_filename = f"{uuid.uuid4()}_{sanitized_filename}"
            original_s3_object_name = f"images/projects/{project_id_for_path}/tasks/{task_id}/{base_filename}"

            try:
                await upload_file_to_s3(io.BytesIO(file_contents), original_s3_object_name)
                uploaded_s3_path = original_s3_object_name
            except Exception as e:
                logger.error(
                    f"S3 upload failed for original image: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="ファイルのアップロード中にエラーが発生しました。",
                )

            # Compress and upload
            try:
                with io.BytesIO(file_contents) as f:
                    compressed_file, output_format, _ = await compress_image_async(f)

                ext = 'jpg' if output_format == 'JPEG' else output_format.lower()
                base_filename_without_ext = os.path.splitext(base_filename)[0]
                compressed_filename = f"{base_filename_without_ext}_compressed.{ext}"

                compressed_s3_object_name = f"images/projects/{project_id_for_path}/tasks/{task_id}/{compressed_filename}"

                # Check if compressed_file is the same as original (no compression benefit)
                if compressed_file.closed or hasattr(compressed_file, 'name'):
                    # If original file was returned (no compression benefit), create a new BytesIO
                    compressed_file = io.BytesIO(file_contents)

                await upload_file_to_s3(compressed_file, compressed_s3_object_name)
                compressed_s3_path = compressed_s3_object_name
            except Exception as e:
                logger.error(
                    f"Image compression or upload failed: {e}", exc_info=True)

        elif final_model_task_type == SubtaskType.VIDEO:
            if not file.content_type or not file.content_type.startswith(ALLOWED_VIDEO_CONTENT_TYPE_PREFIX):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="無効なファイルタイプです。動画ファイルをアップロードしてください。",
                )

            if len(file_contents) > MAX_VIDEO_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"動画ファイルサイズが大きすぎます（最大{MAX_VIDEO_SIZE_BYTES // (1024*1024)}MB）。",
                )

            s3_object_name = f"videos/projects/{project_id_for_path}/tasks/{task_id}/{uuid.uuid4()}_{sanitized_filename}"
            try:
                await upload_file_to_s3(io.BytesIO(file_contents), s3_object_name)
                uploaded_s3_path = s3_object_name
            except Exception as e:
                logger.error(f"S3 upload failed for video: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="ファイルのアップロード中にエラーが発生しました。",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"タスクタイプ '{final_model_task_type.value}' はファイルアップロードをサポートしていません。",
            )

    # 3. 处理 content 数据
    content_data: dict[str, Any] | None = None  # Use Any for datetime

    if uploaded_s3_path:  # File was uploaded, this defines the content
        content_data = {
            'title': name,  # Use subtask name as title for the content
            'description': description or '',
            # Content's task type (picture/video)
            'task_type': final_model_task_type.value,
            's3_path': uploaded_s3_path,
            'compressed_s3_path': compressed_s3_path,
            'author': str(user.id),
            'created_at': datetime.datetime.now().isoformat(),
        }
    # This case is for when no file is uploaded but task_type is specified
    # For non-file subtask types like 'text', 'word', 'excel', etc.
    elif final_model_task_type not in [SubtaskType.PICTURE, SubtaskType.VIDEO]:
        # Example for a text subtask if task_type_str indicated 'text' and no file was uploaded:
        content_data = {
            'title': name,
            'description': description or '',
            'task_type': final_model_task_type.value,  # e.g. 'text', 'word', 'excel'
            's3_path': None,  # No S3 path for non-file subtasks
            'author': str(user.id),
            'created_at': datetime.datetime.now().isoformat(),
        }

    # 4. 创建子任务
    await Subtask.create(
        oid=max_oid + 1,
        name=name,
        # Use the determined task_type ('picture', 'video', 'text', etc.)
        task_type=final_model_task_type,  # Use the SubtaskType enum value
        description=description,
        content=content_data,  # Pass the constructed content_data
        task=task,
        slide_page_number=None,
        status=SubtaskStatus.PENDING,
    )

    return Response(status_code=status.HTTP_201_CREATED)


async def process_multiple_files_background(
    batch_id: str,
    file_info_list: List[tuple],
    project_id: str,
    user_id: str
) -> None:
    """
    并行处理多个PPT文件

    Args:
        batch_id: 批次ID，用于跟踪处理状态
        file_info_list: 文件信息列表，每项为(文件路径, 文件名)的元组
        project_id: 项目ID
        user_id: 用户ID
    """
    # Initialize counters for nonlocal access in process_single_file
    processed_count = 0
    successful_count = 0
    failed_count = 0

    # processor = UploadPPTXProcessor() # Instantiate processor once if it's stateless, or inside loop if stateful per file
    # Since UploadPPTXProcessor is now stateless regarding DB, it can be instantiated once.
    # However, the original code instantiated it inside process_single_file, let's stick to that for minimal changes beyond fixing errors.

    # The init_db and close_db calls for the processor itself were here, they are correctly removed now.
    # try:
    #     await processor.init_db()

    try:
        # Ensure initial status is set for all files if not already present by the main endpoint
        # This part seems to be handled by the main `upload_pptx` endpoint now.
        # status_dict = file_processing_status[batch_id]
        # for _, filename in file_info_list:
        #     if filename not in status_dict['files_status']:
        #         status_dict['files_status'][filename] = {
        #             'status': 'pending',
        #             'message': '等待处理',
        #             'task_id': None,
        #             'subtask_count': 0,
        #             'start_time': datetime.datetime.now().isoformat(),
        #             'end_time': None
        #         }

        async def process_single_file(file_path: str, filename: str):
            """处理单个文件的异步函数"""
            nonlocal processed_count, successful_count, failed_count

            # Each file processing attempt should be independent in terms of processor instance
            # if the processor had any per-file state, though UploadPPTXProcessor is now mostly stateless.
            single_file_processor = UploadPPTXProcessor()

            try:
                # This inner try is for the core processing logic that might raise an exception
                with open(file_path, 'rb') as f_obj:
                    result = await single_file_processor.process_uploaded_pptx(f_obj, filename, project_id, user_id)

                if result['success']:
                    successful_count += 1
                    file_processing_status[batch_id]['files_status'][filename] = {
                        'status': 'success',
                        'task_id': result.get('task_id'),
                        'message': f"タスク {result.get('task_id')} が正常に作成されました"
                    }
                    logger.info(
                        f"Successfully processed {filename}, task ID: {result.get('task_id')}")
                else:
                    failed_count += 1
                    file_processing_status[batch_id]['files_status'][filename] = {
                        'status': 'failed',
                        'message': result.get('message', '不明なエラー')
                    }
                    logger.error(
                        f"Failed to process {filename}: {result.get('message')}")

            except Exception as e_process:  # Catch exceptions from process_uploaded_pptx or file open
                failed_count += 1
                error_message = f"ファイル「{filename}」の処理中に例外が発生しました: {str(e_process)}"
                file_processing_status[batch_id]['files_status'][filename] = {
                    'status': 'failed',
                    'message': error_message
                }
                logger.error(
                    f"Exception during processing of {filename}: {str(e_process)}", exc_info=True)

            finally:
                # This block executes whether the try block succeeded or failed
                if os.path.exists(file_path):
                    os.unlink(file_path)  # Clean up the temporary file
                processed_count += 1
                # Update the main status dictionary after each file is processed (or attempted)
                file_processing_status[batch_id]['processed_files'] = processed_count
                file_processing_status[batch_id]['successful_files'] = successful_count
                file_processing_status[batch_id]['failed_files'] = failed_count
                if filename in file_processing_status[batch_id]['files_status']:
                    file_processing_status[batch_id]['files_status'][filename]['end_time'] = datetime.datetime.now(
                    ).isoformat()

        # Create and run all file processing tasks
        # The original code ran them serially: `for task_to_run in tasks: await task_to_run`
        # Let's maintain that serial execution as it was likely intentional (e.g., to manage DB load)
        for f_path, f_name in file_info_list:
            await process_single_file(f_path, f_name)

    finally:
        # This finally block is for the outer try that might have wrapped processor.init_db()
        # Since init_db/close_db for a shared processor are gone, this block might not be strictly necessary
        # unless there were other setup steps for the batch.
        # For now, let it be, ensuring it has a `pass` if empty or relevant cleanup.
        # The original code had `await processor.close_db()` here, which is removed.
        pass  # Or any other batch-level cleanup if needed

    # 保留状态信息24小时（可根据需要调整）
    asyncio.create_task(cleanup_status(batch_id, 86400))


async def cleanup_status(batch_id: str, delay_seconds: int):
    """在指定延迟后清理状态信息"""
    await asyncio.sleep(delay_seconds)
    if batch_id in file_processing_status:
        del file_processing_status[batch_id]


@router.post('/tasks/upload-pptx', status_code=status.HTTP_201_CREATED)
async def upload_pptx(
    background_tasks: BackgroundTasks,
    project_id: str = Form(...),
    files: List[UploadFile] = File(...),  # 修改为接收多个文件
    current_user: User = Depends(get_current_user)
):
    """
    上传并处理多个PPT文件，创建任务和子任务

    Args:
        background_tasks (BackgroundTasks): FastAPI后台任务
        project_id (str): 项目ID
        files (List[UploadFile]): 上传的多个PPT文件
        current_user (User): 当前用户

    Returns:
        JSONResponse: 包含批次ID和处理状态的响应
    """
    # 验证项目存在性并检查用户权限
    project = await Project.get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='项目不存在'
        )

    if not await project.is_user_can_access(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='您没有权限在此项目中创建任务'
        )

    # 检查是否有文件上传
    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='请选择至少一个文件上传'
        )

    # 生成批次ID用于跟踪处理状态
    batch_id = str(uuid.uuid4())
    temp_file_paths = []  # 用于存储临时文件路径
    valid_files = []  # 存储有效的文件名

    try:
        # 处理每个文件
        for file in files:
            # 验证文件类型
            if not file.filename:
                continue  # 跳过没有文件名的文件

            # 检查文件扩展名
            filename = file.filename.lower()
            if not (filename.endswith('.ppt') or filename.endswith('.pptx')):
                continue  # 跳过非PPT文件

            # 创建临时文件保存上传的内容
            safe_filename = file.filename.replace('/', '_').replace('\\', '_')
            temp_file_path = f"/tmp/{uuid.uuid4()}_{safe_filename}"
            with open(temp_file_path, 'wb') as f:
                content = await file.read()
                # 检查文件大小（500MB限制）
                if len(content) > 500 * 1024 * 1024:  # 500MB in bytes
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f'ファイル {filename} のサイズが500MBの制限を超えています'
                    )
                f.write(content)

            temp_file_paths.append((temp_file_path, filename))
            valid_files.append(filename)

        # 检查是否有有效文件被处理
        if not temp_file_paths:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='没有有效的PPT/PPTX文件'
            )

        # 初始化处理状态
        file_processing_status[batch_id] = {
            'total_files': len(temp_file_paths),
            'processed_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'files_status': {}
        }

        # 添加后台任务处理所有文件
        background_tasks.add_task(
            process_multiple_files_background,
            batch_id,
            temp_file_paths,
            project_id,
            str(current_user.id)
        )

        # 返回批次ID和初始状态
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                'batch_id': batch_id,
                'message': f'已开始处理 {len(temp_file_paths)} 个文件',
                'files': valid_files
            }
        )

    except Exception as e:
        # 清理所有临时文件
        for temp_file_path, _ in temp_file_paths:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'处理文件时出错: {str(e)}'
        )


@router.get('/tasks/upload-status/{batch_id}')
async def get_upload_status(
    batch_id: str,
    user: Annotated[User, Security(get_current_user)]
) -> FileProcessingStatus:
    """
    获取文件上传和处理状态

    Args:
        batch_id: 批次ID
        user: 当前用户

    Returns:
        FileProcessingStatus: 处理状态信息
    """
    if batch_id not in file_processing_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='找不到指定批次的处理状态'
        )

    status_data = file_processing_status[batch_id]
    return FileProcessingStatus(
        batch_id=batch_id,
        total_files=status_data['total_files'],
        processed_files=status_data['processed_files'],
        successful_files=status_data['successful_files'],
        failed_files=status_data['failed_files'],
        files_status=status_data['files_status']
    )


async def process_pptx_background(
    file_path: str,
    filename: str,
    project_id: str,
    user_id: str
) -> None:
    """
    后台处理PPT文件的任务

    Args:
        file_path (str): 临时保存的文件路径
        filename (str): 文件名
        project_id (str): 项目ID
        user_id (str): 用户ID
    """
    # 直接使用UploadPPTXProcessor处理文件
    processor = UploadPPTXProcessor()

    # Remove init_db and close_db calls as UploadPPTXProcessor no longer manages DB connections
    # await processor.init_db()

    try:
        with open(file_path, 'rb') as file:
            result = await processor.process_uploaded_pptx(file, filename, project_id, user_id)

        if not result['success']:
            logger.error(f"处理PPT文件失败: {result.get('message', '未知错误')}")
        else:
            logger.info(
                f"成功处理PPT文件，创建了任务ID: {result['task_id']}，包含 {result['subtask_count']} 个子任务")

    except Exception as e:
        # Added exc_info for more details
        logger.error(
            f"Error processing uploaded file {filename}: {str(e)}", exc_info=True)
    finally:
        # Remove init_db and close_db calls
        # await processor.close_db()

        # 处理完成后删除临时文件
        if os.path.exists(file_path):
            os.unlink(file_path)


async def download_from_presigned_url(presigned_url: str, filename: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(presigned_url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        # 将磁盘写入放到线程池，避免阻塞事件循环

        def _write_file(path: str, content: bytes):
            with open(path, 'wb') as f:
                f.write(content)

        import asyncio
        await asyncio.to_thread(_write_file, filename, response.content)


@router.get('/tasks/{task_id}/export-pptx')
async def export_task_pptx(
    task_id: str,
) -> Response:
    '''Export task as PPTX - works for both PPTX and image-based tasks'''

    # 1. Get task with subtasks
    task = await Task.get_or_none(id=task_id, is_deleted=False).prefetch_related('subtasks')
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Task not found'
        )

    # 2. Check if task has exportable content
    has_pptx_source = task.s3_path and task.s3_path.strip()
    has_image_subtasks = any(
        (s.task_type == SubtaskType.PICTURE or s.task_type ==
         SubtaskType.VIDEO) and s.content
        for s in task.subtasks
    )

    if not has_pptx_source and not has_image_subtasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='No exportable content found for this task'
        )

    # 3. Use service to export
    export_service = PptxExportService()
    try:
        pptx_buffer = await export_service.export_task_to_pptx(task)
        logger.info(f"Successfully exported PPTX for task {task_id}")
    except FileTooLargeError as e:
        # Handle file too large errors (413 Payload Too Large)
        logger.warning(f"File too large for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e)
        )
    except (ContentNotFoundError, InvalidPathError) as e:
        # Handle content not found errors (404 Not Found)
        logger.warning(f"Content error for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        # Handle all other export errors
        logger.error(f"Export failed for task {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to export PPTX'
        )

    # 4. Prepare filename with proper encoding
    base_filename = task.name or f'task_{task_id}'
    annotated_filename = f"{base_filename}_export.pptx"

    # Create ASCII fallback for filename parameter
    ascii_filename = annotated_filename.encode(
        'ascii', 'ignore').decode('ascii')
    if not ascii_filename:
        ascii_filename = f"task_{task_id}_export.pptx"

    # URL-encode the original filename for filename* parameter (RFC 6266)
    utf8_filename = urllib.parse.quote(annotated_filename, safe='')

    # Construct Content-Disposition header with both ASCII and UTF-8 versions
    content_disposition = (
        f'attachment; filename="{ascii_filename}"; '
        f'filename*=UTF-8\'\'{utf8_filename}'
    )

    # Return full bytes (non-streaming) to include Content-Length and avoid chunked encoding issues
    return Response(
        content=pptx_buffer.getvalue(),
        media_type='application/vnd.openxmlformats-officedocument.presentationml.presentation',
        headers={'Content-Disposition': content_disposition}
    )


@router.post('/tasks/upload-document', status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    project_id: str = Form(...),
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    上传并保存docx, txt, excel等文件，为每个文件创建一个任务和对应的子任务。

    Args:
        project_id (str): 项目ID
        files (List[UploadFile]): 上传的文件列表
        background_tasks (BackgroundTasks): FastAPI后台任务
        current_user (User): 当前用户

    Returns:
        JSONResponse: 包含处理结果的响应
    """
    # 验证项目存在性并检查用户权限
    project = await Project.get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='プロジェクトが見つかりません'
        )

    if not await project.is_user_can_access(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='このプロジェクトでタスクを作成する権限がありません'
        )

    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='ファイルを選択してください'
        )

    supported_extensions_map = {
        '.txt': SubtaskType.TEXT,
        '.docx': SubtaskType.WORD,
        '.doc': SubtaskType.WORD,
        '.xlsx': SubtaskType.EXCEL,
        '.xls': SubtaskType.EXCEL,
        '.csv': SubtaskType.EXCEL,  # CSV 也视为表格类
    }

    results = []
    subtask_oid_counter = {}  # 用于跟踪每个任务下的subtask oid

    for file in files:
        if not file.filename:
            # 文件名为空，跳过
            results.append({
                'filename': 'N/A',
                'success': False,
                'message': 'ファイル名がありません',
                'task_id': None,
                'subtask_id': None
            })
            continue

        original_filename = file.filename
        file_ext = os.path.splitext(original_filename)[1].lower()

        if file_ext not in supported_extensions_map:
            results.append({
                'filename': original_filename,
                'success': False,
                'message': 'サポートされていないファイル形式です',
                'task_id': None,
                'subtask_id': None
            })
            continue

        try:
            file_content = await file.read()
            if len(file_content) > 200 * 1024 * 1024:  # 200MB
                results.append({
                    'filename': original_filename,
                    'success': False,
                    'message': 'ファイルサイズが大きすぎます（最大200MB）',
                    'task_id': None,
                    'subtask_id': None
                })
                continue

            # 为文件生成唯一的S3路径
            unique_file_id = str(uuid.uuid4())
            # 保持原始文件名以方便识别，但添加唯一ID确保路径唯一性
            s3_object_name = f"documents/{project_id}/{unique_file_id}/{original_filename}"

            # 上传文件到S3
            file_data = io.BytesIO(file_content)
            await upload_file_to_s3(file_data, s3_object_name)

            # --- 创建任务 ---
            timestamp_tid = datetime.datetime.now().strftime("T%Y%m%d%H%M%S%f")  # 毫秒精度以增加唯一性
            task_name = os.path.splitext(original_filename)[0]
            task_description = f"ドキュメント「{original_filename}」のタスク"

            default_status = await TaskStatus.filter(name='todo').first()
            if not default_status:
                default_status = await TaskStatus.all().first()

            default_priority = await TaskPriority.filter(name='low').first()
            if not default_priority:
                default_priority = await TaskPriority.all().first()

            if not default_status or not default_priority:
                results.append({
                    'filename': original_filename,
                    'success': False,
                    'message': 'プロジェクトの初期ステータスまたは優先度設定が見つかりません',
                    'task_id': None,
                    'subtask_id': None
                })
                continue

            # Task的s3_path在此场景下不直接存储文档路径，因为文档关联到Subtask
            created_task = await Task.create(
                tid=timestamp_tid,
                name=task_name,
                description=task_description,
                s3_path="",  # Task 级别的 s3_path 置空或用于其他目的
                project=project,
                assignee=current_user,
                status=default_status,
                priority=default_priority
            )

            # --- 创建子任务 ---
            subtask_type_enum = supported_extensions_map[file_ext]

            # 获取当前任务下的最大oid
            if str(created_task.id) not in subtask_oid_counter:
                last_subtask_for_task = await Subtask.filter(task_id=created_task.id).order_by('-oid').first()
                subtask_oid_counter[str(
                    created_task.id)] = last_subtask_for_task.oid if last_subtask_for_task else 0

            current_max_oid = subtask_oid_counter[str(created_task.id)]
            new_oid = current_max_oid + 1
            subtask_oid_counter[str(created_task.id)] = new_oid

            subtask_content_data = {
                'title': original_filename,  # 子任务内容标题使用文件名
                'description': f"ファイル「{original_filename}」のコンテンツ",
                # 'text_file', 'word_document', etc.
                'task_type': subtask_type_enum.value,
                's3_path': s3_object_name,
                'author': str(current_user.id),
                'created_at': datetime.datetime.now().isoformat(),
            }

            created_subtask = await Subtask.create(
                oid=new_oid,
                name=original_filename,  # 子任务名也用文件名
                task_type=subtask_type_enum,  # 使用 SubtaskType 枚举成员
                description=f"ファイル「{original_filename}」のレビュー",
                content=subtask_content_data,
                task=created_task,
                status=SubtaskStatus.PENDING,  # 默认状态
            )

            # Add background task for AI classification
            background_tasks.add_task(
                process_subtask_ai_classification, created_subtask.id)

            results.append({
                'filename': original_filename,
                'success': True,
                'message': 'ファイルが正常に処理され、タスクとサブタスクが作成されました',
                'task_id': str(created_task.id),
                'subtask_id': str(created_subtask.id)
            })

        except Exception as e:
            logger.error(
                f"ファイル「{original_filename}」の処理中にエラーが発生しました: {str(e)}", exc_info=True)
            results.append({
                'filename': original_filename,
                'success': False,
                'message': f'処理エラー: {str(e)}',
                'task_id': None,
                'subtask_id': None
            })

    successful_uploads = len([r for r in results if r["success"]])
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            'message': f'{successful_uploads}個のファイルが正常に処理されました。',
            'results': results
        }
    )


@router.post(
    "/projects/{project_id}/tasks/create-from-image",
    response_model=TaskOut,
    status_code=status.HTTP_201_CREATED,
    summary="画像からタスクを作成",
    description="画像をアップロードして新しいタスクと最初のサブタスクを作成します。",
)
async def create_task_from_uploaded_image(
    project_id: uuid.UUID,
    image_file: UploadFile = File(..., description="アップロードする画像ファイル"),
    tag_ids: Optional[List[uuid.UUID]] = Form(None),
    current_user: User = Depends(get_current_user),
):
    """
    Uploads an image and creates a new task with this image as its first subtask.
    - **project_id**: The ID of the project to which the task will belong.
    - **image_file**: The image file to upload.
    """
    if not is_valid_image_or_design_file(image_file.filename, image_file.content_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="無効なファイルタイプです。画像、PSD、またはAIファイルをアップロードしてください。",
        )

    try:
        created_task = await create_task_from_image(
            project_id=project_id,
            image_file=image_file,
            current_user=current_user,
            tag_ids=tag_ids,
        )
        return created_task
    except HTTPException as e:
        # Re-raise known HTTPExceptions (like permission denied, not found)
        raise e
    except Exception as e:
        # Catch-all for other unexpected errors during task creation process
        # Consider logging the error e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"タスクの作成中に予期せぬエラーが発生しました: {str(e)}",
        )


@router.post(
    "/projects/{project_id}/tasks/create-from-video",
    response_model=TaskOut,
    status_code=status.HTTP_201_CREATED,
    summary="動画からタスクを作成",
    description="動画をアップロードして新しいタスクと最初のサブタスクを作成します。",
)
async def create_task_from_uploaded_video(
    project_id: uuid.UUID,
    video_file: UploadFile = File(..., description="アップロードする動画ファイル"),
    tag_ids: Optional[List[uuid.UUID]] = Form(None),
    current_user: User = Depends(get_current_user),
):
    """
    Uploads a video and creates a new task with this video as its first subtask.
    - **project_id**: The ID of the project to which the task will belong.
    - **video_file**: The video file to upload.
    """
    if not video_file.content_type or not video_file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="無効なファイルタイプです。動画ファイルをアップロードしてください。",
        )

    try:
        created_task = await create_task_from_video(
            project_id=project_id,
            video_file=video_file,
            current_user=current_user,
            tag_ids=tag_ids,
        )
        return created_task
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"タスクの作成中に予期せぬエラーが発生しました: {str(e)}",
        )


@router.get('/projects/{project_id}/tasks/all')
async def list_all_project_tasks(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
) -> List[TaskSimpleOut]:
    project = await Project.get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    if not await project.is_user_can_access(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="You don't have permission to access tasks in this project")

    tasks = await Task.filter(project_id=project_id, is_deleted=False).order_by('created_at').all()
    return [TaskSimpleOut(id=task.id, name=task.name) for task in tasks]


class SubtaskCopyPayload(BaseModel):
    target_task_id: str


@router.post('/subtasks/{subtask_id}/copy', status_code=status.HTTP_201_CREATED)
async def copy_subtask(
    user: Annotated[User, Security(get_current_user)],
    subtask_id: str,
    payload: SubtaskCopyPayload,
) -> SubtaskOut:
    """
    Copies a subtask to another task within the same project.
    """
    # 1. Get source subtask
    source_subtask = await Subtask.get_or_none(id=subtask_id).prefetch_related('task__project')
    if not source_subtask:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail='コピー元のサブタスクが見つかりません')

    # 2. Check user permission for source subtask
    source_project = source_subtask.task.project
    if not await source_project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail='この操作を行う権限がありません')

    # 3. Get target task
    target_task = await Task.get_or_none(id=payload.target_task_id).prefetch_related('project')
    if not target_task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail='コピー先のタスクが見つかりません')

    # 4. Check user permission for target task
    target_project = target_task.project
    if not await target_project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail='コピー先のタスクにアクセスする権限がありません')

    # 5. Check if source and target projects are the same.
    if source_project.id != target_project.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail='サブタスクは同じプロジェクト内のタスクにのみコピーできます')

    # 6. Get new oid for the copied subtask
    last_subtask = await Subtask.filter(task_id=target_task.id).order_by('-oid').first()
    new_oid = (last_subtask.oid + 1) if last_subtask else 1

    # 7. Create new subtask (copying data)
    # Add "(copy)" to the name, or handle if it already exists
    new_name = f"{source_subtask.name} (コピー)"
    # A more robust solution might check for existing copies and increment a number

    new_subtask_data = {
        'oid': new_oid,
        'name': new_name,
        'description': source_subtask.description,
        'task_type': source_subtask.task_type,
        'content': source_subtask.content,
        'annotations': source_subtask.annotations,
        'history': [],  # Do not copy history
        'status': SubtaskStatus.PENDING,  # Reset status
        'slide_page_number': source_subtask.slide_page_number,
        'task': target_task,
    }

    new_subtask = await Subtask.create(**new_subtask_data)  # type: ignore

    # 8. Return the new subtask
    response = await SubtaskOut.from_tortoise_orm(new_subtask)
    return response


@router.get('/tasks/{task_id}/suggested-review-sets', response_model=List[ReviewSetOut])
async def suggest_review_sets_for_task(
    task_id: uuid.UUID,
    current_user: User = Security(get_current_user),
):
    """
    For a given task, suggest review sets based on its tags and characters in subtasks.
    """
    # TODO: Permission check to ensure user can access the task
    return await get_suggested_review_sets(task_id)


@router.get('/projects/{project_id}/tasks/navigation',
            response_model=TaskNavigationResponse)
async def list_project_tasks_for_navigation(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    sort_by: Literal["tid", "name", "created_at"] = "created_at",
    order: Literal["asc", "desc"] = "desc"
) -> TaskNavigationResponse:
    """
    获取项目下用于导航的任务列表（轻量版）

    Args:
        project_id: 项目ID
        sort_by: 排序字段 (tid, name, created_at)
        order: 排序方向 (asc, desc)

    Returns:
        TaskNavigationResponse: 包含所有任务的导航数据
    """
    project = await Project.get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="プロジェクトが見つかりません")

    if not await project.is_user_can_access(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="このプロジェクトのタスクにアクセスする権限がありません")

    tasks_queryset = Task.filter(project_id=project_id, is_deleted=False)

    # 构建排序表达式
    order_by_field = f"-{sort_by}" if order == "desc" else sort_by

    total = await tasks_queryset.count()
    tasks = await tasks_queryset.order_by(order_by_field).all()

    items = [TaskNavigationItem.model_validate(task) for task in tasks]

    return TaskNavigationResponse(total=total, items=items)


@router.get('/tasks/{task_id}/thumbnails', response_model=TaskThumbnailsResponse)
async def get_task_thumbnails(
    user: Annotated[User, Security(get_current_user)],
    task_id: str,
    limit: int = 3
) -> TaskThumbnailsResponse:
    """
    获取任务的缩略图信息（前几个图片子任务）

    Args:
        task_id: 任务ID
        limit: 返回的缩略图数量，默认3个
    """
    # 验证任务存在性并检查权限
    task = await Task.get_or_none(id=task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='タスクが見つかりません'
        )

    project = await task.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='このタスクにアクセスする権限がありません'
        )

    # 获取前N个图片类型的子任务
    image_subtasks = await Subtask.filter(
        task_id=task_id,
        task_type=SubtaskType.PICTURE
    ).order_by('oid').limit(limit)

    # 构建缩略图响应
    thumbnails = []
    for subtask in image_subtasks:
        if subtask.content:
            content = subtask.content
            thumbnail = TaskThumbnail(
                subtask_id=subtask.id,
                subtask_name=subtask.name,
                original_s3_path=content.get('s3_path', '') if isinstance(
                    content, dict) else getattr(content, 's3_path', ''),
                compressed_s3_path=content.get('compressed_s3_path') if isinstance(
                    content, dict) else getattr(content, 'compressed_s3_path', None)
            )
            thumbnails.append(thumbnail)

    return TaskThumbnailsResponse(
        task_id=task.id,
        thumbnails=thumbnails
    )
