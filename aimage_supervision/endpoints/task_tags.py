from uuid import UUID
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, Security

from aimage_supervision.models import User
from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.services import task_tag_service
from aimage_supervision.schemas import TaskTagCreate, TaskTagUpdate, TaskTagOut

router = APIRouter(
    prefix="/task-tags",
    tags=["Task Tags"],
)

@router.post("/", response_model=TaskTagOut, status_code=status.HTTP_201_CREATED)
async def create_task_tag(
    tag_create: TaskTagCreate,
    current_user: User = Security(get_current_user),
):
    try:
        tag = await task_tag_service.create_task_tag(tag_create)
        return await task_tag_service.get_task_tag(tag.id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.get("/", response_model=List[TaskTagOut])
async def list_task_tags(
    project_id: UUID,
    current_user: User = Security(get_current_user),
):
    return await task_tag_service.list_task_tags(project_id)

@router.get("/{tag_id}", response_model=TaskTagOut)
async def get_task_tag(
    tag_id: UUID,
    current_user: User = Security(get_current_user),
):
    tag = await task_tag_service.get_task_tag(tag_id)
    if not tag:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task Tag not found")
    return tag

@router.put("/{tag_id}", response_model=TaskTagOut)
async def update_task_tag(
    tag_id: UUID,
    tag_update: TaskTagUpdate,
    current_user: User = Security(get_current_user),
):
    try:
        return await task_tag_service.update_task_tag(tag_id, tag_update)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.delete("/{tag_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task_tag(
    tag_id: UUID,
    current_user: User = Security(get_current_user),
):
    deleted = await task_tag_service.delete_task_tag(tag_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task Tag not found")
    return 