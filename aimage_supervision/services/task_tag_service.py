from typing import List
from uuid import UUID

from tortoise.exceptions import DoesNotExist, IntegrityError

from aimage_supervision.models import Project, TaskTag
from aimage_supervision.schemas import TaskTagCreate, TaskTagOut, TaskTagUpdate


async def create_task_tag(tag_create: TaskTagCreate) -> TaskTag:
    """
    Create a new task tag.
    """
    try:
        project = await Project.get(id=tag_create.project_id)
        tag = await TaskTag.create(
            name=tag_create.name,
            project=project
        )
        return tag
    except IntegrityError:
        raise ValueError(
            f"A tag with name '{tag_create.name}' already exists in this project.")
    except DoesNotExist:
        raise ValueError(f"Project with id {tag_create.project_id} not found.")


async def get_task_tag(tag_id: UUID) -> TaskTagOut | None:
    """
    Get a single task tag by ID.
    """
    tag = await TaskTag.get_or_none(id=tag_id)
    if not tag:
        return None
    return TaskTagOut.model_validate(tag)


async def list_task_tags(project_id: UUID) -> List[TaskTagOut]:
    """
    List all task tags for a given project.
    """
    tags = await TaskTag.filter(project_id=project_id).all()
    # Manually construct the output to avoid circular dependency issues with Pydantic v1
    return [TaskTagOut.model_validate(t) for t in tags]


async def update_task_tag(tag_id: UUID, tag_update: TaskTagUpdate) -> TaskTagOut:
    """
    Update a task tag.
    """
    try:
        tag = await TaskTag.get(id=tag_id)
        # Check for uniqueness before updating
        if await TaskTag.filter(name=tag_update.name, project=tag.project).exclude(id=tag_id).exists():
            raise IntegrityError
        tag.name = tag_update.name
        await tag.save()
        return TaskTagOut.model_validate(tag)
    except DoesNotExist:
        raise ValueError(f"TaskTag with id {tag_id} not found.")
    except IntegrityError:
        raise ValueError(
            f"A tag with name '{tag_update.name}' already exists in this project.")


async def delete_task_tag(tag_id: UUID) -> bool:
    """
    Delete a task tag.
    """
    deleted_count = await TaskTag.filter(id=tag_id).delete()
    if not deleted_count:
        return False
    return True
