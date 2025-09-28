from typing import List, Optional
from uuid import UUID

from tortoise.exceptions import DoesNotExist, IntegrityError

from aimage_supervision.models import (Character, Project,
                                       ReviewPointDefinition,
                                       ReviewPointDefinitionVersion, ReviewSet,
                                       TaskTag)
from aimage_supervision.schemas import (CharacterForReviewSet, ReviewSetCreate,
                                        ReviewSetOut, ReviewSetUpdate,
                                        RPDForReviewSet, TaskTagOut)


async def _create_rpd_for_review_set(rpd: ReviewPointDefinition) -> RPDForReviewSet:
    """
    Helper function to create RPDForReviewSet with current version title.
    """
    # Get the current active version
    current_version = await ReviewPointDefinitionVersion.filter(
        review_point_definition=rpd,
        is_active_version=True
    ).first()

    return RPDForReviewSet(
        id=rpd.id,
        key=rpd.key,
        current_version_title=current_version.title if current_version else None
    )


async def create_review_set(rs_create: ReviewSetCreate) -> ReviewSetOut:
    """
    Create a new review set and associate it with RPDs, characters, and task tags.
    """
    try:
        project = await Project.get(id=rs_create.project_id)
    except DoesNotExist:
        raise ValueError(f"Project with id {rs_create.project_id} not found.")

    try:
        review_set = await ReviewSet.create(
            name=rs_create.name,
            description=rs_create.description,
            project=project
        )

        if rs_create.rpd_ids:
            rpds = await ReviewPointDefinition.filter(id__in=rs_create.rpd_ids, project=project, is_deleted=False)
            await review_set.rpds.add(*rpds)

        if rs_create.character_ids:
            characters = await Character.filter(id__in=rs_create.character_ids, project=project)
            await review_set.characters.add(*characters)

        if rs_create.task_tag_ids:
            task_tags = await TaskTag.filter(id__in=rs_create.task_tag_ids, project=project)
            await review_set.task_tags.add(*task_tags)

        result = await get_review_set(review_set.id)
        if result is None:
            raise RuntimeError(
                f"Failed to retrieve newly created review set {review_set.id}")
        return result

    except IntegrityError:
        raise ValueError(
            f"A review set with name '{rs_create.name}' already exists in this project.")


async def get_review_set(review_set_id: UUID) -> Optional[ReviewSetOut]:
    """
    Get a single review set by ID, including its related entities.
    """
    review_set = await ReviewSet.get_or_none(id=review_set_id).prefetch_related(
        "project", "characters", "task_tags"
    )
    if not review_set:
        return None

    # 手动查询关联的未删除RPD
    active_rpds = await ReviewPointDefinition.filter(
        review_sets=review_set,
        is_deleted=False
    ).all()

    # Create RPDForReviewSet with current version titles
    rpd_for_review_set_list = []
    for rpd in active_rpds:
        rpd_for_review_set_list.append(await _create_rpd_for_review_set(rpd))

    # Manually construct the output schema to handle nested models
    return ReviewSetOut(
        id=review_set.id,
        name=review_set.name,
        description=review_set.description,
        project_id=review_set.project.id,
        created_at=review_set.created_at,
        updated_at=review_set.updated_at,
        rpds=rpd_for_review_set_list,
        characters=[CharacterForReviewSet.from_orm(
            char) for char in review_set.characters],
        task_tags=[TaskTagOut.from_orm(tag) for tag in review_set.task_tags],
    )


async def list_review_sets(project_id: UUID) -> List[ReviewSetOut]:
    """
    List all review sets for a given project.
    """
    review_sets = await ReviewSet.filter(project_id=project_id).prefetch_related(
        "characters", "task_tags"
    ).all()

    results = []
    for review_set in review_sets:
        # 手动查询关联的未删除RPD
        active_rpds = await ReviewPointDefinition.filter(
            review_sets=review_set,
            is_deleted=False
        ).all()

        # Create RPDForReviewSet with current version titles
        rpd_for_review_set_list = []
        for rpd in active_rpds:
            rpd_for_review_set_list.append(await _create_rpd_for_review_set(rpd))

        results.append(ReviewSetOut(
            id=review_set.id,
            name=review_set.name,
            description=review_set.description,
            project_id=project_id,
            created_at=review_set.created_at,
            updated_at=review_set.updated_at,
            rpds=rpd_for_review_set_list,
            characters=[CharacterForReviewSet.from_orm(
                char) for char in review_set.characters],
            task_tags=[TaskTagOut.from_orm(tag)
                       for tag in review_set.task_tags],
        ))
    return results


async def update_review_set(review_set_id: UUID, rs_update: ReviewSetUpdate) -> ReviewSetOut:
    """
    Update a review set, including its associations.
    """
    try:
        review_set = await ReviewSet.get(id=review_set_id).prefetch_related("project")
    except DoesNotExist:
        raise ValueError(f"ReviewSet with id {review_set_id} not found.")

    update_data = rs_update.model_dump(exclude_unset=True)

    # Update simple fields
    if 'name' in update_data:
        review_set.name = update_data['name']
    if 'description' in update_data:
        review_set.description = update_data['description']

    await review_set.save()

    project_id = review_set.project.id

    # Update many-to-many fields
    if rs_update.rpd_ids is not None:
        rpds = await ReviewPointDefinition.filter(id__in=rs_update.rpd_ids, project_id=project_id, is_deleted=False)
        await review_set.rpds.clear()
        await review_set.rpds.add(*rpds)

    if rs_update.character_ids is not None:
        characters = await Character.filter(id__in=rs_update.character_ids, project_id=project_id)
        await review_set.characters.clear()
        await review_set.characters.add(*characters)

    if rs_update.task_tag_ids is not None:
        task_tags = await TaskTag.filter(id__in=rs_update.task_tag_ids, project_id=project_id)
        await review_set.task_tags.clear()
        await review_set.task_tags.add(*task_tags)

    updated_review_set = await get_review_set(review_set_id)
    if not updated_review_set:
        # Should not happen
        raise ValueError("Failed to retrieve updated review set.")
    return updated_review_set


async def delete_review_set(review_set_id: UUID) -> bool:
    """
    Delete a review set.
    """
    deleted_count = await ReviewSet.filter(id=review_set_id).delete()
    if not deleted_count:
        return False
    return True
