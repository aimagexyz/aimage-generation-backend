from typing import List, Set
from uuid import UUID

from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from tortoise.exceptions import DoesNotExist

from ..models import ReviewSet, Subtask, Task


class ReviewSetRecommendation(BaseModel):
    """Review Set recommendation based on character and tags"""
    review_set_id: UUID = Field(...,
                                description="The ID of the recommended Review Set")
    review_set_name: str = Field(...,
                                 description="The name of the recommended Review Set")
    score: float = Field(...,
                         description="Recommendation score", ge=0.0, le=10.0)
    tag_matches: List[str] = Field(default=[], description="Matched tag names")
    character_matches: List[str] = Field(
        default=[], description="Matched character names")
    tag_score: float = Field(...,
                             description="Score from tag matching", ge=0.0, le=6.0)
    character_score: float = Field(
        ..., description="Score from character matching", ge=0.0, le=4.0)


class TaskReviewSetRecommendationsResponse(BaseModel):
    """Response body for task-based Review Set recommendations"""
    task_id: UUID = Field(..., description="The ID of the task")
    task_name: str = Field(..., description="The name of the task")
    recommendations: List[ReviewSetRecommendation] = Field(
        ..., description="List of recommended Review Sets")
    total_recommendations: int = Field(...,
                                       description="Total number of recommendations")


async def get_task_review_set_recommendations(
    task_id: UUID,
    project_id: UUID,
    min_score: float = 0.0
) -> TaskReviewSetRecommendationsResponse:
    """
    Get Review Set recommendations for a task based on characters and tags

    Scoring algorithm:
    - Tag matching: 60% weight (max 6.0 points, 1.0 per tag)
    - Character matching: 40% weight (max 4.0 points, ~0.67 per character)
    """

    # 1. Get task with tags
    try:
        task = await Task.get(id=task_id, project_id=project_id).prefetch_related("tags")
    except DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found in project {project_id}."
        )

    # 2. Get character IDs from subtasks
    subtasks = await Subtask.filter(task_id=task_id)
    character_ids = set()
    for subtask in subtasks:
        if subtask.character_ids:
            for char_id in subtask.character_ids:
                character_ids.add(UUID(char_id))

    # 3. Get task tag IDs
    task_tag_ids = {tag.id for tag in task.tags}

    # 4. Get all review sets in the project with their relationships
    review_sets = await ReviewSet.filter(
        project_id=project_id
    ).prefetch_related("characters", "task_tags")

    # 5. Calculate recommendations with scoring
    recommendations = []

    for review_set in review_sets:
        # Get review set's tag and character IDs
        review_set_tag_ids = {tag.id for tag in review_set.task_tags}
        review_set_character_ids = {char.id for char in review_set.characters}

        # Calculate tag matches
        matched_tag_ids = task_tag_ids & review_set_tag_ids
        matched_tag_names = [
            tag.name for tag in review_set.task_tags if tag.id in matched_tag_ids]
        tag_score = min(len(matched_tag_ids) * 1.0, 6.0)

        # Calculate character matches
        matched_character_ids = character_ids & review_set_character_ids
        matched_character_names = [
            char.name for char in review_set.characters if char.id in matched_character_ids]
        character_score = min(len(matched_character_ids)
                              * (4.0/6.0), 4.0)  # ~0.67 per character

        # Calculate total score
        total_score = tag_score + character_score

        # Skip if below minimum score or no matches at all
        if total_score < min_score or (not matched_tag_ids and not matched_character_ids):
            continue

        recommendations.append(ReviewSetRecommendation(
            review_set_id=review_set.id,
            review_set_name=review_set.name,
            score=round(total_score, 1),
            tag_matches=matched_tag_names,
            character_matches=matched_character_names,
            tag_score=round(tag_score, 1),
            character_score=round(character_score, 1)
        ))

    # 6. Sort by score (descending)
    recommendations.sort(key=lambda x: x.score, reverse=True)

    return TaskReviewSetRecommendationsResponse(
        task_id=task_id,
        task_name=task.name,
        recommendations=recommendations,
        total_recommendations=len(recommendations)
    )
