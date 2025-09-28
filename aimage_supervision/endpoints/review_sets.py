from uuid import UUID
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, Security

from aimage_supervision.models import User
from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.services import review_set_service
from aimage_supervision.schemas import ReviewSetCreate, ReviewSetUpdate, ReviewSetOut

router = APIRouter(
    prefix="/review-sets",
    tags=["Review Sets"],
)

@router.post("/", response_model=ReviewSetOut, status_code=status.HTTP_201_CREATED)
async def create_review_set(
    rs_create: ReviewSetCreate,
    current_user: User = Security(get_current_user),
):
    try:
        return await review_set_service.create_review_set(rs_create)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.get("/", response_model=List[ReviewSetOut])
async def list_review_sets(
    project_id: UUID,
    current_user: User = Security(get_current_user),
):
    return await review_set_service.list_review_sets(project_id)

@router.get("/{review_set_id}", response_model=ReviewSetOut)
async def get_review_set(
    review_set_id: UUID,
    current_user: User = Security(get_current_user),
):
    review_set = await review_set_service.get_review_set(review_set_id)
    if not review_set:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review Set not found")
    return review_set

@router.put("/{review_set_id}", response_model=ReviewSetOut)
async def update_review_set(
    review_set_id: UUID,
    rs_update: ReviewSetUpdate,
    current_user: User = Security(get_current_user),
):
    try:
        return await review_set_service.update_review_set(review_set_id, rs_update)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.delete("/{review_set_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_review_set(
    review_set_id: UUID,
    current_user: User = Security(get_current_user),
):
    deleted = await review_set_service.delete_review_set(review_set_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review Set not found")
    return 