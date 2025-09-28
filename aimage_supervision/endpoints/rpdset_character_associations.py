from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from tortoise.exceptions import DoesNotExist, IntegrityError

from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.models import Character, ReviewSet, User
from aimage_supervision.schemas import (
    CharacterForReviewSet, ReviewSetCharacterAssociationCreate,
    ReviewSetCharacterAssociationOut, ReviewSetCharacterAssociationWithDetails,
    ReviewSetInDB)

router = APIRouter(
    prefix="/review-set-character-associations",
    tags=["Review Set Character Associations"],
)


@router.post("/", response_model=ReviewSetCharacterAssociationOut, status_code=status.HTTP_201_CREATED)
async def create_review_set_character_association(
    association_data: ReviewSetCharacterAssociationCreate,
    current_user: User = Depends(get_current_user)
) -> ReviewSetCharacterAssociationOut:
    """创建ReviewSet和Character的关联"""
    try:
        # 验证ReviewSet和Character是否存在
        review_set = await ReviewSet.get(id=association_data.review_set_id)
        character = await Character.get(id=association_data.character_id)

        # 检查是否在同一个项目中
        await review_set.fetch_related('project')
        await character.fetch_related('project')
        if review_set.project.id != character.project.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ReviewSet和Character必须属于同一个项目"
            )

        # 检查关联是否已存在
        existing_association = await review_set.characters.filter(id=character.id).first()
        if existing_association:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="该ReviewSet和Character已经存在关联"
            )

        # 创建关联
        await review_set.characters.add(character)

        return ReviewSetCharacterAssociationOut(
            review_set_id=association_data.review_set_id,
            character_id=association_data.character_id
        )

    except DoesNotExist as e:
        if "ReviewSet" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定的ReviewSet不存在"
            )
        elif "Character" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定的角色不存在"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="资源不存在"
            )


@router.get("/review_set/{review_set_id}/character/{character_id}", response_model=ReviewSetCharacterAssociationWithDetails)
async def get_review_set_character_association(
    review_set_id: UUID,
    character_id: UUID,
    current_user: User = Depends(get_current_user)
) -> ReviewSetCharacterAssociationWithDetails:
    """获取特定的ReviewSet和Character关联详情"""
    try:
        review_set = await ReviewSet.get(id=review_set_id)
        character = await Character.get(id=character_id)

        # 检查关联是否存在
        existing_association = await review_set.characters.filter(id=character.id).first()
        if not existing_association:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="关联不存在"
            )

        return ReviewSetCharacterAssociationWithDetails(
            review_set_id=review_set_id,
            character_id=character_id,
            review_set=ReviewSetInDB.model_validate(
                review_set, from_attributes=True),
            character=CharacterForReviewSet.model_validate(
                character, from_attributes=True)
        )
    except DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ReviewSet或Character不存在"
        )


@router.get("/review_set/{review_set_id}", response_model=List[ReviewSetCharacterAssociationWithDetails])
async def get_review_set_character_associations_by_review_set(
    review_set_id: UUID,
    current_user: User = Depends(get_current_user)
) -> List[ReviewSetCharacterAssociationWithDetails]:
    """获取特定ReviewSet的所有Character关联"""
    try:
        review_set = await ReviewSet.get(id=review_set_id).prefetch_related('characters')

        associations = []
        for character in review_set.characters:
            associations.append(ReviewSetCharacterAssociationWithDetails(
                review_set_id=review_set_id,
                character_id=character.id,
                review_set=ReviewSetInDB.model_validate(
                    review_set, from_attributes=True),
                character=CharacterForReviewSet.model_validate(
                    character, from_attributes=True)
            ))

        return associations
    except DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定的ReviewSet不存在"
        )


@router.get("/character/{character_id}", response_model=List[ReviewSetCharacterAssociationWithDetails])
async def get_review_set_character_associations_by_character(
    character_id: UUID,
    current_user: User = Depends(get_current_user)
) -> List[ReviewSetCharacterAssociationWithDetails]:
    """获取特定Character的所有ReviewSet关联"""
    try:
        character = await Character.get(id=character_id).prefetch_related('review_sets')

        associations = []
        for review_set in character.review_sets:
            associations.append(ReviewSetCharacterAssociationWithDetails(
                review_set_id=review_set.id,
                character_id=character_id,
                review_set=ReviewSetInDB.model_validate(
                    review_set, from_attributes=True),
                character=CharacterForReviewSet.model_validate(
                    character, from_attributes=True)
            ))

        return associations
    except DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定的角色不存在"
        )


@router.delete("/review_set/{review_set_id}/character/{character_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_review_set_character_association(
    review_set_id: UUID,
    character_id: UUID,
    current_user: User = Depends(get_current_user)
) -> None:
    """删除ReviewSet和Character的关联"""
    try:
        review_set = await ReviewSet.get(id=review_set_id)
        character = await Character.get(id=character_id)

        # 检查关联是否存在
        existing_association = await review_set.characters.filter(id=character.id).first()
        if not existing_association:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="关联不存在"
            )

        # 删除关联
        await review_set.characters.remove(character)

    except DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ReviewSet或Character不存在"
        )
