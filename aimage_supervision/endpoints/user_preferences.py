from typing import Annotated, List

from fastapi import APIRouter, HTTPException, Security, status
from fastapi.responses import JSONResponse

from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.models import LikedImage, User, UserPreferences
from aimage_supervision.schemas import (
    LikedImageRemoveRequest,
    LikedImageRequest,
    LikedImageResponse,
    UserPreferencesResponse,
)

router = APIRouter(prefix='/users/me/preferences', tags=['User Preferences'])


@router.get('/liked-images', response_model=List[LikedImageResponse])
async def get_liked_images(
    user: Annotated[User, Security(get_current_user)]
) -> List[LikedImageResponse]:
    """Get user's liked images with fresh presigned URLs"""
    try:
        # Get liked images for the user
        liked_images = await LikedImage.filter(user=user).order_by('-created_at')
        
        # Fetch fresh S3 URLs for all images
        for liked_image in liked_images:
            await liked_image.fetch_image_url()
        
        # Convert to response format
        return [
            LikedImageResponse(
                id=liked_image.id,
                image_path=liked_image.image_path,
                image_url=liked_image.image_url(),
                source_type=liked_image.source_type,
                source_id=liked_image.source_id,
                display_name=liked_image.display_name,
                tags=liked_image.tags or [],
                created_at=liked_image.created_at,
            )
            for liked_image in liked_images
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving liked images: {str(e)}"
        )


@router.post('/liked-images')
async def add_liked_image(
    request: LikedImageRequest,
    user: Annotated[User, Security(get_current_user)]
) -> dict:
    """Add image to liked list with source tracking"""
    try:
        # Check if image is already liked by this user
        existing = await LikedImage.get_or_none(
            user=user,
            image_path=request.image_path,
            source_type=request.source_type,
            source_id=request.source_id
        )
        
        if existing:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    'message': 'Image already in liked list',
                    'image_path': request.image_path,
                    'liked_image_id': str(existing.id)
                }
            )
        
        # Create new liked image record
        liked_image = await LikedImage.create(
            user=user,
            image_path=request.image_path,
            source_type=request.source_type,
            source_id=request.source_id,
            display_name=request.display_name,
            tags=request.tags or []
        )
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                'message': 'Image added to liked list',
                'image_path': request.image_path,
                'liked_image_id': str(liked_image.id),
                'source_type': request.source_type,
                'source_id': str(request.source_id)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding liked image: {str(e)}"
        )


@router.delete('/liked-images')
async def remove_liked_image(
    request: LikedImageRemoveRequest,
    user: Annotated[User, Security(get_current_user)]
) -> dict:
    """Remove image from liked list"""
    try:
        # Find the liked image record
        liked_image = await LikedImage.get_or_none(
            user=user,
            image_path=request.image_path,
            source_type=request.source_type,
            source_id=request.source_id
        )
        
        if not liked_image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Liked image not found"
            )
        
        # Delete the liked image record
        await liked_image.delete()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                'message': 'Image removed from liked list',
                'image_path': request.image_path,
                'source_type': request.source_type,
                'source_id': str(request.source_id)
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing liked image: {str(e)}"
        )


@router.get('/', response_model=UserPreferencesResponse)
async def get_user_preferences(
    user: Annotated[User, Security(get_current_user)]
) -> UserPreferencesResponse:
    """Get user preferences and settings (liked images available via separate endpoint)"""
    try:
        # Get user preferences (for settings)
        preferences = await UserPreferences.get_or_none(user=user)
        
        return UserPreferencesResponse(
            settings=preferences.settings if preferences else {}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user preferences: {str(e)}"
        ) 