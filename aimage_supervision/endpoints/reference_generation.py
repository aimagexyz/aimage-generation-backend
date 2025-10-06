from typing import List, Optional
from uuid import UUID

from fastapi import (APIRouter, File, Form, HTTPException, Security,
                     UploadFile, status)

from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.models import GeneratedReference, Project, User
from aimage_supervision.schemas import (GeneratedReferenceResponse,
                                        GenerateRequest)
from aimage_supervision.services.reference_generation_service import \
    ReferenceGenerationService
from aimage_supervision.settings import logger

router = APIRouter(prefix='/reference-generation', tags=['Reference Generation'])

@router.post(
    "/projects/{project_id}/generate",
    response_model=List[GeneratedReferenceResponse]
)
async def generate_references(
    project_id: UUID,
    request: GenerateRequest,
    current_user: User = Security(get_current_user)
):
    """Generate character references from text prompt only."""
    # Simple validation
    project = await Project.of_user(current_user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    try:
        return await ReferenceGenerationService.generate_references(
            request, project_id, current_user
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(500, "Generation failed")


@router.post(
    "/projects/{project_id}/generate-from-images",
    response_model=List[GeneratedReferenceResponse]
)
async def generate_references_from_images(
    project_id: UUID,
    base_prompt: str = Form(..., description="Base text prompt"),
    count: int = Form(1, ge=1, le=4),
    aspect_ratio: str = Form("1:1"),
    negative_prompt: Optional[str] = Form(None),
    style: Optional[str] = Form(None),
    pose: Optional[str] = Form(None),
    camera: Optional[str] = Form(None),
    lighting: Optional[str] = Form(None),
    images: List[UploadFile] = File(..., description="Reference images"),
    current_user: User = Security(get_current_user),
):
    """Generate references using a base prompt and reference images. Images are not persisted."""
    # Validate project
    project = await Project.of_user(current_user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    try:
        return await ReferenceGenerationService.generate_references_from_images(
            project_id=project_id,
            user=current_user,
            base_prompt=base_prompt,
            count=count,
            aspect_ratio=aspect_ratio,
            negative_prompt=negative_prompt,
            tags={
                "style": style,
                "pose": pose,
                "camera": camera,
                "lighting": lighting,
            },
            images=images,
        )
    except Exception as e:
        logger.error(f"Image-based generation failed: {e}")
        raise HTTPException(500, "Generation failed")

@router.get(
    "/projects/{project_id}/references",
    response_model=List[GeneratedReferenceResponse]
)
async def list_references(
    project_id: UUID,
    current_user: User = Security(get_current_user)
):
    """List generated references - simple version"""
    
    project = await Project.of_user(current_user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    
    references = await GeneratedReference.filter(
        project_id=project_id
    ).order_by('-created_at').limit(20)  # Simple limit
    
    results = []
    for ref in references:
        await ref.fetch_image_url()
        results.append(GeneratedReferenceResponse(
            id=ref.id,
            base_prompt=ref.base_prompt,
            enhanced_prompt=ref.enhanced_prompt,
            tags=ref.tags,
            image_url=ref.image_url(),
            image_path=ref.image_path,
            created_at=ref.created_at
        ))
    
    return results 