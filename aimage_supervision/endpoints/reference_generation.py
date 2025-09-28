from typing import List
from uuid import UUID
from aimage_supervision.settings import logger
from fastapi import APIRouter, HTTPException, Security, status

from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.models import GeneratedReference, Project, User
from aimage_supervision.schemas import GenerateRequest, GeneratedReferenceResponse
from aimage_supervision.services.reference_generation_service import ReferenceGenerationService


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
    """Generate character references - MVP endpoint"""
    print('request: ', request)
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