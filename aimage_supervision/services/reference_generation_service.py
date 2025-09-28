import io
import uuid
from typing import List
from uuid import UUID

from aimage_supervision.clients.aws_s3 import upload_file_to_s3
from aimage_supervision.clients.image_generation import generate_image_with_gemini
from aimage_supervision.models import GeneratedReference, User
from aimage_supervision.schemas import GenerateRequest, GeneratedReferenceResponse, GenerationTags


class ReferenceGenerationService:
    """Simple generation service - MVP version"""
    
    @staticmethod
    def build_enhanced_prompt(base_prompt: str, tags: GenerationTags) -> str:
        """Simple string concatenation"""
        parts = [base_prompt]
        
        if tags.style:
            parts.append(f"{tags.style} style")
        if tags.pose:
            parts.append(f"{tags.pose}")
        if tags.camera:
            parts.append(f"{tags.camera} shot")
        if tags.lighting:
            parts.append(f"{tags.lighting} lighting")
            
        return ", ".join(parts)
    
    @staticmethod
    async def generate_references(
        request: GenerateRequest,
        project_id: UUID,
        user: User
    ) -> List[GeneratedReferenceResponse]:
        """Simple generation workflow"""

        # 1. Build enhanced prompt
        enhanced_prompt = ReferenceGenerationService.build_enhanced_prompt(
            request.base_prompt, request.tags
        )
        
        # 2. Generate images (reuse existing service)
        generated_images = await generate_image_with_gemini(
            enhanced_prompt,
            request.count,
            request.aspect_ratio,  # Use from request instead of hardcoded "1:1"
            request.negative_prompt  # Pass through negative prompt
        )
        
        # 3. Store and return
        results = []
        for i, generated_image in enumerate(generated_images):
            # Upload to S3
            s3_key = f"references/{project_id}/{uuid.uuid4()}.jpg"
            await upload_file_to_s3(
                io.BytesIO(generated_image.image.image_bytes),
                s3_key
            )
            
            # Save to DB
            reference = await GeneratedReference.create(
                base_prompt=request.base_prompt,
                enhanced_prompt=enhanced_prompt,
                tags=request.tags.model_dump(exclude_none=True),
                image_path=s3_key,
                project_id=project_id,
                created_by=user
            )
            
            await reference.fetch_image_url()
            
            results.append(GeneratedReferenceResponse(
                id=reference.id,
                base_prompt=reference.base_prompt,
                enhanced_prompt=reference.enhanced_prompt,
                tags=reference.tags,
                image_url=reference.image_url(),
                image_path=reference.image_path,
                created_at=reference.created_at
            ))
            
        return results 