import io
import uuid
from typing import List
from uuid import UUID

from fastapi import UploadFile

from aimage_supervision.clients.aws_s3 import upload_file_to_s3
from aimage_supervision.clients.image_generation import (
    generate_image_with_gemini, generate_image_with_gemini_from_images)
from aimage_supervision.models import GeneratedReference, User
from aimage_supervision.schemas import (GeneratedReferenceResponse,
                                        GenerateRequest, GenerationTags)


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
        for i, generated_image in enumerate(generated_images or []):
            # Upload to S3
            s3_key = f"references/{project_id}/{uuid.uuid4()}.jpg"
            await upload_file_to_s3(
                io.BytesIO(getattr(getattr(generated_image, 'image', None), 'image_bytes', b"")),
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

    @staticmethod
    async def generate_references_from_images(
        project_id: UUID,
        user: User,
        base_prompt: str,
        count: int,
        aspect_ratio: str,
        negative_prompt: str | None,
        tags: dict,
        images: list[UploadFile],
    ) -> List[GeneratedReferenceResponse]:
        """Generate images using gemini-2.5-flash-image-preview with reference images.

        Note: uploaded images are used only for generation and not persisted; we only store generated results.
        """
        # Do not enhance prompt for image-based generation
        enhanced_prompt = base_prompt

        # Read image bytes from UploadFile without saving to disk
        image_bytes_list: list[bytes] = []
        for up in images:
            data = await up.read()
            if data:
                image_bytes_list.append(data)

        # Call multimodal model
        generated_bytes = await generate_image_with_gemini_from_images(
            prompt=enhanced_prompt,
            input_images=image_bytes_list,
            aspect_ratio=aspect_ratio,
        )

        if not generated_bytes:
            raise ValueError("Failed to generate image")

        # Store outputs to S3 and DB like text-only path
        results: list[GeneratedReferenceResponse] = []
        s3_key = f"references/{project_id}/{uuid.uuid4()}.jpg"
        await upload_file_to_s3(io.BytesIO(generated_bytes), s3_key)

        reference = await GeneratedReference.create(
            base_prompt=base_prompt,
            enhanced_prompt=enhanced_prompt,
            tags={k: v for k, v in tags.items() if v},
            image_path=s3_key,
            project_id=project_id,
            created_by=user,
        )

        await reference.fetch_image_url()
        results.append(GeneratedReferenceResponse(
            id=reference.id,
            base_prompt=reference.base_prompt,
            enhanced_prompt=reference.enhanced_prompt,
            tags=reference.tags,
            image_url=reference.image_url(),
            image_path=reference.image_path,
            created_at=reference.created_at,
        ))

        return results