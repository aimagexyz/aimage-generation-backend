from typing import Annotated, List
import uuid
import io

from fastapi import (APIRouter, File, HTTPException, Security, UploadFile,
                     status)
from fastapi_pagination import Page

from aimage_supervision.clients.aws_s3 import upload_file_to_s3, get_s3_url_from_path
from aimage_supervision.utils.file_validation import is_valid_image_or_design_file
from aimage_supervision.utils.image_compression import compress_image_async
from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.middlewares.tortoise_paginate import tortoise_paginate
from aimage_supervision.models import (Document, Project, ProjectIn, ProjectOut, ProjectSimpleOut, User)
from aimage_supervision.settings import logger

router = APIRouter(prefix='', tags=['Projects'])

# Constants for image upload
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_CONTENT_TYPE_PREFIX = "image/"


@router.post('/projects')
async def create_project(
    user: Annotated[User, Security(get_current_user)],
    payload: ProjectIn,
) -> ProjectOut:
    project_name: str = payload.name
    description: str | None = payload.description

    existed = await Project.filter(owner=user, name=project_name).exists()

    if existed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='Project with this name already exists',
        )

    project = await Project.create(
        owner=user,
        name=project_name,
        description=description,
    )

    response = await ProjectOut.from_tortoise_orm(project)
    return response


@router.get('/projects')
async def list_projects(
    user: Annotated[User, Security(get_current_user)],
) -> Page[ProjectSimpleOut]:
    # 使用优化的查询，只获取 id 和 name 字段
    projects_queryset = Project.of_user_simple(user)
    response = await tortoise_paginate(
        projects_queryset,
        model=ProjectSimpleOut,  # type: ignore
    )
    return response


@router.get('/projects/{project_id}')
async def get_project(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
) -> ProjectOut:
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found',
        )

    response = await ProjectOut.from_tortoise_orm(project)
    return response


@router.delete('/projects/{project_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_or_quit_project(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
):
    project = await Project.get_or_none(id=project_id)
    if not project or not project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Project not found',
        )

    if project.is_user_owner(user):
        # Delete project
        await project.delete()
        return None
    # Quit project
    if project.is_user_in_coop_members(user):
        await project.coop_members.remove(user)
        return None

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail='You cannot quit this project, because your organization is participating in it',
    )


@router.post("/projects/{project_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload_project_documents(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    file: UploadFile = File(..., description="document file"),
):
    # verify project permission
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or no access permission",
        )

    # upload file to s3
    cloud_file_path = f"projects/{project_id}/documents/{file.filename}"

    # check if file already exists

    await upload_file_to_s3(file.file, cloud_file_path=cloud_file_path)

    # check if file already exists in database
    document = await Document.get_or_none(file_path=cloud_file_path)
    if document:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File already exists",
        )
    # save file metadata to database.
    document = await Document.create(
        project=project,
        file_path=cloud_file_path,
        file_name=file.filename,
        file_size=file.size,
        file_type=file.content_type,
    )

    logger.info(f"Uploaded {file.filename} documents to project {project_id}")

    return status.HTTP_201_CREATED


@router.get("/projects/{project_id}/documents/{document_id}/url")
async def get_document_url(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    document_id: str,
) -> dict[str, str]:
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    document = await Document.get_or_none(id=document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    await document.fetch_s3_url()

    return {"file_url": document.file_url()}


@router.post("/projects/{project_id}/rpd-reference-images")
async def upload_rpd_reference_image(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    file: UploadFile = File(..., description="RPD reference image file"),
):
    """Upload a reference image for RPD copyright detection"""
    # Verify project permission
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or no access permission",
        )

    # Validate file type (including PSD and AI files)
    if not is_valid_image_or_design_file(file.filename, file.content_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload an image, PSD, or AI file."
        )

    # Read file content and validate file size
    file_contents = await file.read()
    if len(file_contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size too large (maximum {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB)."
        )

    try:
        # Compress and convert image (PSD/AI/SVG -> PNG)
        try:
            with io.BytesIO(file_contents) as f:
                compressed_file, output_format, compressed_size = await compress_image_async(f)
            
            # Determine file extension based on compression output
            if output_format == 'JPEG':
                file_extension = 'jpg'
            elif output_format == 'PNG':
                file_extension = 'png'
            elif output_format == 'SVG':
                file_extension = 'svg'
            elif output_format == 'PSD':
                file_extension = 'psd'
            elif output_format == 'AI':
                file_extension = 'ai'
            else:
                file_extension = output_format.lower()
                
        except Exception as e:
            logger.warning(f"Image compression failed for project {project_id}: {e}, using original")
            compressed_file = io.BytesIO(file_contents)
            sanitized_filename = file.filename if file.filename else "reference_image"
            file_extension = sanitized_filename.split('.')[-1] if '.' in sanitized_filename else 'jpg'

        # Generate S3 path for RPD reference image
        s3_object_name = f"projects/{project_id}/rpd-reference-images/{uuid.uuid4()}.{file_extension}"

        # Upload to S3
        if hasattr(compressed_file, 'seek'):
            compressed_file.seek(0)
        await upload_file_to_s3(compressed_file, s3_object_name)

        # Get the uploaded image URL for immediate return
        image_url = await get_s3_url_from_path(s3_object_name)

        logger.info(f"Uploaded RPD reference image to project {project_id}: {s3_object_name}")

        return {
            "s3_path": s3_object_name,
            "url": image_url,
            "filename": sanitized_filename,
            "size": len(file_contents)
        }

    except Exception as e:
        logger.error(f"S3 upload failed for RPD reference image in project {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file to storage. Please try again later."
        )
    finally:
        await file.close()


@router.post("/projects/{project_id}/rpd-reference-images/batch")
async def upload_rpd_reference_images_batch(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    files: List[UploadFile] = File(..., description="RPD reference image files"),
):
    """Batch upload reference images for RPD copyright detection"""
    # Verify project permission
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or no access permission",
        )


    uploaded_files = []
    failed_files = []

    for file in files:
        try:
            # Validate file type (including PSD and AI files)
            if not is_valid_image_or_design_file(file.filename, file.content_type):
                failed_files.append({
                    "filename": file.filename or "unknown",
                    "error": "File must be an image, PSD, or AI file"
                })
                continue

            # Read and validate file size
            file_contents = await file.read()
            if len(file_contents) > MAX_IMAGE_SIZE_BYTES:
                failed_files.append({
                    "filename": file.filename or "unknown",
                    "error": f"File size too large (maximum {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB)"
                })
                continue

            # Compress and convert image (PSD/AI/SVG -> PNG)
            try:
                with io.BytesIO(file_contents) as f:
                    compressed_file, output_format, compressed_size = await compress_image_async(f)
                
                # Determine file extension based on compression output
                if output_format == 'JPEG':
                    file_extension = 'jpg'
                elif output_format == 'PNG':
                    file_extension = 'png'
                elif output_format == 'SVG':
                    file_extension = 'svg'
                elif output_format == 'PSD':
                    file_extension = 'psd'
                elif output_format == 'AI':
                    file_extension = 'ai'
                else:
                    file_extension = output_format.lower()
                    
            except Exception as e:
                logger.warning(f"Image compression failed for batch upload in project {project_id}: {e}, using original")
                compressed_file = io.BytesIO(file_contents)
                sanitized_filename = file.filename if file.filename else "reference_image"
                file_extension = sanitized_filename.split('.')[-1] if '.' in sanitized_filename else 'jpg'

            # Generate S3 path
            s3_object_name = f"projects/{project_id}/rpd-reference-images/{uuid.uuid4()}.{file_extension}"

            # Upload to S3
            if hasattr(compressed_file, 'seek'):
                compressed_file.seek(0)
            await upload_file_to_s3(compressed_file, s3_object_name)

            # Get image URL
            image_url = await get_s3_url_from_path(s3_object_name)

            uploaded_files.append({
                "filename": sanitized_filename,
                "s3_path": s3_object_name,
                "url": image_url,
                "size": len(file_contents)
            })

            logger.info(f"Uploaded RPD reference image to project {project_id}: {s3_object_name}")

        except Exception as e:
            logger.error(f"Failed to upload RPD reference image {file.filename}: {str(e)}")
            failed_files.append({
                "filename": file.filename or "unknown",
                "error": f"Upload failed: {str(e)}"
            })
        finally:
            await file.close()

    # Return detailed upload results
    return {
        "project_id": project_id,
        "uploaded_count": len(uploaded_files),
        "failed_count": len(failed_files),
        "uploaded_files": uploaded_files,
        "failed_files": failed_files,
        "total_uploaded": len(uploaded_files)
    }
