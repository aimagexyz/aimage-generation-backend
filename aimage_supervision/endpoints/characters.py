import io
import mimetypes
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional
from uuid import UUID

from fastapi import (APIRouter, Depends, File, HTTPException, Security,
                     UploadFile, status)
from fastapi.responses import Response, StreamingResponse

from aimage_supervision.clients.aws_s3 import (download_file_content_from_s3,
                                               get_s3_url_from_path,
                                               upload_file_to_s3)
from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.models import (IP, Character, CharacterDetail, IPOut,
                                       Project, User)
from aimage_supervision.schemas import (CharacterCreate, CharacterUpdate,
                                        IPCreate, IPUpdate)
from aimage_supervision.settings import logger
from aimage_supervision.utils.file_validation import \
    is_valid_image_or_design_file
from aimage_supervision.utils.image_compression import compress_image_async

router = APIRouter(prefix='/characters',
                   tags=['characters'])

# 允许的图片MIME类型前缀
ALLOWED_IMAGE_CONTENT_TYPE_PREFIX = "image/"
# 最大图片大小10MB
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


async def process_and_upload_image(file: UploadFile, s3_path_prefix: str, identifier: str) -> str:
    """
    Helper function to process image files (compress/convert) and upload to S3

    Args:
        file: The uploaded file
        s3_path_prefix: S3 path prefix (e.g., "characters/gallery")
        identifier: Unique identifier for the file (e.g., character_id)

    Returns:
        str: The S3 path where the file was uploaded
    """
    # Read file content
    file_content = await file.read()

    # Compress and convert image (PSD/AI/SVG -> PNG)
    try:
        with io.BytesIO(file_content) as f:
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
        logger.warning(f"Image compression failed: {e}, using original")
        compressed_file = io.BytesIO(file_content)
        # Extract extension from original filename
        if file.filename:
            file_extension = Path(file.filename).suffix.lstrip('.') or 'jpg'
        else:
            file_extension = 'jpg'

    # Generate S3 path
    s3_file_key = f"{s3_path_prefix}/{identifier}_{uuid.uuid4()}.{file_extension}"

    # Upload to S3
    if hasattr(compressed_file, 'seek'):
        compressed_file.seek(0)
    await upload_file_to_s3(compressed_file, s3_file_key)

    return s3_file_key


@router.get('/', response_model=List[CharacterDetail])
async def list_characters(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    ip_id: str,
) -> list[CharacterDetail]:  # -> list:# -> list:# -> list:# -> list:
    """获取角色列表，可以按项目或IP筛选"""
    query = Character.all().prefetch_related("project", "ip")

    if project_id:
        project = await Project.of_user(user).get_or_none(id=project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or user lacks access"
            )
        query = query.filter(project_id=project.id)

    if ip_id:
        query = query.filter(ip_id=ip_id)

    characters = await query

    # Convert to Pydantic models with proper image URLs
    character_details = []
    for character in characters:
        # 获取图片URL
        await character.fetch_image_url()
        await character.fetch_gallery_image_urls()

        # 转换为Pydantic模型 (related objects already loaded)
        character_detail = await CharacterDetail.from_tortoise_orm(character)

        # 确保图片URL被正确设置
        if hasattr(character, '_image_url'):
            setattr(character_detail, '_image_url', character._image_url)

        character_details.append(character_detail)

    return character_details


@router.get('/{character_id}', response_model=CharacterDetail)
async def get_character(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    character_id: str,
) -> Character:
    """获取角色详情"""
    character = await Character.get_or_none(id=character_id)

    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 检查用户是否有权限访问该项目
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    # 获取图片URL
    await character.fetch_image_url()
    await character.fetch_gallery_image_urls()

    return character


@router.post('/', status_code=status.HTTP_201_CREATED, response_model=CharacterDetail)
async def create_character(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    character_data: CharacterCreate,
) -> CharacterDetail:
    """创建新角色"""
    # 检查项目是否存在以及用户是否有权访问
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or user lacks access"
        )

    # 检查IP是否存在（如果提供）
    if character_data.ip_id:
        ip = await IP.get_or_none(id=character_data.ip_id)
        if not ip:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="IP not found"
            )

    # 创建角色
    character = await Character.create(
        name=character_data.name,
        alias=character_data.alias,
        description=character_data.description,
        features=character_data.features,
        image_path=character_data.image_path,
        reference_images=character_data.reference_images,
        ip_id=character_data.ip_id,
        project=project
    )

    # 获取图片URL
    await character.fetch_image_url()
    await character.fetch_gallery_image_urls()

    # 确保从数据库获取最新状态并加载关联对象
    await character.fetch_related("project", "ip")

    # 转换为Pydantic模型
    character_detail = await CharacterDetail.from_tortoise_orm(character)

    # 确保图片URL被正确设置
    if hasattr(character, '_image_url'):
        setattr(character_detail, '_image_url', character._image_url)
    if hasattr(character, '_gallery_image_urls'):
        setattr(character_detail, '_gallery_image_urls',
                character._gallery_image_urls)

    return character_detail


@router.put('/{character_id}', response_model=CharacterDetail)
async def update_character(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    character_id: str,
    character_data: CharacterUpdate,
) -> CharacterDetail:
    """更新角色信息"""
    character = await Character.get_or_none(id=character_id)

    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 检查用户是否有权限访问该项目
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    # 检查IP是否存在（如果提供）
    if character_data.ip_id is not None:
        ip = await IP.get_or_none(id=character_data.ip_id)
        if not ip and character_data.ip_id is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="IP not found"
            )

    # 更新角色
    update_data = character_data.model_dump(exclude_unset=True)
    character.update_from_dict(update_data)
    character.ip_id = character_data.ip_id

    await character.save()

    # 获取图片URL
    await character.fetch_image_url()
    await character.fetch_gallery_image_urls()

    # 确保从数据库获取最新状态并加载关联对象
    await character.fetch_related("project", "ip")

    # 转换为Pydantic模型
    character_detail = await CharacterDetail.from_tortoise_orm(character)

    # 确保图片URL被正确设置
    if hasattr(character, '_image_url'):
        setattr(character_detail, '_image_url', character._image_url)
    if hasattr(character, '_gallery_image_urls'):
        setattr(character_detail, '_gallery_image_urls',
                character._gallery_image_urls)

    return character_detail


@router.delete('/{character_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_character(
    user: Annotated[User, Security(get_current_user)],
    character_id: UUID,
) -> Response:
    """删除角色"""
    character = await Character.get_or_none(id=character_id)

    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 检查用户是否有权限访问该项目
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    await character.delete()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{character_id}/upload-image", response_model=CharacterDetail)
async def upload_character_image(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    character_id: str,
    file: UploadFile = File(...),
) -> CharacterDetail:
    """上传角色图片到S3存储"""
    # 验证角色存在性
    character = await Character.get_or_none(id=character_id)
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 验证用户权限
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    try:
        # 验证文件类型 (including PSD and AI files)
        if not is_valid_image_or_design_file(file.filename, file.content_type):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Please upload an image, PSD, or AI file."
            )

        # 读取文件内容并验证文件大小
        file_contents = await file.read()
        if len(file_contents) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size too large (maximum {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB)."
            )

        # Compress and convert image (PSD/AI/SVG -> PNG)
        try:
            # Don't use context manager here as it will close the BytesIO
            f = io.BytesIO(file_contents)
            compressed_file, output_format, compressed_size = await compress_image_async(f)

            # Determine file extension based on compression output
            if output_format == 'JPEG':
                file_extension = '.jpg'
            elif output_format == 'PNG':
                file_extension = '.png'
            elif output_format == 'SVG':
                file_extension = '.svg'
            elif output_format == 'PSD':
                file_extension = '.psd'
            elif output_format == 'AI':
                file_extension = '.ai'
            else:
                file_extension = f'.{output_format.lower()}'

        except Exception as e:
            logger.warning(
                f"Image compression failed for character {character_id}: {e}, using original")
            compressed_file = io.BytesIO(file_contents)
            sanitized_filename = file.filename if file.filename else "untitled_image"
            file_extension = Path(str(sanitized_filename)).suffix

        # 构建S3存储路径
        s3_object_name = f"characters/projects/{project_id}/{character_id}/{uuid.uuid4()}{file_extension}"

        # 上传到S3
        try:
            # Reset file pointer if needed
            if hasattr(compressed_file, 'seek'):
                compressed_file.seek(0)
            await upload_file_to_s3(compressed_file, s3_object_name)
        except Exception as e:
            logger.error(
                f"S3 upload failed for character {character_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to upload file to storage. Please try again later."
            )

        # 更新角色的图片路径
        character.image_path = s3_object_name
        await character.save()

        # 获取并设置图片URL
        await character.fetch_image_url()

        # 确保从数据库获取最新状态并加载关联对象
        await character.fetch_related("project", "ip")

        # 转换为Pydantic模型
        character_detail = await CharacterDetail.from_tortoise_orm(character)

        # 确保图片URL被正确设置
        if hasattr(character, '_image_url') and character._image_url:
            setattr(character_detail, '_image_url', character._image_url)

    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        logger.error(
            f"Error processing character image upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"There was an error processing the file: {str(e)}"
        )
    finally:
        await file.close()

    return character_detail


@router.get("/{character_id}/image")
async def get_character_image(
    user: Annotated[User, Security(get_current_user)],
    character_id: str,
    project_id: str,
) -> Response:
    """从S3直接获取角色图片并返回给前端"""
    character = await Character.get_or_none(id=character_id)
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 验证用户权限
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    # 检查角色是否有图片
    if not character.image_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character has no image"
        )

    try:
        # 从S3下载图片内容
        image_content = await download_file_content_from_s3(character.image_path)

        # 根据文件扩展名推断MIME类型
        file_extension = Path(character.image_path).suffix.lower()
        content_type = mimetypes.guess_type(f"file{file_extension}")[0]
        if not content_type:
            content_type = "application/octet-stream"  # 默认二进制类型

            # 如果没有正确识别扩展名，尝试根据常见图片格式判断
            if file_extension in ['.jpg', '.jpeg']:
                content_type = "image/jpeg"
            elif file_extension == '.png':
                content_type = "image/png"
            elif file_extension == '.gif':
                content_type = "image/gif"
            elif file_extension == '.webp':
                content_type = "image/webp"

        # 返回图片内容
        return Response(
            content=image_content,
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename=\"{character_id}{file_extension}\""
            }
        )
    except Exception as e:
        logger.error(
            f"Error fetching character image from S3: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch character image"
        )


@router.get("/{character_id}/gallery")
async def get_character_gallery(
    user: Annotated[User, Security(get_current_user)],
    character_id: str,
    project_id: str,
) -> Dict[str, Any]:
    """获取角色画廊图片列表"""
    character = await Character.get_or_none(id=character_id)

    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 检查用户是否有权限访问该项目
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    # 获取画廊图片URLs
    await character.fetch_gallery_image_urls()

    return {
        "character_id": character_id,
        "gallery_images": character.gallery_image_urls()
    }


@router.get("/{character_id}/concept-art")
async def get_character_concept_art(
    user: Annotated[User, Security(get_current_user)],
    character_id: str,
    project_id: str,
) -> Dict[str, Any]:
    """获取角色设定集图片列表"""
    character = await Character.get_or_none(id=character_id)

    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 检查用户是否有权限访问该项目
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    # 获取设定集图片URLs
    await character.fetch_concept_art_image_urls()

    return {
        "character_id": character_id,
        "concept_art_images": character.concept_art_image_urls()
    }


@router.post("/{character_id}/gallery", response_model=CharacterDetail)
async def upload_character_gallery_image(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    character_id: str,
    file: UploadFile = File(...),
) -> CharacterDetail:
    """上传角色画廊图片"""
    character = await Character.get_or_none(id=character_id)

    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 检查用户是否有权限访问该项目
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    # 验证文件类型和大小 (including PSD and AI files)
    if not is_valid_image_or_design_file(file.filename, file.content_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image, PSD, or AI file"
        )

    # Check file size before processing
    await file.seek(0)
    file_content = await file.read()
    if len(file_content) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size too large. Maximum allowed: {MAX_IMAGE_SIZE_BYTES} bytes"
        )
    await file.seek(0)

    try:
        # Process, compress/convert and upload to S3
        s3_file_key = await process_and_upload_image(file, "characters/gallery", character_id)

        # 更新角色的reference_images字段
        if character.reference_images is None:
            character.reference_images = []

        character.reference_images.append(s3_file_key)
        await character.save()

        # 获取更新后的URLs
        await character.fetch_image_url()
        await character.fetch_gallery_image_urls()

        logger.info(
            f"Gallery image uploaded for character {character_id}: {s3_file_key}")

        # 确保从数据库获取最新状态并加载关联对象
        await character.fetch_related("project", "ip")

        # 转换为Pydantic模型
        character_detail = await CharacterDetail.from_tortoise_orm(character)

        # 确保图片URL被正确设置
        if hasattr(character, '_image_url'):
            setattr(character_detail, '_image_url', character._image_url)
        if hasattr(character, '_gallery_image_urls'):
            setattr(character_detail, '_gallery_image_urls',
                    character._gallery_image_urls)

        return character_detail

    except Exception as e:
        logger.error(
            f"Failed to upload gallery image for character {character_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload image"
        )


@router.post("/{character_id}/gallery/batch")
async def upload_character_gallery_images_batch(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    character_id: str,
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    """批量上传角色画廊图片"""
    character = await Character.get_or_none(id=character_id)

    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 检查用户是否有权限访问该项目
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    uploaded_files = []
    failed_files = []

    # 确保reference_images字段存在
    if character.reference_images is None:
        character.reference_images = []

    for file in files:
        try:
            # 验证文件类型 (including PSD and AI files)
            if not is_valid_image_or_design_file(file.filename, file.content_type):
                failed_files.append({
                    "filename": file.filename or "unknown",
                    "error": "File must be an image, PSD, or AI file"
                })
                continue

            # 验证文件大小
            await file.seek(0)
            file_content = await file.read()
            if len(file_content) > MAX_IMAGE_SIZE_BYTES:
                failed_files.append({
                    "filename": file.filename or "unknown",
                    "error": f"File size too large (maximum {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB)"
                })
                continue
            await file.seek(0)

            # Process, compress/convert and upload to S3
            s3_file_key = await process_and_upload_image(file, "characters/gallery", character_id)

            # 添加到reference_images列表
            character.reference_images.append(s3_file_key)

            uploaded_files.append({
                "filename": file.filename or "unknown",
                "s3_key": s3_file_key,
                "size": len(file_content)
            })

            logger.info(
                f"Gallery image uploaded for character {character_id}: {s3_file_key}")

        except Exception as e:
            logger.error(f"Failed to upload file {file.filename}: {str(e)}")
            failed_files.append({
                "filename": file.filename or "unknown",
                "error": f"Upload failed: {str(e)}"
            })

    # 保存更新后的reference_images
    try:
        await character.save()

        # 获取更新后的URLs
        await character.fetch_image_url()
        await character.fetch_gallery_image_urls()

    except Exception as e:
        logger.error(f"Failed to save character after batch upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded images"
        )

    # 返回详细的上传结果
    return {
        "character_id": character_id,
        "uploaded_count": len(uploaded_files),
        "failed_count": len(failed_files),
        "uploaded_files": uploaded_files,
        "failed_files": failed_files,
        "total_gallery_images": len(character.reference_images) if character.reference_images else 0
    }


@router.delete("/{character_id}/gallery/{image_index}")
async def delete_character_gallery_image(
    user: Annotated[User, Security(get_current_user)],
    character_id: str,
    project_id: str,
    image_index: int,
) -> Dict[str, Any]:
    """删除角色画廊中的指定图片"""
    character = await Character.get_or_none(id=character_id)

    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 检查用户是否有权限访问该项目
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    if not character.reference_images or image_index >= len(character.reference_images):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found in gallery"
        )

    # 移除指定索引的图片
    removed_image = character.reference_images.pop(image_index)
    await character.save()

    logger.info(
        f"Gallery image removed for character {character_id}: {removed_image}")

    return {"message": "Image removed from gallery", "removed_image": removed_image}


@router.post("/{character_id}/concept-art", response_model=CharacterDetail)
async def upload_character_concept_art_image(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    character_id: str,
    file: UploadFile = File(...),
) -> CharacterDetail:
    """上传角色设定集图片"""
    character = await Character.get_or_none(id=character_id)

    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 检查用户是否有权限访问该项目
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    # 验证文件类型和大小 (including PSD and AI files)
    if not is_valid_image_or_design_file(file.filename, file.content_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image, PSD, or AI file"
        )

    # Check file size before processing
    await file.seek(0)
    file_content = await file.read()
    if len(file_content) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size too large. Maximum allowed: {MAX_IMAGE_SIZE_BYTES} bytes"
        )
    await file.seek(0)

    try:
        # Process, compress/convert and upload to S3
        s3_file_key = await process_and_upload_image(file, "characters/concept-art", character_id)

        # 更新角色的concept_art_images字段
        if character.concept_art_images is None:
            character.concept_art_images = []

        character.concept_art_images.append(s3_file_key)
        await character.save()

        # 获取更新后的URLs
        await character.fetch_image_url()
        await character.fetch_gallery_image_urls()
        await character.fetch_concept_art_image_urls()

        logger.info(
            f"Concept art image uploaded for character {character_id}: {s3_file_key}")

        # 确保从数据库获取最新状态并加载关联对象
        await character.fetch_related("project", "ip")

        # 转换为Pydantic模型
        character_detail = await CharacterDetail.from_tortoise_orm(character)

        # 确保图片URL被正确设置
        if hasattr(character, '_image_url'):
            setattr(character_detail, '_image_url', character._image_url)
        if hasattr(character, '_gallery_image_urls'):
            setattr(character_detail, '_gallery_image_urls',
                    character._gallery_image_urls)
        if hasattr(character, '_concept_art_image_urls'):
            setattr(character_detail, '_concept_art_image_urls',
                    character._concept_art_image_urls)

        return character_detail

    except Exception as e:
        logger.error(
            f"Failed to upload concept art image for character {character_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload image"
        )


@router.post("/{character_id}/concept-art/batch")
async def upload_character_concept_art_images_batch(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    character_id: str,
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    """批量上传角色设定集图片"""
    character = await Character.get_or_none(id=character_id)

    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 检查用户是否有权限访问该项目
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    uploaded_files = []
    failed_files = []

    # 确保concept_art_images字段存在
    if character.concept_art_images is None:
        character.concept_art_images = []

    for file in files:
        try:
            # 验证文件类型 (including PSD and AI files)
            if not is_valid_image_or_design_file(file.filename, file.content_type):
                failed_files.append({
                    "filename": file.filename or "unknown",
                    "error": "File must be an image, PSD, or AI file"
                })
                continue

            # 验证文件大小
            await file.seek(0)
            file_content = await file.read()
            if len(file_content) > MAX_IMAGE_SIZE_BYTES:
                failed_files.append({
                    "filename": file.filename or "unknown",
                    "error": f"File size too large (maximum {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB)"
                })
                continue
            await file.seek(0)

            # Process, compress/convert and upload to S3
            s3_file_key = await process_and_upload_image(file, "characters/concept-art", character_id)

            # 添加到concept_art_images列表
            character.concept_art_images.append(s3_file_key)

            uploaded_files.append({
                "filename": file.filename or "unknown",
                "s3_key": s3_file_key,
                "size": len(file_content)
            })

            logger.info(
                f"Concept art image uploaded for character {character_id}: {s3_file_key}")

        except Exception as e:
            logger.error(f"Failed to upload file {file.filename}: {str(e)}")
            failed_files.append({
                "filename": file.filename or "unknown",
                "error": f"Upload failed: {str(e)}"
            })

    # 保存更新后的concept_art_images
    try:
        await character.save()

        # 获取更新后的URLs
        await character.fetch_image_url()
        await character.fetch_gallery_image_urls()
        await character.fetch_concept_art_image_urls()

    except Exception as e:
        logger.error(f"Failed to save character after batch upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded images"
        )

    # 返回详细的上传结果
    return {
        "character_id": character_id,
        "uploaded_count": len(uploaded_files),
        "failed_count": len(failed_files),
        "uploaded_files": uploaded_files,
        "failed_files": failed_files,
        "total_concept_art_images": len(character.concept_art_images) if character.concept_art_images else 0
    }


@router.delete("/{character_id}/concept-art/{image_index}")
async def delete_character_concept_art_image(
    user: Annotated[User, Security(get_current_user)],
    character_id: str,
    project_id: str,
    image_index: int,
) -> Dict[str, Any]:
    """删除角色设定集中的指定图片"""
    character = await Character.get_or_none(id=character_id)

    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # 检查用户是否有权限访问该项目
    project = await character.project
    if not await project.is_user_can_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have access to the project this character belongs to"
        )

    if not character.concept_art_images or image_index >= len(character.concept_art_images):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found in concept art collection"
        )

    # 移除指定索引的图片
    removed_image = character.concept_art_images.pop(image_index)
    await character.save()

    logger.info(
        f"Concept art image removed for character {character_id}: {removed_image}")

    return {"message": "Image removed from concept art collection", "removed_image": removed_image}


# IP 相关路由
ip_router = APIRouter(prefix='/ips', tags=['ips'])


@ip_router.get('/', response_model=List[IPOut])
async def list_ips(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
) -> List[IP]:
    """获取IP列表，可以按项目筛选"""
    if project_id:
        project = await Project.of_user(user).get_or_none(id=project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or user lacks access"
            )
        ips = await IP.filter(projects=project)
    else:
        ips = await IP.all()

    return ips


@ip_router.get('/{ip_id}', response_model=IPOut)
async def get_ip(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    ip_id: str,
) -> IP:
    """获取IP详情"""
    ip = await IP.get_or_none(id=ip_id)

    if not ip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="IP not found"
        )

    if project_id:
        # Verify access to the specified project
        auth_project = await Project.of_user(user).get_or_none(id=project_id)
        if not auth_project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or user lacks access"
            )
        # Check if the IP is linked to this authorized project
        is_linked = await ip.projects.filter(id=auth_project.id).exists()
        if not is_linked:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="IP not found in the specified project context"
            )

    return ip


@ip_router.post('/', status_code=status.HTTP_201_CREATED, response_model=IPOut)
async def create_ip(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    ip_data: IPCreate,
) -> IP:
    """创建新IP"""
    # 创建IP
    ip = await IP.create(
        name=ip_data.name,
        description=ip_data.description,
    )

    # 添加关联的项目（如果提供）
    if ip_data.project_ids:
        for pid in ip_data.project_ids:
            project = await Project.of_user(user).get_or_none(id=pid)
            if project:
                await ip.projects.add(project)
            else:
                logger.warning(
                    f"Skipping association for IP {ip.id}: Project {pid} not found or user lacks access.")

    return ip


@ip_router.put('/{ip_id}', response_model=IPOut)
async def update_ip(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    ip_id: str,
    ip_data: IPUpdate,
) -> IP:
    """更新IP信息"""
    ip = await IP.get_or_none(id=ip_id)

    if not ip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="IP not found"
        )

    # 更新IP
    if ip_data.name is not None:
        ip.name = ip_data.name
    if ip_data.description is not None:
        ip.description = ip_data.description

    await ip.save()

    # 更新关联的项目（如果提供）
    if ip_data.project_ids is not None:
        # 删除现有关联
        await ip.projects.clear()

        # 添加新关联
        for pid in ip_data.project_ids:
            project = await Project.of_user(user).get_or_none(id=pid)
            if project:
                await ip.projects.add(project)
            else:
                logger.warning(
                    f"Skipping association for IP {ip.id}: Project {pid} not found or user lacks access.")

    await ip.save()

    return ip


@ip_router.delete('/{ip_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_ip(
    user: Annotated[User, Security(get_current_user)],
    project_id: str,
    ip_id: str,
) -> None:
    """删除IP"""
    ip = await IP.get_or_none(id=ip_id)

    if not ip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="IP not found"
        )

    await ip.delete()
    return None


# --- RPD Association Management (Simplified) ---

@router.post("/{character_id}/rpds/{rpd_id}",
             status_code=201,
             summary="Add RPD association to character")
async def add_rpd_to_character(
    character_id: UUID,
    rpd_id: UUID,
    user: Annotated[User, Security(get_current_user)]
):
    """Add an RPD association to a character using direct many-to-many relationship."""
    try:
        # Get Character and RPD
        character = await Character.get_or_none(id=character_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")

        # Check user permissions
        project = await character.project
        if not await project.is_user_can_access(user):
            raise HTTPException(
                status_code=403, detail="User does not have access to this character's project")

        # Import here to avoid circular imports
        from aimage_supervision.models import ReviewPointDefinition
        rpd = await ReviewPointDefinition.get_or_none(id=rpd_id)
        if not rpd:
            raise HTTPException(
                status_code=404, detail="ReviewPointDefinition not found")

        # Add the relationship
        await character.associated_rpds.add(rpd)

        return {"message": "RPD association added successfully"}

    except Exception as e:
        logger.error(f"Error adding RPD to character: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to add RPD association")


@router.delete("/{character_id}/rpds/{rpd_id}",
               status_code=200,
               summary="Remove RPD association from character")
async def remove_rpd_from_character(
    character_id: UUID,
    rpd_id: UUID,
    user: Annotated[User, Security(get_current_user)]
):
    """Remove an RPD association from a character."""
    try:
        # Get Character and RPD
        character = await Character.get_or_none(id=character_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")

        # Check user permissions
        project = await character.project
        if not await project.is_user_can_access(user):
            raise HTTPException(
                status_code=403, detail="User does not have access to this character's project")

        # Import here to avoid circular imports
        from aimage_supervision.models import ReviewPointDefinition
        rpd = await ReviewPointDefinition.get_or_none(id=rpd_id)
        if not rpd:
            raise HTTPException(
                status_code=404, detail="ReviewPointDefinition not found")

        # Remove the relationship
        await character.associated_rpds.remove(rpd)

        return {"message": "RPD association removed successfully"}

    except Exception as e:
        logger.error(f"Error removing RPD from character: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to remove RPD association")


@router.get("/{character_id}/rpds",
            summary="Get RPDs associated with character")
async def get_character_rpds(
    character_id: UUID,
    user: Annotated[User, Security(get_current_user)]
):
    """Get all RPDs associated with a character."""
    try:
        character = await Character.get_or_none(id=character_id).prefetch_related('associated_rpds')
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")

        # Check user permissions
        project = await character.project
        if not await project.is_user_can_access(user):
            raise HTTPException(
                status_code=403, detail="User does not have access to this character's project")

        rpds = await character.associated_rpds.all()

        return {
            "character_id": character_id,
            "rpds": [
                {
                    "id": str(rpd.id),
                    "key": rpd.key,
                    "is_active": rpd.is_active,
                    "created_at": rpd.created_at.isoformat(),
                    "updated_at": rpd.updated_at.isoformat()
                }
                for rpd in rpds
            ]
        }

    except Exception as e:
        logger.error(f"Error getting character RPDs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get RPDs")
