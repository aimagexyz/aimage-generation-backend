import asyncio
import io
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import (APIRouter, BackgroundTasks, Body, Depends, File, Form,
                     HTTPException, Security, UploadFile, status)
from pydantic import BaseModel, Field  # Add this import

from aimage_supervision.clients.aws_s3 import \
    download_image_from_url_or_s3_path
from aimage_supervision.enums import AiReviewMode
from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.models import Character, ReviewPointDefinition, User
from aimage_supervision.schemas import (PromptRewriteRequest,
                                        PromptRewriteResponse)
from aimage_supervision.schemas import \
    ReviewPointDefinition as ReviewPointDefinitionSchema
from aimage_supervision.schemas import (ReviewPointDefinitionCreate,
                                        ReviewPointDefinitionVersionBase,
                                        ReviewPointDefinitionVersionInDB)
from aimage_supervision.services.ai_review_service import \
    analyze_single_image_with_rpd_data
from aimage_supervision.services.appellation_service import (
    get_all_appellations_from_s3, parse_and_upload_appellation_file,
    validate_appellation_data)
from aimage_supervision.services.autofill import (AutofillRequest,
                                                  AutofillResponse,
                                                  get_rpd_title_suggestions)
from aimage_supervision.services.rpd_create_service import (
    RPD_Preprocess, RPD_Preprocess_wrapper, create_description,
    create_new_version_for_review_point, create_review_point_definition,
    delete_review_point_definition, generate_image_description,
    generate_rpd_content, get_review_point_definition,
    list_review_point_definitions, rewrite_prompt_with_ai,
    update_review_point_definition_status)
from aimage_supervision.settings import logger

# --- New Router for Review Point Definitions ---
router = APIRouter(
    prefix="/review-point-definitions",
    tags=["Review Point Definitions"]
)


@router.post(
    "/",
    response_model=ReviewPointDefinitionSchema,
    status_code=status.HTTP_201_CREATED
)
async def create_review_point_definition_endpoint(
    rpd_create: ReviewPointDefinitionCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Security(get_current_user)
):
    print(f"Creating review point definition: {rpd_create}")
    try:
        created_by_identifier = str(
            current_user.id) if current_user else "system"
        rpd = await create_review_point_definition(
            definition_create=rpd_create,
            created_by_id=created_by_identifier
        )
        if rpd_create.key == 'general_ng_review' or rpd_create.key == 'visual_review':
            # 使用asyncio.create_task在线程池中执行，避免阻塞主事件循环
            task = asyncio.create_task(RPD_Preprocess_wrapper(rpd.id))
            # 将任务添加到背景任务集合以防止垃圾回收
            task.add_done_callback(
                lambda t: t.exception() if t.exception() else None)
        return rpd
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # Log the exception e
        # Basic logging
        print(
            f"Unexpected error in create_review_point_definition_endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred creating review point definition.")


@router.get("/", response_model=List[ReviewPointDefinitionSchema])
async def list_review_point_definitions_endpoint(
    project_id: UUID | None = None,
    active_only: bool = True,
):
    return await list_review_point_definitions(
        project_id=project_id,
        active_only=active_only
    )


class AppellationDataResponse(BaseModel):
    """称呼表数据响应模型"""
    success: bool
    message: str
    data: Dict[str, Dict[str, str]] = {}
    characters: List[str] = []
    character_count: int = 0


@router.get(
    "/appellation-data/",
    response_model=AppellationDataResponse,
    status_code=status.HTTP_200_OK,
    summary="Get appellation table data",
    description="Retrieve the complete appellation table data from S3 storage."
)
async def get_appellation_data_endpoint(
    s3_url: str,
    current_user: User = Security(get_current_user)
):
    """
    获取称呼表数据

    Args:
        s3_url: 称呼表文件的S3 URL
        current_user: 当前用户

    Returns:
        AppellationDataResponse: 包含完整称呼表数据的响应
    """
    try:
        # 验证S3 URL格式
        if not s3_url.startswith('s3://'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无效的S3 URL格式"
            )

        # 从S3获取数据
        appellation_data = await get_all_appellations_from_s3(s3_url)

        if appellation_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="称呼表文件不存在或无法访问"
            )

        characters = list(appellation_data.keys())

        return AppellationDataResponse(
            success=True,
            message=f"成功获取称呼表数据，包含 {len(characters)} 个角色",
            data=appellation_data,
            characters=characters,
            character_count=len(characters)
        )

    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        # 记录和处理意外错误
        logger.error(f"Unexpected error in get_appellation_data_endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取称呼表数据时发生意外错误"
        )


@router.get("/{rpd_id}/", response_model=ReviewPointDefinitionSchema)
async def get_review_point_definition_endpoint(
    rpd_id: UUID,
):
    rpd = await get_review_point_definition(rpd_id)
    if not rpd:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Review Point Definition not found")
    return rpd


@router.delete(
    "/{rpd_id}/",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a Review Point Definition",
    description="Marks a Review Point Definition as deleted by setting its `is_deleted` flag to true. "
    "This makes it inaccessible through the API but preserves all data in the database, "
    "including all versions and generated AI review findings. This action is reversible in the database."
)
async def delete_review_point_definition_endpoint(
    rpd_id: UUID,
    current_user: User = Security(get_current_user)
):
    """
    Soft-delete a Review Point Definition by ID.

    This operation marks the RPD as 'deleted' but does not permanently remove it
    or any of its associated data from the database.
    """
    try:
        deleted = await delete_review_point_definition(rpd_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review Point Definition not found or already deleted"
            )
        # Return 204 No Content on successful deletion (no response body)
        return
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        # Log the exception for debugging
        print(
            f"Unexpected error in delete_review_point_definition_endpoint for RPD {rpd_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting the review point definition.")


@router.post(
    "/{rpd_id}/versions/",
    response_model=ReviewPointDefinitionVersionInDB,
    status_code=status.HTTP_201_CREATED
)
async def create_new_version_for_review_point_endpoint(
    rpd_id: UUID,
    version_create: ReviewPointDefinitionVersionBase,
    background_task: BackgroundTasks,
    current_user: User = Security(get_current_user)
):
    try:
        created_by_identifier = str(
            current_user.id) if current_user else "system"

        # 获取RPD的key以判断是否需要预处理
        rpd = await ReviewPointDefinition.get_or_none(id=rpd_id)
        if not rpd:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review Point Definition not found"
            )

        new_version = await create_new_version_for_review_point(
            rpd_id=rpd_id,
            version_create=version_create,
            created_by_id=created_by_identifier
        )

        # 根据RPD的key判断是否需要预处理
        if rpd.key == 'general_ng_review' or rpd.key == 'visual_review':
            # 使用asyncio.create_task在线程池中执行，避免阻塞主事件循环
            task = asyncio.create_task(RPD_Preprocess_wrapper(rpd_id))
            # 将任务添加到背景任务集合以防止垃圾回收
            task.add_done_callback(
                lambda t: t.exception() if t.exception() else None)
        return new_version
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # Basic logging
        print(
            f"Unexpected error in create_new_version_for_review_point_endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred creating new version.")


class RPDStatusUpdate(BaseModel):
    is_active: bool


@router.patch("/{rpd_id}/status/",
              response_model=ReviewPointDefinitionSchema)
async def update_rpd_status_endpoint(
    rpd_id: UUID,
    status_update: RPDStatusUpdate = Body(...),
    current_user: User = Security(get_current_user)
):
    try:
        updated_rpd = await update_review_point_definition_status(
            rpd_id=rpd_id,
            is_active=status_update.is_active
        )
        if not updated_rpd:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review Point Definition not found during status update")
        return updated_rpd
    except Exception as e:
        # Basic logging
        print(f"Unexpected error in update_rpd_status_endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred updating RPD status.")


class GenerateDescriptionRequest(BaseModel):
    tag: str
    image_url: str


class GenerateDescriptionResponse(BaseModel):
    eng_visual_characteristics: str
    eng_key_considerations: str
    jpn_visual_characteristics: str
    jpn_key_considerations: str


@router.post(
    "/generate-description/",
    response_model=GenerateDescriptionResponse,
    status_code=status.HTTP_200_OK
)
async def generate_description_endpoint(
    request: GenerateDescriptionRequest,
    current_user: User = Security(get_current_user)
):
    """
    根据图片和标签生成AI描述

    Args:
        request: 包含tag和image_url的请求体
        current_user: 当前用户

    Returns:
        GenerateDescriptionResponse: 生成的英日文描述
    """
    try:
        # 下载图片
        image_bytesio = await download_image_from_url_or_s3_path(request.image_url)
        image_bytes = image_bytesio.read()

        # 调用生成描述函数
        description = create_description(image_bytes, request.tag)

        return GenerateDescriptionResponse(
            eng_visual_characteristics=description.eng_visual_characteristics,
            eng_key_considerations=description.eng_key_considerations,
            jpn_visual_characteristics=description.jpn_visual_characteristics,
            jpn_key_considerations=description.jpn_key_considerations
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"参数错误: {str(e)}"
        )
    except Exception as e:
        print(f"生成描述时发生错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="生成描述时发生内部错误"
        )


@router.post(
    "/rewrite-prompt",
    response_model=PromptRewriteResponse,
    status_code=status.HTTP_200_OK,
    summary="Rewrite user prompt with AI based on image",
    description="Uses AI to transform a simple user prompt into a more complete, professional, and effective prompt. "
    "This endpoint analyzes the image content from the specified subtask to create a detailed and context-aware prompt."
)
async def rewrite_prompt_endpoint(
    request: PromptRewriteRequest,
):
    """
    使用AI基于图片内容将用户输入的简单prompt转写成更完整、专业的prompt

    此功能可以帮助用户：
    - 基于图片内容分析用户的需求
    - 将简单的想法转换为结构化的prompt
    - 添加必要的细节和上下文
    - 提高prompt的清晰度和有效性

    Args:
        request: 包含原始prompt、子任务ID和语言选项的请求体
        current_user: 当前用户（用于认证）

    Returns:
        PromptRewriteResponse: 包含转写结果和置信度的响应
    """

    try:
        # 验证输入
        if not request.original_prompt.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="原始prompt不能为空"
            )

        # 调用AI服务进行prompt转写
        rewrite_result = await rewrite_prompt_with_ai(
            original_prompt=request.original_prompt,
            rpd_type=request.rpd_type,
            context=request.context,
        )
        # 构建响应
        response = PromptRewriteResponse(
            original_prompt=request.original_prompt,
            rewritten_prompt=rewrite_result.rewritten_prompt,
            rewritten_prompt_jpn=rewrite_result.rewritten_prompt_jpn,
            confidence=rewrite_result.confidence,
        )

        return response

    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except ValueError as e:
        # 处理服务层的业务逻辑错误
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # 记录和处理意外错误
        print(f"Unexpected error in prompt rewrite endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="处理prompt转写请求时发生意外错误"
        )


# --- Generate Complete RPD Content ---

class GenerateRPDContentRequest(BaseModel):
    """生成完整RPD内容的请求模型"""
    user_input: str
    image_url: Optional[str] = None  # Optional image URL
    context: Optional[str] = None    # Optional context


class GenerateRPDContentResponse(BaseModel):
    """生成完整RPD内容的响应模型"""
    title: str
    description_for_ai: str
    description_for_ai_jpn: str
    suggested_tag: str


@router.post(
    "/generate-rpd-content",
    response_model=GenerateRPDContentResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate complete RPD content with AI",
    description="Uses AI to generate complete RPD content including key, title, descriptions, and tags based on user input and optional image."
)
async def generate_rpd_content_endpoint(
    request: GenerateRPDContentRequest,
):
    """
    基于用户输入和可选图片生成完整的RPD内容

    此功能可以帮助用户：
    - 自动推荐合适的RPD类型key
    - 生成简洁的标题
    - 生成完整的中英日文AI描述
    - 推荐相关标签
    - 提供生成质量的置信度评分

    Args:
        request: 包含用户输入、图片URL和上下文的请求体

    Returns:
        GenerateRPDContentResponse: 包含完整RPD内容的响应
    """
    try:
        # 验证输入
        if not request.user_input.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户输入不能为空"
            )

        # 下载图片（如果提供）
        image_bytes = None
        if request.image_url:
            try:
                image_bytesio = await download_image_from_url_or_s3_path(
                    request.image_url)
                image_bytes = image_bytesio.read() if image_bytesio else None
            except Exception as e:
                print(f"Failed to download image: {e}")
                # 继续执行，不让图片下载失败阻止整个流程

        # 调用AI服务生成完整RPD内容
        rpd_result = await generate_rpd_content(
            user_input=request.user_input,
            image_bytes=image_bytes,
            context=request.context,
        )

        # 构建响应
        response = GenerateRPDContentResponse(
            title=rpd_result.title,
            description_for_ai=rpd_result.description_for_ai,
            description_for_ai_jpn=rpd_result.description_for_ai_jpn,
            suggested_tag=rpd_result.suggested_tag,
        )

        return response

    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except ValueError as e:
        # 处理服务层的业务逻辑错误
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # 记录和处理意外错误
        print(f"Unexpected error in generate RPD content endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="生成RPD内容时发生意外错误"
        )


# --- Generate Image Description ---

class GenerateImageDescriptionRequest(BaseModel):
    """生成图片详细描述的请求模型"""
    image_url: str
    context: Optional[str] = None  # Optional context
    # RPD type for generating appropriate description
    rpd_type: str = "general_ng_review"
    rpd_title: str = ""  # RPD title for context


class GenerateImageDescriptionResponse(BaseModel):
    """生成图片详细描述的响应模型"""
    detailed_description: str
    key_elements: List[str]
    style_analysis: str
    suggested_keywords: List[str]
    confidence: float


@router.post(
    "/generate-image-description",
    response_model=GenerateImageDescriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate detailed image description with AI",
    description="Uses AI to generate detailed image description including visual analysis, key elements, style analysis, and suggested keywords."
)
async def generate_image_description_endpoint(
    request: GenerateImageDescriptionRequest,
):
    """
    基于图片和上下文信息生成详细的图片描述

    此功能可以帮助用户：
    - 生成详细的日文图片描述
    - 识别图片中的关键元素
    - 进行风格和技术分析
    - 提供相关关键词建议
    - 提供分析质量的置信度评分

    Args:
        request: 包含图片URL、上下文和类型的请求体

    Returns:
        GenerateImageDescriptionResponse: 包含详细图片描述的响应
    """
    try:
        # 验证输入
        if not request.image_url.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="图片URL不能为空"
            )

        # 下载图片
        try:
            image_bytesio = await download_image_from_url_or_s3_path(
                request.image_url)
            if not image_bytesio:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="无法下载图片"
                )
            image_bytes = image_bytesio.read()
        except Exception as e:
            print(f"Failed to download image: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"图片下载失败: {str(e)}"
            )

        # 调用AI服务生成图片描述
        description_result = await generate_image_description(
            image_bytes=image_bytes,
            context=request.context,
            rpd_type=request.rpd_type,
            rpd_title=request.rpd_title,
        )

        # 构建响应
        response = GenerateImageDescriptionResponse(
            detailed_description=description_result.detailed_description,
            key_elements=description_result.key_elements,
            style_analysis=description_result.style_analysis,
            suggested_keywords=description_result.suggested_keywords,
            confidence=description_result.confidence,
        )

        return response

    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except ValueError as e:
        # 处理服务层的业务逻辑错误
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # 记录和处理意外错误
        print(f"Unexpected error in generate image description endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="生成图片描述时发生意外错误"
        )


@router.post('/autofill-title', response_model=AutofillResponse)
async def autofill_rpd_title(
    request: AutofillRequest
):
    """
    RPDタイトルの智能补全API

    ユーザーの入力に基づいてGemini AIが生成した
    RPDタイトルの補完候補を3-5個返します
    """
    try:
        # 调用智能补全服务
        result = await get_rpd_title_suggestions(
            user_input=request.user_input,
            context=request.context,
            rpd_type=request.rpd_type,
            max_suggestions=request.max_suggestions
        )

        return result

    except Exception as e:
        logger.error(f"Error in autofill_rpd_title: {str(e)}")
        return AutofillResponse(
            suggestions=[],
            original_input=request.user_input,
            success=False,
            error_message=f"智能补全服务错误: {str(e)}"
        )


# --- Test RPD with Single Image ---

class RPDVersionTestData(BaseModel):
    """用于测试的RPD Version数据"""
    title: str = Field(..., description="RPD标题")
    parent_key: str = Field(
        ..., description="RPD类型: general_ng_review, copyright_review, visual_review")
    description_for_ai: Optional[str] = Field(
        None, description="AI描述（用于copyright_review和visual_review）")
    # eng_description_for_ai: Deprecated, using description_for_ai instead
    tag_list: Optional[List[str]] = Field(
        default=[], description="标签列表（用于general_ng_review）")
    reference_images: Optional[List[str]] = Field(
        default=[], description="参考图片S3路径列表（用于visual_review）")


class ImageRPDTestRequest(BaseModel):
    """单张图片RPD测试的请求模型"""
    rpd_version_data: RPDVersionTestData = Field(
        ..., description="要测试的RPD版本数据")
    cr_check: bool = False
    mode: AiReviewMode = AiReviewMode.QUALITY


class ImageRPDTestResponse(BaseModel):
    """单张图片RPD测试的响应模型"""
    success: bool
    message: str
    findings_count: int
    findings: List[dict] = []  # 简化的findings列表
    processing_time_seconds: Optional[float] = None
    rpd_title: str = ""
    rpd_type: str = ""


@router.post(
    "/test-rpd-with-image",
    response_model=ImageRPDTestResponse,
    status_code=status.HTTP_200_OK,
    summary="Test RPD analysis with a single image",
    description="Directly analyze a single uploaded image with a specific RPD version data for testing purposes. "
    "This endpoint is designed for RPD creation and testing workflows, providing immediate feedback "
    "without creating permanent tasks or subtasks."
)
async def test_rpd_with_image_endpoint(
    image_file: UploadFile = File(..., description="要分析的图片文件"),
    rpd_title: str = Form(..., description="RPD标题"),
    rpd_parent_key: str = Form(
        ..., description="RPD类型: general_ng_review, copyright_review, visual_review"),
    rpd_description_for_ai: Optional[str] = Form(
        None, description="AI描述（copyright_review和visual_review用）"),
    # rpd_eng_description_for_ai: Deprecated, using rpd_description_for_ai instead
    rpd_tag_list: Optional[str] = Form(
        None, description="标签列表，用逗号分隔（general_ng_review用）"),
    rpd_reference_images: Optional[str] = Form(
        None, description="参考图片S3路径，用逗号分隔（visual_review用）"),
    cr_check: bool = Form(False, description="是否启用CR检查"),
    mode: str = Form("quality", description="分析模式: quality 或 speed"),
    current_user: User = Security(get_current_user)
):
    """
    直接对单张图片进行单个RPD测试分析

    此功能专为RPD创建和测试流程设计，特点：
    - 直接上传图片进行分析，无需创建task/subtask
    - 支持传入完整的RPD version数据进行测试
    - 快速返回分析结果
    - 不在数据库中创建永久记录
    - 适合RPD开发和调试使用

    Args:
        image_file: 要分析的图片文件
        rpd_title: RPD标题
        rpd_parent_key: RPD类型
        rpd_description_for_ai: AI描述
        # rpd_eng_description_for_ai: Deprecated
        rpd_tag_list: 标签列表（逗号分隔）
        rpd_reference_images: 参考图片路径（逗号分隔）
        cr_check: 是否启用CR检查
        mode: 分析模式（quality/speed）
        current_user: 当前用户

    Returns:
        ImageRPDTestResponse: 包含分析结果的响应
    """
    import time
    start_time = time.time()

    try:
        # 1. 验证图片文件
        if not image_file.content_type or not image_file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无效的文件类型。请上传图片文件。"
            )

        # 2. 读取图片内容
        image_contents = await image_file.read()
        if len(image_contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="图片文件为空"
            )

        # 3. 验证并构建RPD version数据（已在前端验证）

        # 解析标签列表
        parsed_tag_list = []
        if rpd_tag_list and rpd_tag_list.strip():
            parsed_tag_list = [tag.strip()
                               for tag in rpd_tag_list.split(",") if tag.strip()]

        # 解析参考图片列表
        parsed_reference_images = []
        if rpd_reference_images and rpd_reference_images.strip():
            parsed_reference_images = [
                img.strip() for img in rpd_reference_images.split(",") if img.strip()]

        # 4. 验证分析模式
        try:
            analysis_mode = AiReviewMode(mode)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的分析模式: {mode}。支持的模式: quality, speed"
            )

        # 5. 构建RPD version数据
        rpd_version_data = RPDVersionTestData(
            title=rpd_title,
            parent_key=rpd_parent_key,
            description_for_ai=rpd_description_for_ai,
            # eng_description_for_ai: Deprecated, using description_for_ai instead
            tag_list=parsed_tag_list,
            reference_images=parsed_reference_images
        )

        # 6. 调用AI分析服务进行图片分析
        analysis_result = await analyze_single_image_with_rpd_data(
            image_bytes=image_contents,
            image_filename=image_file.filename or "test_image",
            rpd_version_data=rpd_version_data.dict(),  # 转换为字典
            cr_check=cr_check,
            mode=analysis_mode,
            user_id=current_user.id
        )

        # 7. 计算处理时间
        processing_time = time.time() - start_time

        # 8. 构建响应
        simplified_findings = []
        if analysis_result.get("findings"):
            for finding in analysis_result["findings"]:
                simplified_findings.append({
                    "rpd_key": finding.get("rpd_key", "unknown"),
                    "description": finding.get("description", ""),
                    "severity": finding.get("severity", ""),
                    "suggestion": finding.get("suggestion", ""),
                    "confidence": finding.get("confidence", 0.0),
                    "tag": finding.get("tag", "")
                })

        return ImageRPDTestResponse(
            success=True,
            message=f"RPD测试完成，发现 {len(simplified_findings)} 个问题点",
            findings_count=len(simplified_findings),
            findings=simplified_findings,
            processing_time_seconds=round(processing_time, 2),
            rpd_title=rpd_title,
            rpd_type=rpd_parent_key
        )

    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        # 记录和处理意外错误
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error in test_rpd_with_image_endpoint: {e}")

        return ImageRPDTestResponse(
            success=False,
            message=f"RPD测试失败: {str(e)}",
            findings_count=0,
            findings=[],
            processing_time_seconds=round(processing_time, 2),
            rpd_title=rpd_title,
            rpd_type=rpd_parent_key
        )


# --- 称呼表相关API ---

class AppellationUploadResponse(BaseModel):
    """称呼表上传响应模型"""
    success: bool
    message: str
    s3_url: str = ""
    file_type: str = ""
    characters: List[str] = []
    character_count: int = 0
    total_appellations: int = 0
    validation_warnings: List[str] = []


@router.post(
    "/upload-appellation-file/",
    response_model=AppellationUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and parse appellation table file",
    description="Upload an Excel or JSON file containing character appellation mappings. "
    "Excel files will be converted to JSON format and stored in S3. "
    "The file should contain a matrix where rows represent speakers and columns represent targets."
)
async def upload_appellation_file_endpoint(
    file: UploadFile = File(...,
                            description="称呼表文件 (Excel .xlsx/.xls 或 JSON)"),
    project_id: str = Form(..., description="项目ID"),
    session_id: Optional[str] = Form(None, description="会话ID（可选，用于临时文件）"),
    current_user: User = Security(get_current_user)
):
    """
    上传并解析称呼表文件

    此API支持：
    1. Excel格式 (.xlsx, .xls)：
       - 第一行和第一列为角色名称
       - 行表示某个角色被各种角色的称呼
       - 列表示某个角色对其他角色的称呼
       - 对角线上是第一人称

    2. JSON格式：
       - 格式：{"A":{"A":"第一人称","B":"A称呼B",...},"B":{...}...}

    Args:
        file: 称呼表文件
        project_id: 项目ID
        session_id: 可选的会话ID，用于临时文件存储
        current_user: 当前用户

    Returns:
        AppellationUploadResponse: 包含上传结果和解析信息的响应
    """
    try:
        # 验证文件类型
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文件名不能为空"
            )

        filename = file.filename.lower()
        if not (filename.endswith(('.xlsx', '.xls', '.json'))):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不支持的文件类型。仅支持 Excel (.xlsx, .xls) 和 JSON 文件。"
            )

        # 验证文件大小（限制为10MB）
        file_size = 0
        content = await file.read()
        file_size = len(content)

        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="文件大小不能超过10MB"
            )

        # 重新创建UploadFile对象，因为内容已经被读取
        file_copy = UploadFile(
            filename=file.filename,
            file=io.BytesIO(content),
            headers=file.headers
        )

        # 解析并上传文件
        result = await parse_and_upload_appellation_file(
            file=file_copy,
            project_id=project_id,
            session_id=session_id
        )

        # 验证数据
        validation_result = validate_appellation_data(result["data"])

        return AppellationUploadResponse(
            success=True,
            message=f"称呼表文件上传成功。识别到 {len(result['characters'])} 个角色。",
            s3_url=result["s3_url"],
            file_type=result["file_type"],
            characters=result["characters"],
            character_count=len(result["characters"]),
            total_appellations=validation_result["total_appellations"],
            validation_warnings=validation_result.get("warnings", [])
        )

    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except ValueError as e:
        # 处理解析错误
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"文件解析失败: {str(e)}"
        )
    except Exception as e:
        # 记录和处理意外错误
        logger.error(
            f"Unexpected error in upload_appellation_file_endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="上传称呼表文件时发生意外错误"
        )


# --- Text Review 测试相关API ---

class TextRPDTestRequest(BaseModel):
    """文本对话RPD测试的请求模型"""
    dialogue_text: str = Field(..., description="要测试的对话文本")
    rpd_title: str = Field(..., description="RPD标题")
    appellation_file_s3_url: str = Field(..., description="称呼表文件的S3 URL")
    special_rules: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="特殊规则列表")
    project_id: str = Field(..., description="项目ID")


class TextRPDTestResponse(BaseModel):
    """文本对话RPD测试的响应模型"""
    success: bool
    message: str
    analysis: str = ""
    processing_time_seconds: Optional[float] = None
    rpd_title: str = ""
    detected_speaker: Optional[str] = None
    detected_targets: List[str] = []


@router.post(
    "/test-rpd-with-text",
    response_model=TextRPDTestResponse,
    status_code=status.HTTP_200_OK,
    summary="Test text review RPD with dialogue text",
    description="Test a text review RPD by analyzing a single dialogue line. "
    "This endpoint checks character appellations and special rules against the provided dialogue text."
)
async def test_rpd_with_text_endpoint(
    request: TextRPDTestRequest,
    current_user: User = Security(get_current_user)
):
    """
    对单句对话进行text review RPD测试分析

    此功能专为text review RPD的创建和测试流程设计，特点：
    - 直接输入对话文本进行分析，无需创建task/subtask
    - 支持称呼表和特殊规则的检查
    - 快速返回分析结果
    - 不在数据库中创建永久记录
    - 适合RPD开发和调试使用

    Args:
        request: 包含对话文本和RPD配置的请求
        current_user: 当前用户

    Returns:
        TextRPDTestResponse: 包含分析结果的响应
    """
    import time
    start_time = time.time()

    try:
        # 1. 验证输入参数
        if not request.dialogue_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="对话文本不能为空"
            )

        if not request.appellation_file_s3_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="称呼表文件S3 URL不能为空"
            )

        # 2. 从S3加载称呼表数据
        from aimage_supervision.services.appellation_service import \
            get_all_appellations_from_s3

        appellation_data = await get_all_appellations_from_s3(request.appellation_file_s3_url)
        if not appellation_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无法加载称呼表数据，请检查S3 URL是否正确"
            )

        # 3. 调用text review分析服务
        from aimage_supervision.services.ai_review_service import check_speaker

        # 准备数据格式（模拟现有的数据结构）
        prepared_data = {
            'original_index': 1,
            'speaker': '未知',  # 让AI来识别
            'line': request.dialogue_text.strip()
        }

        # 执行检查
        analysis_result = await check_speaker(
            data=prepared_data,
            model_name="gemini-2.5-flash",  # 使用快速模型进行测试
            max_retries=3,
            retry_delay=1.0,
            alias_dict=appellation_data,
            extra_info=request.special_rules or [],
            api_timeout=30.0,
            sheet_index=0
        )

        # 4. 处理分析结果
        processing_time = time.time() - start_time

        # 提取分析结果
        detected_speaker = analysis_result.get('speaker', '未识别')
        detected_targets = analysis_result.get('target', [])

        # 处理称呼检查结果
        if analysis_result.get('status') == "error":
            analysis_text = "❌ エラー: "+analysis_result.get('message', '')
        else:
            analysis_text = "✅ 問題なし"

        return TextRPDTestResponse(
            success=True,
            message="文本分析完成",
            analysis=analysis_text,
            processing_time_seconds=round(processing_time, 2),
            rpd_title=request.rpd_title,
            detected_speaker=detected_speaker,
            detected_targets=detected_targets
        )

    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        # 记录和处理意外错误
        processing_time = time.time() - start_time
        logger.error(f"Unexpected error in test_rpd_with_text_endpoint: {e}")

        return TextRPDTestResponse(
            success=False,
            message=f"文本分析失败: {str(e)}",
            analysis="",
            processing_time_seconds=round(processing_time, 2),
            rpd_title=request.rpd_title,
            detected_speaker=None,
            detected_targets=[]
        )


# --- Character Association Management (Simplified) ---

@router.post("/{rpd_id}/characters/{character_id}",
             status_code=201,
             summary="Add character association to RPD")
async def add_character_to_rpd(
    rpd_id: UUID,
    character_id: UUID,
    _: User = Depends(get_current_user)
):
    """Add a character association to an RPD using direct many-to-many relationship."""
    try:
        # Get RPD and Character
        rpd = await ReviewPointDefinition.get_or_none(id=rpd_id)
        if not rpd:
            raise HTTPException(
                status_code=404, detail="ReviewPointDefinition not found")

        character = await Character.get_or_none(id=character_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")

        # Add the relationship
        await rpd.characters.add(character)

        return {"message": "Character association added successfully"}

    except Exception as e:
        logger.error(f"Error adding character to RPD: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to add character association")


@router.delete("/{rpd_id}/characters/{character_id}",
               status_code=200,
               summary="Remove character association from RPD")
async def remove_character_from_rpd(
    rpd_id: UUID,
    character_id: UUID,
    _: User = Depends(get_current_user)
):
    """Remove a character association from an RPD."""
    try:
        # Get RPD and Character
        rpd = await ReviewPointDefinition.get_or_none(id=rpd_id)
        if not rpd:
            raise HTTPException(
                status_code=404, detail="ReviewPointDefinition not found")

        character = await Character.get_or_none(id=character_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")

        # Remove the relationship
        await rpd.characters.remove(character)

        return {"message": "Character association removed successfully"}

    except Exception as e:
        logger.error(f"Error removing character from RPD: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to remove character association")


@router.get("/{rpd_id}/characters",
            summary="Get characters associated with RPD")
async def get_rpd_characters(
    rpd_id: UUID,
    _: User = Depends(get_current_user)
):
    """Get all characters associated with an RPD."""
    try:
        rpd = await ReviewPointDefinition.get_or_none(id=rpd_id).prefetch_related('characters')
        if not rpd:
            raise HTTPException(
                status_code=404, detail="ReviewPointDefinition not found")

        characters = await rpd.characters.all()

        return {
            "rpd_id": rpd_id,
            "characters": [
                {
                    "id": str(char.id),
                    "name": char.name,
                    "alias": char.alias,
                    "description": char.description
                }
                for char in characters
            ]
        }

    except Exception as e:
        logger.error(f"Error getting RPD characters: {e}")
        raise HTTPException(status_code=500, detail="Failed to get characters")
