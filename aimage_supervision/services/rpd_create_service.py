import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
from uuid import UUID

from fastapi import BackgroundTasks
from google import genai
from pydantic import BaseModel, Field
from tortoise.exceptions import DoesNotExist, IntegrityError
from tortoise.expressions import Q
from tortoise.transactions import in_transaction

from aimage_supervision.clients.aws_s3 import \
    download_file_content_from_s3_sync
from aimage_supervision.models import (ReviewPointDefinition,
                                       ReviewPointDefinitionVersion)
from aimage_supervision.prompts.create_description_prompts import (
    general_prompt_rewrite_prompt, generate_ng_review_prompt,
    generate_visual_review_prompt, prompt_group)
from aimage_supervision.prompts.review_point_prompts import (
    generate_rpd_content_prompt, generate_rpd_type_prompt,
    visual_review_prompt)
from aimage_supervision.prompts.visual_review_multi_agent_prompts import (
    build_constraint_prompt, build_enum_prompt, build_evidence_prompt,
    build_guideline_prompt)
from aimage_supervision.schemas import AiReview as AiReviewSchema
from aimage_supervision.schemas import \
    AiReviewFindingEntry as AiReviewFindingEntrySchema
from aimage_supervision.schemas import \
    ReviewPointDefinition as ReviewPointDefinitionSchema
from aimage_supervision.schemas import (ReviewPointDefinitionCreate,
                                        ReviewPointDefinitionVersionBase,
                                        ReviewPointDefinitionVersionInDB)
from aimage_supervision.settings import logger

# --- RPD Content Generation Models ---


class RPDDescription(BaseModel):
    eng_visual_characteristics: str
    eng_key_considerations: str
    jpn_visual_characteristics: str
    jpn_key_considerations: str


class RPDContentResult(BaseModel):
    """Complete RPD content generation result"""
    title: str = Field(..., description="生成的标题")
    description_for_ai: str = Field(..., description="AI用的完整prompt内容")
    description_for_ai_jpn: str = Field(..., description="AI用的完整prompt内容（日文）")
    suggested_tag: str = Field(default="", description="建议的标签")


class RPDType(BaseModel):
    type: str = Field(..., description="RPD类型")

# --- Image Description Models ---


class ImageDescription(BaseModel):
    """Image description generation result"""
    detailed_description: str = Field(..., description="详细的图片描述（日文）")
    key_elements: List[str] = Field(default=[], description="图片中的关键元素列表")
    style_analysis: str = Field(..., description="风格和技术分析（日文）")
    suggested_keywords: List[str] = Field(default=[], description="建议的关键词标签")
    confidence: float = Field(..., description="描述质量的置信度评分", ge=0.0, le=1.0)


# --- TTA Guideline and Constraint Models ---

@dataclass
class ProcessingConfig:
    """Multi-Agent处理配置"""
    guideline_thinking: int = 8192           # Guideline生成时的thinking预算
    constraint_thinking: int = 0             # Constraint编译的thinking预算
    classify_enum_thinking: int = 0          # 分类枚举的thinking预算
    classify_evidence_thinking_low: int = 0   # 低confidence时的thinking
    classify_evidence_thinking_high: int = 8192  # 高confidence时的thinking
    tta_scales: List[float] = field(default_factory=lambda: [1.0])
    # 0.8, 1.0, 1.2])  # TTA缩放比例
    confidence_threshold: float = 0.8        # confidence阈值


class GuidelineItem(BaseModel):
    category: str
    criteria: str
    reference_features: Optional[str] = None
    expected_result: Optional[str] = None
    # multi-image keys (optional for flexibility)
    common_features: Optional[str] = None
    acceptable_variations: Optional[str] = None


class DetailedAnalysis(BaseModel):
    colors: Optional[List[str]] = None
    shapes: Optional[List[str]] = None
    proportions: Optional[List[str]] = None
    textures: Optional[List[str]] = None


class ComparisonAnalysis(BaseModel):
    consistencies: Optional[List[str]] = None
    variations: Optional[List[str]] = None
    critical_features: Optional[List[str]] = None


class GuidelinesSchema(BaseModel):
    visual_guidelines: List[GuidelineItem] = Field(default_factory=list)
    detailed_analysis: Optional[DetailedAnalysis] = None
    comparison_analysis: Optional[ComparisonAnalysis] = None
    key_checkpoints: Optional[List[str]] = None
    labels: Optional[List[str]] = None


class ConstraintItem(BaseModel):
    constraint_id: str
    category: Optional[str] = None
    check_method: Optional[str] = None
    success_criteria: Optional[str] = None
    failure_indicators: Optional[str] = None

    class SeverityMapping(BaseModel):
        critical: Optional[str] = None
        moderate: Optional[str] = None
        minor: Optional[str] = None

    severity_mapping: Optional[SeverityMapping] = None


class ConstraintsSchema(BaseModel):
    verification_constraints: List[ConstraintItem] = Field(
        default_factory=list)
    priority_order: Optional[List[str]] = None


class GuidelineGeneratorAgent:
    """从参考图片生成视觉检查指南"""

    def __init__(self, gemini_client: genai.Client, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.gemini_client = gemini_client

    async def generate(
        self,
        base_prompt: str,
        reference_images: List[bytes],
        description_for_ai: str,
        rpd_type: str,
        thinking_budget: int = 8192,
        model_name: Optional[str] = None,
    ) -> Optional[Dict]:
        """从参考图片生成视觉检查指南"""
        try:
            if not model_name:
                logger.error("model_name is required for guideline generation")
                return None
            image_count = len(reference_images)
            prompt = build_guideline_prompt(
                rpd_type=rpd_type,
                base_prompt=base_prompt,
                description_for_ai=description_for_ai,
                image_count=image_count,
            )

            if not self.gemini_client:
                logger.error(
                    "GEMINI_API_KEY not set. Cannot generate visual review findings.")
                return None

            image_parts = [
                genai.types.Part.from_bytes(
                    data=img_bytes, mime_type='image/jpeg')
                for img_bytes in reference_images
            ]
            contents = [prompt] + image_parts

            response = self.gemini_client.models.generate_content(
                model=model_name,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_budget=thinking_budget,
                        include_thoughts=False
                    ) if thinking_budget and thinking_budget > 0 else None,
                    response_mime_type='application/json',
                    response_schema=GuidelinesSchema,
                    temperature=0.1,
                    max_output_tokens=8192,
                )
            )

            if response and getattr(response, 'parsed', None):
                parsed = response.parsed
                if hasattr(parsed, 'model_dump'):
                    result = parsed.model_dump()
                elif hasattr(parsed, 'dict'):
                    result = parsed.dict()
                else:
                    result = None
                if result is not None:
                    if rpd_type != 'classification tasks':
                        # 合规任务的labels强制为固定集合，避免模型幻觉
                        result['labels'] = ["safe", "alert", "risk"]
                    return result
            return None

        except Exception as e:
            logger.exception("Guideline generation error")
            return None


class ConstraintCompilerAgent:
    """约束编译器 - 将指南转换为可验证的约束条件"""

    def __init__(self, gemini_client: genai.Client, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.gemini_client = gemini_client

    async def compile(
        self,
        guidelines: Dict,
        rpd_title: str,
        thinking_budget: int = 0,
        base_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        将指南编译为可验证的约束条件

        参数:
            guidelines: 来自GuidelineGenerator的指南
            rpd_title: RPD
            thinking_budget: 思考预算 (没有think必要)

        返回:
            编译后的约束条件
        """
        try:
            if not model_name:
                logger.error(
                    "model_name is required for constraint compilation")
                return None
            logger.info(
                "Starting constraint compilation for RPD: %s", rpd_title)
            prompt = build_constraint_prompt(
                rpd_title=rpd_title,
                guidelines_json_str=json.dumps(
                    guidelines, ensure_ascii=True, indent=2),
            )

            if not self.gemini_client:
                logger.error("Gemini client not initialized")
                return None

            response = self.gemini_client.models.generate_content(
                model=model_name,
                contents=[prompt],
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_budget=thinking_budget,
                        include_thoughts=False
                    ) if thinking_budget and thinking_budget > 0 else None,
                    response_mime_type='application/json',
                    response_schema=ConstraintsSchema,
                    temperature=0.1,
                    max_output_tokens=8192,
                )
            )
            if response and getattr(response, 'parsed', None):
                parsed = response.parsed
                # 转为dict
                if hasattr(parsed, 'model_dump'):
                    result = parsed.model_dump()
                elif hasattr(parsed, 'dict'):
                    result = parsed.dict()
                else:
                    result = None
                if result is not None:
                    logger.info(
                        "JSON parsing successful in constraint compilation")
                    return result
            # 容错：从文本中恢复JSON
            raw_text = getattr(response, 'text', None) if response else None
            if raw_text:
                snippet = raw_text.strip()[:300].replace('\n', ' ')
                logger.info("[constraint] raw text snippet: %s", snippet)
                try:
                    cleaned = raw_text.strip()
                    if cleaned.startswith('```'):
                        cleaned = cleaned.strip('`')
                        cleaned = cleaned.replace('json\n', '')
                    l = cleaned.find('{')
                    r = cleaned.rfind('}')
                    if l != -1 and r != -1 and r > l:
                        candidate = cleaned[l:r+1]
                        data = json.loads(candidate)
                        if isinstance(data, dict) and 'verification_constraints' in data:
                            logger.info(
                                "[constraint] recovered JSON from fallback parsing")
                            return data
                except Exception:
                    pass
            logger.warning(
                "No valid response returned in constraint compilation")
            return None

        except Exception as e:
            logger.exception("Constraint compilation error")
            return None


def create_description(image_bytes: bytes, tag: str):
    """
    Generate description for RPD based on image and tag
    """
    task_gemini_client = genai.Client(api_key=os.getenv(
        'GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None

    if not task_gemini_client:
        raise ValueError(
            "GEMINI_API_KEY not set. Cannot generate description.")

    system_prompt = generate_ng_review_prompt
    # Using a model that supports JSON output and good instruction following. Adjust if needed.
    model_name = "gemini-2.5-flash"

    prompt_parts = [
        f"ng item: {tag}",
        genai.types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
    ]

    response = task_gemini_client.models.generate_content(
        model=model_name,
        contents=prompt_parts,
        config=genai.types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=RPDDescription,
            system_instruction=system_prompt,
            temperature=0.1,
        )
    )

    return response.parsed


# --- Image Description Service ---


async def generate_image_description(
    image_bytes: bytes,
    context: Optional[str] = None,
    rpd_type: str = "general_ng_review",
    rpd_title: str = ""
) -> ImageDescription:
    """
    基于图片和上下文信息生成详细的图片描述

    Args:
        image_bytes: 图片的字节数据
        context: 可选的上下文信息，帮助理解图片用途或背景
        language: 输出语言选择 ("en"=英文, "jpn"=日文, "both"=双语)

    Returns:
        ImageDescription: 包含详细描述和分析的结果对象
    """
    task_gemini_client = genai.Client(api_key=os.getenv(
        'GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None

    if not task_gemini_client:
        print("Error: GEMINI_API_KEY not set. Cannot generate image description.")
        return ImageDescription(
            detailed_description="Unable to generate description - API key not configured",
            key_elements=[],
            style_analysis="Analysis unavailable",
            suggested_keywords=[],
            confidence=0.0
        )

    # 构建上下文信息
    focus_context_group = {
        "general_ng_review": "Focus on the abstract shape of the object in the image instead of the specific details. ignore the background of the image.",
        "copyright_review": "Focus on the copyright information in the image.",
        "visual_review": "Focus on the visual elements and the content of the image. The shape, the relative length, and the relative position of the object or character in the image.",
        "text_review": "Focus on the text content and the content of the image.",
        "settings_review": "Focus on the abstract settings in the image.",
        "design_review": "Focus on the texts, design elements in the image."
    }

    context_info = focus_context_group[rpd_type]

    # 创建系统指令
    system_instruction = """You are an expert image analyst specializing in detailed visual description and analysis. 
Your task is to provide comprehensive, accurate descriptions of images that capture both obvious and subtle details.
The description you generated is used for AI review on images or text. You should change your description style based on the type, title and context information. 
The review type is {type}. 
The title of the RPD is: {title}
The context information is: {context_info}

Provide:
1. A comprehensive description covering all visual elements
2. Key elements/subjects present in the image
3. Style and technical analysis
4. Suggested keywords for categorization
5. Confidence score for the analysis quality

Provide descriptions in Japanese.
Be objective, detailed, and professional.""".format(
        type=rpd_type,
        context_info=focus_context_group[rpd_type],
        title=rpd_title
    )

    # 构建prompt
    prompt_text = f"""Please analyze this image in detail, {context_info}
Be thorough but concise, focusing on the most important visual aspects."""

    prompt_parts = [
        prompt_text,
        genai.types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
    ]

    try:
        response = task_gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_parts,
            config=genai.types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=ImageDescription,
                system_instruction=system_instruction,
                temperature=0.2,
            )
        )

        result = response.parsed
        if not isinstance(result, ImageDescription):
            # Fallback response
            return ImageDescription(
                detailed_description="This image contains visual content that requires analysis.",
                key_elements=["visual content"],
                style_analysis="Style analysis not available",
                suggested_keywords=["image", "visual"],
                confidence=0.2
            )

        return result

    except Exception as e:
        print(f"Error in image description generation: {e}")
        return ImageDescription(
            detailed_description=f"Error occurred during image analysis: {str(e)}",
            key_elements=[],
            style_analysis="Analysis failed due to error",
            suggested_keywords=[],
            confidence=0.1
        )


# --- Prompt Rewrite Service Functions ---


class PromptRewriteResult(BaseModel):
    """Internal model for prompt rewrite results"""
    rewritten_prompt: str = Field(..., description="转写后的完整prompt")
    rewritten_prompt_jpn: str = Field(..., description="转写后的完整prompt（日文）")
    confidence: float = Field(..., description="置信度评分", ge=0.0, le=1.0)


async def rewrite_prompt_with_ai(
    original_prompt: str,
    rpd_type: str,
    context: Optional[str] = None,
) -> PromptRewriteResult:
    """
    使用Gemini AI将用户输入的简单prompt转写成更完整、更专业的prompt

    Args:
        original_prompt: 用户输入的原始prompt
        image_bytes: 可选的图片数据
        context: 可选的上下文信息

    Returns:
        PromptRewriteResult: 包含转写结果和改进信息的结果对象
    """
    task_gemini_client = genai.Client(api_key=os.getenv(
        'GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None
    if not task_gemini_client:
        print("Error: GEMINI_API_KEY not set. Cannot rewrite prompt.")
        return PromptRewriteResult(
            rewritten_prompt=original_prompt,
            rewritten_prompt_jpn=original_prompt,
            confidence=0.3
        )

    # 构建上下文信息
    context_info = prompt_group[rpd_type]

    system_instruction = general_prompt_rewrite_prompt.format(
        type=rpd_type,
        context_info=context_info
    )

    prompt_parts = ['original_prompt: ' + original_prompt
                    ]

    try:
        response = task_gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_parts,
            config=genai.types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=PromptRewriteResult,
                system_instruction=system_instruction,
                temperature=0.3,  # 稍微增加一些创造性
            )
        )

        result = response.parsed
        if not isinstance(result, PromptRewriteResult):
            return PromptRewriteResult(
                rewritten_prompt=f"[Rewritten version] {original_prompt}",
                rewritten_prompt_jpn=f"[Rewritten version] {original_prompt}",
                confidence=0.5
            )

        return result

    except Exception as e:
        # 返回一个安全的fallback结果
        print(f"Error in prompt rewriting: {e}")
        return PromptRewriteResult(
            rewritten_prompt=f"[Rewritten version] {original_prompt}",
            rewritten_prompt_jpn=f"[Rewritten version] {original_prompt}",
            confidence=0.3
        )


# --- RPD Content Generation Service ---

async def generate_rpd_content(
    user_input: str,
    image_bytes: Optional[bytes] = None,
    context: Optional[str] = None,
) -> RPDContentResult:
    """
    基于用户输入和图片生成完整的RPD内容，包括推荐的key、title、prompt内容和tags

    Args:
        user_input: 用户输入的简单描述
        image_bytes: 可选的图片数据
        context: 可选的上下文信息

    Returns:
        RPDContentResult: 包含完整RPD内容的结果对象
    """
    task_gemini_client = genai.Client(api_key=os.getenv(
        'GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None

    if not task_gemini_client:
        print("Error: GEMINI_API_KEY not set. Cannot generate RPD content.")
        return RPDContentResult(
            title=user_input[:20] +
            "..." if len(user_input) > 20 else user_input,
            description_for_ai=f"[Generated from user input] {user_input}",
            description_for_ai_jpn=f"[Generated from user input] {user_input}",
            suggested_tag='',
        )

    # 构建prompt
    context_info = f"\n\nContext: {context}" if context else ""

    prompt_text = f"Here is the user's input: {user_input}{context_info}"

    if image_bytes:
        prompt_parts = [
            prompt_text,
            genai.types.Part.from_bytes(
                data=image_bytes, mime_type='image/jpeg')
        ]
    else:
        prompt_parts = [prompt_text]

    system_instruction = generate_rpd_type_prompt

    try:
        response = task_gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_parts,
            config=genai.types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=RPDType,
                system_instruction=system_instruction,
                temperature=0.2,  # 保持相对稳定的输出
            )
        )

        result = response.parsed
        if not isinstance(result, RPDType):
            review_type = "general_ng_review"
        else:
            review_type = result.type
    except Exception as e:
        print(f"Error in RPD type generation: {e}")
        review_type = "general_ng_review"

    if review_type not in prompt_group:
        return RPDContentResult(
            title=user_input[:47] +
            "..." if len(user_input) > 50 else user_input,
            description_for_ai=f"Review items related to: {user_input}. Check for compliance with established guidelines and standards.",
            description_for_ai_jpn=f"{user_input}に関する項目を確認してください。確立されたガイドラインと基準への準拠をチェックしてください。",
            suggested_tag=''
        )
    system_instruction = generate_rpd_content_prompt.format(
        type_prompt=prompt_group[review_type])

    prompt_parts += [f"The type of RPD is: {review_type}"]

    try:
        response = task_gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_parts,
            config=genai.types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=RPDContentResult,
                system_instruction=system_instruction,
                temperature=0.2,  # 保持相对稳定的输出
            )
        )

        result = response.parsed
        if not isinstance(result, RPDContentResult):
            # Fallback response
            return RPDContentResult(
                title=user_input[:47] +
                "..." if len(user_input) > 50 else user_input,
                description_for_ai=f"Review items related to: {user_input}. Check for compliance with established guidelines and standards.",
                description_for_ai_jpn=f"{user_input}に関する項目を確認してください。確立されたガイドラインと基準への準拠をチェックしてください。",
                suggested_tag=''
            )

        return result

    except Exception as e:
        print(f"Error in RPD content generation: {e}")
        return RPDContentResult(
            title=user_input[:47] +
            "..." if len(user_input) > 50 else user_input,
            description_for_ai=f"Review items related to: {user_input}. Check for compliance with established guidelines and standards.",
            description_for_ai_jpn=f"{user_input}に関する項目を確認してください。確立されたガイドラインと基準への準拠をチェックしてください。",
            suggested_tag=''
        )


async def _map_rpd_model_to_schema(rpd_model: ReviewPointDefinition) -> ReviewPointDefinitionSchema:
    """Helper to map ORM model to Pydantic schema, fetching current active version."""
    current_active_version_model = await ReviewPointDefinitionVersion.filter(
        review_point_definition=rpd_model,
        is_active_version=True
    ).first()

    versions_in_db = await rpd_model.versions.all()
    versions_schema = []
    for v in versions_in_db:
        # 为旧数据提供fallback逻辑
        version_dict = {
            'id': v.id,
            'review_point_definition_id': rpd_model.id,
            'version_number': v.version_number,
            'title': v.title,
            'description_for_ai': v.description_for_ai,
            'is_active_version': v.is_active_version,
            'created_at': v.created_at,
            'created_by': v.created_by,
            'reference_images': v.reference_images or [],
            'tag_list': v.tag_list or [],
            'reference_files': getattr(v, 'reference_files', []),
            'special_rules': getattr(v, 'special_rules', []),
            'ng_subcategory': getattr(v, 'ng_subcategory', None),
            # 为旧数据提供fallback：如果没有user_instruction，使用description_for_ai
            'user_instruction': getattr(v, 'user_instruction', None) or v.description_for_ai
        }
        versions_schema.append(
            ReviewPointDefinitionVersionInDB.model_validate(version_dict))

    rpd_schema = ReviewPointDefinitionSchema(
        id=rpd_model.id,
        created_at=rpd_model.created_at,
        updated_at=rpd_model.updated_at,
        key=cast(Literal['general_ng_review', 'visual_review', 'settings_review',
                 'design_review', 'text_review', 'copyright_review'], rpd_model.key),
        is_active=rpd_model.is_active,
        versions=versions_schema,
        current_version=ReviewPointDefinitionVersionInDB.from_orm(
            current_active_version_model) if current_active_version_model else None,
        current_version_num=current_active_version_model.version_number if current_active_version_model else None
    )
    return rpd_schema


async def create_review_point_definition(
    definition_create: ReviewPointDefinitionCreate,
    created_by_id: Optional[str] = "system"  # User ID or system identifier
) -> ReviewPointDefinitionSchema:
    """
    Creates a ReviewPointDefinition and its initial version.
    """
    async with in_transaction():
        try:
            print(f"Creating review point definition: {definition_create}")
            # Verify project exists
            from aimage_supervision.models import Project
            project = await Project.get(id=definition_create.project_id)

            description_for_ai = definition_create.user_instruction

            tag_list = definition_create.tag_list or []

            rpd = await ReviewPointDefinition.create(
                key=definition_create.key,
                is_active=definition_create.is_active,
                project=project
            )

            initial_version = await ReviewPointDefinitionVersion.create(
                review_point_definition=rpd,
                version_number=1,
                title=definition_create.title,
                is_ready_for_ai_review=False,
                user_instruction=definition_create.user_instruction,
                description_for_ai=description_for_ai,
                ng_subcategory=definition_create.ng_subcategory,
                is_active_version=True,
                created_by=created_by_id,
                reference_images=definition_create.reference_images,
                tag_list=tag_list,
                reference_files=definition_create.reference_files or [],
                special_rules=definition_create.special_rules
            )
        except IntegrityError as e:
            # Handle cases like database constraint violations
            raise ValueError(f"Could not create review point definition: {e}")
        except DoesNotExist:
            raise ValueError(
                f"Project with id {definition_create.project_id} not found")

        return await _map_rpd_model_to_schema(rpd)


async def get_review_point_definition(rpd_id: UUID) -> Optional[ReviewPointDefinitionSchema]:
    """
    Fetches a single ReviewPointDefinition by its ID, including its current active version.
    It will not return soft-deleted RPDs.
    """
    try:
        rpd_model = await ReviewPointDefinition.get(id=rpd_id, is_deleted=False)
        return await _map_rpd_model_to_schema(rpd_model)
    except DoesNotExist:
        return None


async def list_review_point_definitions(
    active_only: bool = True,
    project_id: Optional[UUID] = None
) -> List[ReviewPointDefinitionSchema]:
    """
    Lists ReviewPointDefinitions, optionally filtering for active ones.
    It does not include soft-deleted RPDs.
    """
    query = ReviewPointDefinition.filter(is_deleted=False)
    if active_only:
        query = query.filter(is_active=True)
    if project_id:
        query = query.filter(project_id=project_id)

    rpd_models = await query.all()

    # Map all models to schemas concurrently
    rpd_schemas = await asyncio.gather(
        *[_map_rpd_model_to_schema(rpd) for rpd in rpd_models]
    )
    return rpd_schemas


async def create_new_version_for_review_point(
    rpd_id: UUID,
    version_create: ReviewPointDefinitionVersionBase,
    created_by_id: Optional[str] = "system"
) -> ReviewPointDefinitionVersionInDB:
    """
    Creates a new version for an existing ReviewPointDefinition.
    Sets the new version as active and deactivates previous active versions for this RPD.
    """
    async with in_transaction():
        try:
            rpd = await ReviewPointDefinition.get(id=rpd_id)
        except DoesNotExist:
            raise ValueError("ReviewPointDefinition not found.")

        # Deactivate previous active versions for this RPD
        await ReviewPointDefinitionVersion.filter(
            review_point_definition=rpd,
            is_active_version=True
        ).update(is_active_version=False)

        # Determine the new version number
        last_version = await ReviewPointDefinitionVersion.filter(
            review_point_definition=rpd
        ).order_by('-version_number').first()
        new_version_number = (
            last_version.version_number + 1) if last_version else 1

        # Process user instruction to create description_for_ai
        description_for_ai = version_create.user_instruction

        new_version = await ReviewPointDefinitionVersion.create(
            review_point_definition=rpd,
            version_number=new_version_number,
            title=version_create.title,
            is_ready_for_ai_review=False,
            user_instruction=version_create.user_instruction,
            description_for_ai=description_for_ai,
            ng_subcategory=version_create.ng_subcategory,
            is_active_version=True,  # New version is active
            created_by=created_by_id,
            reference_images=version_create.reference_images,
            tag_list=version_create.tag_list,
            reference_files=version_create.reference_files or [],
            special_rules=version_create.special_rules
        )
        rpd.updated_at = new_version.created_at  # Touch the parent RPD
        await rpd.save()

        return ReviewPointDefinitionVersionInDB.from_orm(new_version)


async def update_review_point_definition_status(
    rpd_id: UUID,
    is_active: bool
) -> Optional[ReviewPointDefinitionSchema]:
    """
    Updates the is_active status of a ReviewPointDefinition.
    """
    async with in_transaction():
        try:
            rpd = await ReviewPointDefinition.get(id=rpd_id)
            rpd.is_active = is_active
            await rpd.save()
            return await _map_rpd_model_to_schema(rpd)
        except DoesNotExist:
            return None


async def delete_review_point_definition(rpd_id: UUID) -> bool:
    """
    Soft-deletes a ReviewPointDefinition by setting its `is_deleted` flag to True.
    Also removes associated records from relationship tables.
    Returns True if soft-deletion was successful, False if RPD was not found.
    """
    async with in_transaction():
        try:
            rpd = await ReviewPointDefinition.get(id=rpd_id, is_deleted=False)

            # 删除review_sets_rpds关联表中的记录
            # 由于使用软删除而不是物理删除，需要手动清理这些关联
            # 使用 Tortoise ORM 的方法来清理关联，避免原始 SQL 的数据库兼容性问题
            from aimage_supervision.models import ReviewSet
            review_sets_with_rpd = await ReviewSet.filter(rpds=rpd_id).all()
            for review_set in review_sets_with_rpd:
                await review_set.rpds.remove(rpd)

            # 清除角色关联（新的简化多对多关系会自动清理）
            await rpd.characters.clear()

            # 执行软删除
            rpd.is_deleted = True
            await rpd.save()

            # logger.info(
            #     f"Successfully soft-deleted RPD {rpd_id} and cleaned up associations")
            return True
        except DoesNotExist:
            # logger.warning(
            #     f"RPD {rpd_id} not found for soft deletion or already deleted")
            return False
        except Exception as e:
            # logger.error(f"Unexpected error soft-deleting RPD {rpd_id}: {e}")
            raise


def split_review_prompt(review_prompt: str) -> Tuple[Any, Any]:
    gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    class RewritePrompt(BaseModel):
        prompt1: str
        prompt2: str

    system_prompt = """I have a ng review prompt, I want to split it into 2 prompts, one is for the detection of the specific ng elements, another is for assess the detection in the first step and control the output sensitivity of the ng detection.
    Please split and rewrite the prompt into 2 prompts, and return the 2 prompts in the following format, Please return in Japanese:
    {
        "prompt1": "prompt1",
        "prompt2": "prompt2"
    }
    
    example:
    input:

    Cruel expression
    ● Killing people or Pokemon, cutting up with bleeding, abuse Pokemon without any reason or background
    ● Humans eat Pokemon as living creatures
    ○ (1) Excludes secretions such as Yadon's tail, which regenerates immediately and clearly does not cause Pokemon pain, or Mitsu Honey's Mitsu.
    ○ (2) It is acceptable when expressed as a text as a matter of the "past" such as "humans were eating Pokemon in the past."
    ● Knifes, military-related expressions, products, expressions and products that lead to anti-ethical actions
    ○ (To express things that could threaten life or lead to bullying together with Pokemon)
    
    output:
    prompt1:
    请仔细分析这张图片，根据以下暴力元素定义判断图片中是否包含暴力内容：

    Cruel expression
    ● Killing people or Pokemon, cutting up with bleeding, abuse Pokemon without any reason or background
    ● Humans eat Pokemon as living creatures
    ○ (1) Excludes secretions such as Yadon's tail, which regenerates immediately and clearly does not cause Pokemon pain, or Mitsu Honey's Mitsu.
    ○ (2) It is acceptable when expressed as a text as a matter of the "past" such as "humans were eating Pokemon in the past."
    ● Knifes, military-related expressions, products, expressions and products that lead to anti-ethical actions
    ○ (To express things that could threaten life or lead to bullying together with Pokemon)

    请详细分析图片内容，并给出是否包含暴力元素的判断。如果包含，请详细说明具体的暴力元素类型和程度。

    分析结果请按以下格式输出：
    - 是否包含暴力元素：[是/否]
    - 具体暴力元素：[详细描述发现的暴力元素，如果没有则写"无"]
    - 暴力程度：[alert/risk，如果无暴力元素则写"safe"]
    - 分析理由：[详细解释判断理由]
    
    prompt2:
    请评判以下暴力元素检测结果的准确性和合理性：

    检测结果：
    {ng_detection_result}

    评判标准：
    注意：你应该注意到宝可梦之间的正常战斗场景是安全的（有时会有火焰武器、爆炸和刀具），只要不是关于虐待的内容。

    请根据以下要求进行评判：
    1. 检测结果是否准确识别了真正的暴力元素
    2. 是否将正常的宝可梦战斗误判为暴力内容
    3. 暴力程度的评估是否合理
    4. 是否存在误报或漏报的情况

    请给出最终评判：
    - 检测准确性：[准确/部分准确/不准确]
    - 最终暴力判定：[是/否]
    - 建议处理方式：[通过/警告/拒绝]
    - 置信度分数：[0-100]
    - 评判理由：[详细说明评判依据]
    """
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=review_prompt,
        config=genai.types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=RewritePrompt,
            system_instruction=system_prompt,
            temperature=0.1,
        )
    )
    return response.parsed.prompt1, response.parsed.prompt2


async def RPD_classification(user_input: str) -> str:
    """classify the prompt into one of the following types: right/wrong tasks, classification tasks, numerical tasks, orientation tasks, color tasks, consecutive judgement tasks, others

    Args:
        user_input (str): the prompt to classify

    Returns:
        str: the type of the task
    """
    gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    class RPDClassification(BaseModel):
        type: str

    system_prompt = """ I have a prompt for review visual elements in a image. I want you to classify the prompt into one of the following types:
    1. right/wrong tasks: If the user input is about the right or wrong of the visual elements in the image, for example, the shape fo the character's eyes. it should be right/wrong tasks.
    2. classification tasks: If the user input is about the classification of the visual elements in the image, for example, whether there is pocket on the character's clothes. it should be classification tasks.
    3. Numerical tasks: If the user input is about the numerical value of the visual elements in the image, for example, the number of buttons on the character's clothes. it should be numerical tasks.
    4. orientation tasks: If the user input is about the orientation of the visual elements in the image, for example, the direction of the character's hair. it should be orientation tasks.
    5. Color tasks: If the user input is about the color of the visual elements in the image, for example, the color of the character's hair. it should be color tasks.
    6. consecutive judgement tasks: If the user input is about the consecutive judgement of the visual elements in the image, for example, whether the character's hair is too long. it should be consecutive judgement tasks.
    7. others: If you think the task is not one of the above types, please return "others".
    
    please output the type of the task in the following format:
    {"type": "right/wrong tasks"}
    """
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input,
        config=genai.types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=RPDClassification,
            system_instruction=system_prompt,
            temperature=0.1,
        )
    )

    if not isinstance(response.parsed, RPDClassification):
        return "others"
    else:
        return response.parsed.type


async def generate_guideline_and_constraints(title: str, user_input: str, reference_image_bytes: List[bytes], rpd_type: str) -> Tuple[Any, Any]:
    gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    model_name = "gemini-2.5-flash"

    config = ProcessingConfig(
        guideline_thinking=int(
            os.getenv('MULTI_AGENT_GUIDELINE_THINKING', '8192')),
        constraint_thinking=int(
            os.getenv('MULTI_AGENT_CONSTRAINT_THINKING', '0'))
    )

    if os.getenv('GEMINI_API_KEY') is None:
        print("Error: GEMINI_API_KEY not set. Cannot rewrite prompt.")
        return None, None

    # 初始化各Agent
    guideline_generator = GuidelineGeneratorAgent(
        gemini_client, config)
    constraint_compiler = ConstraintCompilerAgent(
        gemini_client, config)

    base_prompt = visual_review_prompt
    # 使用传入的reference_image_bytes
    ref_images = reference_image_bytes or []

    # Phase 1: 指南生成
    guidelines = await guideline_generator.generate(
        base_prompt=base_prompt,
        reference_images=ref_images,
        description_for_ai=user_input,
        rpd_type=rpd_type,
        thinking_budget=config.guideline_thinking,
        model_name=model_name,
    )

    if not guidelines:
        logger.warning("Phase 1: guideline generation failed")
        return None, None

    # Phase 2: 约束编译
    logger.info("[%s] Phase 2: constraint compilation", title)
    constraints = await constraint_compiler.compile(
        guidelines=guidelines,
        rpd_title=title,
        thinking_budget=config.constraint_thinking,
        base_prompt=base_prompt,
        model_name=model_name,
    )
    if not constraints:
        logger.warning("Phase 2: constraint generation failed")
        return None, None
    return guidelines, constraints


async def RPD_prompt_rewrite(title: str, user_input: str, reference_image_bytes: List[bytes], rpd_type: str) -> tuple[Dict[str, Any], Dict[str, Any], str, str]:
    """rewrite the prompt into a more detailed prompt

    Args:
        title (str): the title for the RPD
        user_input (str): the prompt to rewrite
        reference_image_bytes (List[bytes]): reference images as bytes

    Returns:
        tuple[str, str, str, str]: (guidelines, constraints, detector, assessor) prompts
    """
    guidelines, constraints = await generate_guideline_and_constraints(title, user_input, reference_image_bytes, rpd_type)

    detector, assessor = split_review_prompt(user_input)

    # NOTE: For future simple review system, description_for_ai may be needed
    # description_for_ai = user_input

    return guidelines or {}, constraints or {}, detector or "", assessor or ""


async def RPD_Preprocess(rpd_id: UUID) -> None:
    """RPD preprocess, including classification and prompt rewrite
    Args:
        rpd_id (UUID): The ID of the RPD to preprocess

    Returns:
        None
    """
    rpd = await ReviewPointDefinition.get(id=rpd_id)
    rpd_version = await ReviewPointDefinitionVersion.get(
        review_point_definition=rpd,
        is_active_version=True
    )

    # Initialize variables with default values
    rpd_type = ""
    guidelines: Dict[str, Any] = {}
    constraints: Dict[str, Any] = {}
    detector = ""
    assessor = ""

    # Currently not used but may be needed for classification
    if rpd_version.user_instruction:
        rpd_type = "others"
        if rpd.key == "visual_review":
            try:
                rpd_type = await RPD_classification(rpd_version.user_instruction)
            except Exception as e:
                logger.warning(f"Failed to classify RPD type: {e}")
                rpd_type = "others"

        if rpd_version.title:
            try:
                # Download reference images from S3 if available
                reference_image_bytes: List[bytes] = []
                if rpd_version.reference_images:
                    for image_path in rpd_version.reference_images:
                        try:
                            image_bytes = download_file_content_from_s3_sync(
                                image_path)
                            reference_image_bytes.append(image_bytes)
                        except Exception as e:
                            logger.warning(
                                f"Failed to download reference image {image_path}: {e}")
                            continue

                # Generate the AI prompts
                guidelines, constraints, detector, assessor = await RPD_prompt_rewrite(
                    title=rpd_version.title,
                    user_input=rpd_version.user_instruction,
                    reference_image_bytes=reference_image_bytes,
                    rpd_type=rpd_type
                )
            except Exception as e:
                logger.warning(f"Failed to generate AI prompts: {e}")
                # Continue with empty prompts if generation fails

        # Update the RPD version with the preprocessed data
        rpd_version.rpd_type = rpd_type
        rpd_version.guidelines = guidelines
        rpd_version.constraints = constraints
        rpd_version.detector = detector
        rpd_version.assessor = assessor
        rpd_version.is_ready_for_ai_review = True

        await rpd_version.save()

    return None


async def RPD_Preprocess_wrapper(rpd_id: UUID) -> None:
    """
    包装器函数，将同步操作分离到线程池中执行，避免阻塞主事件循环

    Args:
        rpd_id (UUID): The ID of the RPD to preprocess

    Returns:
        None
    """
    try:
        # 在主事件循环中进行数据库查询
        rpd = await ReviewPointDefinition.get(id=rpd_id)
        rpd_version = await ReviewPointDefinitionVersion.get(
            review_point_definition=rpd,
            is_active_version=True
        )

        # 初始化变量
        rpd_type = "others"
        detector = ""
        assessor = ""
        guidelines = {}
        constraints = {}

        # 如果没有用户指令，直接返回
        if not rpd_version.user_instruction:
            rpd_version.is_ready_for_ai_review = True
            await rpd_version.save()
            return

        # 在线程池中执行同步的AI分类调用
        if rpd.key == "visual_review":
            try:
                def sync_classify():
                    # 创建新的事件循环用于AI调用
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(RPD_classification(rpd_version.user_instruction))
                    finally:
                        loop.close()

                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    rpd_type = await loop.run_in_executor(executor, sync_classify)
            except Exception as e:
                logger.warning(f"Failed to classify RPD type: {e}")
                rpd_type = "others"

        if rpd_version.title:
            try:
                # 在线程池中下载参考图片
                reference_image_bytes = []
                if rpd_version.reference_images:
                    for image_path in rpd_version.reference_images:
                        try:
                            def sync_download():
                                return download_file_content_from_s3_sync(image_path)

                            loop = asyncio.get_event_loop()
                            with ThreadPoolExecutor(max_workers=1) as executor:
                                image_bytes = await loop.run_in_executor(executor, sync_download)
                            reference_image_bytes.append(image_bytes)
                        except Exception as e:
                            logger.warning(
                                f"Failed to download reference image {image_path}: {e}")
                            continue

                # 在线程池中执行AI prompt重写
                def sync_prompt_rewrite():
                    # 创建新的事件循环用于AI调用
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(RPD_prompt_rewrite(
                            title=rpd_version.title,
                            user_input=rpd_version.user_instruction,
                            reference_image_bytes=reference_image_bytes,
                            rpd_type=rpd_type
                        ))
                    finally:
                        loop.close()

                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    guidelines, constraints, detector, assessor = await loop.run_in_executor(executor, sync_prompt_rewrite)
            except Exception as e:
                logger.warning(f"Failed to generate AI prompts: {e}")

        # 在主事件循环中更新数据库
        rpd_version.rpd_type = rpd_type
        rpd_version.guidelines = guidelines
        rpd_version.constraints = constraints
        rpd_version.detector = detector
        rpd_version.assessor = assessor
        rpd_version.is_ready_for_ai_review = True

        await rpd_version.save()

    except Exception as e:
        logger.error(f"Error in RPD_Preprocess_wrapper: {e}")
        # 不要重新抛出异常，因为这是后台任务
