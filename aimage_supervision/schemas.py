from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from aimage_supervision.enums import SubtaskStatus, SubtaskType


class SubtaskContent(BaseModel):
    title: str
    s3_path: str
    description: str
    task_type: SubtaskType
    author: Optional[str] = None
    created_at: Optional[str] = None
    slide_page_number: Optional[int] = None
    compressed_s3_path: Optional[str] = None


class Pos(BaseModel):
    x: float
    y: float


class Size(BaseModel):
    width: float
    height: float


class Rect(Pos, Size):
    pass


class SubtaskAnnotationMeta(BaseModel):
    id: str  # Annotation ID (uuid)
    type: Optional[Literal['annotation', 'comment',
                           'ai-annotation', 'ai-comment']] = None
    version: Optional[int] = None
    timestamp: Optional[str] = None
    author: Optional[str] = None
    to: Optional[str] = None
    solved: Optional[bool] = None
    attachment_image_url: Optional[str] = None
    # reference_urls: Optional[List[str]] = None


class SubtaskAnnotation(SubtaskAnnotationMeta):
    text: str
    rect: Optional[Rect] = None
    color: Optional[str] = None
    tool: Optional[Literal['cursor', 'rect',
                           'circle', 'arrow', 'text', 'pen']] = None
    start_at: Optional[float] = None
    end_at: Optional[float] = None
    # JSON string from fabric.js for handwritten annotations
    drawing_data: Optional[str] = None


class SubtaskStatusUser(BaseModel):
    user_id: str  # User ID (uuid)
    user_name: str


class StatusHistoryEntry(BaseModel):
    updated_at: str
    updated_by: SubtaskStatusUser
    status_from: Optional[SubtaskStatus] = None
    status_to: SubtaskStatus


class Comment(BaseModel):
    id: str  # Comment ID (uuid)
    author: str  # User ID (uuid)
    to: str  # Target ID (uuid)
    marked_id: Optional[str] = None
    content: str


class TaskUpdate(BaseModel):
    """用于更新任务的输入模型"""
    title: Optional[str] = None
    description: Optional[str] = None
    status_id: Optional[str] = None
    priority_id: Optional[str] = None
    assignee_id: Optional[str] = None
    due_date: Optional[datetime] = None


class SubtaskUpdate(BaseModel):
    """用于更新子任务基本信息的输入模型"""
    name: Optional[str] = None
    description: Optional[str] = None


class SubtaskCharactersUpdate(BaseModel):
    """用于更新子任务角色关联的输入模型"""
    character_ids: List[str] = Field(
        ..., description="Character UUIDs to associate with this subtask")

    model_config = {
        "json_schema_extra": {
            "example": {
                "character_ids": ["12345678-1234-1234-1234-123456789abc", "87654321-4321-4321-4321-cba987654321"]
            }
        }
    }


class SubtaskAnnotationUpdate(BaseModel):
    """用于更新注释的输入模型"""
    text: Optional[str] = Field(
        None, min_length=1, max_length=5000, description="注释内容")
    rect: Optional[Rect] = Field(None, description="注释位置和大小")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "修正后的评论内容",
                "rect": {"x": 100, "y": 50, "width": 200, "height": 150}
            }
        }
    }


class TaskSimpleOut(BaseModel):
    id: UUID
    name: str


class TaskNavigationItem(BaseModel):
    """任务导航专用的轻量级数据模型项目"""
    id: UUID
    name: str
    tid: str
    description: str

    model_config = {
        "from_attributes": True,
    }


class TaskNavigationResponse(BaseModel):
    """任务导航专用的轻量级数据模型"""
    total: int
    items: List[TaskNavigationItem]


class SubtaskCreate(BaseModel):
    """用于创建子任务的输入模型"""
    name: str
    task_type: SubtaskType = SubtaskType.PICTURE
    description: Optional[str] = None
    content: Optional[SubtaskContent] = None


class ReferenceCardOut(BaseModel):
    id: str
    title: str
    description: str
    source: Literal['anime', 'history', 'character']


# --- All Schemas below this line are the NEW, CORRECT ones for AI Review ---

# --- Helper Schemas for reusability ---

class FindingArea(BaseModel):
    """Represents a bounding box area for a detected element or finding."""
    x: int
    y: int
    width: int
    height: int


# --- Literals ---
ReviewPointKey = Literal[
    "general_ng_review",
    "visual_review",
    "settings_review",
    "design_review",
    "text_review",
    "copyright_review"
]

NGSubcategory = Literal[
    "concrete_shape",  # 具体形状
    "abstract_type"    # 抽象类型
]

Severity = Literal["risk", "alert", "safe", "high", "medium", "low"]

FindingStatus = Literal[
    "pending_ai_review",        # AI is generating findings for this point
    "pending_human_review",     # AI findings generated, awaiting human review
    "ai_suggestion_accepted",   # Human accepted the AI suggestion as is
    # Human modified the AI suggestion (implies new human finding)
    "ai_suggestion_modified",
    "ai_suggestion_rejected",   # Human rejected the AI suggestion
    "human_added",              # Finding added purely by a human
    "resolved",                 # Finding has been addressed
    "ignored",                  # Finding will not be addressed
    "promoted_to_kb",           # Finding was promoted to knowledge base
    "superseded"                # Finding was replaced by a newer version or different finding
]


# --- AI Detected Elements (to be stored in AiReview.ai_review_output_json) ---

class AiDetectedElement(BaseModel):
    name: str
    confidence: float
    label: Literal['character', 'object', 'text']
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    area: FindingArea = Field(...,
                              description="Bounding box of the detected element.")
    character_id: Optional[UUID] = Field(None,
                                         description="引用到已知角色的ID，如果检测到了已知角色")

    model_config = {
        "from_attributes": True  # If it ever needs to be mapped from an ORM object directly
    }


class AiDetectedElements(BaseModel):
    """Container for all detected elements from an image, including an overall description."""
    description: Optional[str] = Field(
        None, description="Overall textual description of the image content based on AI detection.")
    elements: List[AiDetectedElement] = Field(
        default_factory=list, description="List of individual elements detected in the image.")

    model_config = {
        "from_attributes": True  # If it ever needs to be mapped from an ORM object directly
    }


# --- Review Point Definition ---

class ReviewPointDefinitionBase(BaseModel):
    key: ReviewPointKey
    is_active: bool = True


class ReviewPointDefinitionCreate(ReviewPointDefinitionBase):
    title: str  # For the first version
    user_instruction: str = Field(description="User instruction for AI review")
    description_for_ai: Optional[str] = Field(
        None, description="Optional, will be generated from user_instruction if not provided")
    # eng_description_for_ai: Deprecated, using description_for_ai instead
    ai_description_groups: Optional[List[dict]] = Field(
        None, description="Groups of AI descriptions for general_ng_review")
    project_id: UUID  # Add project_id for project-specific RPDs
    reference_images: Optional[List[str]] = Field(
        default=[],
        description="List of S3 paths for reference images for the initial version"
    )
    tag_list: Optional[List[str]] = Field(
        default=[],
        description="List of tags for NG review filtering"
    )
    reference_files: Optional[List[str]] = Field(
        default=[],
        description="List of S3 URLs for reference files (e.g., appellation table for text_review)"
    )
    special_rules: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        description="Special rules in JSON format: [{'speaker': '角色A', 'target': '角色B', 'alias': '特殊称呼', 'conditions': ['条件1', '条件2']}]"
    )
    ng_subcategory: Optional[NGSubcategory] = Field(
        None,
        description="Subcategory for general_ng_review: 'concrete_shape' (具体形状) or 'abstract_type' (抽象类型)"
    )


class ReviewPointDefinitionVersionBase(BaseModel):
    title: str
    user_instruction: Optional[str] = Field(
        None, description="User instruction for AI review")
    description_for_ai: Optional[str] = Field(
        None, description="generated prompt for AI review")
    # eng_description_for_ai: Deprecated, using description_for_ai instead
    is_active_version: bool = True
    reference_images: Optional[List[str]] = Field(
        default=[],
        description="List of S3 paths for reference images that provide visual context and guidance for AI reviews"
    )
    tag_list: Optional[List[str]] = Field(
        default=[],
        description="List of tags for NG review filtering"
    )
    reference_files: Optional[List[str]] = Field(
        default=[],
        description="List of S3 URLs for reference files (e.g., appellation table for text_review)"
    )
    special_rules: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        description="Special rules in JSON format: [{'speaker': '角色A', 'target': '角色B', 'alias': '特殊称呼', 'conditions': ['条件1', '条件2']}]"
    )
    ng_subcategory: Optional[NGSubcategory] = Field(
        None,
        description="Subcategory for general_ng_review: 'concrete_shape' (具体形状) or 'abstract_type' (抽象类型)"
    )

    # RPD预处理相关字段
    rpd_type: Optional[str] = Field(
        None,
        description="RPD classification type (e.g., 'right/wrong tasks', 'classification tasks', etc.)"
    )
    guidelines: Optional[Dict] = Field(
        None,
        description="AI-generated guidelines for the review"
    )
    constraints: Optional[Dict] = Field(
        None,
        description="AI-generated constraints for the review"
    )
    detector: Optional[str] = Field(
        None,
        description="AI-generated detector prompt for the review"
    )
    assessor: Optional[str] = Field(
        None,
        description="AI-generated assessor prompt for the review"
    )
    is_ready_for_ai_review: bool = Field(
        default=False,
        description="Whether this RPD version has been preprocessed and is ready for AI review"
    )


class ReviewPointDefinitionVersionCreate(ReviewPointDefinitionVersionBase):
    review_point_definition_id: UUID


class ReviewPointDefinitionVersionInDB(ReviewPointDefinitionVersionBase):
    id: UUID
    review_point_definition_id: UUID
    parent_key: Optional[ReviewPointKey] = None
    version_number: int
    created_at: datetime
    created_by: Optional[str] = None  # User ID or "AI"

    model_config = {
        "from_attributes": True
    }


class ReviewPointDefinitionInDB(ReviewPointDefinitionBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    versions: List[ReviewPointDefinitionVersionInDB] = Field(
        default_factory=list)

    model_config = {
        "from_attributes": True
    }


class ReviewPointDefinition(ReviewPointDefinitionInDB):
    current_version: Optional[ReviewPointDefinitionVersionInDB] = None
    current_version_num: Optional[int] = None


# --- AI Review Finding Entry ---

class AiReviewFindingEntryBase(BaseModel):
    description: str
    severity: Severity
    suggestion: Optional[str] = None
    area: FindingArea = Field(...,
                              description="Bounding box area of the finding.")
    reference_images: Optional[List[str]] = Field(
        default=[], description="Reference images for the finding.")
    reference_source: Optional[str] = Field(
        None, description="Textual description of the reference source or document.")
    tag: Optional[str] = Field(
        None, description="Tag for categorizing the finding.")
    is_ai_generated: bool
    is_fixed: bool = Field(
        default=False, description="Whether this finding is marked as fixed/resolved by user")
    status: FindingStatus

    # 内容类型，复用现有的 SubtaskType 枚举
    content_type: SubtaskType = Field(
        default=SubtaskType.PICTURE, description="Type of content being reviewed (picture, video, text, etc.)")

    # 类型特定的元数据
    content_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Type-specific metadata (e.g., video timestamps, text positions)")


class FindingGenData(AiReviewFindingEntryBase):
    review_point_definition_version_id: UUID


class AiReviewFindingEntryCreate(AiReviewFindingEntryBase):
    ai_review_id: UUID
    review_point_definition_version_id: UUID
    original_ai_finding_id: Optional[UUID] = None


class AiReviewFindingEntryInDB(AiReviewFindingEntryBase):
    id: UUID
    ai_review_id: UUID
    review_point_definition_version_id: UUID
    original_ai_finding_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }


class AiReviewFindingEntry(AiReviewFindingEntryInDB):
    pass


# --- Schema for updating a finding's status ---
class FindingStatusUpdate(BaseModel):
    status: FindingStatus


# --- Schema for updating a finding's is_fixed status ---
class FindingFixedStatusUpdate(BaseModel):
    is_fixed: bool


# --- Schema for updating a finding's content ---
class FindingContentUpdate(BaseModel):
    description: Optional[str] = None
    severity: Optional[Severity] = None
    suggestion: Optional[str] = None


class FindingBoundingBoxUpdate(BaseModel):
    area: FindingArea


# --- Schema for overriding an AI finding with a human one ---
class AiReviewFindingHumanOverrideCreate(BaseModel):
    description: str
    severity: Severity
    suggestion: Optional[str] = None
    area: FindingArea = Field(...,
                              description="Bounding box area for the human override.")
    reference_images: Optional[List[str]] = Field(
        default=[], description="Reference images for the human override.")
    reference_source: Optional[str] = Field(
        None, description="Textual description of the reference source or document.")
    status: FindingStatus


# --- Schema for adding a purely human finding to a review ---
class AiReviewFindingHumanCreate(BaseModel):
    review_point_definition_version_id: UUID
    description: str
    severity: Severity
    suggestion: Optional[str] = None
    area: FindingArea = Field(...,
                              description="Bounding box area for the human-added finding.")
    reference_images: Optional[List[str]] = Field(
        default=[], description="Reference images for the human-added finding.")
    reference_source: Optional[str] = Field(
        None, description="Textual description of the reference source or document.")
    status: FindingStatus


# --- AI Review (Overall review instance for a Subtask) ---

class AiReviewBase(BaseModel):
    subtask_id: UUID


class AiReviewCreate(AiReviewBase):
    pass


class AiReviewInDB(AiReviewBase):
    id: UUID
    version: int
    is_latest: bool
    review_timestamp: datetime
    initiated_by_user_id: Optional[UUID] = None
    last_modified_by_user_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    findings: List[AiReviewFindingEntry] = Field(default=[])

    # 新增状态字段
    processing_status: Optional[str] = None
    error_message: Optional[str] = None
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    should_cancel: Optional[bool] = None

    model_config = {
        "from_attributes": True
    }


class AiReview(AiReviewInDB):
    # TODO: Should be AiDetectedElements
    detected_elements: Optional[List[AiDetectedElement]] = None
    detected_elements_summary: Optional[AiDetectedElements] = Field(
        None, description="Summary of detected elements in the image, including an overall description and list of elements.")


# --- Promoted Finding (Knowledge Base Entry) ---

class PromotedFindingBase(BaseModel):
    finding_entry_id: UUID
    subtask_id_context: UUID  # The subtask where this finding was identified as valuable
    notes: Optional[str] = None
    # Simple list of string tags for now
    tags: List[str] = Field(default_factory=list)
    sharing_scope: Optional[str] = None  # e.g., "team", "project", "global"


class PromotedFindingCreate(PromotedFindingBase):
    promoted_by_user_id: UUID


class PromotedFindingInDB(PromotedFindingBase):
    id: UUID
    promoted_by_user_id: UUID
    promotion_timestamp: datetime
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }


class PromotedFinding(PromotedFindingInDB):
    # Include the full finding details
    finding_entry: Optional[AiReviewFindingEntry] = None


# --- Schemas for specific API request/response if needed, composing above ---

# Example: what a full subtask review might return
class SubtaskReviewResponse(AiReview):
    pass


# Example for an endpoint returning a specific RPD
class ReviewPointDefinitionDetail(ReviewPointDefinition):
    pass


# --- Schemas for Promoting Findings (Knowledge Base) ---

class PromoteFindingRequestBody(BaseModel):
    """Request body for promoting a finding. User provides these details."""
    finding_entry_id: UUID
    subtask_id_context: UUID  # Contextual subtask ID from which finding is being promoted
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    sharing_scope: Optional[str] = None  # e.g., "team", "project", "global"


# Add other specific request/response schemas as endpoints are refactored.

# --- Review Set and Task Tag Schemas ---

class TaskTagBase(BaseModel):
    name: str


class TaskTagCreate(TaskTagBase):
    project_id: UUID


class TaskTagUpdate(TaskTagBase):
    pass


class TaskTagInDB(TaskTagBase):
    id: UUID
    project_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }


class TaskTagOut(TaskTagInDB):
    pass


class ReviewSetBase(BaseModel):
    name: str
    description: Optional[str] = None


class ReviewSetCreate(ReviewSetBase):
    project_id: UUID
    rpd_ids: List[UUID] = Field(default_factory=list)
    character_ids: List[UUID] = Field(default_factory=list)
    task_tag_ids: List[UUID] = Field(default_factory=list)


class ReviewSetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    rpd_ids: Optional[List[UUID]] = None
    character_ids: Optional[List[UUID]] = None
    task_tag_ids: Optional[List[UUID]] = None


class ReviewSetInDB(ReviewSetBase):
    id: UUID
    project_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }

# Forward declaration for recursive models


class CharacterForReviewSet(BaseModel):
    id: UUID
    name: str
    thumbnail_url: Optional[str] = None

    model_config = {
        "from_attributes": True
    }


class RPDForReviewSet(BaseModel):
    id: UUID
    key: str
    # Add current version title to display in ReviewSetDetailPanel
    current_version_title: Optional[str] = None

    model_config = {
        "from_attributes": True
    }


class ReviewSetOut(ReviewSetInDB):
    rpds: List[RPDForReviewSet] = Field(default_factory=list)
    characters: List[CharacterForReviewSet] = Field(default_factory=list)
    task_tags: List[TaskTagOut] = Field(default_factory=list)


# ReviewSet-Character Association Schemas
class ReviewSetCharacterAssociationCreate(BaseModel):
    review_set_id: UUID
    character_id: UUID


class ReviewSetCharacterAssociationOut(BaseModel):
    review_set_id: UUID
    character_id: UUID

    model_config = {
        "from_attributes": True
    }


class ReviewSetCharacterAssociationWithDetails(BaseModel):
    review_set_id: UUID
    character_id: UUID
    review_set: ReviewSetInDB
    character: CharacterForReviewSet

    model_config = {
        "from_attributes": True
    }


# --- Character Management Schemas ---

class CharacterCreate(BaseModel):
    """用于创建角色的输入模型"""
    name: str
    alias: Optional[str] = None
    description: Optional[str] = None
    features: Optional[str] = None
    image_path: Optional[str] = None
    reference_images: Optional[List[str]] = None
    ip_id: Optional[UUID] = None
    project_id: UUID


class CharacterUpdate(BaseModel):
    """用于更新角色的输入模型"""
    name: Optional[str] = None
    alias: Optional[str] = None
    description: Optional[str] = None
    features: Optional[str] = None
    image_path: Optional[str] = None
    reference_images: Optional[List[str]] = None
    ip_id: Optional[UUID] = None


class IPCreate(BaseModel):
    """用于创建IP的输入模型"""
    name: str
    description: Optional[str] = None
    project_ids: Optional[List[UUID]] = None


class IPUpdate(BaseModel):
    """用于更新IP的输入模型"""
    name: Optional[str] = None
    description: Optional[str] = None
    project_ids: Optional[List[UUID]] = None


class CharacterDetection(BaseModel):
    """角色检测结果模型"""
    character_id: UUID
    name: str
    confidence: float
    area: FindingArea


# Item related schemas
class ItemCreate(BaseModel):
    """用于创建Item的输入模型"""
    filename: str
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    project_id: Optional[UUID] = None


class ItemUpdate(BaseModel):
    """用于更新Item的输入模型"""
    filename: Optional[str] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    project_id: Optional[UUID] = None


class PDFResponse(BaseModel):
    """PDF response model"""
    id: UUID
    filename: str
    s3_path: str
    file_size: int
    total_pages: int
    extraction_session_id: Optional[str] = None
    extracted_at: datetime
    extraction_method: str
    extraction_stats: Optional[Dict[str, Any]] = None
    project_id: UUID
    uploaded_by: UUID
    created_at: datetime
    updated_at: datetime


class ItemResponse(BaseModel):
    """Item response model"""
    id: UUID
    filename: str
    s3_path: str
    s3_url: Optional[str] = None
    image_url: str
    content_type: Optional[str] = None
    file_size: Optional[int] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    project_id: Optional[UUID] = None
    uploaded_by: UUID
    created_at: datetime
    updated_at: datetime

    # PDF-related fields
    source_type: str = 'direct_upload'
    source_pdf_id: Optional[UUID] = None
    pdf_page_number: Optional[int] = None
    pdf_image_index: Optional[int] = None


class ItemBatchUploadResponse(BaseModel):
    """Batch upload response model"""
    uploaded_items: List[ItemResponse]
    failed_uploads: List[dict]
    total_uploaded: int
    total_failed: int


# --- Prompt Rewrite Schemas ---

class PromptRewriteRequest(BaseModel):
    """用于prompt转写请求的输入模型"""
    original_prompt: str = Field(..., description="用户输入的原始prompt")
    context: Optional[str] = Field(None, description="可选的上下文信息，例如项目描述、任务类型等")
    rpd_type: str = Field(..., description="RPD类型")
    target_language: Optional[Literal["japanese", "english", "chinese"]] = Field(
        default="english", description="目标语言"
    )
    image_url: Optional[str] = Field(None, description="图片URL（可选）")


class PromptRewriteResponse(BaseModel):
    """Prompt转写响应模型"""
    original_prompt: str = Field(..., description="用户输入的原始prompt")
    rewritten_prompt: str = Field(..., description="AI转写后的完整prompt")
    rewritten_prompt_jpn: str = Field(..., description="AI转写后的完整prompt（日文）")
    confidence: float = Field(..., description="转写质量的置信度评分", ge=0.0, le=1.0)

    model_config = {
        "from_attributes": True
    }


# --- Character Reference Generation Schemas ---

class GenerationTags(BaseModel):
    """4 basic tag categories only"""
    style: Optional[str] = None      # "anime", "realistic", "chibi"
    pose: Optional[str] = None       # "standing", "sitting", "action"
    camera: Optional[str] = None     # "close-up", "full-body", "portrait"
    lighting: Optional[str] = None   # "natural", "dramatic", "soft"


class GenerateRequest(BaseModel):
    base_prompt: str = Field(min_length=1)
    tags: GenerationTags
    count: int = Field(default=1, ge=1, le=4)
    # Add missing parameters for better user control
    aspect_ratio: str = Field(
        default="1:1",
        description="Image aspect ratio",
        pattern="^(1:1|16:9|9:16|4:3|3:4)$"
    )
    negative_prompt: Optional[str] = Field(
        default=None, description="Things to avoid in generation")


class GeneratedReferenceResponse(BaseModel):
    id: UUID
    base_prompt: str
    enhanced_prompt: str
    tags: dict
    image_url: str
    image_path: str  # Include S3 path for liked images functionality
    created_at: datetime


# --- User Preferences Schemas ---

class LikedImageRequest(BaseModel):
    """Request schema for adding liked images with source tracking"""
    image_path: str = Field(..., min_length=1, max_length=1024,
                            description="S3 path/key for the image")
    source_type: str = Field(..., min_length=1, max_length=50,
                             description="Type of source object")
    source_id: UUID = Field(..., description="UUID of the source object")
    display_name: Optional[str] = Field(
        None, max_length=255, description="Optional display name")
    tags: Optional[List[str]] = Field(
        default=[], description="Optional tags")


class LikedImageRemoveRequest(BaseModel):
    """Request schema for removing liked images"""
    image_path: str = Field(..., min_length=1, max_length=1024,
                            description="S3 path/key for the image")
    source_type: str = Field(..., min_length=1, max_length=50,
                             description="Type of source object")
    source_id: UUID = Field(..., description="UUID of the source object")


class LikedImageResponse(BaseModel):
    """Response schema for liked images with fresh URLs"""
    id: UUID
    image_path: str
    image_url: str  # Fresh presigned URL
    source_type: str
    source_id: UUID
    display_name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime

    model_config = {
        "from_attributes": True
    }


class UserPreferencesResponse(BaseModel):
    """Response schema for user preferences and settings"""
    settings: dict = Field(
        default_factory=dict, description="User settings and configuration preferences")

    # Note: liked_images are now handled by dedicated LikedImage endpoints
    # Use GET /users/me/preferences/liked-images to retrieve liked images

    model_config = {
        "from_attributes": True
    }

# --- Task Thumbnail Schemas ---


class TaskThumbnail(BaseModel):
    """单个缩略图信息"""
    subtask_id: UUID
    subtask_name: str
    original_s3_path: str
    compressed_s3_path: Optional[str] = None

    model_config = {
        "from_attributes": True
    }


class TaskThumbnailsResponse(BaseModel):
    """任务缩略图响应模型"""
    task_id: UUID
    thumbnails: List[TaskThumbnail] = Field(
        default_factory=list,
        description="任务的前几个图片子任务缩略图（默认前3个）"
    )

    model_config = {
        "from_attributes": True
    }
