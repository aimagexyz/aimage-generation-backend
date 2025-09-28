import traceback
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import (APIRouter, BackgroundTasks, HTTPException, Query,
                     Security, status)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field  # Add this import

from aimage_supervision.enums import AiReviewMode
from aimage_supervision.enums import \
    AiReviewProcessingStatus as ProcessingStatusEnum
from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.models import (Character, ReviewPointDefinition,
                                       Subtask, User)
from aimage_supervision.schemas import AiReview as AiReviewSchema
from aimage_supervision.schemas import (AiReviewInDB, FindingBoundingBoxUpdate,
                                        FindingContentUpdate,
                                        FindingFixedStatusUpdate)
from aimage_supervision.services import ai_review_service, batch_process
from aimage_supervision.services.ai_review_service import (
    get_latest_rpd_execution_summary, update_finding_bounding_box)
from aimage_supervision.services.rpd_filter import filter_rpd
from aimage_supervision.settings import MAX_CONCURRENT

from ..services.review_set_recommendation_service import \
    TaskReviewSetRecommendationsResponse as \
    TaskReviewSetRecommendationsResponseService

CHARACTER_PREDICTION_PLACEHOLDER = "Character_Prediction_Placeholder"


class AIReviewInitiateRequest(BaseModel):
    rpd_ids: Optional[List[UUID]] = Field(
        None,
        description="A list of Review Point Definition IDs to run. If not provided, no RPD-based review will run (unless cr_check=True)."
    )


# --- New Router for AI Review Process ---
router = APIRouter(
    prefix="/ai-reviews",
    tags=["AI Reviews"]
)


@router.post(
    "/subtasks/{subtask_id}/initiate",
    status_code=status.HTTP_201_CREATED,
    summary="Initiate a new AI Review for a Subtask",
    description="Triggers the full AI review process for the specified subtask, "
    "including element detection and finding generation for all active review points. "
    "A new, versioned AIReview record will be created.")
async def initiate_ai_review_endpoint(
    subtask_id: UUID,
    request: AIReviewInitiateRequest,
    background_tasks: BackgroundTasks,
    cr_check: Optional[bool] = False,
    mode: AiReviewMode = Query(
        AiReviewMode.QUALITY,
        description="AI Review Mode: 'quality' for precision, 'speed' for faster results."),
    current_user: User = Security(get_current_user)
) -> JSONResponse:
    try:
        # 后台任务现在会自动处理状态更新
        background_tasks.add_task(
            ai_review_service.initiate_ai_review_for_subtask,
            subtask_id=subtask_id,
            initiated_by_user_id=current_user.id,
            cr_check=cr_check,
            rpd_ids=request.rpd_ids,
            mode=mode
        )

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                'message': f'AI Review initiated for subtask {subtask_id}',
                'status': 'accepted'
            }
        )
    except Exception as e:
        # 简化错误处理
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/{ai_review_id}",
    response_model=AiReviewSchema,
    summary="Get a specific AI Review by ID",
    description="Retrieves the full details of a specific AI Review, including all its findings and detected elements."
)
async def get_ai_review_by_id_endpoint(
    ai_review_id: UUID,
    # Added for consistency, can be used for authZ later
    current_user: User = Security(get_current_user)
) -> AiReviewSchema:
    # current_user can be used here for authorization if needed in the future
    review = await ai_review_service.get_ai_review_by_id(ai_review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AI Review with ID {ai_review_id} not found."
        )
    return review


@router.get(
    "/subtasks/{subtask_id}/latest",
    response_model=AiReviewSchema,
    summary="Get the latest AI Review for a Subtask",
    description="Retrieves the most recent (latest version) AI Review for the specified subtask, including all findings and detected elements."
)
async def get_latest_ai_review_for_subtask_endpoint(
    subtask_id: UUID,
    current_user: User = Security(get_current_user)
) -> AiReviewSchema:
    review = await ai_review_service.get_latest_ai_review_for_subtask(subtask_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No AI Review found for subtask with ID {subtask_id}."
        )
    return review


@router.get(
    "/subtasks/{subtask_id}/all",
    response_model=List[AiReviewSchema],
    summary="List all AI Review versions for a Subtask",
    description="Retrieves a list of all AI Review versions for the specified subtask, ordered from newest to oldest. Each review includes all findings and detected elements."
)
async def list_ai_reviews_for_subtask_endpoint(
    subtask_id: UUID,
    current_user: User = Security(get_current_user)
) -> List[AiReviewSchema]:
    reviews = await ai_review_service.list_ai_reviews_for_subtask(subtask_id)
    # No need to check if empty, an empty list is a valid response (200 OK)
    return reviews


class BatchInitiateAiReviewRequest(BaseModel):
    project_id: UUID = Field(...,
                             description="The ID of the project these tasks belong to.")
    task_ids: List[UUID] = Field(...,
                                 description="A list of task IDs to initiate AI review for.")


class BatchInitiateCustomAiReviewRequest(BaseModel):
    project_id: UUID = Field(...,
                             description="The ID of the project these tasks belong to.")
    task_ids: List[UUID] = Field(...,
                                 description="A list of task IDs to initiate AI review for.")
    rpd_ids: Optional[List[UUID]] = Field(
        None,
        description="A list of Review Point Definition IDs to run. If not provided, will use all active RPDs."
    )
    review_set_ids: Optional[List[UUID]] = Field(
        None,
        description="A list of Review Set IDs to expand into RPD IDs."
    )
    mode: AiReviewMode = Field(
        AiReviewMode.QUALITY,
        description="AI Review Mode: 'quality' for precision, 'speed' for faster results."
    )


@router.post(
    "/batch/cr-check/initiate",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch initiate CR Check AI Reviews for multiple tasks",
    description="Triggers the AI review process for all subtasks within the specified tasks, with CR Check enabled."
)
async def batch_initiate_cr_check_review_endpoint(
    request: BatchInitiateAiReviewRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Security(get_current_user)
) -> JSONResponse:
    """
    Batch initiates AI review for all subtasks of the given tasks.
    Uses parallel processing within a single background task for better performance.
    """
    try:
        subtasks = await ai_review_service.get_subtasks_for_tasks(task_ids=request.task_ids)

        if not subtasks:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    'message': 'No subtasks found for the provided task IDs.',
                    'subtask_count': 0
                }
            )

        # 使用新的并行处理函数作为单个后台任务
        background_tasks.add_task(
            batch_process.batch_initiate_cr_check_parallel,
            subtasks=subtasks,
            initiated_by_user_id=current_user.id,
            max_concurrent=MAX_CONCURRENT
        )

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                'message': f'Accepted batch CR Check request for {len(subtasks)} subtasks. Processing will continue in the background with parallel execution.',
                'subtask_count': len(subtasks),
                'max_concurrent': MAX_CONCURRENT
            }
        )
    except Exception as e:
        # Log the exception e
        print(
            f"Unexpected error in batch_initiate_cr_check_review_endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while initiating the batch AI review.")


@router.post(
    "/batch/initiate",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch initiate AI Reviews for multiple tasks with custom RPDs",
    description="Triggers the AI review process for all subtasks within the specified tasks, with custom RPD selection."
)
async def batch_initiate_custom_ai_review_endpoint(
    request: BatchInitiateCustomAiReviewRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Security(get_current_user)
) -> JSONResponse:
    """
    Batch initiates AI review for all subtasks of the given tasks with custom RPD selection.
    Uses parallel processing within a single background task for better performance.
    """
    try:
        subtasks = await ai_review_service.get_subtasks_for_tasks(task_ids=request.task_ids)

        if not subtasks:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    'message': 'No subtasks found for the provided task IDs.',
                    'subtask_count': 0
                }
            )

        batch_id = str(uuid.uuid4())[:8]

        # 使用新的并行处理函数作为单个后台任务
        background_tasks.add_task(
            batch_process.batch_initiate_parallel,
            subtasks=subtasks,
            initiated_by_user_id=current_user.id,
            rpd_ids=request.rpd_ids,
            review_set_ids=request.review_set_ids,
            mode=request.mode,
            max_concurrent=MAX_CONCURRENT,
            batch_id=batch_id  # 传递batch_id到服务层
        )

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                'message': f'Accepted batch AI review request for {len(subtasks)} subtasks. Processing will continue in the background with parallel execution.',
                'batch_id': batch_id,  # 🆕 返回batch_id供前端使用
                'subtask_count': len(subtasks),
                'max_concurrent': MAX_CONCURRENT,
                'mode': request.mode.value,
                'rpd_count': len(request.rpd_ids) if request.rpd_ids else 0,
                'review_set_count': len(request.review_set_ids) if request.review_set_ids else 0
            }
        )
    except Exception as e:
        # Log the exception e
        print(
            f"Unexpected error in batch_initiate_custom_ai_review_endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while initiating the batch AI review.")


class CharacterPredictionRequest(BaseModel):
    """Request body for character prediction"""
    subtask_id: UUID = Field(...,
                             description="The ID of the subtask to predict character for")


class CharacterPredictionResponse(BaseModel):
    """Response body for character prediction"""
    character_id: UUID = Field(...,
                               description="The ID of the predicted character")
    character_name: str = Field(...,
                                description="The name of the predicted character")
    confidence: float = Field(...,
                              description="AI confidence score for the prediction",
                              ge=0.0,
                              le=1.0)
    prediction_method: str = Field(
        ..., description="Method used for prediction (e.g., 'random', 'ai_model')")
    predicted: bool = Field(...,
                            description="Whether character prediction was successful")


class CharacterPredictionResponseList(BaseModel):
    """Response body for character prediction"""
    characters: List[CharacterPredictionResponse] = Field(...,
                                                          description="List of predicted characters")


class RPDRecommendation(BaseModel):
    """RPD recommendation based on character"""
    rpd_id: UUID = Field(..., description="The ID of the recommended RPD")
    rpd_title: str = Field(..., description="The title of the recommended RPD")
    rpd_key: str = Field(..., description="The key of the recommended RPD")
    reason: str = Field(..., description="Reason for recommendation")


class CharacterRPDRecommendationsResponse(BaseModel):
    """Response body for character-based RPD recommendations"""
    character_id: UUID = Field(..., description="The ID of the character")
    character_name: str = Field(..., description="The name of the character")
    recommendations: List[RPDRecommendation] = Field(
        ..., description="List of recommended RPDs")
    total_recommendations: int = Field(...,
                                       description="Total number of recommendations")


class AiReviewProcessingStatus(BaseModel):
    """Response body for AI review processing status"""
    subtask_id: UUID = Field(..., description="The ID of the subtask")
    is_processing: bool = Field(...,
                                description="Whether AI review is currently being processed")
    is_completed: bool = Field(...,
                               description="Whether AI review has been completed")
    is_cancelled: Optional[bool] = Field(False,
                                         description="Whether AI review has been cancelled")
    latest_review_id: Optional[UUID] = Field(
        None, description="The ID of the latest AI review if available")
    processing_started_at: Optional[datetime] = Field(
        None, description="When the processing started")
    completed_at: Optional[datetime] = Field(
        None, description="When the processing completed")
    findings_count: int = Field(0, description="Number of findings generated")
    message: str = Field(..., description="Status message")

    # 新增字段
    processing_status: Optional[str] = Field(None, description="明确的状态字符串")
    error_message: Optional[str] = Field(None, description="错误信息")
    should_cancel: Optional[bool] = Field(False, description="中断信号")


def _generate_status_message(review: AiReviewSchema) -> str:
    """生成状态消息的统一函数"""
    if review.processing_status == ProcessingStatusEnum.COMPLETED.value:
        findings_count = len(review.findings) if review.findings else 0
        return f"AI review completed with {findings_count} findings"
    elif review.processing_status == ProcessingStatusEnum.PROCESSING.value:
        return "AI review is currently being processed"
    elif review.processing_status == ProcessingStatusEnum.CANCELLED.value:
        return "AI review was cancelled by user"
    elif review.processing_status == ProcessingStatusEnum.FAILED.value:
        error_msg = review.error_message or "Unknown error"
        return f"AI review failed: {error_msg}"
    else:  # PENDING or None
        return "AI review is pending"


@router.post(
    "/subtasks/{subtask_id}/interrupt",
    status_code=status.HTTP_200_OK,
    summary="Interrupt AI Review processing for a Subtask",
    description="Interrupts the currently running AI review process for the specified subtask. "
    "The review status will be set to 'cancelled' and the background processing will be stopped."
)
async def interrupt_ai_review_endpoint(
    subtask_id: UUID,
    current_user: User = Security(get_current_user)
) -> JSONResponse:
    """
    中断指定子任务的AI审查处理。

    将当前正在进行的AI审查任务标记为已取消，并停止后台处理。
    """
    try:
        success = await ai_review_service.interrupt_ai_review_for_subtask(subtask_id)

        if success:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    'message': f'AI Review interrupted for subtask {subtask_id}',
                    'status': 'interrupted',
                    'subtask_id': str(subtask_id)
                }
            )
        else:
            # 没有找到可中断的review或review已经完成
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active AI review found to interrupt for this subtask"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error interrupting AI review for subtask {subtask_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while interrupting AI review for subtask {subtask_id}."
        )


@router.get(
    "/subtasks/{subtask_id}/processing-status",
    response_model=AiReviewProcessingStatus,
    summary="Check AI Review processing status for a Subtask",
    description="Returns the current processing status of AI review for the specified subtask. "
    "This endpoint can be used for polling to check if the background AI review process has completed."
)
async def get_ai_review_processing_status_endpoint(
    subtask_id: UUID,
    current_user: User = Security(get_current_user)
) -> AiReviewProcessingStatus:
    """
    检查指定子任务的AI审查处理状态。

    用于前端轮询检查后台AI审查任务是否完成。
    """
    try:
        latest_review = await ai_review_service.get_latest_ai_review_for_subtask(subtask_id)

        if not latest_review:
            return AiReviewProcessingStatus(
                subtask_id=subtask_id,
                is_processing=False,
                is_completed=False,
                is_cancelled=False,
                latest_review_id=None,
                processing_started_at=None,
                completed_at=None,
                findings_count=0,
                message="No AI review found for this subtask",
                processing_status=None,
                error_message=None,
                should_cancel=False
            )

        # 状态判断变得非常简单（KISS原则）
        is_processing = latest_review.processing_status == ProcessingStatusEnum.PROCESSING.value
        is_completed = latest_review.processing_status == ProcessingStatusEnum.COMPLETED.value
        is_cancelled = latest_review.processing_status == ProcessingStatusEnum.CANCELLED.value

        # 生成状态消息
        message = _generate_status_message(latest_review)

        return AiReviewProcessingStatus(
            subtask_id=subtask_id,
            is_processing=is_processing,
            is_completed=is_completed,
            is_cancelled=is_cancelled,
            latest_review_id=latest_review.id,
            processing_started_at=latest_review.processing_started_at,
            completed_at=latest_review.processing_completed_at,
            findings_count=len(
                latest_review.findings) if latest_review.findings else 0,
            message=message,
            processing_status=latest_review.processing_status,
            error_message=latest_review.error_message,
            should_cancel=latest_review.should_cancel
        )

    except Exception as e:
        print(
            f"Error checking AI review processing status for subtask {subtask_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while checking AI review processing status for subtask {subtask_id}."
        )


@router.post(
    "/subtasks/{subtask_id}/predict-character",
    status_code=status.HTTP_201_CREATED,
    summary="Predict character for a subtask",
    description="Uses AI to predict the most likely character present in the given subtask's content. "
    "Runs as a background task to avoid blocking the UI.")
async def predict_character_for_subtask_endpoint(
    subtask_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Security(get_current_user)
) -> JSONResponse:
    """
    Predict the character present in a subtask using AI analysis.

    This runs as a background task to avoid blocking the UI.
    The prediction result will be saved to the subtask's character_ids field.
    """
    try:
        # 1. Fetch the subtask and verify it exists
        subtask = await Subtask.get_or_none(id=subtask_id)
        if not subtask:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Subtask with ID {subtask_id} not found."
            )

        # 2. Get the project ID through the task relationship
        await subtask.fetch_related("task__project")
        project = subtask.task.project

        # 3. Check if user has access to the project (basic authorization)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found for the given subtask."
            )

        # 4. Get all characters available in this project
        characters = await Character.filter(project_id=project.id).all()

        if not characters:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No characters found in the project. Please add characters before predicting.")

        # 5. Set initial processing status (we'll use the existing character_ids field)
        # 暂时设置一个特殊的状态值来表示正在处理
        subtask.character_ids = ["PROCESSING"]
        await subtask.save(update_fields=['character_ids', 'updated_at'])

        # 6. Add background task for AI prediction
        background_tasks.add_task(
            _predict_character_background_task,
            subtask_id=subtask_id,
            user_id=current_user.id
        )

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                'message': f'Character prediction initiated for subtask {subtask_id}',
                'status': 'processing'
            }
        )

    except HTTPException:
        # Re-raise HTTP exceptions (404, 400, etc.)
        raise
    except Exception as e:
        # Log unexpected errors
        print(
            f"Unexpected error in predict_character_for_subtask_endpoint for subtask {subtask_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while predicting character for subtask {subtask_id}.")


@router.patch(
    "/findings/{finding_id}/fixed-status",
    summary="Update finding fixed status",
    description="Update the is_fixed status of a specific AI review finding entry. "
    "This allows users to mark findings as fixed/resolved to keep them visible in future reviews."
)
async def update_finding_fixed_status_endpoint(
    finding_id: UUID,
    status_update: FindingFixedStatusUpdate,
    current_user: User = Security(get_current_user)
) -> JSONResponse:
    """
    更新AI审查发现条目的is_fixed状态

    Args:
        finding_id: 发现条目的ID
        status_update: 包含is_fixed状态的更新数据
        current_user: 当前用户
    """
    try:
        updated_finding = await ai_review_service.update_finding_fixed_status(
            finding_id=finding_id,
            is_fixed=status_update.is_fixed
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                'message': f'Finding {finding_id} fixed status updated successfully',
                'finding_id': str(updated_finding.id),
                'is_fixed': updated_finding.is_fixed
            }
        )

    except ValueError as e:
        # Finding not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        # Log unexpected errors
        print(
            f"Unexpected error updating finding fixed status for {finding_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating finding fixed status."
        )


@router.get(
    "/subtasks/{subtask_id}/findings-summary",
    response_model=AiReviewSchema,
    summary="Get optimized AI Review for subtask with relevant findings",
    description="Get an AI Review for a subtask with optimized findings data, including latest findings and fixed historical findings. "
    "This optimized endpoint avoids scanning all historical versions while maintaining the same response format as get_latest_ai_review_for_subtask."
)
async def get_findings_summary_endpoint(
    subtask_id: UUID,
    current_user: User = Security(get_current_user)
) -> AiReviewSchema:
    """
    获取子任务的优化AI审查结果，包含最新版本和已修复的历史发现
    返回格式与 get_latest_ai_review_for_subtask 相同，但性能更优

    Args:
        subtask_id: 子任务ID
        current_user: 当前用户
    """
    try:
        review = await ai_review_service.get_findings_summary_for_subtask(subtask_id)
        if not review:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No AI Review found for subtask with ID {subtask_id}."
            )
        return review

    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        # Log unexpected errors
        print(
            f"Unexpected error getting findings summary for subtask {subtask_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while getting findings summary for subtask {subtask_id}."
        )


@router.patch(
    "/findings/{finding_id}/content",
    summary="Update finding content",
    description="Update the content of a specific AI review finding entry (description, severity, suggestion). "
    "This allows users to modify the finding details while keeping the original AI finding for reference."
)
async def update_finding_content_endpoint(
    finding_id: UUID,
    content_update: FindingContentUpdate,
    current_user: User = Security(get_current_user)
) -> JSONResponse:
    """
    更新AI审查发现条目的内容

    Args:
        finding_id: 发现条目的ID
        content_update: 包含要更新的内容数据
        current_user: 当前用户
    """
    try:
        updated_finding = await ai_review_service.update_finding_content(
            finding_id=finding_id,
            description=content_update.description,
            severity=content_update.severity,
            suggestion=content_update.suggestion
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                'message': f'Finding {finding_id} content updated successfully',
                'finding_id': str(updated_finding.id),
                'description': updated_finding.description,
                'severity': updated_finding.severity,
                'suggestion': updated_finding.suggestion,
                'updated_at': updated_finding.updated_at.isoformat()
            }
        )

    except ValueError as e:
        # Finding not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        # Log unexpected errors
        print(
            f"Unexpected error updating finding content for {finding_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating finding content."
        )


@router.get(
    "/characters/{character_id}/rpd-recommendations",
    response_model=CharacterRPDRecommendationsResponse,
    summary="Get RPD recommendations for a character",
    description="Returns RPD recommendations based on the character's associations, sorted by weight/importance."
)
async def get_character_rpd_recommendations_endpoint(
    character_id: UUID,
    project_id: UUID = Query(..., description="Project ID to filter RPDs"),
    current_user: User = Security(get_current_user)
) -> CharacterRPDRecommendationsResponse:
    """
    Get RPD recommendations for a character based on associations

    Returns a list of RPDs that are associated with the character.
    """
    try:
        # 1. Get character with associated RPDs
        character_with_rpds = await Character.get_or_none(
            id=character_id,
            project_id=project_id
        ).prefetch_related('associated_rpds__versions')

        if not character_with_rpds:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character with ID {character_id} not found in project {project_id}."
            )

        # 2. Filter active RPDs and build recommendations
        recommendations = []
        for rpd in character_with_rpds.associated_rpds:
            if not rpd.is_active:
                continue

            # Get latest version title
            latest_version = None
            if rpd.versions:
                latest_version = max(
                    rpd.versions, key=lambda v: v.version_number)

            rpd_title = latest_version.title if latest_version else rpd.key

            # Since we don't have weight/association_type anymore, use default values
            reason = "関連キャラクター"

            recommendations.append(RPDRecommendation(
                rpd_id=rpd.id,
                rpd_title=rpd_title,
                rpd_key=rpd.key,
                reason=reason
            ))

        return CharacterRPDRecommendationsResponse(
            character_id=character_id,
            character_name=character_with_rpds.name,
            recommendations=recommendations,
            total_recommendations=len(recommendations)
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors
        print(
            f"Unexpected error getting RPD recommendations for character {character_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while getting RPD recommendations."
        )


@router.patch(
    "/findings/{finding_id}/bounding-box",
    summary="Update finding bounding box",
    description="Update the bounding box area of a specific AI review finding entry. "
    "This allows users to modify the position and size of the finding's bounding box."
)
async def update_finding_bounding_box_endpoint(
    finding_id: UUID,
    bounding_box_update: FindingBoundingBoxUpdate,
    current_user: User = Security(get_current_user)
) -> JSONResponse:
    """
    更新AI审查发现条目的边界框

    Args:
        finding_id: 发现条目的ID
        bounding_box_update: 包含新的边界框数据
        current_user: 当前用户
    """
    try:
        updated_finding = await update_finding_bounding_box(
            finding_id=finding_id,
            area=bounding_box_update.area.dict()
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                'message': f'Finding {finding_id} bounding box updated successfully',
                'finding_id': str(updated_finding.id),
                'area': updated_finding.area,
                'updated_at': updated_finding.updated_at.isoformat()
            }
        )

    except ValueError as e:
        # Finding not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        # Log unexpected errors
        print(
            f"Unexpected error updating finding bounding box for {finding_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating finding bounding box."
        )


# Background task functions
async def _predict_character_background_task(subtask_id: UUID, user_id: UUID):
    """
    Background task for character prediction.
    """
    try:
        # Fetch the subtask and related data
        subtask = await Subtask.get_or_none(id=subtask_id)
        if not subtask:
            print(f"Subtask {subtask_id} not found in background task")
            return

        await subtask.fetch_related("task__project")
        project = subtask.task.project

        if not project:
            print(f"Project not found for subtask {subtask_id}")
            return

        # Get all characters available in this project
        characters = await Character.filter(project_id=project.id).all()

        if not characters:
            print(f"No characters found in project {project.id}")
            return

        # AI prediction using Gemini
        predict_characters, _ = await ai_review_service.predict_character_for_subtask(
            subtask=subtask,
            character_candidates=characters
        )
        if not predict_characters:
            # AI could not predict any character
            print(
                f"AI could not predict any character for subtask {subtask_id}")
            subtask.character_ids = [str(UUID(int=0))]
            await subtask.save(update_fields=['character_ids', 'updated_at'])
            return

        # Save AI prediction result to database
        subtask.character_ids = [str(predict_character.id)
                                 for predict_character in predict_characters]
        await subtask.save(update_fields=['character_ids', 'updated_at'])

        print(f"Character prediction completed for subtask {subtask_id}")

    except Exception as e:
        print(
            f"Error in character prediction background task for subtask {subtask_id}: {e}")
        traceback.print_exc()

        # 设置错误状态
        try:
            subtask = await Subtask.get_or_none(id=subtask_id)
            if subtask:
                subtask.character_ids = ["FAILED"]
                await subtask.save(update_fields=['character_ids', 'updated_at'])
        except Exception as save_error:
            print(
                f"Failed to save error status to subtask character_ids: {save_error}")


class LatestExecutedRPD(BaseModel):
    """最近执行的RPD信息"""
    rpd_key: str = Field(..., description="RPD标识符")
    rpd_title: str = Field(..., description="RPD标题")
    version_number: int = Field(..., description="RPD版本号")
    executed_at: datetime = Field(..., description="执行时间")
    ai_review_version: int = Field(..., description="AI审核版本号")
    finding_count: int = Field(..., description="发现的问题数量")


class LatestExecutedRPDsResponse(BaseModel):
    """最近执行的RPD列表响应"""
    ai_review_version: Optional[int] = Field(None, description="AI审核版本号")
    executed_at: Optional[datetime] = Field(None, description="执行时间")
    executed_rpds: List[LatestExecutedRPD] = Field(
        default_factory=list, description="执行的RPD列表")
    total_findings: int = Field(0, description="总发现问题数")


@router.get(
    "/subtasks/{subtask_id}/latest-executed-rpds",
    response_model=LatestExecutedRPDsResponse,
    summary="Get latest executed RPDs for a subtask",
    description="获取子任务最近一次AI审核执行的所有RPD信息"
)
async def get_latest_executed_rpds(
    subtask_id: UUID,
    current_user: User = Security(get_current_user)
) -> LatestExecutedRPDsResponse:
    """获取子任务最近一次执行的RPD列表"""
    try:
        # 验证subtask存在
        subtask = await Subtask.get_or_none(id=subtask_id)
        if not subtask:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Subtask {subtask_id} not found"
            )

        # 获取最新AI审核的执行结果

        result = await get_latest_rpd_execution_summary(subtask_id)

        if not result['ai_review_version']:
            return LatestExecutedRPDsResponse(
                ai_review_version=None,
                executed_at=None,
                executed_rpds=[],
                total_findings=0
            )

        # 转换为响应格式
        executed_rpds = []
        for rpd_data in result['executed_rpds']:
            executed_rpds.append(LatestExecutedRPD(
                rpd_key=rpd_data['rpd_key'],
                rpd_title=rpd_data['rpd_title'],
                version_number=rpd_data['version_number'],
                executed_at=result['executed_at'],
                ai_review_version=result['ai_review_version'],
                finding_count=rpd_data['finding_count']
            ))

        return LatestExecutedRPDsResponse(
            ai_review_version=result['ai_review_version'],
            executed_at=result['executed_at'],
            executed_rpds=executed_rpds,
            total_findings=result['total_findings']
        )

    except HTTPException:
        raise
    except Exception as e:
        print(
            f"Error getting latest executed RPDs for subtask {subtask_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while getting latest executed RPDs."
        )


@router.post(
    "/{ai_review_id}/cancel",
    status_code=status.HTTP_200_OK,
    summary="Cancel AI Review processing",
    description="Cancel an ongoing AI Review process. This will set the cancellation signal "
    "and the processing will stop at the next checkpoint."
)
async def cancel_ai_review(
    ai_review_id: UUID,
    current_user: User = Security(get_current_user)
) -> JSONResponse:
    """
    取消正在进行的AI审查处理
    """
    try:
        from aimage_supervision.models import AiReview

        # 获取AI Review记录
        ai_review = await AiReview.get_or_none(id=ai_review_id)
        if not ai_review:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI Review not found"
            )

        # 检查当前状态是否可以取消
        if ai_review.processing_status == ProcessingStatusEnum.COMPLETED:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": "AI Review is already completed",
                    "ai_review_id": str(ai_review_id),
                    "current_status": ai_review.processing_status
                }
            )
        elif ai_review.processing_status == ProcessingStatusEnum.CANCELLED:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": "AI Review is already cancelled",
                    "ai_review_id": str(ai_review_id),
                    "current_status": ai_review.processing_status
                }
            )
        elif ai_review.processing_status == ProcessingStatusEnum.FAILED:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": "AI Review has already failed",
                    "ai_review_id": str(ai_review_id),
                    "current_status": ai_review.processing_status
                }
            )
        elif ai_review.processing_status == ProcessingStatusEnum.PENDING:
            # 对于PENDING状态，直接标记为CANCELLED
            ai_review.processing_status = ProcessingStatusEnum.CANCELLED
            ai_review.should_cancel = True
            await ai_review.save()
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": "AI Review cancelled before processing started",
                    "ai_review_id": str(ai_review_id),
                    "current_status": "cancelled"
                }
            )
        elif ai_review.processing_status != ProcessingStatusEnum.PROCESSING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel AI Review with status: {ai_review.processing_status}"
            )

        # 设置取消信号
        ai_review.should_cancel = True
        await ai_review.save()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "AI Review cancellation signal sent successfully",
                "ai_review_id": str(ai_review_id)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error cancelling AI Review {ai_review_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while cancelling AI Review."
        )


@router.get(
    "/tasks/{task_id}/review-set-recommendations",
    response_model=TaskReviewSetRecommendationsResponseService,
    summary="Get Review Set recommendations for a task",
    description="Returns Review Set recommendations based on the task's characters and tags, with 60/40 scoring algorithm."
)
async def get_task_review_set_recommendations_endpoint(
    task_id: UUID,
    project_id: UUID = Query(...,
                             description="Project ID to filter Review Sets"),
    min_score: float = Query(
        0.0, description="Minimum score to include", ge=0.0, le=10.0),
    current_user: User = Security(get_current_user)
) -> TaskReviewSetRecommendationsResponseService:
    """
    Get Review Set recommendations for a task based on associations

    Scoring algorithm:
    - Tag matching: 60% weight (1.0 point per matched tag, max 6.0)
    - Character matching: 40% weight (~0.67 point per matched character, max 4.0)
    """
    try:
        from ..services.review_set_recommendation_service import \
            get_task_review_set_recommendations

        return await get_task_review_set_recommendations(task_id, project_id, min_score)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors
        print(
            f"Unexpected error getting Review Set recommendations for task {task_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while getting Review Set recommendations."
        )
