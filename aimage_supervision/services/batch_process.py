import asyncio
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
from uuid import UUID

import httpx
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field
from tortoise.exceptions import DoesNotExist, IntegrityError
from tortoise.expressions import Q
from tortoise.transactions import in_transaction

from aimage_supervision.clients.aws_s3 import (
    download_file_content_from_s3, download_file_content_from_s3_sync,
    download_image_from_s3, download_video_from_s3, get_s3_url_from_path)
from aimage_supervision.enums import (AiReviewMode, AiReviewProcessingStatus,
                                      SubtaskType)
from aimage_supervision.models import (AiReview, AiReviewFindingEntry,
                                       Character, ReviewPointDefinition,
                                       ReviewPointDefinitionVersion, Subtask,
                                       User)
from aimage_supervision.prompts.review_point_prompts import (
    copyright_review_prompt, ng_review_prompt, visual_review_prompt)
from aimage_supervision.schemas import AiDetectedElement, AiDetectedElements
from aimage_supervision.schemas import AiReview as AiReviewSchema
from aimage_supervision.schemas import \
    AiReviewFindingEntry as AiReviewFindingEntrySchema
from aimage_supervision.schemas import (AiReviewFindingEntryInDB, AiReviewInDB,
                                        FindingArea)
from aimage_supervision.schemas import \
    ReviewPointDefinition as ReviewPointDefinitionSchema
from aimage_supervision.schemas import (ReviewPointDefinitionCreate,
                                        ReviewPointDefinitionVersionBase,
                                        ReviewPointDefinitionVersionInDB,
                                        Severity)
from aimage_supervision.services.ai_review_service import \
    initiate_ai_review_for_subtask
from aimage_supervision.settings import MAX_CONCURRENT, logger


async def batch_initiate_cr_check_parallel(
    subtasks: List[Subtask],
    initiated_by_user_id: UUID,
    max_concurrent: Optional[int] = None,
    mode: AiReviewMode = AiReviewMode.QUALITY
) -> None:
    """
    并行处理多个子任务的CR检查。

    这个函数设计为后台任务使用，所有结果通过日志记录。
    由于是后台任务，返回值会被FastAPI框架丢弃。

    Args:
        subtasks: 需要处理的子任务列表
        initiated_by_user_id: 发起用户的ID
        max_concurrent: 最大并发数量，默认从 settings 中读取

    Returns:
        None: 后台任务不需要返回值，所有信息通过日志输出
    """

    batch_start_time = time.time()
    batch_id = str(uuid.uuid4())[:8]  # 为这个批次创建一个短ID用于日志追踪

    # ========== 数据库记录 - 开始 ==========
    from datetime import datetime, timezone

    from aimage_supervision.enums import BatchJobStatus
    from aimage_supervision.models import BatchProcessJob, User

    # 获取项目ID（从第一个subtask获取，假设所有subtask属于同一个项目）
    project_id = None
    if subtasks:
        first_subtask = subtasks[0]
        await first_subtask.fetch_related('task__project')
        project_id = first_subtask.task.project.id if first_subtask.task and first_subtask.task.project else None

    # 创建批处理任务记录
    batch_job = await BatchProcessJob.create(
        batch_id=batch_id,
        job_name=f"AI_CR_Check_{batch_id}_{len(subtasks)}tasks",
        job_type="ai_review_cr_check",
        status=BatchJobStatus.RUNNING,
        created_by_id=initiated_by_user_id,
        project_id=project_id,
        total_items=len(subtasks),
        processed_items=0,
        successful_items=0,
        failed_items=0,
        started_at=datetime.now(timezone.utc),
        max_concurrent=max_concurrent if max_concurrent is not None else MAX_CONCURRENT,
        parameters={
            "subtask_ids": [str(subtask.id) for subtask in subtasks],
            "max_concurrent": max_concurrent if max_concurrent is not None else MAX_CONCURRENT,
            "initiated_by_user_id": str(initiated_by_user_id)
        }
    )

    logger.info(f"🗄️ [批次 {batch_id}] 创建数据库记录: {batch_job.id}")

    # 创建信号量来控制并发数量
    if max_concurrent is None:
        max_concurrent = MAX_CONCURRENT
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_subtask_with_limit(subtask: Subtask, task_index: int) -> Dict[str, Any]:
        """带并发限制的单个子任务处理"""
        subtask_short_id = str(subtask.id)[:8]

        # 等待获取信号量的时间
        wait_start_time = time.time()

        async with semaphore:
            wait_time = time.time() - wait_start_time
            actual_start_time = time.time()
            try:
                result = await initiate_ai_review_for_subtask(
                    subtask_id=subtask.id,
                    initiated_by_user_id=initiated_by_user_id,
                    cr_check=True,
                    mode=mode
                )

                processing_time = time.time() - actual_start_time
                total_time_from_batch_start = time.time() - batch_start_time
                # 计算整体严重程度
                overall_severity = "safe"  # 默认值
                if result.findings:
                    # 获取所有findings的severity
                    severities = [
                        finding.severity for finding in result.findings]
                    # 按严重程度排序：risk > alert > safe
                    severity_order = {"risk": 3, "alert": 2, "safe": 1,
                                      # 向后兼容旧值
                                      "high": 3, "medium": 2, "low": 1}
                    max_severity_value = max(
                        severities, key=lambda s: severity_order.get(s, 0))
                    overall_severity = max_severity_value
                return {
                    "subtask_id": str(subtask.id),
                    "subtask_short_id": subtask_short_id,
                    "task_index": task_index + 1,
                    "status": "success",
                    "ai_review_id": str(result.id),
                    "findings_count": len(result.findings) if result.findings else 0,
                    "severity": overall_severity,
                    "wait_time": wait_time,
                    "processing_time": processing_time,
                    "total_time_from_batch_start": total_time_from_batch_start,
                    "start_timestamp": actual_start_time,
                    "end_timestamp": time.time(),
                    "error": None
                }

            except Exception as e:
                processing_time = time.time() - actual_start_time
                total_time_from_batch_start = time.time() - batch_start_time
                error_msg = f"处理子任务 {subtask.id} 时出错: {str(e)}"
                return {
                    "subtask_id": str(subtask.id),
                    "subtask_short_id": subtask_short_id,
                    "task_index": task_index + 1,
                    "status": "error",
                    "ai_review_id": None,
                    "findings_count": 0,
                    "severity": None,
                    "wait_time": wait_time,
                    "processing_time": processing_time,
                    "total_time_from_batch_start": total_time_from_batch_start,
                    "start_timestamp": actual_start_time,
                    "end_timestamp": time.time(),
                    "error": error_msg
                }

    tasks = [process_single_subtask_with_limit(
        subtask, i) for i, subtask in enumerate(subtasks)]

    # 记录任务创建完成时间
    tasks_created_time = time.time()

    # 并行执行所有任务，使用 return_exceptions=True 确保单个任务失败不影响其他任务
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 记录所有任务完成时间
    all_tasks_completed_time = time.time()

    # 处理可能的异常结果
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # 如果 gather 本身出现异常
            logger.error(f"🔥 [批次 {batch_id}] 任务 {i+1} gather异常: {str(result)}")
            processed_results.append({
                "subtask_id": str(subtasks[i].id),
                "subtask_short_id": str(subtasks[i].id)[:8],
                "task_index": i + 1,
                "status": "error",
                "ai_review_id": None,
                "findings_count": 0,
                "severity": None,
                "wait_time": 0,
                "processing_time": 0,
                "total_time_from_batch_start": all_tasks_completed_time - batch_start_time,
                "start_timestamp": batch_start_time,
                "end_timestamp": all_tasks_completed_time,
                "error": f"批量处理异常: {str(result)}"
            })
        else:
            processed_results.append(result)  # type: ignore[arg-type]

    # 计算统计信息
    total_time = all_tasks_completed_time - batch_start_time
    successful_count = sum(
        1 for r in processed_results if r["status"] == "success")
    failed_count = len(processed_results) - successful_count

    # 创建 subtask_id 到 severity 的映射字典
    subtask_severity_map = {}
    for result in processed_results:
        if result["status"] == "success" and result["severity"]:
            subtask_severity_map[str(result["subtask_id"])
                                 ] = result["severity"]
        else:
            subtask_severity_map[str(result["subtask_id"])] = "failed"

    # 计算并行效率
    if successful_count > 0:
        avg_processing_time = sum(r["processing_time"] for r in processed_results if r["status"]
                                  == "success") / successful_count
        theoretical_serial_time = avg_processing_time * len(subtasks)
        parallel_efficiency = (theoretical_serial_time /
                               total_time) * 100 if total_time > 0 else 0

    # ========== 数据库记录 - 结束 ==========
    try:
        # 更新批处理任务记录
        if successful_count == 0 and failed_count > 0:
            # 全部失败
            batch_job.status = BatchJobStatus.FAILED
        else:
            # 有成功的任务（可能也有失败的）
            batch_job.status = BatchJobStatus.COMPLETED
        batch_job.processed_items = len(processed_results)
        batch_job.successful_items = successful_count
        batch_job.failed_items = failed_count
        batch_job.completed_at = datetime.now(timezone.utc)

        # 保存详细结果
        batch_job.results = {
            "summary": {
                "total_tasks": len(subtasks),
                "successful_count": successful_count,
                "failed_count": failed_count,
                "total_time_seconds": total_time,
                "parallel_efficiency_percent": parallel_efficiency if successful_count > 0 else 0,
                "avg_processing_time_seconds": avg_processing_time if successful_count > 0 else 0,
                "severity_counts": subtask_severity_map
            },
            "performance_metrics": {
                "batch_start_time": batch_start_time,
                "tasks_created_time": tasks_created_time,
                "all_tasks_completed_time": all_tasks_completed_time,
                "theoretical_serial_time": theoretical_serial_time if successful_count > 0 else 0
            }
        }

        await batch_job.save()

        logger.info(
            f"🗄️ [批次 {batch_id}] 更新数据库记录完成: 状态={batch_job.status.value}, 成功={successful_count}/{len(subtasks)}")

    except Exception as db_error:
        logger.error(f"🗄️ [批次 {batch_id}] 更新数据库记录失败: {str(db_error)}")
        # 即使数据库更新失败，也不影响主要任务的完成


async def expand_review_sets_to_rpd_ids(project_id: UUID, review_set_ids: Optional[List[UUID]] = None) -> List[UUID]:
    """
    将 Review Set IDs 展开为 RPD IDs

    Args:
        project_id: 项目ID
        review_set_ids: Review Set IDs 列表

    Returns:
        展开后的 RPD IDs 列表
    """
    if not review_set_ids:
        return []

    try:
        from aimage_supervision.models import ReviewSet

        # 查询所有指定的Review Sets，并预加载关联的RPDs
        review_sets = await ReviewSet.filter(
            id__in=review_set_ids,
            project_id=project_id,
        ).prefetch_related('rpds')

        # 收集所有RPD IDs
        rpd_ids = []
        for review_set in review_sets:
            for rpd in review_set.rpds:
                rpd_ids.append(rpd.id)

        # 去重并返回
        unique_rpd_ids = list(set(rpd_ids))

        logger.info(
            f"Expanded {len(review_set_ids)} review sets to {len(unique_rpd_ids)} RPD IDs "
            f"for project {project_id}"
        )

        return unique_rpd_ids

    except Exception as e:
        logger.error(f"Failed to expand review sets to RPD IDs: {str(e)}")
        return []


async def batch_initiate_parallel(
    subtasks: List[Subtask],
    initiated_by_user_id: UUID,
    rpd_ids: Optional[List[UUID]] = None,
    review_set_ids: Optional[List[UUID]] = None,
    mode: AiReviewMode = AiReviewMode.QUALITY,
    max_concurrent: Optional[int] = None,
    batch_id: Optional[str] = None
) -> None:
    """
    并行处理多个子任务的自定义AI审查。

    这个函数设计为后台任务使用，所有结果通过日志记录。
    由于是后台任务，返回值会被FastAPI框架丢弃。

    Args:
        subtasks: 需要处理的子任务列表
        initiated_by_user_id: 发起用户的ID
        rpd_ids: 可选的RPD IDs列表，如果提供则使用这些RPD
        review_set_ids: 可选的Review Set IDs列表，将被展开为RPD IDs
        mode: AI审查模式（quality/speed）
        max_concurrent: 最大并发数量，默认从 settings 中读取
        batch_id: 可选的批次ID，如果没有提供则自动生成

    Returns:
        None: 后台任务不需要返回值，所有信息通过日志输出
    """

    batch_start_time = time.time()
    # 使用传入的batch_id，如果没有则生成一个
    if not batch_id:
        batch_id = str(uuid.uuid4())[:8]

    # ========== 数据库记录 - 开始 ==========
    from datetime import datetime, timezone

    from aimage_supervision.enums import BatchJobStatus
    from aimage_supervision.models import BatchProcessJob, User

    # 获取项目ID（从第一个subtask获取，假设所有subtask属于同一个项目）
    project_id = None
    if subtasks:
        first_subtask = subtasks[0]
        await first_subtask.fetch_related('task__project')
        project_id = first_subtask.task.project.id if first_subtask.task and first_subtask.task.project else None

    # 展开Review Sets到RPD IDs
    expanded_rpd_ids = []
    if review_set_ids and project_id:
        expanded_rpd_ids = await expand_review_sets_to_rpd_ids(project_id, review_set_ids)

    # 合并所有RPD IDs
    final_rpd_ids = []
    if rpd_ids:
        final_rpd_ids.extend(rpd_ids)
    if expanded_rpd_ids:
        final_rpd_ids.extend(expanded_rpd_ids)

    # 去重
    final_rpd_ids = list(set(final_rpd_ids))

    # 创建批处理任务记录
    batch_job = await BatchProcessJob.create(
        batch_id=batch_id,
        job_name=f"AI_Review_{mode.value}_{batch_id}_{len(subtasks)}tasks",
        job_type="ai_review_custom",
        status=BatchJobStatus.RUNNING,
        created_by_id=initiated_by_user_id,
        project_id=project_id,
        total_items=len(subtasks),
        processed_items=0,
        successful_items=0,
        failed_items=0,
        started_at=datetime.now(timezone.utc),
        max_concurrent=max_concurrent if max_concurrent is not None else MAX_CONCURRENT,
        parameters={
            "subtask_ids": [str(subtask.id) for subtask in subtasks],
            "rpd_ids": [str(rpd_id) for rpd_id in final_rpd_ids] if final_rpd_ids else None,
            "review_set_ids": [str(rs_id) for rs_id in review_set_ids] if review_set_ids else None,
            "mode": mode.value,
            "max_concurrent": max_concurrent if max_concurrent is not None else MAX_CONCURRENT,
            "initiated_by_user_id": str(initiated_by_user_id)
        }
    )

    logger.info(f"🗄️ [批次 {batch_id}] 创建数据库记录: {batch_job.id}")

    # 创建信号量来控制并发数量
    if max_concurrent is None:
        max_concurrent = MAX_CONCURRENT
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_subtask_with_limit(subtask: Subtask, task_index: int) -> Dict[str, Any]:
        """带并发限制的单个子任务处理"""
        subtask_short_id = str(subtask.id)[:8]

        # 等待获取信号量的时间
        wait_start_time = time.time()

        async with semaphore:
            wait_time = time.time() - wait_start_time
            actual_start_time = time.time()
            try:
                result = await initiate_ai_review_for_subtask(
                    subtask_id=subtask.id,
                    initiated_by_user_id=initiated_by_user_id,
                    cr_check=False,  # 普通AI审查，不是CR检查
                    rpd_ids=final_rpd_ids,
                    mode=mode
                )

                processing_time = time.time() - actual_start_time
                total_time_from_batch_start = time.time() - batch_start_time
                # 计算整体严重程度
                overall_severity = "safe"  # 默认值
                if result.findings:
                    # 获取所有findings的severity
                    severities = [
                        finding.severity for finding in result.findings]
                    # 按严重程度排序：risk > alert > safe
                    severity_order = {"risk": 3, "alert": 2, "safe": 1,
                                      # 向后兼容旧值
                                      "high": 3, "medium": 2, "low": 1}
                    max_severity_value = max(
                        severities, key=lambda s: severity_order.get(s, 0))
                    overall_severity = max_severity_value
                return {
                    "subtask_id": str(subtask.id),
                    "subtask_short_id": subtask_short_id,
                    "task_index": task_index + 1,
                    "status": "success",
                    "ai_review_id": str(result.id),
                    "findings_count": len(result.findings) if result.findings else 0,
                    "severity": overall_severity,
                    "wait_time": wait_time,
                    "processing_time": processing_time,
                    "total_time_from_batch_start": total_time_from_batch_start,
                    "start_timestamp": actual_start_time,
                    "end_timestamp": time.time(),
                    "error": None
                }

            except Exception as e:
                processing_time = time.time() - actual_start_time
                total_time_from_batch_start = time.time() - batch_start_time
                error_msg = f"处理子任务 {subtask.id} 时出错: {str(e)}"
                return {
                    "subtask_id": str(subtask.id),
                    "subtask_short_id": subtask_short_id,
                    "task_index": task_index + 1,
                    "status": "error",
                    "ai_review_id": None,
                    "findings_count": 0,
                    "severity": None,
                    "wait_time": wait_time,
                    "processing_time": processing_time,
                    "total_time_from_batch_start": total_time_from_batch_start,
                    "start_timestamp": actual_start_time,
                    "end_timestamp": time.time(),
                    "error": error_msg
                }

    tasks = [process_single_subtask_with_limit(
        subtask, i) for i, subtask in enumerate(subtasks)]

    # 记录任务创建完成时间
    tasks_created_time = time.time()

    # 并行执行所有任务，使用 return_exceptions=True 确保单个任务失败不影响其他任务
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 记录所有任务完成时间
    all_tasks_completed_time = time.time()

    # 处理可能的异常结果
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # 如果 gather 本身出现异常
            logger.error(f"🔥 [批次 {batch_id}] 任务 {i+1} gather异常: {str(result)}")
            processed_results.append({
                "subtask_id": str(subtasks[i].id),
                "subtask_short_id": str(subtasks[i].id)[:8],
                "task_index": i + 1,
                "status": "error",
                "ai_review_id": None,
                "findings_count": 0,
                "severity": None,
                "wait_time": 0,
                "processing_time": 0,
                "total_time_from_batch_start": all_tasks_completed_time - batch_start_time,
                "start_timestamp": batch_start_time,
                "end_timestamp": all_tasks_completed_time,
                "error": f"批量处理异常: {str(result)}"
            })
        else:
            processed_results.append(result)  # type: ignore[arg-type]

    # 计算统计信息
    total_time = all_tasks_completed_time - batch_start_time
    successful_count = sum(
        1 for r in processed_results if r["status"] == "success")
    failed_count = len(processed_results) - successful_count

    # 创建 subtask_id 到 severity 的映射字典
    subtask_severity_map = {}
    for result in processed_results:
        if result["status"] == "success" and result["severity"]:
            subtask_severity_map[str(result["subtask_id"])
                                 ] = result["severity"]
        else:
            subtask_severity_map[str(result["subtask_id"])] = "failed"

    # 计算并行效率
    if successful_count > 0:
        avg_processing_time = sum(r["processing_time"] for r in processed_results if r["status"]
                                  == "success") / successful_count
        theoretical_serial_time = avg_processing_time * len(subtasks)
        parallel_efficiency = (theoretical_serial_time /
                               total_time) * 100 if total_time > 0 else 0

    # ========== 数据库记录 - 结束 ==========
    try:
        # 更新批处理任务记录
        if successful_count == 0 and failed_count > 0:
            # 全部失败
            batch_job.status = BatchJobStatus.FAILED
        else:
            # 有成功的任务（可能也有失败的）
            batch_job.status = BatchJobStatus.COMPLETED
        batch_job.processed_items = len(processed_results)
        batch_job.successful_items = successful_count
        batch_job.failed_items = failed_count
        batch_job.completed_at = datetime.now(timezone.utc)

        # 保存详细结果
        batch_job.results = {
            "summary": {
                "total_tasks": len(subtasks),
                "successful_count": successful_count,
                "failed_count": failed_count,
                "total_time_seconds": total_time,
                "parallel_efficiency_percent": parallel_efficiency if successful_count > 0 else 0,
                "avg_processing_time_seconds": avg_processing_time if successful_count > 0 else 0,
                "severity_counts": subtask_severity_map,
                "mode": mode.value,
                "rpd_count": len(final_rpd_ids) if final_rpd_ids else 0,
                "review_set_count": len(review_set_ids) if review_set_ids else 0
            },
            "performance_metrics": {
                "batch_start_time": batch_start_time,
                "tasks_created_time": tasks_created_time,
                "all_tasks_completed_time": all_tasks_completed_time,
                "theoretical_serial_time": theoretical_serial_time if successful_count > 0 else 0
            }
        }

        await batch_job.save()

        logger.info(
            f"🗄️ [批次 {batch_id}] 更新数据库记录完成: 状态={batch_job.status.value}, 成功={successful_count}/{len(subtasks)}")

    except Exception as db_error:
        logger.error(f"🗄️ [批次 {batch_id}] 更新数据库记录失败: {str(db_error)}")
        # 即使数据库更新失败，也不影响主要任务的完成
