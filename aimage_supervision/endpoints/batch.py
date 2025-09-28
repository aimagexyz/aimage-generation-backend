import asyncio
import json
import os
import tempfile
import time
import urllib.parse
import uuid
from typing import Annotated, Any, Dict, List, Optional
from uuid import UUID

import boto3
import pandas as pd
from fastapi import (APIRouter, Depends, HTTPException, Query, Response,
                     Security, status)
from fastapi.responses import PlainTextResponse
from openpyxl import load_workbook
from pydantic import BaseModel
from tortoise.contrib.fastapi import HTTPNotFoundError

from ..clients.aws_s3 import download_file_content_from_s3_sync
from ..enums import BatchJobStatus
from ..middlewares.auth import get_current_user
from ..models import BatchProcessJob, BatchProcessJobOut, User
from ..settings import MAX_CONCURRENT


class BatchProcessingStats(BaseModel):
    """批处理统计信息响应模型"""
    total_batches: int
    success_rate_percentage: float
    failed_batches: int
    average_processing_time_seconds: float
    total_tasks_processed: int
    successful_batches: int


class BatchTaskResult(BaseModel):
    """批处理任务结果详情"""
    task_id: str  # 这里是subtask_id，保持原有字段名
    task_name: str
    status: str  # 'success' | 'failed' | 'skipped'
    error_message: Optional[str] = None
    findings_count: Optional[int] = None
    severity: Optional[str] = None  # 'high' | 'medium' | 'low' | None
    created_at: str
    updated_at: str
    # 添加navigation需要的字段
    subtask_id: str  # subtask的UUID
    parent_task_id: Optional[str] = None  # 父task的UUID


class BatchProcessJobListItem(BaseModel):
    """批处理任务列表项响应模型 - 简化版，只包含列表显示需要的字段"""
    id: str
    batch_id: str
    job_type: str  # 处理类型
    status: str
    created_at: str
    completed_at: Optional[str] = None
    total_items: int
    successful_items: int
    failed_items: int
    duration_seconds: Optional[float] = None
    # 用户信息 - 只包含显示和搜索需要的字段
    initiated_by_user_name: Optional[str] = None
    initiated_by_user_email: Optional[str] = None


class BatchProcessingRecord(BaseModel):
    """批处理记录详情响应模型"""
    id: str
    batch_id: str
    project_id: Optional[str] = None
    processing_type: str  # job_type 映射到 processing_type
    status: str
    initiated_by_user_id: Optional[str] = None
    initiated_by_user_name: Optional[str] = None
    initiated_by_user_email: Optional[str] = None
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    max_concurrent_tasks: int = MAX_CONCURRENT  # 最大并发任务数
    total_processing_time_seconds: float
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    task_results: List[BatchTaskResult] = []
    error_summary: Optional[str] = None


router = APIRouter(
    prefix='/batch',
    tags=['Batch Process Jobs'],
)


@router.get(
    '',
    response_model=List[BatchProcessJobListItem],
    description='List all batch process jobs.',
)
async def get_batch_process_jobs(
    user: Annotated[User, Security(get_current_user)],
    project_id: Optional[UUID] = None,
) -> List[BatchProcessJobListItem]:
    """
    Retrieve batch process jobs, optionally filtered by project_id.
    TODO: Add pagination.
    """
    # 构建查询条件
    if project_id:
        jobs = await BatchProcessJob.filter(project_id=project_id).prefetch_related('created_by').order_by('-created_at')
    else:
        jobs = await BatchProcessJob.all().prefetch_related('created_by').order_by('-created_at')

    # 手动构建简化响应，只包含列表显示必要的字段
    job_list = []
    for job in jobs:
        job_item = BatchProcessJobListItem(
            id=str(job.id),
            batch_id=job.batch_id,
            job_type=job.job_type,
            status=job.status.value,  # 确保返回字符串值
            created_at=job.created_at.isoformat() if job.created_at else '',
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            total_items=job.total_items,
            successful_items=job.successful_items,
            failed_items=job.failed_items,
            duration_seconds=job.duration_seconds(),
            # 用户信息 - 只包含显示和搜索需要的字段
            initiated_by_user_name=job.created_by.display_name if job.created_by else None,
            initiated_by_user_email=job.created_by.email if job.created_by else None,
        )
        job_list.append(job_item)

    return job_list


@router.get(
    '/stats',
    response_model=BatchProcessingStats,
    description='Get batch processing statistics.',
)
async def get_batch_processing_stats(
    user: Annotated[User, Security(get_current_user)],
    project_id: Optional[UUID] = None,
) -> BatchProcessingStats:
    """
    获取批处理统计信息，包括总批次数、成功率、失败数、平均处理时间等。
    可以按项目过滤。
    """
    # 获取批处理任务，根据项目过滤
    if project_id:
        all_jobs = await BatchProcessJob.filter(project_id=project_id)
    else:
        all_jobs = await BatchProcessJob.all()

    if not all_jobs:
        return BatchProcessingStats(
            total_batches=0,
            success_rate_percentage=0.0,
            failed_batches=0,
            average_processing_time_seconds=0.0,
            total_tasks_processed=0,
            successful_batches=0,
        )

    # 计算统计信息
    total_batches = len(all_jobs)
    successful_batches = 0
    failed_batches = 0
    total_tasks_processed = 0
    total_processing_time = 0.0
    processing_time_count = 0

    for job in all_jobs:
        # 累计处理的任务总数
        total_tasks_processed += job.processed_items

        # 判断批次状态
        if job.status == BatchJobStatus.COMPLETED:
            successful_batches += 1
        elif job.status == BatchJobStatus.FAILED:
            failed_batches += 1

        # 计算处理时间
        duration = job.duration_seconds()
        if duration is not None and duration > 0:
            total_processing_time += duration
            processing_time_count += 1

    # 计算成功率
    success_rate_percentage = (
        successful_batches / total_batches * 100.0) if total_batches > 0 else 0.0

    # 计算平均处理时间
    average_processing_time_seconds = (
        total_processing_time / processing_time_count) if processing_time_count > 0 else 0.0

    return BatchProcessingStats(
        total_batches=total_batches,
        success_rate_percentage=round(success_rate_percentage, 1),
        failed_batches=failed_batches,
        average_processing_time_seconds=round(
            average_processing_time_seconds, 1),
        total_tasks_processed=total_tasks_processed,
        successful_batches=successful_batches,
    )


@router.get(
    '/{job_id}/detail',
    response_model=BatchProcessingRecord,
    description='Get detailed batch processing record by ID.',
    responses={404: {'description': 'Batch process job not found'}},
)
async def get_batch_processing_detail(
    job_id: UUID,
    user: Annotated[User, Security(get_current_user)],
) -> BatchProcessingRecord:
    """
    获取批处理任务的详细信息，包括任务结果明细。
    """
    # 获取批处理任务，预加载关联的用户和项目信息
    job = await BatchProcessJob.get_or_none(id=job_id).prefetch_related('created_by', 'project')

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Batch process job not found',
        )

    # 解析任务结果
    task_results = []
    if job.results and isinstance(job.results, dict):
        summary = job.results.get('summary', {})
        severity_counts = summary.get('severity_counts', {})

        # 获取所有相关的subtask信息，同时预加载task信息
        subtask_info_map = {}
        if severity_counts:
            from ..models import Subtask
            subtask_ids = [UUID(sid) for sid in severity_counts.keys() if sid]
            subtasks = await Subtask.filter(id__in=subtask_ids).prefetch_related('task').all()
            subtask_info_map = {
                str(subtask.id): subtask for subtask in subtasks}

        # 如果有详细的任务结果，解析它们
        if 'task_details' in job.results:
            task_details = job.results['task_details']
            for task_detail in task_details:
                subtask_id = task_detail.get('subtask_id', '')
                subtask = subtask_info_map.get(subtask_id)
                task_result = BatchTaskResult(
                    task_id=subtask_id,
                    task_name=subtask.name if subtask else f"Task {task_detail.get('task_index', '')}",
                    status=task_detail.get('status', 'unknown'),
                    error_message=task_detail.get('error'),
                    findings_count=task_detail.get('findings_count'),
                    severity=severity_counts.get(subtask_id),
                    created_at=job.created_at.isoformat() if job.created_at else '',
                    updated_at=job.updated_at.isoformat() if job.updated_at else '',
                    subtask_id=subtask_id,
                    parent_task_id=str(
                        subtask.task.id) if subtask and subtask.task else None,
                )
                task_results.append(task_result)
        else:
            # 从 severity_counts 创建任务结果
            for subtask_id, severity in severity_counts.items():
                subtask = subtask_info_map.get(subtask_id)

                task_result = BatchTaskResult(
                    task_id=subtask_id,
                    task_name=subtask.name if subtask else f"Subtask {subtask_id[:8]}",
                    status='success',  # severity_counts中的都是成功的
                    error_message=None,
                    findings_count=None,  # 暂时无法从summary获取
                    severity=severity,
                    created_at=job.created_at.isoformat() if job.created_at else '',
                    updated_at=job.updated_at.isoformat() if job.updated_at else '',
                    subtask_id=subtask_id,
                    parent_task_id=str(
                        subtask.task.id) if subtask and subtask.task else None,
                )
                task_results.append(task_result)

    # 按严重程度排序：high > medium > low，然后按 task_name 排序
    def get_severity_priority(severity):
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        return severity_order.get(severity, 3)  # 无severity的排在最后

    task_results.sort(key=lambda x: (
        get_severity_priority(x.severity), x.task_name))

    # 如果没有任何结果数据，且批处理已完成，创建基础汇总信息
    # 但对于正在运行或待处理的batch，不应该显示假数据
    if (not task_results and job.total_items > 0 and
            job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]):
        # 创建汇总的任务结果条目
        for i in range(min(job.total_items, 10)):  # 限制显示数量
            status = 'success' if i < job.successful_items else (
                'failed' if i < job.successful_items + job.failed_items else 'skipped')
            task_result = BatchTaskResult(
                task_id=f"task_{i+1}",
                task_name=f"Task {i+1}",
                status=status,
                severity=None,
                created_at=job.created_at.isoformat() if job.created_at else '',
                updated_at=job.updated_at.isoformat() if job.updated_at else '',
                subtask_id=f"task_{i+1}",  # 无实际数据时的占位符
                parent_task_id=None,
            )
            task_results.append(task_result)

    # 映射 job_type 到 processing_type
    processing_type_mapping = {
        'ai_review_cr_check': 'copyright_check',
        'scenario_check': 'scenario_check',
        'image_analysis': 'image_analysis',
        'content_review': 'content_review',
    }
    processing_type = processing_type_mapping.get(job.job_type, job.job_type)

    # 映射状态
    status_mapping = {
        BatchJobStatus.PENDING: 'pending',
        BatchJobStatus.RUNNING: 'running',
        BatchJobStatus.COMPLETED: 'completed',
        BatchJobStatus.FAILED: 'failed',
        BatchJobStatus.CANCELLED: 'failed',
    }
    mapped_status = status_mapping.get(job.status, str(job.status))

    # 计算跳过的任务数 (total - processed)
    skipped_tasks = max(0, job.total_items - job.processed_items)

    return BatchProcessingRecord(
        id=str(job.id),
        batch_id=job.batch_id,
        project_id=str(job.project.id) if job.project else None,
        processing_type=processing_type,
        status=mapped_status,
        initiated_by_user_id=str(
            job.created_by.id) if job.created_by else None,
        initiated_by_user_name=job.created_by.display_name if job.created_by else None,
        initiated_by_user_email=job.created_by.email if job.created_by else None,
        total_tasks=job.total_items,
        successful_tasks=job.successful_items,
        failed_tasks=job.failed_items,
        skipped_tasks=skipped_tasks,
        max_concurrent_tasks=job.max_concurrent,  # 返回实际的最大并发数
        total_processing_time_seconds=job.duration_seconds() or 0.0,
        created_at=job.created_at.isoformat() if job.created_at else '',
        updated_at=job.updated_at.isoformat() if job.updated_at else '',
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        task_results=task_results,
        error_summary=job.error_message,
    )


@router.get(
    '/{job_id}',
    response_model=BatchProcessJobOut,
    description='Get a single batch process job by ID.',
    responses={404: {'description': 'Job not found'}},
)
async def get_batch_process_job(
    job_id: UUID,
    user: Annotated[User, Security(get_current_user)],
) -> BatchProcessJobOut:
    """
    Retrieve a single batch process job by its UUID.
    """
    job = await BatchProcessJob.get_or_none(id=job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Batch process job not found',
        )
    return await BatchProcessJobOut.from_tortoise_orm(job)
