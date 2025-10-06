# -*- coding: utf-8 -*-
"""
Multi-Agent视觉审查管道

特点：
1. 独立新文件，不影响现有代码
2. 与原函数相同的接口签名
3. 集成bounding box重点区域分析
4. Gemini 2.5三阶段Multi-Agent处理
5. 可直接替代原函数使用
"""

import asyncio
import os
from typing import Optional, Tuple

from aimage_supervision.clients.aws_s3 import \
    download_file_content_from_s3_sync
from aimage_supervision.prompts.review_point_prompts import \
    visual_review_prompt
from aimage_supervision.schemas import ReviewPointDefinitionVersionInDB
from aimage_supervision.services.ai_review_pipeline import BoundingBoxes
from aimage_supervision.services.visual_review_multi_agent import (
    ProcessingConfig, VisualReviewMultiAgent)
from aimage_supervision.settings import logger

# ================================================================================================
# Multi-Agent Visual Review System - Direct Integration
# ================================================================================================


def generate_findings_for_visual_review_multi_agent_sync(
    image_bytes: bytes,
    rpd_version: ReviewPointDefinitionVersionInDB,
    bounding_boxes: BoundingBoxes,
    model_name: str
) -> Tuple[str, str, str]:
    """
    Multi-Agent视觉审查

    独立的纯同步实现，遵循原系统架构模式。
    用于在asyncio.to_thread()线程池中执行，确保线程安全。

    输入：与异步版本一致
    输出：(description, severity, suggestion) - 日语结果
    """
    logger.info("[Multi-Agent Sync] 开始处理RPD: %s", rpd_version.title)

    try:
        # 获取API密钥
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("[Multi-Agent Sync] GEMINI_API_KEY not found")
            return (
                "Multi-Agentシステムの初期化に失敗しましたが、安全な結果を返します。",
                "safe",
                "",
            )

        # 创建配置（与异步版本相同）
        tta_scales_str = os.getenv('MULTI_AGENT_TTA_SCALES', '1.0')
        tta_scales = [float(x.strip()) for x in tta_scales_str.split(',')]

        config = ProcessingConfig(
            classify_enum_thinking=int(
                os.getenv('MULTI_AGENT_CLASSIFY_ENUM_THINKING', '0')),
            classify_evidence_thinking_low=int(
                os.getenv('MULTI_AGENT_EVIDENCE_THINKING_LOW', '2048')),
            classify_evidence_thinking_high=int(
                os.getenv('MULTI_AGENT_EVIDENCE_THINKING_HIGH', '4096')),
            tta_scales=tta_scales,
            confidence_threshold=float(
                os.getenv('MULTI_AGENT_CONFIDENCE_THRESHOLD', '0.8')),
            classify_label_thinking=int(
                os.getenv('MULTI_AGENT_CLASSIFY_LABEL_THINKING', '4096')),
        )

        # 初始化Multi-Agent系统
        multi_agent = VisualReviewMultiAgent(api_key, config)

        # 从S3下载参考图片到内存（同步）
        reference_image_bytes = []
        if rpd_version.reference_images:
            for s3_path in rpd_version.reference_images:
                image_data = download_file_content_from_s3_sync(s3_path)
                reference_image_bytes.append(image_data)

        # 使用asyncio.run调用Multi-Agent处理（在独立线程中安全）
        result = asyncio.run(multi_agent.process_single_rpd(
            image_bytes=image_bytes,
            rpd=rpd_version,
            reference_image_bytes=reference_image_bytes,
            base_prompt=visual_review_prompt,
            bounding_boxes=bounding_boxes,
            model_name=model_name,
        ))

        logger.info("[Multi-Agent Sync] RPD处理完成: %s -> severity=%s",
                    rpd_version.title, result[1])
        return result

    except Exception as e:
        logger.exception("[Multi-Agent Sync] 处理失败 %s", rpd_version.title)
        return (
            "Multi-Agentシステム（同期）でエラーが発生しましたが、安全な結果を返します。",
            "safe",
            "",
        )


# ================================================================================================
# 模块说明文档
# ================================================================================================
"""
只需要在ai_review_service的调用处替换函数名即可
   
   ```python
   # 原代码
   result = await _generate_findings_for_visual_review(image, rpd, boxes, model)
   
   # 使用Multi-Agent
   from aimage_supervision.services.ai_review_pipeline_multi_agent import generate_findings_for_visual_review_multi_agent
   result = await generate_findings_for_visual_review_multi_agent(image, rpd, boxes, model)
   ```
"""
