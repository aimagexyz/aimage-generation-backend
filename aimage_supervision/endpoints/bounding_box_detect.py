import asyncio
import os
from typing import List, Optional

import httpx
import numpy as np

from aimage_supervision.schemas import ReviewPointDefinitionVersionInDB
from aimage_supervision.services.ai_review_pipeline import (BoundingBox,
                                                            BoundingBoxes)
from aimage_supervision.settings import AGI_API_KEY, AGI_SERVER_URL, logger


async def check_task_status(task_id: str):
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
        logger.debug(f"Checking task status for {task_id} at {AGI_SERVER_URL}")

        response = await client.get(
            f"{AGI_SERVER_URL}/api/v1/process-video/status/{task_id}",
            headers={
                "X-API-Key": AGI_API_KEY
            }
        )
        return response.json()


async def detect_bounding_boxes(image_bytes: bytes, text: str):
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
        logger.debug(f"Detecting bounding boxes for text: {text}")

        # 使用 files 参数来发送 multipart/form-data
        files = {
            "image_file": ("image.png", image_bytes, "image/png")
        }
        data = {
            "text": text
        }

        response = await client.post(
            f"{AGI_SERVER_URL}/api/v1/grounding/detect",
            headers={
                "X-API-Key": AGI_API_KEY
            },
            files=files,
            data=data
        )
        response.raise_for_status()  # 检查HTTP错误
        return response.json()


def compute_iou(box1, box2):
    """计算两个边界框的IoU（交集比并集）"""
    # box格式: [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x1 >= x2 or y1 >= y2:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def intersection_box(box1, box2):
    """返回两个框的交集区域，若无交集返回None"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 < x2 and y1 < y2:
        return [x1, y1, x2, y2]
    else:
        return None


def merge_overlapping_boxes_to_intersection(boxes):
    """尽量两两合并重叠框为交集，直到所有框无重叠"""
    if len(boxes) == 0:
        return np.array([])

    boxes = [list(b) for b in boxes]
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            merged = False
            for j in range(i+1, len(boxes)):
                if used[j]:
                    continue
                iou = compute_iou(boxes[i], boxes[j])
                if iou > 0:
                    inter = intersection_box(boxes[i], boxes[j])
                    if inter is not None:
                        new_boxes.append(inter)
                        used[i] = used[j] = True
                        changed = True
                        merged = True
                        break
            if not merged:
                new_boxes.append(boxes[i])
                used[i] = True
        boxes = new_boxes
    return np.array(boxes)


def _convert_api_response_to_bounding_boxes(api_response: dict, image_width: Optional[int] = None, image_height: Optional[int] = None) -> BoundingBoxes:
    """
    将新API返回的结果转换为现有的BoundingBoxes格式，并应用边界框合并

    新API返回格式:
    {
      "detections": [
        {
          "box": [xmin, ymin, xmax, ymax],  # 像素坐标
          "label": "...",
          "score": 0.xxx
        }
      ]
    }

    现有格式需要坐标标准化到0-1000范围
    """
    detections = api_response.get("detections", [])
    if not detections:
        return BoundingBoxes(bounding_boxes=[])

    # 第一步：收集所有标准化后的边界框和置信度
    normalized_boxes = []
    scores = []

    for detection in detections:
        box = detection.get("box", [])
        score = detection.get("score", 0.0)

        if len(box) == 4:
            xmin, ymin, xmax, ymax = box

            # 坐标标准化逻辑
            if image_width is None or image_height is None:
                if max(xmin, ymin, xmax, ymax) > 100:
                    # 假设是像素坐标，使用默认图像尺寸标准化
                    estimated_width = 1920
                    estimated_height = 1080
                    normalized_xmin = int(xmin * 1000 / estimated_width)
                    normalized_ymin = int(ymin * 1000 / estimated_height)
                    normalized_xmax = int(xmax * 1000 / estimated_width)
                    normalized_ymax = int(ymax * 1000 / estimated_height)
                else:
                    # 假设已经是归一化坐标（0-1）
                    normalized_xmin = int(xmin * 1000)
                    normalized_ymin = int(ymin * 1000)
                    normalized_xmax = int(xmax * 1000)
                    normalized_ymax = int(ymax * 1000)
            else:
                # 使用提供的图像尺寸标准化
                normalized_xmin = int(xmin * 1000 / image_width)
                normalized_ymin = int(ymin * 1000 / image_height)
                normalized_xmax = int(xmax * 1000 / image_width)
                normalized_ymax = int(ymax * 1000 / image_height)

            # 确保坐标在0-1000范围内
            normalized_xmin = max(0, min(1000, normalized_xmin))
            normalized_ymin = max(0, min(1000, normalized_ymin))
            normalized_xmax = max(0, min(1000, normalized_xmax))
            normalized_ymax = max(0, min(1000, normalized_ymax))

            # 确保max坐标不小于min坐标
            if normalized_xmax <= normalized_xmin:
                normalized_xmax = normalized_xmin + 1
            if normalized_ymax <= normalized_ymin:
                normalized_ymax = normalized_ymin + 1

            normalized_boxes.append(
                [normalized_xmin, normalized_ymin, normalized_xmax, normalized_ymax])
            scores.append(score)

    # 第二步：应用边界框合并算法
    if len(normalized_boxes) > 1:
        # 合并重叠的边界框
        merged_boxes = merge_overlapping_boxes_to_intersection(
            normalized_boxes)

        if len(merged_boxes) == 0:
            return BoundingBoxes(bounding_boxes=[])

        # 为合并后的边界框分配置信度（使用参与合并的边界框的最高置信度）
        final_boxes = []
        for merged_box in merged_boxes:
            # 找到与合并框最相关的原始框的置信度
            max_score = 0.0
            for i, original_box in enumerate(normalized_boxes):
                iou = compute_iou(merged_box, original_box)
                if iou > 0:
                    max_score = max(max_score, scores[i])

            # 如果没有找到重叠，使用平均置信度
            if max_score < 1e-6:  # 避免浮点数精确比较
                max_score = sum(scores) / len(scores) if scores else 0.0

            final_boxes.append((merged_box, max_score))
    else:
        # 只有一个或零个边界框，无需合并
        final_boxes = [(box, score)
                       for box, score in zip(normalized_boxes, scores)]

    # 第三步：创建BoundingBox对象
    bounding_boxes = []
    for (box, score) in final_boxes:
        xmin, ymin, xmax, ymax = box
        bounding_box = BoundingBox(
            xmin=int(xmin),
            ymin=int(ymin),
            xmax=int(xmax),
            ymax=int(ymax),
            confidence=float(score)
        )
        bounding_boxes.append(bounding_box)

    logger.debug(
        f"Converted {len(detections)} detections to {len(bounding_boxes)} merged bounding boxes")
    return BoundingBoxes(bounding_boxes=bounding_boxes)


async def detect_bounding_boxes_new(image_bytes: bytes, text: str) -> BoundingBoxes:
    """
    使用新的API检测边界框，替换旧的_detect_bounding_boxes_sync函数
    """
    try:
        # 调用新的API
        api_response = await detect_bounding_boxes(image_bytes, text)

        # 转换API响应为现有格式
        bounding_boxes = _convert_api_response_to_bounding_boxes(api_response)

        logger.debug(
            f"Detected {len(bounding_boxes.bounding_boxes)} bounding boxes using new API")
        return bounding_boxes

    except Exception as e:
        logger.error(f"Error in detect_bounding_boxes_new: {str(e)}")
        return BoundingBoxes(bounding_boxes=[])


def detect_bounding_boxes_new_sync(image_bytes: bytes, text: str) -> BoundingBoxes:
    """
    同步版本的新边界框检测函数，用于在线程中执行
    """
    try:
        # 在新的事件循环中运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                detect_bounding_boxes_new(image_bytes, text))
            return result
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Error in detect_bounding_boxes_new_sync: {str(e)}")
        return BoundingBoxes(bounding_boxes=[])

