from aimage_supervision.settings import logger
import os
from typing import Dict, List, Optional, Tuple

from google import genai
from pydantic import BaseModel, Field

from aimage_supervision.clients.aws_s3 import \
    download_file_content_from_s3_sync
from aimage_supervision.prompts.review_point_prompts import (
    copyright_review_prompt, ng_review_prompt, visual_review_prompt)
from aimage_supervision.schemas import (FindingArea,
                                        ReviewPointDefinitionVersionInDB,
                                        Severity)
from aimage_supervision.services.ng_check_agents import NGCheckPipeline



class GeminiDetectedElements(BaseModel):
    description: str = Field(
        ..., description="A comprehensive textual description of the key visual elements in the image.")
    characters: List[str] = Field(...,
                                  description="A list of detected characters in the image.")
    objects: List[str] = Field(...,
                               description="A list of detected objects in the image. Max 3 objects.")


class AiReviewFinding(BaseModel):
    description: str = Field(...,
                             description="Detailed description of the finding in Japanese!")
    severity: Severity = Field(...,
                               description="Severity of the finding (high, medium, or low).")
    suggestion: Optional[str] = Field(
        None, description="Suggestion to fix the issue in Japanese!")
    area: FindingArea = Field(...,
                              description="Bounding box area of the finding.")


class AiReviewFindings(BaseModel):
    description: str = Field(...,
                             description="Detailed description of input image.")
    findings: List[AiReviewFinding]


class Copyright(BaseModel):
    copyright_mark: str = Field(...,
                                description="The copyright mark in the image.")

# --- Service function to get findings for one review point from Gemini ---


class BoundingBox(BaseModel):
    xmin: int = Field(...,
                      description="The x-coordinate normalized to 0-1000.")
    ymin: int = Field(...,
                      description="The y-coordinate normalized to 0-1000.")
    xmax: int = Field(...,
                      description="The x-coordinate normalized to 0-1000.")
    ymax: int = Field(...,
                      description="The y-coordinate normalized to 0-1000.")
    confidence: float = Field(...,
                              description="The confidence score of the bounding box.")


class BoundingBoxes(BaseModel):
    bounding_boxes: List[BoundingBox] = Field(...,
                                              description="The bounding boxes of the objects in the image.")


async def _detect_bounding_boxes(image_bytes: bytes, rpd_version: ReviewPointDefinitionVersionInDB, model_name: str) -> BoundingBoxes:
    """
    Detects bounding boxes in an image using Gemini.
    """
    # 为这个任务创建独立的 Gemini 客户端
    task_gemini_client = genai.Client(api_key=os.getenv(
        'GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None
    if not task_gemini_client:
        logger.error("GEMINI_API_KEY not set. Cannot detect bounding boxes.")
        return BoundingBoxes(bounding_boxes=[])

    prompt_parts = [
        "Image:",
        genai.types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
    ]

    if rpd_version.parent_key == 'copyright_review':
        system_instruction = "Detect the 2d bounding boxes of the copyright, which is the small black text in the bottom of the image.  box_2d [xmin, ymin, xmax, ymax] and normalized to 0-1000. Please return the confidence score for each bounding box you detected."
    else:
        system_instruction = f"Detect all objects that are related to {rpd_version.title} in the image. box_2d [xmin, ymin, xmax, ymax] and normalized to 0-1000. Please return the confidence score for each bounding box you detected."
    # Using a model that supports JSON output and good instruction following. Adjust if needed.
    logger.info(f'system_instruction: {system_instruction}')

    logger.info(
        f'rpd_version details: ID={rpd_version.id}, title={rpd_version.title}, parent_key={rpd_version.parent_key}')
    response = task_gemini_client.models.generate_content(
        model=model_name,
        contents=prompt_parts,
        config=genai.types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=BoundingBoxes,
            system_instruction=system_instruction,
            temperature=0.1,
        )
    )
    logger.info(f'parsed response: {response.parsed}')
    return response.parsed


def convert_bounding_boxes_to_x_y_width_height(bounding_boxes: BoundingBoxes, image_width: int, image_height: int) -> List[Dict[str, int]]:
    """
    Converts bounding boxes to x, y, width, height.
    """
    # convert to x, y, width, height
    _bounding_boxes = []
    for bounding_box in bounding_boxes.bounding_boxes:
        x = int(bounding_box.xmin * image_width / 1000)
        y = int(bounding_box.ymin * image_height / 1000)
        width = int((bounding_box.xmax - bounding_box.xmin) *
                    image_width / 1000)
        height = int((bounding_box.ymax - bounding_box.ymin) *
                     image_height / 1000)
        _bounding_boxes.append({
            'x': x,
            'y': y,
            'width': width,
            'height': height
        })

    return _bounding_boxes


def _generate_findings_for_ng_review_sync(
    image_bytes: bytes,
    rpd_version: ReviewPointDefinitionVersionInDB,
    model_name: str
) -> List[Dict[str, str | None]]:
    """
    同步版本的NG审查findings生成，用于在线程中执行
    """
    # 为这个任务创建独立的 Gemini 客户端
    task_gemini_client = genai.Client(api_key=os.getenv(
        'GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None
    if not task_gemini_client:
        print("Error: GEMINI_API_KEY not set. Cannot generate NG review findings.")
        return []
    print("rpd_version.ng_subcategory: ", rpd_version.ng_subcategory)
    if rpd_version.ng_subcategory == 'abstract_type':
        print("rpd_version.ng_subcategory is abstract_type")
        if not rpd_version.description_for_ai:
            if not rpd_version.user_instruction:
                raise ValueError("description_for_ai or user_instruction is required.")
            if "\n********\n" not in rpd_version.user_instruction:
                raise ValueError("user_instruction must contain \n********\n")
            prompt1, prompt2 = rpd_version.user_instruction.split("\n********\n")
        else:
            if "\n********\n" not in rpd_version.description_for_ai:
                raise ValueError("description_for_ai must contain \n********\n")
            prompt1, prompt2 = rpd_version.description_for_ai.split("\n********\n")
            if "{ng_detection_result}" not in prompt2:
                prompt2 = prompt2 + "\n\n检测結果：{ng_detection_result}"

        ng_check_pipeline = NGCheckPipeline()
        output = ng_check_pipeline.process_image(image_bytes, prompt1, prompt2)

        if output['confidence_score'] > 90:
            if output['ng_detected'] == False:
                result = [{
                    'description': 'NG要素は見つかりませんでした。',
                    'severity': 'safe',
                    'suggestion': None
                }]
            elif output['assessment_result'].suggestion == '承認':
                result = [{
                    'description': output['detection_result'].ng_type,
                    'tag': rpd_version.title,
                    'severity': output['detection_result'].ng_degree,
                    'suggestion': output['detection_result'].analysis_reason
                }]
            else:
                result = [{
                    'description': rpd_version.title+"について",
                    'tag': rpd_version.title,
                    'severity': 'alert',
                    'suggestion': output['assessment_result'].evaluation_reason
                }]
        else:
            if output['ng_detected'] == False:
                result = [{
                    'description': 'AIはNG要素は見つかりませんでしたが、信頼度は低いため、手動で確認してください。',
                    'severity': 'alert',
                    'suggestion': None
                }]
            else:
                result = [{
                    'description': 'AIはNG要素は見つかりませんでしたが、信頼度は低いため、手動で確認してください。' + output['detection_result'].ng_type + 'です。',
                    'tag': rpd_version.title,
                    'severity': output['detection_result'].ng_degree,
                    'suggestion': output['detection_result'].analysis_reason
                }]

    else:
        system_instruction = ng_review_prompt.format(
            ng_word=rpd_version.title, checklist=rpd_version.description_for_ai, tag_list=rpd_version.tag_list)

        prompt_parts = [
            "Image:",
            genai.types.Part.from_bytes(
                data=image_bytes, mime_type='image/jpeg'),
        ]

        class ScoreDict(BaseModel):
            shape: int
            color: int
            meaning: int

        class PotentialNGItems(BaseModel):
            title: str
            tag: str
            score: ScoreDict
            reasoning: str

        class ImageAnalysisResult(BaseModel):
            potential_ng_items: List[PotentialNGItems]

        generation_config = genai.types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=ImageAnalysisResult,
            system_instruction=system_instruction,
            temperature=0.0,  # 0.0で固定
        )

        response = task_gemini_client.models.generate_content(
            model=model_name,
            contents=prompt_parts,
            config=generation_config,
        )

        findings = response.parsed
        print("findings: ", findings)
        result = []
        for finding in findings.potential_ng_items:
            total_score = finding.score.shape + finding.score.color + finding.score.meaning
            if finding.score.shape == 0:
                continue
            if finding.score.shape < 2:
                severity = 'safe'
            elif total_score == 10:
                severity = 'risk'
            else:
                severity = 'alert'

            result.append({
                'description': finding.title,
                'tag': finding.tag,
                'severity': severity,
                'suggestion': finding.reasoning
            })
        if len(result) == 0:
            result.append({
                'description': 'NG要素は見つかりませんでした。',
                'severity': 'safe',
                'suggestion': None
            })

    return result


def _detect_bounding_boxes_sync(image_bytes: bytes, rpd_version: ReviewPointDefinitionVersionInDB, model_name: str) -> BoundingBoxes:
    """
    同步版本的边界框检测，用于在线程中执行
    """
    # 为这个任务创建独立的 Gemini 客户端
    task_gemini_client = genai.Client(api_key=os.getenv(
        'GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None
    if not task_gemini_client:
        logger.error("GEMINI_API_KEY not set. Cannot detect bounding boxes.")
        return BoundingBoxes(bounding_boxes=[])

    prompt_parts = [
        "Image:",
        genai.types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
    ]

    if rpd_version.parent_key == 'copyright_review':
        system_instruction = "Detect the 2d bounding boxes of the copyright, which is the small black text in the bottom of the image.  box_2d [xmin, ymin, xmax, ymax] and normalized to 0-1000. Please return the confidence score for each bounding box you detected."
    else:
        system_instruction = f"Detect all objects that are related to {rpd_version.title} in the image. box_2d [xmin, ymin, xmax, ymax] and normalized to 0-1000. Please return the confidence score for each bounding box you detected."

    # 详细调试日志 - 写入文件以便在线程中查看
    logger.info(f'system_instruction: {system_instruction}')
    logger.info(
        f'rpd_version details: ID={rpd_version.id}, title={rpd_version.title}, parent_key={rpd_version.parent_key}')

    try:
        response = task_gemini_client.models.generate_content(
            model=model_name,
            contents=prompt_parts,
            config=genai.types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=BoundingBoxes,
                system_instruction=system_instruction,
                temperature=0.1,
            )
        )
        logger.info(f'Parsed response: {response.parsed}')

        # 检查解析结果
        if response.parsed and response.parsed.bounding_boxes:
            for i, bbox in enumerate(response.parsed.bounding_boxes):
                logger.info(
                    f'边界框 {i}: xmin={bbox.xmin}, ymin={bbox.ymin}, xmax={bbox.xmax}, ymax={bbox.ymax}, confidence={bbox.confidence}')
        else:
            logger.warning(f'未检测到边界框')

        return response.parsed

    except Exception as e:
        logger.error(f'API调用失败: {str(e)}')
        return BoundingBoxes(bounding_boxes=[])


def _generate_findings_for_copyright_review_sync(
    image_bytes: bytes,
    bounding_boxes: BoundingBoxes,
    rpd_version: ReviewPointDefinitionVersionInDB,
    model_name: str
) -> Tuple[str, str, str]:
    """
    同步版本的版权审查findings生成，用于在线程中执行
    """
    # 为这个任务创建独立的 Gemini 客户端
    task_gemini_client = genai.Client(api_key=os.getenv(
        'GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None
    if not task_gemini_client:
        print("Error: GEMINI_API_KEY not set. Cannot generate copyright review findings.")
        return "", "low", ""

    description = ''
    severity = ''
    suggestion = ''

    system_instruction = copyright_review_prompt
    prompt_parts = [
        "Image:",
        genai.types.Part.from_bytes(
            data=image_bytes, mime_type='image/jpeg'),
        rpd_version.description_for_ai,
        "The bounding boxes are:",
    ]
    for bounding_box in bounding_boxes.bounding_boxes:
        prompt_parts.append(
            f"Bounding box: {bounding_box.xmin}, {bounding_box.ymin}, {bounding_box.xmax}, {bounding_box.ymax}"
        )

    class AiReviewFindingWithoutArea(BaseModel):
        description: str = Field(...,
                                 description="Detailed description of the finding in Japanese!")
        severity: Severity = Field(...,
                                   description="Severity of the finding (risk, alert, or safe).")
        suggestion: Optional[str] = Field(
            None, description="Suggestion to fix the issue in Japanese!")

    response = task_gemini_client.models.generate_content(
        model=model_name,
        contents=prompt_parts,
        config=genai.types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=AiReviewFindingWithoutArea,
            system_instruction=system_instruction,
            temperature=0.1,
        )
    )
    finding = response.parsed
    description = finding.description
    severity = finding.severity
    suggestion = finding.suggestion

    return description, severity, suggestion


def _generate_findings_for_visual_review_sync(
    image_bytes: bytes,
    rpd_version: ReviewPointDefinitionVersionInDB,
    bounding_boxes: BoundingBoxes,
    model_name: str
) -> Tuple[str, str, str]:
    """
    同步版本的视觉审查findings生成，用于在线程中执行
    """
    # 为这个任务创建独立的 Gemini 客户端
    task_gemini_client = genai.Client(api_key=os.getenv(
        'GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None
    if not task_gemini_client:
        print("Error: GEMINI_API_KEY not set. Cannot generate visual review findings.")
        return "", "low", ""

    description = ''
    severity = ''
    suggestion = ''
    reference_images = []
    if rpd_version.reference_images:
        for reference_image in rpd_version.reference_images:
            data = download_file_content_from_s3_sync(reference_image)
            reference_images.append(genai.types.Part.from_bytes(
                data=data, mime_type='image/jpeg'))
    system_instruction = visual_review_prompt
    prompt_parts = [
        "Review Image:",
        genai.types.Part.from_bytes(
            data=image_bytes, mime_type='image/jpeg'),
        "The review point is:",
        rpd_version.description_for_ai,
        "The reference images are:"]+reference_images+[
        "The bounding boxes are:",
    ]
    for bounding_box in bounding_boxes.bounding_boxes:
        prompt_parts.append(
            f"Bounding box: {bounding_box.xmin}, {bounding_box.ymin}, {bounding_box.xmax}, {bounding_box.ymax}"
        )

    class AiReviewFindingVisualReview(BaseModel):
        description: str = Field(...,
                                 description="Detailed description of the finding in Japanese!")
        severity: Severity = Field(...,
                                   description="Severity of the finding (risk, alert, or safe).")
        suggestion: Optional[str] = Field(
            None, description="Suggestion to fix the issue in Japanese!")

    response = task_gemini_client.models.generate_content(
        model=model_name,
        contents=prompt_parts,
        config=genai.types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=AiReviewFindingVisualReview,
            system_instruction=system_instruction,
            temperature=0.1,
        )
    )
    finding = response.parsed
    description = finding.description
    severity = finding.severity
    suggestion = finding.suggestion

    return description, severity, suggestion

