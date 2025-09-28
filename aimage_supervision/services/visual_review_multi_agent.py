# -*- coding: utf-8 -*-

import io
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from PIL import Image
from pydantic import BaseModel, Field

from aimage_supervision.settings import logger
from aimage_supervision.prompts.visual_review_multi_agent_prompts import (
    build_classification_prompt, build_enum_prompt, build_evidence_prompt,
    build_label_set_prompt)
from aimage_supervision.schemas import ReviewPointDefinitionVersionInDB


def _truncate_for_log(text: Optional[str], limit: int = 1200) -> str:
    """Return a truncated string for safe logging."""
    if text is None:
        return "<None>"
    try:
        s = str(text)
    except Exception:
        return "<unprintable>"
    if len(s) <= limit:
        return s
    return s[:limit] + f" ... [truncated {len(s)-limit} chars]"


class EnumResultSchema(BaseModel):
    label: str = Field(description='"potential_issues" or "likely_safe"')


class EvidenceResultSchema(BaseModel):
    severity: str = Field(description='"risk"/"alert"/"safe"')
    confidence: float
    description: Optional[str] = None
    suggestion: Optional[str] = None
    evidence: Optional[List[str]] = None


@dataclass
class ProcessingConfig:
    """Multi-Agent处理配置"""
    classify_enum_thinking: int = 0
    classify_evidence_thinking_low: int = 0
    classify_evidence_thinking_high: int = 8192
    tta_scales: List[float] = field(default_factory=lambda: [1.0])
    confidence_threshold: float = 0.8
    classify_label_thinking: int = 4096


class EvidenceClassifierAgent:
    """合规任务专用证据分类器

    说明：
    - 本 Agent 仅用于【合规判定】类任务（compliance）。
    - 实现两阶段流程：先进行快速枚举（classify_enum），对可疑情况再进入证据验证（verify_evidence）。
    - 不负责【分类任务】（classification），分类由 ClassificationAgent 处理。
    """

    def __init__(self, api_key: str, config: Optional[ProcessingConfig] = None):
        self.api_key = api_key
        self.config = config or ProcessingConfig()
        if api_key:
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            self.gemini_client = None
            logger.error("Gemini client not initialized")

    async def classify_enum(
        self,
        image_bytes: bytes,
        constraints: Dict,
        scale: float = 1.0,
        thinking_budget: int = 0,
        bounding_boxes_text: str = "",
        model_name: Optional[str] = None,
        rpd_title: Optional[str] = None,
        task_description: Optional[str] = None,
    ) -> Optional[str]:
        """
        合规任务第一阶段：高速枚举分类（potential_issues / likely_safe）

        参数:
            image_bytes: 图片字节数据
            constraints: 合规检查约束条件（来自预处理后的 RPD constraints）
            scale: TTA用缩放比例
            thinking_budget: 思考预算

        返回:
            分类结果（"potential_issues"/"likely_safe"）
        """
        try:
            if not model_name:
                logger.error("model_name is required for enum classification")
                return None
            logger.info(
                "Starting enum classification with scale=%s, image_size=%s bytes", scale, len(image_bytes))

            prompt = build_enum_prompt(
                rpd_title=rpd_title or "",
                task_description=task_description or "",
                constraints_json_str=json.dumps(
                    constraints, ensure_ascii=True, separators=(',', ':')),
                bbox_text=bounding_boxes_text,
            )
            logger.info("[Compliance][Enum] Prompt: %s",
                        _truncate_for_log(prompt))
            # TTA图片缩放后转换为genai.types.Part
            scaled_image = VisualReviewMultiAgent._scale_image_bytes(
                image_bytes, scale)

            image_part = genai.types.Part.from_bytes(
                data=scaled_image, mime_type='image/jpeg')

            response = self.gemini_client.models.generate_content(
                model=model_name,
                contents=[prompt, image_part],
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_budget=thinking_budget,
                        include_thoughts=False
                    ) if thinking_budget and thinking_budget > 0 else None,
                    response_mime_type='application/json',
                    response_schema=EnumResultSchema,
                    temperature=0.0,
                    max_output_tokens=1024,
                )
            )
            # 仅基于 schema 的结果
            if response and getattr(response, 'parsed', None):
                parsed = response.parsed
                if isinstance(parsed, EnumResultSchema):
                    logger.info("[Compliance][Enum] Parsed result: %s", parsed)
                    return parsed.label
                if isinstance(parsed, dict) and 'label' in parsed:
                    logger.info(
                        "[Compliance][Enum] Parsed dict label: %s", parsed)
                    return str(parsed['label'])
            return None

        except Exception as e:
            logger.exception("Enum classification error")
            return None

    async def verify_evidence(
        self,
        image_bytes: bytes,
        constraints: Dict,
        rpd_title: str,
        scale: float = 1.0,
        thinking_budget: int = 4096,
        bounding_boxes_text: str = "",
        base_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        合规任务第二阶段：详细证据验证（输出 severity/description/suggestion 等）

        参数:
            image_bytes: 图片字节数据
            constraints: 合规检查约束条件（来自预处理后的 RPD constraints）
            rpd_title: RPD标题
            scale: TTA用缩放比例
            thinking_budget: 思考预算

        返回:
            详细的验证结果（JSON）：severity/confidence/description/suggestion/evidence
        """
        try:
            if not model_name:
                logger.error(
                    "model_name is required for evidence verification")
                return None
            prompt = build_evidence_prompt(
                rpd_title=rpd_title,
                constraints_json_str=json.dumps(
                    constraints, ensure_ascii=True, indent=2),
                bbox_text=bounding_boxes_text,
                base_prompt=base_prompt,
            )
            logger.info("[Compliance][Evidence] Prompt: %s",
                        _truncate_for_log(prompt))

            if not self.gemini_client:
                logger.error("Gemini client not initialized")
                return None

            # TTA图片缩放后转换为genai.types.Part
            scaled_image = VisualReviewMultiAgent._scale_image_bytes(
                image_bytes, scale)
            image_part = genai.types.Part.from_bytes(
                data=scaled_image, mime_type='image/jpeg')

            response = self.gemini_client.models.generate_content(
                model=model_name,
                contents=[prompt, image_part],
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_budget=thinking_budget,
                        include_thoughts=False
                    ) if thinking_budget and thinking_budget > 0 else None,
                    response_mime_type='application/json',
                    response_schema=EvidenceResultSchema,
                    temperature=0.1,
                    max_output_tokens=4096,
                )
            )
            result_obj: Optional[Dict] = None
            if response and getattr(response, 'parsed', None):
                parsed = response.parsed
                if hasattr(parsed, 'model_dump'):
                    result_obj = parsed.model_dump()
                elif hasattr(parsed, 'dict'):
                    result_obj = parsed.dict()
                elif isinstance(parsed, dict):
                    result_obj = parsed
            if result_obj is not None:
                logger.info(
                    "[Compliance][Evidence] Parsed result: %s", result_obj)
            # 容错：从文本恢复
            raw_text = getattr(response, 'text', None) if response else None
            if not result_obj and raw_text:
                logger.info(
                    "[Compliance][Evidence] Raw text snippet: %s", _truncate_for_log(raw_text))
                try:
                    cleaned = raw_text.strip()
                    if cleaned.startswith('```'):
                        cleaned = cleaned.strip('`').replace('json\n', '')
                    l = cleaned.find('{')
                    r = cleaned.rfind('}')
                    if l != -1 and r != -1 and r > l:
                        data = json.loads(cleaned[l:r+1])
                        if isinstance(data, dict) and 'severity' in data and 'confidence' in data:
                            result_obj = data
                except Exception:
                    pass

            return result_obj

        except Exception as e:
            logger.exception("Evidence verification error")
            return None


class ClassificationResultSchema(BaseModel):
    label: str
    confidence: float
    rationale: Optional[str] = None


class ClassificationAgent:
    """分类任务Agent：根据constraints进行单轮分类输出label与rationale"""

    def __init__(self, api_key: str, config: Optional[ProcessingConfig] = None):
        self.api_key = api_key
        self.config = config or ProcessingConfig()
        self.gemini_client = genai.Client(api_key=api_key) if api_key else None

    async def classify_label(
        self,
        image_bytes: bytes,
        constraints: Dict,
        rpd_title: str,
        thinking_budget: int = 4096,
        bounding_boxes_text: str = "",
        base_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        allowed_labels: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        try:
            if not model_name:
                logger.error("model_name is required for classification")
                return None

            prompt = build_classification_prompt(
                rpd_title=rpd_title,
                constraints_json_str=json.dumps(
                    constraints, ensure_ascii=True, indent=2),
                bbox_text=bounding_boxes_text,
                base_prompt=base_prompt,
                allowed_labels=allowed_labels,
            )
            logger.info("[Classification] Prompt: %s",
                        _truncate_for_log(prompt))

            if not self.gemini_client:
                logger.error("Gemini client not initialized")
                return None

            # 单轮分类（无TTA，仅1.0缩放）
            scaled_image = VisualReviewMultiAgent._scale_image_bytes(
                image_bytes, 1.0)
            image_part = genai.types.Part.from_bytes(
                data=scaled_image, mime_type='image/jpeg')

            response = self.gemini_client.models.generate_content(
                model=model_name,
                contents=[prompt, image_part],
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_budget=thinking_budget,
                        include_thoughts=False
                    ) if thinking_budget and thinking_budget > 0 else None,
                    response_mime_type='application/json',
                    response_schema=ClassificationResultSchema,
                    temperature=0.1,
                    max_output_tokens=2048,
                )
            )

            label_source = None
            if response and getattr(response, 'parsed', None):
                parsed = response.parsed
                if hasattr(parsed, 'model_dump'):
                    obj = parsed.model_dump()
                    label_source = 'schema_parsed_model_dump'
                    logger.info("[Classification] Parsed(model_dump): %s", obj)
                    return obj
                if hasattr(parsed, 'dict'):
                    obj = parsed.dict()
                    label_source = 'schema_parsed_dict_method'
                    logger.info("[Classification] Parsed(dict): %s", obj)
                    return obj
                if isinstance(parsed, dict):
                    label_source = 'schema_parsed_plain_dict'
                    logger.info(
                        "[Classification] Parsed(plain dict): %s", parsed)
                    return parsed
            return None

        except Exception:
            logger.exception("Classification error")
            return None


class LabelSetSchema(BaseModel):
    labels: List[str]
    notes: Optional[str] = None


class LabelExtractorAgent:
    """基于 description_for_ai 的候选标签抽取 Agent（分类任务临时方案）"""

    def __init__(self, api_key: str, config: ProcessingConfig = None):
        self.api_key = api_key
        self.config = config or ProcessingConfig()
        self.gemini_client = genai.Client(api_key=api_key) if api_key else None

    async def extract_labels(
        self,
        rpd_title: str,
        description_for_ai: str,
        base_prompt: Optional[str],
        model_name: Optional[str],
        thinking_budget: int,
    ) -> Optional[List[str]]:
        try:
            if not model_name:
                logger.error("model_name is required for label extraction")
                return None
            prompt = build_label_set_prompt(
                rpd_title=rpd_title,
                description_for_ai=description_for_ai,
            )
            logger.info("[LabelSet] Prompt: %s", _truncate_for_log(prompt))

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
                    response_schema=LabelSetSchema,
                    temperature=0.1,
                    max_output_tokens=1024,
                )
            )

            labels: Optional[List[str]] = None
            if response and getattr(response, 'parsed', None):
                parsed = response.parsed
                if hasattr(parsed, 'model_dump'):
                    obj = parsed.model_dump()
                    labels = obj.get('labels') if isinstance(
                        obj, dict) else None
                    logger.info("[LabelSet] Parsed(model_dump): %s", obj)
                elif hasattr(parsed, 'dict'):
                    obj = parsed.dict()
                    labels = obj.get('labels') if isinstance(
                        obj, dict) else None
                    logger.info("[LabelSet] Parsed(dict): %s", obj)
                elif isinstance(parsed, dict):
                    labels = parsed.get('labels')
                    logger.info("[LabelSet] Parsed(plain dict): %s", parsed)

            if not labels:
                raw_text = getattr(
                    response, 'text', None) if response else None
                if raw_text:
                    logger.info("[LabelSet] Raw text snippet: %s",
                                _truncate_for_log(raw_text))
                    try:
                        cleaned = raw_text.strip()
                        if cleaned.startswith('```'):
                            cleaned = cleaned.strip('`').replace('json\n', '')
                        l = cleaned.find('{')
                        r = cleaned.rfind('}')
                        if l != -1 and r != -1 and r > l:
                            data = json.loads(cleaned[l:r+1])
                            if isinstance(data, dict) and 'labels' in data:
                                labels = data.get('labels')
                                logger.info(
                                    "[LabelSet] Fallback JSON parsed: %s", data)
                    except Exception:
                        pass

            # 清洗、去重
            if labels:
                seen = set()
                cleaned_labels: List[str] = []
                for lab in labels:
                    lab_s = str(lab).strip()
                    if lab_s and lab_s not in seen:
                        seen.add(lab_s)
                        cleaned_labels.append(lab_s)
                labels = cleaned_labels

            logger.info("[LabelSet] Extracted labels: %s", labels)
            return labels

        except Exception:
            logger.exception("Label extraction error")
            return None


class VisualReviewMultiAgent:
    """Multi-Agent视觉审查系统"""

    def __init__(self, api_key: str, config: Optional[ProcessingConfig] = None):
        """初始化Multi-Agent系统"""
        self.api_key = api_key
        self.config = config or ProcessingConfig()

        # 分类与合规处理器
        self.classification_agent = ClassificationAgent(api_key, self.config)
        self.label_extractor = LabelExtractorAgent(api_key, self.config)
        self.evidence_classifier = EvidenceClassifierAgent(
            api_key, self.config)

    @staticmethod
    def _scale_image_bytes(image_bytes: bytes, scale: float) -> bytes:
        """根据scale缩放图片"""
        img = Image.open(io.BytesIO(image_bytes))

        # JPEG不支持带alpha的模式，统一转换
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # 仅应用TTA缩放
        if scale != 1.0:
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize(
                (new_width, new_height), Image.Resampling.LANCZOS)

        output = io.BytesIO()
        img.save(output, format='JPEG')
        return output.getvalue()

    @staticmethod
    def _format_bounding_boxes(bounding_boxes) -> str:
        """
        静态方法避免重复调用
        """
        if not bounding_boxes or not hasattr(bounding_boxes, 'bounding_boxes') or not bounding_boxes.bounding_boxes:
            return ""

        # 与单Agent一致：逐行输出 "Bounding box: xmin, ymin, xmax, ymax"
        lines = []
        for bbox in bounding_boxes.bounding_boxes:
            lines.append(
                f"Bounding box: {bbox.xmin}, {bbox.ymin}, {bbox.xmax}, {bbox.ymax}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json_from_text(value: Any) -> Optional[Dict]:
        """将文本或对象解析为dict；失败返回None。

        支持以下情况：
        - 已经是dict则直接返回
        - 纯JSON字符串
        - 带有Markdown代码块包裹的JSON
        - 含有额外文本时，提取首个完整的{...}片段尝试解析
        """
        try:
            if value is None:
                return None
            if isinstance(value, dict):
                return value
            if not isinstance(value, str):
                print("value is not a string")
                return None
            s = value.strip()
            s = s.replace("\'", "\"")
            s = s.replace('None', 'null')
            if not s:
                print("s is not a string")
                return None
            # 去掉``` 包裹
            if s.startswith('```'):
                s = s.strip('`')
                s = s.replace('json\n', '')
            # 直接尝试JSON解析
            try:
                return json.loads(s)
            except Exception:
                print("json.loads(s) failed, s:", s)
                pass
            # 尝试提取首个花括号包裹的JSON片段
            l = s.find('{')
            r = s.rfind('}')
            if l != -1 and r != -1 and r > l:
                candidate = s[l:r+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    print("json.loads(candidate) failed, candidate:", candidate)
                    return None
            return None
        except Exception:
            print("json.loads(s) failed")
            return None

    async def process_single_rpd(
        self,
        image_bytes: bytes,
        rpd: ReviewPointDefinitionVersionInDB,
        reference_image_bytes: Optional[List[bytes]] = None,
        base_prompt: Optional[str] = None,
        bounding_boxes: Optional[Any] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
            单个RPD的处理

            参数:
                image_bytes: 图片的字节数据  
                rpd: 视觉审查RPD定义
                reference_image_bytes: 参考图片字节数据列表
                base_prompt: 基础提示词（prmpts/review_point_prompts/visual_review_prompt）
                bounding_boxes: 已检测的边界框(若无则None)

            返回:
                Tuple[str, str, str]: (description, severity, suggestion)
        """
        start_time = time.time()

        try:
            bbox_text = self._format_bounding_boxes(bounding_boxes)

            if base_prompt is None:
                base_prompt = "あなたは日本のアニメーション制作における視覚的品質管理の専門家です。"

            # 使用RPD预处理阶段的constraints（必须存在）
            constraints: Optional[Dict] = self._parse_json_from_text(
                getattr(rpd, 'constraints', None)
            )
            if not constraints:
                logger.error(
                    "[%s] No valid preprocessed constraints found on RPD. Aborting.", rpd.title)
                return self._create_fallback_result(rpd)

            # 使用RPD预处理阶段的labels（必须存在）
            guidelines: Optional[Dict] = self._parse_json_from_text(
                getattr(rpd, 'guidelines', None)
            )
            if not guidelines:
                logger.error(
                    "[%s] No valid preprocessed guidelines found on RPD. Aborting.", rpd.title)
                return self._create_fallback_result(rpd)

            # 任务分流：分类任务 or 合规任务
            is_classification = False
            try:
                if getattr(rpd, 'rpd_type', None) == 'classification tasks':
                    is_classification = True
            except Exception:
                pass

            logger.info("[Routing] Task type: %s",
                        'classification' if is_classification else 'compliance')

            if is_classification:
                # 分类路径（无TTA，单轮分类）
                allowed_labels: Optional[List[str]] = None
                # 1) 优先从 guidelines 读取 classes
                try:
                    if isinstance(guidelines, dict):
                        classes = guidelines.get('labels')
                        if isinstance(classes, (list, tuple)):
                            allowed_labels = [str(x).strip()
                                              for x in classes if str(x).strip()]
                            if allowed_labels:
                                logger.info(
                                    "[LabelSet] Source=guidelines.labels; Labels=%s", allowed_labels)
                except Exception:
                    pass

                # 2) 如无，则从 description_for_ai 通过 LLM 抽取
                if not allowed_labels:
                    labels = await self.label_extractor.extract_labels(
                        rpd_title=rpd.title,
                        description_for_ai=rpd.description_for_ai or "",
                        base_prompt=base_prompt,
                        model_name=model_name,
                        thinking_budget=self.config.classify_label_thinking,
                    )
                    if labels:
                        allowed_labels = labels
                        logger.info(
                            "[LabelSet] Source=description_for_ai; Labels=%s", allowed_labels)
                    else:
                        logger.info(
                            "[LabelSet] Extraction failed; proceeding without allowed labels")

                classify_res = await self.classification_agent.classify_label(
                    image_bytes=image_bytes,
                    constraints=constraints,
                    rpd_title=rpd.title,
                    thinking_budget=self.config.classify_label_thinking,
                    bounding_boxes_text=bbox_text,
                    base_prompt=base_prompt,
                    model_name=model_name,
                    allowed_labels=allowed_labels,
                )
                if not classify_res:
                    return self._create_classification_fallback_result(rpd)

                label = str(classify_res.get('label', '')).strip()
                rationale = classify_res.get('rationale') or ''
                logger.info(
                    "[Classification] Final label: %s; confidence: %s; rationale_snippet: %s",
                    label,
                    classify_res.get('confidence'),
                    _truncate_for_log(rationale),
                )
                description = f"分類結果: {label}。根拠: {rationale or 'N/A'}"
                severity = 'alert'  # 固定为 alert
                suggestion = ""
                return (description, severity, suggestion)
            else:
                # 合规路径：保持现有第3步流程（TTA → enum → evidence → 聚合）
                logger.info("[%s] Phase 3: TTA voting (compliance)", rpd.title)
                enum_results = []
                evidence_results = []

                for scale in self.config.tta_scales:
                    enum_result = await self.evidence_classifier.classify_enum(
                        image_bytes=image_bytes,
                        constraints=constraints,
                        scale=scale,
                        thinking_budget=self.config.classify_enum_thinking,
                        bounding_boxes_text=bbox_text,
                        model_name=model_name,
                        rpd_title=rpd.title,
                        task_description=rpd.description_for_ai or "",
                    )

                    if enum_result:
                        logger.info(
                            "[Compliance] Enum label@scale=%s: %s", scale, enum_result)
                        enum_results.append(enum_result)

                    if enum_result == "potential_issues":
                        thinking_budget = (
                            self.config.classify_evidence_thinking_high
                            if len(enum_results) == 1
                            else self.config.classify_evidence_thinking_low
                        )

                        evidence_result = await self.evidence_classifier.verify_evidence(
                            image_bytes=image_bytes,
                            constraints=constraints,
                            rpd_title=rpd.title,
                            scale=scale,
                            thinking_budget=thinking_budget,
                            bounding_boxes_text=bbox_text,
                            base_prompt=base_prompt,
                            model_name=model_name,
                        )

                        if evidence_result:
                            logger.info(
                                "[Compliance] Evidence result@scale=%s: %s", scale, evidence_result)
                            evidence_results.append(evidence_result)

                final_result = self._aggregate_tta_results(
                    enum_results, evidence_results)
                logger.info(
                    "[Compliance] Final aggregated result: %s", final_result)

                if final_result["confidence"] < self.config.confidence_threshold:
                    logger.info(
                        "[%s] Low confidence; running additional verification", rpd.title)
                    additional_result = await self.evidence_classifier.verify_evidence(
                        image_bytes=image_bytes,
                        constraints=constraints,
                        rpd_title=rpd.title,
                        scale=1.0,
                        thinking_budget=self.config.classify_evidence_thinking_high * 2,
                        bounding_boxes_text=bbox_text,
                        base_prompt=base_prompt,
                        model_name=model_name,
                    )
                    if additional_result:
                        evidence_results.append(additional_result)
                        final_result = self._aggregate_tta_results(
                            enum_results, evidence_results)

                description = final_result["description"]
                severity = final_result["severity"]
                suggestion = final_result.get("suggestion") or ""
                return (description, severity, suggestion)

        except Exception as e:
            logger.exception("[%s] Processing failed", rpd.title)
            return self._create_fallback_result(rpd)

    def _aggregate_tta_results(self, enum_results: List[str], evidence_results: List[Dict]) -> Dict:
        """综合TTA结果"""
        try:
            if not evidence_results:
                return {
                    "severity": "safe",
                    "confidence": 0.9,
                    "description": "検査の結果、品質上の問題は発見されませんでした。",
                    "suggestion": None
                }

            # 重要度加重平均
            severity_scores = {"safe": 0, "alert": 1, "risk": 2}
            weighted_severity = 0
            weighted_confidence = 0
            total_weight = 0

            descriptions = []
            suggestions = []

            for result in evidence_results:
                weight = result.get("confidence", 0.5)
                severity_score = severity_scores.get(
                    result.get("severity", "safe"), 0)

                weighted_severity += severity_score * weight
                weighted_confidence += result.get("confidence", 0.5) * weight
                total_weight += weight

                if result.get("description"):
                    descriptions.append(result["description"])
                if result.get("suggestion"):
                    suggestions.append(result["suggestion"])

            if total_weight > 0:
                avg_severity_score = weighted_severity / total_weight
                avg_confidence = weighted_confidence / total_weight
            else:
                avg_severity_score = 0
                avg_confidence = 0.5

            # 最终严重度判定
            if avg_severity_score >= 1.5:
                final_severity = "risk"
            elif avg_severity_score >= 0.5:
                final_severity = "alert"
            else:
                final_severity = "safe"

            # 描述的整合
            if descriptions:
                final_description = descriptions[0]
            else:
                final_description = "検査を実施しました。"

            # 建议的整合
            final_suggestion = suggestions[0] if suggestions else None

            return {
                "severity": final_severity,
                "confidence": min(avg_confidence, 1.0),
                "description": final_description,
                "suggestion": final_suggestion
            }

        except Exception as e:
            logger.exception("TTA aggregation error")
            return {
                "severity": "safe",
                "confidence": 0.5,
                "description": "処理中にエラーが発生しましたが、安全な結果を返します。",
                "suggestion": None
            }

    def _create_fallback_result(self, rpd: ReviewPointDefinitionVersionInDB) -> Tuple[str, str, str]:
        """フォールバック結果の作成"""
        return (
            "処理中にエラーが発生しましたが、安全な結果を返します。",  # description
            "safe",  # severity
            ""  # suggestion
        )

    def _create_classification_fallback_result(self, rpd: ReviewPointDefinitionVersionInDB) -> Tuple[str, str, str]:
        """分類タスクのフォールバック結果（固定 alert）"""
        return (
            "分類の判定に失敗しました。手動で確認してください。",
            "alert",
            ""
        )
