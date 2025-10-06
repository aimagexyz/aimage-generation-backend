import json
from aimage_supervision.settings import logger
import os
from typing import List, Optional

import dotenv
from google import genai
from pydantic import BaseModel

dotenv.load_dotenv()

# Pydantic models for request/response


class AutofillRequest(BaseModel):
    """智能补全请求模型"""
    user_input: str
    context: Optional[str] = None  # 可选的上下文信息
    rpd_type: str  # RPD类型（如：general_ng_review等）
    max_suggestions: int = 5  # 最大建议数量


class AutofillSuggestion(BaseModel):
    """单个补全建议"""
    text: str
    confidence: float  # 置信度 (0-1)
    reason: Optional[str] = None  # 建议理由


class AutofillSuggestions(BaseModel):
    """补全建议列表"""
    suggestions: List[AutofillSuggestion]


class AutofillResponse(BaseModel):
    """智能补全响应模型"""
    suggestions: List[AutofillSuggestion]
    original_input: str
    success: bool
    error_message: Optional[str] = None


class RPDTitleAutofillService:
    """RPD标题智能补全服务"""

    def __init__(self):
        self.gemini_client = None
        self._init_gemini_client()

    def _init_gemini_client(self):
        """初始化Gemini客户端"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                logger.error("GEMINI_API_KEY not set")
                return

            self.gemini_client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")

    def _get_rpd_context_prompt(self, rpd_type: str) -> str:
        """根据RPD类型获取上下文提示"""
        rpd_contexts = {
            "general_ng_review": "Review points related to general NG items. Title example: 青少年に悪い影響を与える要素、宗教要素、暴力要素、性別差別要素",
            "visual_review": "Review points related to visual features such as object shapes, character hair, eyes, etc. Title example: キャラクターの髪の長さ、目の大きさ、オブジェクトの形状",
            "settings_review": "Review points related to world background settings. Title example: **背景の設定、**世界観の設定",
            "design_review": "Review points related to UI/UX design, layout, and color scheme. Title example: バーナーのデザインの確認、レイアウトの確認、配色の確認",
            "text_review": "Review points related to text content, wording, and notation. Title example: テキストの確認、表記の確認、文言の確認",
            "copyright_review": "Review points related to check the copyright marks in the image and whether the text is right. Title example:　コピーライトチェック"
        }
        return rpd_contexts.get(rpd_type, "general review")

    def _create_prompt(self, context: Optional[str] = None, max_suggestions: int = 5) -> str:
        """補完用プロンプトを作成"""

        prompt = """You are a professional in anime industry, when user want to create a title, you can help them finish the title.
                    You will be given the user's input, and the type of the title for. sometimes some context information.
                    {context}
                    Please generate {max_suggestions} suggestions for the user's input.
                    The suggestions should be practical and professional.
                    The suggestions should be in Japanese.
                    The suggestions should be in 10-30 characters.
                    The suggestions should not be too similar to each other.
                    The suggestions should not change the user's input, it should be a suffix of the user's input.
                    The suggestions should not be too long.
                    

                    Please return the suggestions in the following JSON format:
                    {{
                    "suggestions": [
                        {{
                        "text": "補完されたタイトル",
                        "confidence": 0.9,
                        "reason": "選択理由の簡潔な説明"
                        }}
                    ]
                    }}

                    If the user's input is empty or too short, please return an empty list:
                    {{
                       "suggestions": []
                    }}。

                    Do not include punctuation."""

        return prompt.format(context=context, max_suggestions=max_suggestions)

    async def get_autofill_suggestions(self, request: AutofillRequest) -> AutofillResponse:
        """智能补全建议获取"""
        try:
            if not self.gemini_client:
                return AutofillResponse(
                    suggestions=[],
                    original_input=request.user_input,
                    success=False,
                    error_message="Gemini client not available"
                )

            # 输入验证
            # if not request.user_input.strip():
            #     # 空输入时提供默认建议
            #     default_suggestions = self._get_default_suggestions(
            #         request.rpd_type or "")
            #     return AutofillResponse(
            #         suggestions=default_suggestions,
            #         original_input=request.user_input,
            #         success=True
            #     )

            # 创建提示
            prompt = self._create_prompt(
                self._get_rpd_context_prompt(request.rpd_type),
                request.max_suggestions)

            if request.context:
                prompt_parts = [
                    'user_input: ' + request.user_input + '\n',
                    'context: ' + request.context + '\n',
                    'rpd_type: ' + request.rpd_type + '\n',
                ]
            else:
                prompt_parts = [
                    'user_input: ' + request.user_input + '\n',
                    'rpd_type: ' + request.rpd_type + '\n',
                ]

            # 调用Gemini API
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt_parts,
                config=genai.types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=AutofillSuggestions,
                    system_instruction=prompt,
                    temperature=0.2,  # 保持相对稳定的输出
                )
            )
            if not response or not response.parsed:
                logger.error("Empty response from Gemini")
                return AutofillResponse(
                    suggestions=[],
                    original_input=request.user_input,
                    success=False,
                    error_message="Empty response from AI"
                )

            # 按置信度排序
            suggestions = response.parsed.suggestions

            suggestions.sort(key=lambda x: x.confidence, reverse=True)

            return AutofillResponse(
                suggestions=suggestions[:request.max_suggestions],
                original_input=request.user_input,
                success=True
            )

        except Exception as e:
            logger.error(f"Error in get_autofill_suggestions: {str(e)}")
            return AutofillResponse(
                suggestions=[],
                original_input=request.user_input,
                success=False,
                error_message=str(e)
            )

    def _get_default_suggestions(self, rpd_type: str) -> List[AutofillSuggestion]:
        """获取默认建议（当用户输入为空时）"""
        default_by_type = {
            "general_ng_review": [
                "基本品質チェック",
                "一般的なNG項目の確認",
                "品質基準の審査",
                "標準的な問題点の検出"
            ],
            "visual_review": [
                "ビジュアル要素の検証",
                "視覚的品質の確認",
                "デザイン表現の審査",
                "画像・グラフィックの品質チェック"
            ],
            "settings_review": [
                "システム設定の確認",
                "設定値の検証",
                "設定項目の審査",
                "構成パラメータの確認"
            ],
            "design_review": [
                "デザイン品質の評価",
                "UI/UXデザインの確認",
                "デザイン要素の審査",
                "レイアウト・配色の検証"
            ],
            "text_review": [
                "テキスト内容の確認",
                "文字情報の検証",
                "表記・文言の審査",
                "テキスト品質のチェック"
            ],
            "copyright_review": [
                "コピーライト表記の確認",
                "著作権マークの検証",
                "権利表示の審査",
                "ライセンス表記のチェック"
            ]
        }

        suggestions_text = default_by_type.get(rpd_type, [
            "レビューポイントの定義",
            "品質チェック項目",
            "審査基準の設定",
            "チェックポイントの確認"
        ])

        return [
            AutofillSuggestion(
                text=text,
                confidence=0.8,
                reason="デフォルト推奨項目"
            ) for text in suggestions_text
        ]


# 创建全局服务实例
autofill_service = RPDTitleAutofillService()


# 公共接口函数
async def get_rpd_title_suggestions(
    user_input: str,
    context: Optional[str] = None,
    rpd_type: str = "general_ng_review",
    max_suggestions: int = 5
) -> AutofillResponse:
    """
    获取RPD标题的智能补全建议

    Args:
        user_input: 用户输入的文本
        context: 可选的上下文信息
        rpd_type: RPD类型（如：general_ng_review等）
        max_suggestions: 最大建议数量

    Returns:
        AutofillResponse: 包含建议列表的响应
    """
    request = AutofillRequest(
        user_input=user_input,
        context=context,
        rpd_type=rpd_type,
        max_suggestions=max_suggestions
    )

    return await autofill_service.get_autofill_suggestions(request)
