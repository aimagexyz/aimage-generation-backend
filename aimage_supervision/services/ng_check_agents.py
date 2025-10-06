import base64
import io
import os
from typing import Any, Dict, TypedDict

import dotenv
from google import genai
from langgraph.graph import StateGraph
from PIL import Image
from pydantic import BaseModel

dotenv.load_dotenv()


class DetectResult(BaseModel):
    is_ng: bool
    ng_type: str
    ng_degree: str
    analysis_reason: str


class AssessmentResult(BaseModel):
    accuracy: str
    final_decision: bool
    suggestion: str
    confidence_score: int
    evaluation_reason: str

# 定义状态结构


class AgentState(TypedDict):
    image_data: bytes  # Base64编码的图片数据
    ng_prompt: str  # 第一个代理的prompt
    ng_assessment_prompt: str  # 第二个代理的prompt
    ng_detection_result: DetectResult  # 第一个代理的结果
    ng_level_assessment: AssessmentResult  # 第二个代理的评判结果
    final_decision: str  # 最终决定
    confidence_score: float  # 置信度分数


class NGCheckPipeline:
    def __init__(self):
        """初始化NG检测pipeline"""
        if os.getenv('GEMINI_API_KEY'):
            self.gemini_client = genai.Client(
                api_key=os.getenv('GEMINI_API_KEY'))
        else:
            self.gemini_client = None
        self.graph = self._build_graph()

    def _ng_detection_agent(self, state: AgentState) -> AgentState:
        """第一个代理：检测图片中的NG元素，返回检测结果"""

        ng_prompt = state["ng_prompt"]

        try:
            # 根据状态中的图片数据创建图片对象
            image = Image.open(io.BytesIO(state["image_data"]))

            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[ng_prompt, image],
                config=genai.types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=DetectResult,
                )
            )

            state["ng_detection_result"] = response.parsed
            print(f"NG检测代理结果: {response.text}")

        except Exception as e:
            state["ng_detection_result"] = DetectResult(
                is_ng=False, ng_type="", ng_degree="", analysis_reason="")
            print(f"NG检测代理错误: {str(e)}")

        return state

    def _ng_assessment_agent(self, state: AgentState) -> AgentState:
        """第二个代理：评判NG元素检测结果"""
        # print(state["ng_assessment_prompt"])
        # print(state["ng_detection_result"])
        ng_assessment_prompt = state["ng_assessment_prompt"].format(
            ng_detection_result=state["ng_detection_result"])

        image = Image.open(io.BytesIO(state["image_data"]))

        prompt_parts = [ng_assessment_prompt, image]

        try:
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_parts,
                config=genai.types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=AssessmentResult,
                )
            )
            state["ng_level_assessment"] = response.parsed

            state["confidence_score"] = response.parsed.confidence_score

            # 提取最终决定
            state["final_decision"] = response.parsed.final_decision

            print(f"评判代理结果: {response.text}")

        except Exception as e:
            state["ng_level_assessment"] = AssessmentResult(
                accuracy="", final_decision=False, suggestion="", confidence_score=0, evaluation_reason="")
            state["confidence_score"] = 0.0
            state["final_decision"] = "错误"
            print(f"评判代理错误: {str(e)}")

        return state

    def _build_graph(self):
        """构建LangGraph工作流图"""

        # 创建状态图
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("ng_detection", self._ng_detection_agent)
        workflow.add_node("ng_assessment", self._ng_assessment_agent)

        # 设置入口点
        workflow.set_entry_point("ng_detection")

        # 添加边
        workflow.add_edge("ng_detection", "ng_assessment")
        workflow.add_edge("ng_assessment", "__end__")

        return workflow.compile()

    def process_image(self, image_bytes: bytes, ng_prompt: str, ng_assessment_prompt: str) -> Dict[str, Any]:
        """处理单张图片"""

        # 初始化状态
        initial_state = AgentState(
            image_data=image_bytes,
            ng_prompt=ng_prompt,
            ng_assessment_prompt=ng_assessment_prompt,
            ng_detection_result=DetectResult(
                is_ng=False, ng_type="", ng_degree="", analysis_reason=""),
            ng_level_assessment=AssessmentResult(
                accuracy="", final_decision=False, suggestion="", confidence_score=0, evaluation_reason=""),
            final_decision="",
            confidence_score=0.0
        )

        # 执行工作流
        result = self.graph.invoke(initial_state)

        return {
            "success": True,
            "ng_detected": result["final_decision"],
            "confidence_score": result["confidence_score"],
            "detection_result": result["ng_detection_result"],
            "assessment_result": result["ng_level_assessment"]
        }

        # except Exception as e:
        #     return {
        #         "success": False,
        #         "image_path": image_path,
        #         "error": str(e),
        #         "ng_detected": False,
        #         "final_decision": "错误",
        #         "confidence_score": 0.0
        #     }
