import base64
import tempfile
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Security, UploadFile, status
from fastapi.params import Form
from pydantic import BaseModel, Field

from aimage_supervision.middlewares.auth import get_current_user
from aimage_supervision.models import User
from aimage_supervision.utils.video_utils import (
    detect_video_scenes_and_extract_frames, is_video_format)

router = APIRouter(prefix='/video', tags=['Video'])


class SceneFrameResponse(BaseModel):
    """场景关键帧响应模型"""
    first_frame: str = Field(..., description="第一帧的Base64编码图片数据")
    middle_frame: str = Field(..., description="中间帧的Base64编码图片数据")
    last_frame: str = Field(..., description="最后帧的Base64编码图片数据")
    scene_number: int = Field(..., description="场景编号")


class SceneInfo(BaseModel):
    """场景信息模型"""
    scene_number: int
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int
    frame_count: int


class VideoScenesResponse(BaseModel):
    """视频场景检测响应模型"""
    scenes: list[SceneInfo]
    total_scenes: int
    detector_used: str
    threshold_used: float
    message: str = Field(default="", description="额外信息")


class SceneFramesWithInfoResponse(BaseModel):
    """带场景信息的帧响应模型"""
    frames: SceneFrameResponse
    scene_info: SceneInfo


@router.post('/scenes/analyze', response_model=VideoScenesResponse)
async def analyze_video_scenes(
    file: UploadFile,
    detector_type: Annotated[str, Form()] = "adaptive",
    threshold: Annotated[Optional[float], Form()] = None,
    current_user: User = Security(get_current_user)
) -> VideoScenesResponse:
    """
    分析视频场景切换点

    - **file**: 视频文件 (支持 MP4, MOV, AVI 等格式)
    - **detector_type**: 检测器类型 ("adaptive" 或 "content")
    - **threshold**: 检测阈值 (可选，使用默认值)
    """

    # 验证文件类型
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file is required"
        )

    # 读取文件内容
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"read file failed: {str(e)}"
        )

    # 验证是否为视频文件
    if not is_video_format(file_content, file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="unsupported file format, please upload a video file"
        )

    # 验证检测器类型
    if detector_type not in ["adaptive", "content"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="detector type must be 'adaptive' or 'content'"
        )

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    try:
        # 检测场景
        result = detect_video_scenes_and_extract_frames(
            video_path=temp_file_path,
            detector_type=detector_type,
        )

        # 转换为响应模型
        scenes = [
            SceneInfo(
                scene_number=scene["scene_number"],
                start_time=scene["start_time"],
                end_time=scene["end_time"],
                duration=scene["duration"],
                start_frame=scene["start_frame"],
                end_frame=scene["end_frame"],
                frame_count=scene["frame_count"]
            )
            for scene in result["scenes"]
        ]

        return VideoScenesResponse(
            scenes=scenes,
            total_scenes=result["total_scenes"],
            detector_used=result["detector_used"],
            threshold_used=result["threshold_used"],
            message=result.get("message", "")
        )

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PySceneDetect 库未安装，请联系管理员"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"视频处理失败: {str(e)}"
        )
    finally:
        # 清理临时文件
        try:
            import os
            os.unlink(temp_file_path)
        except:
            pass


@router.post('/scenes/{scene_number}/frames', response_model=SceneFrameResponse)
async def get_scene_frames(
    scene_number: int,
    file: UploadFile,
    detector_type: Annotated[str, Form()] = "adaptive",
    current_user: User = Security(get_current_user)
) -> SceneFrameResponse:
    """
    获取指定场景的关键帧（第一帧、中间帧、最后帧）

    - **scene_number**: 场景编号 (从1开始)
    - **file**: 视频文件
    - **detector_type**: 检测器类型 ("adaptive" 或 "content")
    """

    if scene_number < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="场景编号必须大于0"
        )

    # 验证文件类型
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="未提供文件名"
        )

    # 读取文件内容
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"读取文件失败: {str(e)}"
        )

    # 验证是否为视频文件
    if not is_video_format(file_content, file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不支持的文件格式，请上传视频文件"
        )

    # 验证检测器类型
    if detector_type not in ["adaptive", "content"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="检测器类型必须是 'adaptive' 或 'content'"
        )

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    try:
        # 获取场景帧
        frames = get_scene_representative_frames(
            video_path=temp_file_path,
            scene_number=scene_number,
            detector_type=detector_type
        )

        # 将字节数据转换为Base64编码
        first_frame_b64 = base64.b64encode(
            frames["first_frame"]).decode('utf-8')
        middle_frame_b64 = base64.b64encode(
            frames["middle_frame"]).decode('utf-8')
        last_frame_b64 = base64.b64encode(frames["last_frame"]).decode('utf-8')

        return SceneFrameResponse(
            first_frame=first_frame_b64,
            middle_frame=middle_frame_b64,
            last_frame=last_frame_b64,
            scene_number=scene_number
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PySceneDetect 库未安装，请联系管理员"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"视频处理失败: {str(e)}"
        )
    finally:
        # 清理临时文件
        try:
            import os
            os.unlink(temp_file_path)
        except:
            pass


@router.post('/scenes/{scene_number}/frames-with-info', response_model=SceneFramesWithInfoResponse)
async def get_scene_frames_with_info(
    scene_number: int,
    file: UploadFile,
    detector_type: Annotated[str, Form()] = "adaptive",
    threshold: Annotated[Optional[float], Form()] = None,
    current_user: User = Security(get_current_user)
) -> SceneFramesWithInfoResponse:
    """
    获取指定场景的关键帧和详细信息

    - **scene_number**: 场景编号 (从1开始)
    - **file**: 视频文件
    - **detector_type**: 检测器类型 ("adaptive" 或 "content")
    - **threshold**: 检测阈值 (可选)
    """

    if scene_number < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="场景编号必须大于0"
        )

    # 验证文件类型
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="未提供文件名"
        )

    # 读取文件内容
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"读取文件失败: {str(e)}"
        )

    # 验证是否为视频文件
    if not is_video_format(file_content, file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不支持的文件格式，请上传视频文件"
        )

    # 验证检测器类型
    if detector_type not in ["adaptive", "content"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="检测器类型必须是 'adaptive' 或 'content'"
        )

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    try:
        # 获取完整的场景信息
        result = detect_video_scenes_and_extract_frames(
            video_path=temp_file_path,
            detector_type=detector_type,
            threshold=threshold,
        )

        # 验证场景编号是否有效
        if scene_number > len(result["scenes"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"场景编号 {scene_number} 超出范围，共检测到 {len(result['scenes'])} 个场景"
            )

        # 获取指定场景的信息
        scene = result["scenes"][scene_number - 1]

        # 转换字节数据为Base64
        first_frame_b64 = base64.b64encode(
            scene["first_frame_bytes"]).decode('utf-8')
        middle_frame_b64 = base64.b64encode(
            scene["middle_frame_bytes"]).decode('utf-8')
        last_frame_b64 = base64.b64encode(
            scene["last_frame_bytes"]).decode('utf-8')

        frames = SceneFrameResponse(
            first_frame=first_frame_b64,
            middle_frame=middle_frame_b64,
            last_frame=last_frame_b64,
            scene_number=scene_number
        )

        scene_info = SceneInfo(
            scene_number=scene["scene_number"],
            start_time=scene["start_time"],
            end_time=scene["end_time"],
            duration=scene["duration"],
            start_frame=scene["start_frame"],
            end_frame=scene["end_frame"],
            frame_count=scene["frame_count"]
        )

        return SceneFramesWithInfoResponse(
            frames=frames,
            scene_info=scene_info
        )

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PySceneDetect 库未安装，请联系管理员"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"视频处理失败: {str(e)}"
        )
    finally:
        # 清理临时文件
        try:
            import os
            os.unlink(temp_file_path)
        except:
            pass
