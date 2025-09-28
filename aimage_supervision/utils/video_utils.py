import io
import json
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
from PIL import Image

try:
    from scenedetect import (AdaptiveDetector, ContentDetector,  # type: ignore
                             detect)
    from scenedetect.frame_timecode import FrameTimecode  # type: ignore
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    PYSCENEDETECT_AVAILABLE = False


def is_video_format(image_bytes: bytes, filename: Optional[str] = None) -> bool:
    """
    检测字节数据是否为视频格式

    Args:
        image_bytes: 二进制数据
        filename: 可选的文件名，用于检查扩展名

    Returns:
        bool: True 如果是视频格式，False 如果是图片格式
    """
    # 如果提供了文件名，检查扩展名
    if filename:
        # 获取小写扩展名
        ext = os.path.splitext(filename)[1].lower()

        # 视频扩展名列表
        video_extensions = ['.mp4', '.webm',
                            '.avi', '.mov', '.mkv', '.flv', '.wmv']

        # 图片扩展名列表
        image_extensions = ['.jpg', '.jpeg', '.png',
                            '.gif', '.bmp', '.webp', '.webpm']

        # 如果是已知的图片扩展名，直接返回False
        if ext in image_extensions:
            return False

        # 如果是已知的视频扩展名，继续检查文件头
        if ext in video_extensions:
            # 继续检查文件头
            pass

    # 检查常见的视频文件头
    video_signatures = [
        b'\x00\x00\x00\x20\x66\x74\x79\x70',  # MP4
        b'\x00\x00\x00\x18\x66\x74\x79\x70',  # MP4
        b'\x00\x00\x00\x14\x66\x74\x79\x70',  # MP4 variant
    ]

    # 检查前20个字节
    header = image_bytes[:20]

    for signature in video_signatures:
        if header.startswith(signature):
            return True

    # 特殊检查 AVI 格式
    if header.startswith(b'RIFF') and b'AVI ' in header:
        return True

    # 特殊检查 WEBM 格式 - 需要更严格的检测
    if header.startswith(b'\x1a\x45\xdf\xa3'):
        # WEBM 格式的更多特征检查
        # WEBM 视频文件通常在文件头后有特定的标记
        # 检查是否包含视频轨道标识符
        if b'webm' in image_bytes[:100] and b'\x83\x81\x01' in image_bytes[:1000]:
            return True
        # 如果没有视频轨道标识符，可能是 webpm 图片，返回 False
        return False

    return False


def rgb_to_bgr_numpy(rgb_array: np.ndarray) -> np.ndarray:
    """
    使用 numpy 将 RGB 数组转换为 BGR 数组

    Args:
        rgb_array: RGB 格式的 numpy 数组

    Returns:
        BGR 格式的 numpy 数组
    """
    # RGB 转 BGR: 交换第0和第2通道
    return rgb_array[:, :, ::-1]


def get_middle_frame_of_video(video_bytes: bytes) -> bytes:
    """
    从视频字节数据中提取中间帧并返回为JPEG格式的字节数据

    Args:
        video_bytes: 视频的二进制数据

    Returns:
        bytes: 中间帧图片的JPEG格式字节数据

    Raises:
        Exception: 如果无法处理视频或提取中间帧
    """
    # 创建临时文件保存视频数据
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    try:
        # 使用 moviepy 加载视频并获取中间帧
        video = VideoFileClip(temp_video_path)

        # 获取视频时长并计算中间时间点
        duration = video.duration
        middle_time = duration / 2

        # 获取中间帧 (RGB格式的numpy数组)
        frame = video.get_frame(middle_time)

        # 关闭视频文件
        video.close()

        # 将 numpy 数组转换为 PIL Image (RGB格式)
        pil_image = Image.fromarray(frame.astype(np.uint8))

        # 将 PIL Image 编码为 JPEG 格式的字节数据
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)

        return img_byte_arr.getvalue()

    except Exception as e:
        raise Exception(f"提取视频中间帧失败: {str(e)}")
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_video_path)
        except:
            pass


def get_frame_at_timestamp(video: VideoFileClip, timestamp_sec: float) -> bytes:
    """
    从视频文件路径中提取指定时间戳的帧并返回为JPEG格式的字节数据

    Args:
        video_path: 视频的文件路径
        timestamp_sec: 目标帧的时间戳（秒）

    Returns:
        bytes: 帧图片的JPEG格式字节数据

    Raises:
        Exception: 如果无法处理视频或提取帧
    """
    try:

        # 检查时间戳是否在视频时长范围内
        if timestamp_sec < 0 or timestamp_sec > video.duration:
            raise ValueError(
                f"时间戳 {timestamp_sec}s 超出视频时长 {video.duration}s 的范围")

        # 获取指定时间戳的帧 (RGB格式的numpy数组)
        frame = video.get_frame(timestamp_sec)

        # 将 numpy 数组转换为 PIL Image (RGB格式)
        pil_image = Image.fromarray(frame.astype(np.uint8))

        # 将 PIL Image 编码为 JPEG 格式的字节数据
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)

        return img_byte_arr.getvalue()

    except Exception as e:
        raise Exception(
            f"提取时间戳 {timestamp_sec}s 的帧失败: {str(e)}")


def get_frame_at_timestamp_from_path(video_path: str, timestamp_sec: float) -> bytes:
    """
    从视频文件路径中提取指定时间戳的帧并返回为JPEG格式的字节数据
    """
    video = VideoFileClip(video_path)
    image_bytes = get_frame_at_timestamp(video, timestamp_sec)
    video.close()
    return image_bytes


def concat_4_frames_from_video(video: VideoFileClip, timestamp_sec: List[float]) -> bytes:
    """
    从视频中提取指定时间戳的帧，将4张帧拼接成2x2网格，并返回为JPEG格式的字节数据

    Args:
        video: VideoFileClip对象
        timestamp_sec: 4个时间戳的列表（秒）

    Returns:
        bytes: 拼接后图片的JPEG格式字节数据

    Raises:
        ValueError: 如果时间戳数量不是4个或时间戳超出视频时长范围
    """
    try:
        if len(timestamp_sec) != 4:
            raise ValueError(f"需要提供4个时间戳，但收到了{len(timestamp_sec)}个")

        frames = []

        for timestamp in timestamp_sec:
            # 检查时间戳是否在视频时长范围内
            if timestamp < 0 or timestamp > video.duration:
                raise ValueError(
                    f"时间戳 {timestamp}s 超出视频时长 {video.duration}s 的范围")

            # 获取指定时间戳的帧 (RGB格式的numpy数组)
            frame = video.get_frame(timestamp)

            # 将 numpy 数组转换为 PIL Image
            pil_image = Image.fromarray(frame.astype(np.uint8))
            frames.append(pil_image)

        # 获取第一张图片的尺寸作为参考
        width, height = frames[0].size

        # 确保所有图片尺寸一致，如果不一致则调整到第一张图片的尺寸
        for i in range(1, len(frames)):
            if frames[i].size != (width, height):
                frames[i] = frames[i].resize(
                    (width, height), Image.Resampling.LANCZOS)

        # 创建2x2网格的拼接图片
        grid_width = width * 2
        grid_height = height * 2
        grid_image = Image.new('RGB', (grid_width, grid_height))

        # 拼接图片：
        # 位置0: 左上角 (0, 0)
        # 位置1: 右上角 (width, 0)
        # 位置2: 左下角 (0, height)
        # 位置3: 右下角 (width, height)
        positions = [
            (0, 0),           # 左上角
            (width, 0),       # 右上角
            (0, height),      # 左下角
            (width, height)   # 右下角
        ]

        for i, frame in enumerate(frames):
            grid_image.paste(frame, positions[i])

        # 将拼接后的图片编码为JPEG格式的字节数据
        img_byte_arr = io.BytesIO()
        grid_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)

        return img_byte_arr.getvalue()
    except Exception as e:
        raise Exception(f"failed to concat 4 frames from video: {str(e)}")


def process_image_or_video_bytes(image_bytes: bytes) -> bytes:
    """
    处理图片或视频字节数据。如果是视频，提取中间帧；如果是图片，直接返回

    Args:
        image_bytes: 图片或视频的二进制数据

    Returns:
        bytes: 图片的字节数据（对于视频返回中间帧的JPEG数据）
    """

    if is_video_format(image_bytes):
        # 如果是视频，提取中间帧
        return get_middle_frame_of_video(image_bytes)
    else:
        # 如果是图片，直接返回原始数据
        return image_bytes


def get_middle_frame_as_pil_image(video_bytes: bytes) -> Image.Image:
    """
    从视频字节数据中提取中间帧并返回为PIL Image对象

    Args:
        video_bytes: 视频的二进制数据

    Returns:
        PIL.Image.Image: 中间帧的PIL Image对象
    """
    frame_bytes = get_middle_frame_of_video(video_bytes)
    return Image.open(io.BytesIO(frame_bytes))


def convert_image_format(image_bytes: bytes, target_format: str = 'JPEG', quality: int = 95) -> bytes:
    """
    转换图片格式，使用 PIL 替代 OpenCV

    Args:
        image_bytes: 原始图片字节数据
        target_format: 目标格式 ('JPEG', 'PNG', 'WEBP' 等)
        quality: 图片质量 (1-100，仅对JPEG有效)

    Returns:
        bytes: 转换后的图片字节数据
    """
    try:
        # 使用 PIL 打开图片
        image = Image.open(io.BytesIO(image_bytes))

        # 如果是RGBA图片转换为JPEG，需要先转换为RGB
        if target_format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA'):
            # 创建白色背景
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()
                                 [-1])  # 使用alpha通道作为mask
            else:
                background.paste(image)
            image = background

        # 保存到字节流
        img_byte_arr = io.BytesIO()
        if target_format.upper() == 'JPEG':
            image.save(img_byte_arr, format=target_format,
                       quality=quality, optimize=True)
        else:
            image.save(img_byte_arr, format=target_format)
        img_byte_arr.seek(0)

        return img_byte_arr.getvalue()

    except Exception as e:
        raise Exception(f"图片格式转换失败: {str(e)}")


def detect_video_scenes_and_extract_frames(
    video_path: str,
    detector_type: str = "content",
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    使用 PySceneDetect 检测视频场景切换点，并抽取每个片段的第一帧、最后一帧和中间帧

    Args:
        video_path: 视频文件路径
        detector_type: 检测器类型 ("adaptive", "content", "threshold")
        threshold: 检测阈值 (可选，如果不提供将使用默认值)
        output_dir: 输出目录 (可选，如果不提供将不保存图片文件)

    Returns:
        Dict: 包含场景信息和帧数据的字典
        {
            "scenes": [
                {
                    "scene_number": 1,
                    "start_time": 0.0,
                    "end_time": 5.2,
                    "start_frame": 0,
                    "end_frame": 156,
                    "first_frame_bytes": bytes,
                    "middle_frame_bytes": bytes,
                    "last_frame_bytes": bytes,
                },
                ...
            ],
            "total_scenes": 5,
            "detector_used": "adaptive",
            "threshold_used": 27.0
        }

    Raises:
        ImportError: 如果未安装 scenedetect 库
        Exception: 如果处理视频失败
    """
    if not PYSCENEDETECT_AVAILABLE:
        raise ImportError(
            "需要安装 scenedetect 库: pip install scenedetect[opencv]")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    try:
        # 选择检测器
        if detector_type.lower() == "adaptive":
            detector = AdaptiveDetector(
                threshold=threshold) if threshold else AdaptiveDetector()
            actual_threshold = threshold if threshold else 3.0  # AdaptiveDetector 默认阈值
        elif detector_type.lower() == "content":
            detector = ContentDetector(
                threshold=threshold) if threshold else ContentDetector()
            actual_threshold = threshold if threshold else 27.0  # ContentDetector 默认阈值
        else:
            raise ValueError(f"不支持的检测器类型: {detector_type}")

        # 检测场景
        video = VideoFileClip(video_path)
        video_width, video_height = video.size

        scene_list = detect(video_path, detector)

        if not scene_list:
            fps = video.fps
            total_frames = int(video.duration * fps)
            start_time = FrameTimecode(0, fps)
            end_time = FrameTimecode(total_frames, fps)
            scene_list = [(start_time, end_time)]

        scenes_data = []

        for i, (start_time, end_time) in enumerate(scene_list):
            scene_number = i + 1

            # 获取时间戳（秒）
            start_seconds = start_time.get_seconds()
            end_seconds = end_time.get_seconds()-0.1

            # 计算中间时间戳
            middle_seconds = (start_seconds + end_seconds) / 2

            # random frame
            random_frame_seconds = random.uniform(start_seconds, end_seconds)

            # 抽取帧
            try:
                # first_frame_bytes = get_frame_at_timestamp(
                #     video, start_seconds)
                middle_frame_bytes = get_frame_at_timestamp(
                    video, middle_seconds)
                # 对于最后一帧，稍微往前一点以避免可能的越界
                # last_frame_time = max(start_seconds, end_seconds - 0.1)
                # last_frame_bytes = get_frame_at_timestamp(
                #     video, last_frame_time)
                concat_4_frames_bytes = concat_4_frames_from_video(
                    video, [start_seconds, middle_seconds, end_seconds, random_frame_seconds])

                scene_info = {
                    "scene_number": scene_number,
                    "start_time": start_seconds,
                    "end_time": end_seconds,
                    "duration": end_seconds - start_seconds,
                    "start_frame": start_time.frame_num,
                    "end_frame": end_time.frame_num,
                    "frame_count": end_time.frame_num - start_time.frame_num,
                    # "first_frame_bytes": first_frame_bytes,
                    "middle_frame_bytes": middle_frame_bytes,
                    # "last_frame_bytes": last_frame_bytes
                    "concat_4_frames_bytes":  concat_4_frames_bytes
                }

                scenes_data.append(scene_info)

            except Exception as e:
                print(f"警告: 处理场景 {scene_number} 时出错: {str(e)}")
                continue

        video.close()

        result = {
            "scenes": scenes_data,
            "total_scenes": len(scenes_data),
            "detector_used": detector_type,
            "threshold_used": actual_threshold,
            "width": video_width,
            "height": video_height
        }

        return result

    except Exception as e:
        raise Exception(f"视频场景检测和帧抽取失败: {str(e)}")
