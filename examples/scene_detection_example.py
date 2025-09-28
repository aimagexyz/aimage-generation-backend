#!/usr/bin/env python3
"""
视频场景切分和帧抽取示例

这个示例展示了如何使用 PySceneDetect 来：
1. 检测视频中的场景切换点
2. 抽取每个场景的第一帧、中间帧和最后一帧
3. 保存结果到文件

使用方法:
    python scene_detection_example.py <video_path> [output_dir]

依赖:
    pip install scenedetect[opencv]
"""

from aimage_supervision.utils.video_utils import (
    detect_video_scenes_and_extract_frames, get_scene_representative_frames)
import os
import sys
from pathlib import Path

# 添加父目录到 Python 路径以便导入模块
sys.path.append(str(Path(__file__).parent.parent))


def main():
    if len(sys.argv) < 2:
        print(
            "用法: python scene_detection_example.py <video_path> [output_dir]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        sys.exit(1)

    print(f"正在处理视频: {video_path}")
    print("=" * 50)

    try:
        # 使用自适应检测器进行场景检测
        result = detect_video_scenes_and_extract_frames(
            video_path=video_path,
            detector_type="adaptive",
            output_dir=output_dir
        )

        print(f"检测完成!")
        print(f"使用的检测器: {result['detector_used']}")
        print(f"检测阈值: {result['threshold_used']}")
        print(f"共检测到 {result['total_scenes']} 个场景")
        print()

        # 显示每个场景的信息
        for scene in result['scenes']:
            print(f"场景 {scene['scene_number']}:")
            print(
                f"  时间范围: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s")
            print(f"  持续时间: {scene['duration']:.2f}s")
            print(f"  帧范围: {scene['start_frame']} - {scene['end_frame']}")
            print(f"  帧数量: {scene['frame_count']}")

            if output_dir:
                print(f"  保存的图片:")
                print(f"    第一帧: {scene['first_frame_path']}")
                print(f"    中间帧: {scene['middle_frame_path']}")
                print(f"    最后一帧: {scene['last_frame_path']}")
            print()

        if output_dir:
            print(f"场景信息已保存到: {result.get('scenes_info_path', 'N/A')}")

        # 演示获取特定场景的帧
        if result['total_scenes'] > 0:
            print("=" * 50)
            print("演示: 获取第一个场景的代表性帧")
            scene_frames = get_scene_representative_frames(
                video_path=video_path,
                scene_number=1,
                detector_type="adaptive"
            )

            print(f"第一个场景的帧数据:")
            print(f"  第一帧大小: {len(scene_frames['first_frame'])} bytes")
            print(f"  中间帧大小: {len(scene_frames['middle_frame'])} bytes")
            print(f"  最后帧大小: {len(scene_frames['last_frame'])} bytes")

    except ImportError as e:
        print(f"错误: {e}")
        print("请安装 PySceneDetect: pip install scenedetect[opencv]")
        sys.exit(1)
    except Exception as e:
        print(f"处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
