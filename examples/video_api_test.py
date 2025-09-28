#!/usr/bin/env python3
"""
视频场景检测 API 测试脚本

这个脚本演示了如何使用新创建的视频场景检测API端点。

使用方法:
    python video_api_test.py <video_file_path>

依赖:
    pip install httpx
"""

import asyncio
import base64
import sys
from pathlib import Path

import httpx

# API 基础URL（根据实际部署环境调整）
BASE_URL = "http://localhost:8000/api/v1"

# 认证令牌（请替换为实际的JWT令牌）
# 在实际使用中，你需要先通过认证端点获取令牌
AUTH_TOKEN = "your_jwt_token_here"


async def test_analyze_video_scenes(video_path: str):
    """测试视频场景分析API"""
    print("=" * 60)
    print("测试: 分析视频场景")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # 准备文件
        with open(video_path, 'rb') as f:
            video_content = f.read()

        # 准备请求
        files = {
            'file': (Path(video_path).name, video_content, 'video/mp4')
        }
        data = {
            'detector_type': 'adaptive',
            # 'threshold': 3.0  # 可选参数
        }
        headers = {
            'Authorization': f'Bearer {AUTH_TOKEN}'
        }

        try:
            response = await client.post(
                f"{BASE_URL}/video/scenes/analyze",
                files=files,
                data=data,
                headers=headers,
                timeout=300.0  # 5分钟超时，视频处理可能需要较长时间
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ 分析成功!")
                print(f"检测器类型: {result['detector_used']}")
                print(f"检测阈值: {result['threshold_used']}")
                print(f"总场景数: {result['total_scenes']}")
                print()

                if result['scenes']:
                    print("场景列表:")
                    for scene in result['scenes']:
                        print(f"  场景 {scene['scene_number']}:")
                        print(
                            f"    时间: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s")
                        print(f"    持续: {scene['duration']:.2f}s")
                        print(f"    帧数: {scene['frame_count']}")
                        print()

                return result
            else:
                print(f"❌ 请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None

        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
            return None


async def test_get_scene_frames(video_path: str, scene_number: int):
    """测试获取场景关键帧API"""
    print("=" * 60)
    print(f"测试: 获取场景 {scene_number} 的关键帧")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # 准备文件
        with open(video_path, 'rb') as f:
            video_content = f.read()

        # 准备请求
        files = {
            'file': (Path(video_path).name, video_content, 'video/mp4')
        }
        data = {
            'detector_type': 'adaptive'
        }
        headers = {
            'Authorization': f'Bearer {AUTH_TOKEN}'
        }

        try:
            response = await client.post(
                f"{BASE_URL}/video/scenes/{scene_number}/frames",
                files=files,
                data=data,
                headers=headers,
                timeout=300.0
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ 获取成功!")
                print(f"场景编号: {result['scene_number']}")

                # 保存帧图片（可选）
                save_frames = input("是否保存帧图片到本地? (y/n): ").lower() == 'y'
                if save_frames:
                    video_name = Path(video_path).stem

                    # 保存第一帧
                    first_frame_data = base64.b64decode(result['first_frame'])
                    with open(f"{video_name}_scene{scene_number}_first.jpg", 'wb') as f:
                        f.write(first_frame_data)

                    # 保存中间帧
                    middle_frame_data = base64.b64decode(
                        result['middle_frame'])
                    with open(f"{video_name}_scene{scene_number}_middle.jpg", 'wb') as f:
                        f.write(middle_frame_data)

                    # 保存最后帧
                    last_frame_data = base64.b64decode(result['last_frame'])
                    with open(f"{video_name}_scene{scene_number}_last.jpg", 'wb') as f:
                        f.write(last_frame_data)

                    print(f"帧图片已保存:")
                    print(f"  第一帧: {video_name}_scene{scene_number}_first.jpg")
                    print(
                        f"  中间帧: {video_name}_scene{scene_number}_middle.jpg")
                    print(f"  最后帧: {video_name}_scene{scene_number}_last.jpg")

                return result
            else:
                print(f"❌ 请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None

        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
            return None


async def test_get_scene_frames_with_info(video_path: str, scene_number: int):
    """测试获取场景关键帧和详细信息API"""
    print("=" * 60)
    print(f"测试: 获取场景 {scene_number} 的关键帧和信息")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # 准备文件
        with open(video_path, 'rb') as f:
            video_content = f.read()

        # 准备请求
        files = {
            'file': (Path(video_path).name, video_content, 'video/mp4')
        }
        data = {
            'detector_type': 'adaptive',
            # 'threshold': 3.0  # 可选参数
        }
        headers = {
            'Authorization': f'Bearer {AUTH_TOKEN}'
        }

        try:
            response = await client.post(
                f"{BASE_URL}/video/scenes/{scene_number}/frames-with-info",
                files=files,
                data=data,
                headers=headers,
                timeout=300.0
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ 获取成功!")

                # 显示场景信息
                scene_info = result['scene_info']
                print(f"场景信息:")
                print(f"  编号: {scene_info['scene_number']}")
                print(
                    f"  时间: {scene_info['start_time']:.2f}s - {scene_info['end_time']:.2f}s")
                print(f"  持续: {scene_info['duration']:.2f}s")
                print(
                    f"  帧范围: {scene_info['start_frame']} - {scene_info['end_frame']}")
                print(f"  帧数量: {scene_info['frame_count']}")
                print()

                # 显示帧信息
                frames = result['frames']
                print(f"关键帧:")
                print(
                    f"  第一帧大小: {len(base64.b64decode(frames['first_frame']))} bytes")
                print(
                    f"  中间帧大小: {len(base64.b64decode(frames['middle_frame']))} bytes")
                print(
                    f"  最后帧大小: {len(base64.b64decode(frames['last_frame']))} bytes")

                return result
            else:
                print(f"❌ 请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None

        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
            return None


async def main():
    if len(sys.argv) < 2:
        print("用法: python video_api_test.py <video_file_path>")
        print()
        print("注意:")
        print("1. 请确保服务器正在运行 (通常在 http://localhost:8000)")
        print("2. 请在脚本中设置正确的 AUTH_TOKEN")
        print("3. 确保已安装 httpx: pip install httpx")
        sys.exit(1)

    video_path = sys.argv[1]

    if not Path(video_path).exists():
        print(f"错误: 视频文件不存在: {video_path}")
        sys.exit(1)

    print(f"开始测试视频 API")
    print(f"视频文件: {video_path}")
    print(f"API 基础URL: {BASE_URL}")
    print()

    # 测试1: 分析视频场景
    scenes_result = await test_analyze_video_scenes(video_path)

    if scenes_result and scenes_result['total_scenes'] > 0:
        print()
        # 测试2: 获取第一个场景的关键帧
        await test_get_scene_frames(video_path, 1)

        print()
        # 测试3: 获取第一个场景的关键帧和详细信息
        await test_get_scene_frames_with_info(video_path, 1)

    print()
    print("=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 检查依赖
    try:
        import httpx
    except ImportError:
        print("错误: 需要安装 httpx")
        print("请运行: pip install httpx")
        sys.exit(1)

    asyncio.run(main())
