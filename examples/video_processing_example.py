"""
视频和图片处理示例

演示如何使用 aimage_supervision.utils.video_utils 中的功能来:
1. 检测字节数据是图片还是视频
2. 从视频中提取第一帧
3. 自动处理图片或视频数据
"""

import asyncio

from aimage_supervision.clients.aws_s3 import \
    download_file_content_from_s3_sync
from aimage_supervision.utils.video_utils import (get_first_frame_as_pil_image,
                                                  get_first_frame_of_video,
                                                  is_video_format,
                                                  process_image_or_video_bytes)


async def example_process_media_from_s3():
    """
    示例：从S3下载媒体文件并自动处理
    """
    # 假设你从S3获得了一些字节数据
    s3_path = "your/media/file/path"  # 替换为实际的S3路径

    try:
        # 从S3下载文件内容
        media_bytes = download_file_content_from_s3_sync(s3_path)

        # 方法1: 先检测是否为视频，然后分别处理
        if is_video_format(media_bytes):
            print("检测到视频格式，提取第一帧...")
            first_frame_bytes = get_first_frame_of_video(media_bytes)
            print(f"成功提取第一帧，大小: {len(first_frame_bytes)} bytes")
        else:
            print("检测到图片格式，直接使用")
            first_frame_bytes = media_bytes

        # 方法2: 使用自动处理函数（推荐）
        processed_bytes = process_image_or_video_bytes(media_bytes)
        print(f"自动处理完成，结果大小: {len(processed_bytes)} bytes")

        return processed_bytes

    except Exception as e:
        print(f"处理媒体文件时出错: {e}")
        return None


def example_local_file_processing():
    """
    示例：处理本地文件
    """
    # 读取本地文件（图片或视频）
    file_path = "path/to/your/media/file"  # 替换为实际路径

    try:
        with open(file_path, 'rb') as f:
            media_bytes = f.read()

        # 检测文件类型
        if is_video_format(media_bytes):
            print(f"文件 {file_path} 是视频格式")

            # 提取第一帧作为PIL Image对象
            first_frame_pil = get_first_frame_as_pil_image(media_bytes)
            print(f"第一帧尺寸: {first_frame_pil.size}")

            # 或者提取第一帧作为JPEG字节数据
            first_frame_bytes = get_first_frame_of_video(media_bytes)
            print(f"第一帧JPEG大小: {len(first_frame_bytes)} bytes")

        else:
            print(f"文件 {file_path} 是图片格式")

        # 自动处理
        result_bytes = process_image_or_video_bytes(media_bytes)

        # 保存处理结果
        output_path = "processed_image.jpg"
        with open(output_path, 'wb') as f:
            f.write(result_bytes)
        print(f"处理结果已保存到: {output_path}")

    except Exception as e:
        print(f"处理文件时出错: {e}")


def example_integrate_with_ai_model():
    """
    示例：在AI模型中集成使用
    """
    def process_media_for_ai_analysis(media_bytes: bytes) -> bytes:
        """
        为AI分析准备媒体数据
        无论输入是图片还是视频，都返回可用于AI分析的图片数据
        """
        try:
            # 自动处理：如果是视频提取第一帧，如果是图片直接返回
            processed_bytes = process_image_or_video_bytes(media_bytes)

            print("媒体数据已准备就绪，可用于AI分析")
            return processed_bytes

        except Exception as e:
            print(f"准备AI分析数据时出错: {e}")
            raise

    # 使用示例
    # media_bytes = your_media_data  # 来自任何源的媒体数据
    # ai_ready_bytes = process_media_for_ai_analysis(media_bytes)
    # # 现在可以将 ai_ready_bytes 传递给任何期望图片数据的AI模型


if __name__ == "__main__":
    print("=== 视频和图片处理示例 ===")

    print("\n1. 本地文件处理示例:")
    # example_local_file_processing()  # 取消注释并提供实际文件路径

    print("\n2. S3媒体处理示例:")
    # asyncio.run(example_process_media_from_s3())  # 取消注释并提供实际S3路径

    print("\n3. AI集成示例:")
    example_integrate_with_ai_model()

    print("\n提示：取消注释相应的函数调用并提供实际的文件路径或S3路径来运行完整示例")
