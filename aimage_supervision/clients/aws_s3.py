import io
import os
import tempfile
import uuid
from typing import Any, BinaryIO

import boto3
import httpx

from aimage_supervision.settings import (AWS_ACCESS_KEY_ID, AWS_BUCKET_NAME,
                                         AWS_REGION, AWS_SECRET_ACCESS_KEY,
                                         boto3_session)


async def get_s3_url_from_path(cloud_file_path: str | None, expires_in: int = 3600*3, bucket_name: str = AWS_BUCKET_NAME) -> str:
    '''
    Generates a presigned URL for S3 objects that expires after 1 hour.

    Args:
    cloud_file_path (str): The path to the file in the S3 bucket.
    expires_in (int): The number of seconds the URL will be valid for. Defaults to 3 hour (3600s*3).

    Returns:
    str: A presigned URL to access the S3 object.
    '''
    if not cloud_file_path:
        return ''

    cleaned_path = cloud_file_path
    if cloud_file_path.startswith('s3://'):
        parts = cloud_file_path.split('/', 3)
        if len(parts) > 3:
            cleaned_path = parts[3]

    try:

        async with boto3_session.client('s3') as s3_client:
            presigned_url = await s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={
                    'Bucket': bucket_name,
                    'Key': cleaned_path,
                },
                ExpiresIn=expires_in,
            )
            return presigned_url
    except Exception as e:
        print(f"Error generating presigned URL for {cleaned_path}: {str(e)}")
        return ''


async def get_path_with_s3_url(cloud_file_path: str | None, expires_in: int = 3600*3, bucket_name: str = AWS_BUCKET_NAME) -> tuple[str, str]:
    if not cloud_file_path:
        return '', ''
    url = await get_s3_url_from_path(cloud_file_path, expires_in, bucket_name)
    return cloud_file_path, url


async def upload_file_to_s3(file_data: BinaryIO, cloud_file_path: str, bucket_name: str = AWS_BUCKET_NAME):
    if not cloud_file_path or not file_data:
        raise ValueError('cloud_file_path and file_data are required')

    async with boto3_session.client('s3') as s3_client:
        await s3_client.upload_fileobj(
            file_data,
            bucket_name,
            cloud_file_path,
        )


async def download_file_from_s3(cloud_file_path: str, local_file_path: str, bucket_name: str = AWS_BUCKET_NAME) -> None:
    '''
    从S3下载文件到本地

    Args:
        cloud_file_path (str): S3上的文件路径
        local_file_path (str): 要保存到的本地文件路径
    '''
    if not cloud_file_path or not local_file_path:
        raise ValueError('cloud_file_path and local_file_path are required')

    async with boto3_session.client('s3') as s3_client:
        await s3_client.download_file(
            bucket_name,
            cloud_file_path,
            local_file_path,
        )


async def download_file_content_from_s3(cloud_file_path: str, bucket_name: str = AWS_BUCKET_NAME) -> bytes:
    '''
    直接从S3下载文件内容到内存

    Args:
        cloud_file_path (str): S3上的文件路径
        bucket_name (str): S3存储桶名称，默认使用AWS_BUCKET_NAME

    Returns:
        bytes: 文件内容
    '''
    if not cloud_file_path:
        raise ValueError('cloud_file_path is required')

    # 处理以s3://开头的路径
    cleaned_path = cloud_file_path
    if cloud_file_path.startswith('s3://'):
        parts = cloud_file_path.split('/', 3)
        if len(parts) > 3:
            cleaned_path = parts[3]

    try:
        # 创建一个内存缓冲区来接收文件内容
        file_content = io.BytesIO()

        async with boto3_session.client('s3') as s3_client:
            await s3_client.download_fileobj(
                bucket_name,
                cleaned_path,
                file_content
            )

        # 重置缓冲区位置，以便读取内容
        file_content.seek(0)
        return file_content.getvalue()
    except Exception as e:
        raise ValueError(f"Error downloading file from S3: {str(e)}")


def download_file_content_from_s3_sync(cloud_file_path: str, bucket_name: str = AWS_BUCKET_NAME) -> bytes:
    '''
    直接从S3下载文件内容到内存

    Args:
        cloud_file_path (str): S3上的文件路径
        bucket_name (str): S3存储桶名称，默认使用AWS_BUCKET_NAME

    Returns:
        bytes: 文件内容
    '''
    if not cloud_file_path:
        raise ValueError('cloud_file_path is required')

    # 处理以s3://开头的路径
    cleaned_path = cloud_file_path
    if cloud_file_path.startswith('s3://'):
        parts = cloud_file_path.split('/', 3)
        if len(parts) > 3:
            cleaned_path = parts[3]

    try:
        # 创建一个内存缓冲区来接收文件内容
        file_content = io.BytesIO()

        # 使用标准的boto3创建同步客户端
        s3_client = boto3.client(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        s3_client.download_fileobj(
            bucket_name,
            cleaned_path,
            file_content
        )

        # 重置缓冲区位置，以便读取内容
        file_content.seek(0)
        return file_content.getvalue()
    except Exception as e:
        raise ValueError(f"Error downloading file from S3: {str(e)}")


async def download_file_content_as_bytesio_from_s3(cloud_file_path: str, bucket_name: str = AWS_BUCKET_NAME) -> BinaryIO:
    '''
    从S3下载文件内容并返回BytesIO对象，用于图片embedding

    Args:
        cloud_file_path (str): S3上的文件路径
        bucket_name (str): S3存储桶名称，默认使用AWS_BUCKET_NAME

    Returns:
        BinaryIO: BytesIO对象
    '''
    if not cloud_file_path:
        raise ValueError('cloud_file_path is required')

    # 处理以s3://开头的路径
    cleaned_path = cloud_file_path
    if cloud_file_path.startswith('s3://'):
        parts = cloud_file_path.split('/', 3)
        if len(parts) > 3:
            cleaned_path = parts[3]

    try:
        # 创建一个内存缓冲区来接收文件内容
        file_content = io.BytesIO()

        async with boto3_session.client('s3') as s3_client:
            await s3_client.download_fileobj(
                bucket_name,
                cleaned_path,
                file_content
            )

        # 重置缓冲区位置，以便读取内容
        file_content.seek(0)
        return file_content
    except Exception as e:
        raise ValueError(f"Error downloading file from S3: {str(e)}")


async def download_image_from_url_or_s3_path(image_url_or_path: str, bucket_name: str = AWS_BUCKET_NAME) -> BinaryIO:
    '''
    智能下载图片：支持HTTP URL (presigned URL) 和 S3路径两种格式

    Args:
        image_url_or_path (str): 图片URL或S3路径
        bucket_name (str): S3存储桶名称，默认使用AWS_BUCKET_NAME

    Returns:
        BinaryIO: BytesIO对象
    '''
    if not image_url_or_path:
        raise ValueError('image_url_or_path is required')

    # 判断是HTTP URL还是S3路径
    if image_url_or_path.startswith(('http://', 'https://')):
        # HTTP URL - 使用HTTP请求下载（presigned URL）
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(image_url_or_path)
                response.raise_for_status()  # 如果状态码不是2xx，抛出异常

                # 创建BytesIO对象
                file_content = io.BytesIO(response.content)
                file_content.seek(0)
                return file_content

        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"HTTP error downloading image: {e.response.status_code} - {e.response.text}")
        except httpx.TimeoutException:
            raise ValueError("Timeout downloading image from URL")
        except Exception as e:
            raise ValueError(f"Error downloading image from URL: {str(e)}")

    else:
        # S3路径 - 使用S3 SDK下载
        return await download_file_content_as_bytesio_from_s3(image_url_or_path, bucket_name)


# === 临时存储相关函数 ===

async def upload_file_to_s3_with_ttl(
    file_data: BinaryIO,
    s3_path: str,
    ttl_hours: int = 2,
    content_type: str = "image/jpeg",
    bucket_name: str = AWS_BUCKET_NAME
) -> None:
    """
    上传文件到S3并设置TTL标签

    Args:
        file_data: 文件数据流
        s3_path: S3存储路径
        ttl_hours: TTL时间（小时），默认2小时
        content_type: 文件MIME类型
        bucket_name: S3存储桶名称
    """
    if not s3_path or not file_data:
        raise ValueError('s3_path 和 file_data 是必需的')

    from datetime import datetime, timedelta

    # 计算过期时间
    expire_time = datetime.utcnow() + timedelta(hours=ttl_hours)
    expire_timestamp = int(expire_time.timestamp())

    try:
        async with boto3_session.client('s3') as s3_client:
            # 上传文件并设置标签
            await s3_client.upload_fileobj(
                file_data,
                bucket_name,
                s3_path,
                ExtraArgs={
                    'ContentType': content_type,
                    'Tagging': f'session_type=temp&expire_at={expire_timestamp}'
                }
            )
        print(f"已上传 {s3_path}，TTL: {ttl_hours} 小时")
    except Exception as e:
        print(f"上传文件到S3失败 {s3_path}: {e}")
        raise


async def cleanup_s3_prefix(prefix: str, bucket_name: str = AWS_BUCKET_NAME) -> tuple[int, list[str]]:
    """
    清理S3指定前缀下的所有文件

    Args:
        prefix: S3路径前缀
        bucket_name: S3存储桶名称

    Returns:
        tuple[int, list[str]]: (删除的文件数量, 删除的文件路径列表)
    """
    if not prefix:
        raise ValueError('prefix 是必需的')

    deleted_count = 0
    deleted_files: list[str] = []

    try:
        async with boto3_session.client('s3') as s3_client:
            # 列出所有匹配前缀的文件
            response = await s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                return deleted_count, deleted_files

            # 批量删除文件
            objects_to_delete = [{'Key': obj['Key']}
                                 for obj in response['Contents']]

            if objects_to_delete:
                delete_response = await s3_client.delete_objects(  # type: ignore
                    Bucket=bucket_name,
                    Delete={'Objects': objects_to_delete}
                )

                deleted_count = len(delete_response.get('Deleted', []))
                deleted_files = [obj['Key']
                                 for obj in delete_response.get('Deleted', [])]

        print(f"清理前缀 {prefix}: 删除了 {deleted_count} 个文件")
        return deleted_count, deleted_files

    except Exception as e:
        print(f"清理S3前缀失败 {prefix}: {e}")
        raise


async def list_files_with_prefix(prefix: str, bucket_name: str = AWS_BUCKET_NAME) -> list[str]:
    """
    列出S3指定前缀下的所有文件

    Args:
        prefix: S3路径前缀
        bucket_name: S3存储桶名称

    Returns:
        list[str]: 文件路径列表
    """
    if not prefix:
        return []

    try:
        async with boto3_session.client('s3') as s3_client:
            response = await s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                return []

            return [obj['Key'] for obj in response['Contents']]

    except Exception as e:
        print(f"列出S3前缀文件失败 {prefix}: {e}")
        return []


async def move_s3_files(from_prefix: str, to_prefix: str, bucket_name: str = AWS_BUCKET_NAME) -> tuple[list[str], list[str]]:
    """
    将S3文件从一个前缀移动到另一个前缀

    Args:
        from_prefix: 源前缀
        to_prefix: 目标前缀
        bucket_name: S3存储桶名称

    Returns:
        tuple[list[str], list[str]]: (成功移动的文件列表, 失败的文件列表)
    """
    if not from_prefix or not to_prefix:
        raise ValueError('from_prefix 和 to_prefix 是必需的')

    moved_files = []
    failed_files = []

    try:
        # 获取所有源文件
        source_files = await list_files_with_prefix(from_prefix, bucket_name)

        async with boto3_session.client('s3') as s3_client:
            for source_key in source_files:
                try:
                    # 生成目标路径
                    relative_path = source_key[len(from_prefix):].lstrip('/')
                    target_key = f"{to_prefix.rstrip('/')}/{relative_path}"

                    # 复制文件
                    await s3_client.copy_object(
                        Bucket=bucket_name,
                        CopySource={'Bucket': bucket_name, 'Key': source_key},
                        Key=target_key
                    )

                    # 删除源文件
                    await s3_client.delete_object(
                        Bucket=bucket_name,
                        Key=source_key
                    )

                    moved_files.append(f"s3://{bucket_name}/{target_key}")

                except Exception as file_error:
                    print(f"移动文件失败 {source_key}: {file_error}")
                    failed_files.append(source_key)

        print(f"移动文件: 成功 {len(moved_files)} 个，失败 {len(failed_files)} 个")
        return moved_files, failed_files

    except Exception as e:
        print(f"批量移动文件失败: {e}")
        raise


async def download_image_from_s3(s3_path: str) -> str:
    """Downloads an image from S3 and saves it to a temporary file."""
    # This helper was implicitly part of the old ai_review.py, making it explicit here.
    presigned_url = await get_s3_url_from_path(s3_path)
    image_path = os.path.join(tempfile.gettempdir(),
                              f"{uuid.uuid4()}.jpg")
    async with httpx.AsyncClient() as client:
        response = await client.get(presigned_url)
        response.raise_for_status()
        with open(image_path, "wb") as f:
            f.write(response.content)
    return image_path


async def download_video_from_s3(s3_path: str) -> str:
    """Downloads a video from S3 and saves it to a temporary file."""
    presigned_url = await get_s3_url_from_path(s3_path)
    video_path = os.path.join(tempfile.gettempdir(),
                              f"{uuid.uuid4()}.mp4")
    async with httpx.AsyncClient() as client:
        response = await client.get(presigned_url)
        response.raise_for_status()
        with open(video_path, "wb") as f:
            f.write(response.content)
    return video_path

