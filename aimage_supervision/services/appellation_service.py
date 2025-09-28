import io
import json
import uuid
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import UploadFile

from aimage_supervision.clients.aws_s3 import upload_file_to_s3
from aimage_supervision.settings import AWS_BUCKET_NAME


async def parse_and_upload_appellation_file(
    file: UploadFile,
    project_id: str,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    解析称呼表文件（Excel或JSON）并上传到S3

    Args:
        file: 上传的文件（Excel或JSON）
        project_id: 项目ID
        session_id: 会话ID（可选，用于临时文件）

    Returns:
        Dict包含：
        - s3_url: 上传后的S3路径
        - file_type: 文件类型 ('excel' 或 'json')
        - data: 解析后的JSON数据
        - characters: 角色名称列表
    """

    # 读取文件内容
    file_content = await file.read()

    # 根据文件扩展名判断类型
    filename = file.filename or ""
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ""

    # 解析文件内容
    if file_extension in ['xlsx', 'xls']:
        # Excel文件处理
        appellation_data = _parse_excel_appellation(file_content)
        file_type = 'excel'
    elif file_extension == 'json':
        # JSON文件处理
        appellation_data = _parse_json_appellation(file_content)
        file_type = 'json'
    else:
        raise ValueError(
            f"不支持的文件类型: {file_extension}。仅支持 Excel (.xlsx, .xls) 和 JSON 文件。")

    # 生成S3路径
    unique_id = str(uuid.uuid4())
    if session_id:
        # 临时文件路径（用于RPD创建过程中）
        s3_path = f"temp/appellations/{session_id}/{unique_id}_appellation.json"
    else:
        # 永久文件路径（用于已保存的RPD）
        s3_path = f"projects/{project_id}/appellations/{unique_id}_appellation.json"

    # 将解析后的数据转换为JSON并上传到S3
    json_data = json.dumps(appellation_data, ensure_ascii=False, indent=2)
    json_bytes = io.BytesIO(json_data.encode('utf-8'))

    await upload_file_to_s3(
        file_data=json_bytes,
        cloud_file_path=s3_path
    )

    # 提取角色名称列表
    characters = list(appellation_data.keys()) if appellation_data else []

    return {
        "s3_url": f"s3://{AWS_BUCKET_NAME}/{s3_path}",
        "file_type": file_type,
        "data": appellation_data,
        "characters": characters,
        "original_filename": filename
    }


def _parse_excel_appellation(file_content: bytes) -> Dict[str, Dict[str, str]]:
    """
    解析Excel格式的称呼表

    Excel格式：
    - 第一行和第一列为角色名称
    - 行表示某个角色被各种角色的称呼
    - 列表示某个角色对其他角色的称呼
    - 对角线上是第一人称

    Args:
        file_content: Excel文件的字节内容

    Returns:
        Dict[str, Dict[str, str]]: 称呼表数据
    """
    try:
        # 使用pandas读取Excel文件
        excel_file = io.BytesIO(file_content)
        df = pd.read_excel(excel_file, index_col=0)

        # 验证数据格式
        if df.empty:
            raise ValueError("Excel文件为空")

        # 获取角色名称（行和列的标签）
        characters = df.index.tolist()
        columns = df.columns.tolist()

        # 验证行列是否匹配（应该是方形矩阵）
        if len(characters) != len(columns):
            raise ValueError("Excel文件格式错误：行数和列数不匹配，应该是方形的称呼表")

        # 构建称呼表字典
        appellation_data: Dict[str, Dict[str, str]] = {}

        for speaker in characters:
            appellation_data[str(speaker)] = {}
            for target in columns:
                # 获取称呼内容，处理NaN值
                appellation = df.loc[speaker, target]
                if pd.isna(appellation):
                    # 如果是对角线（自己称呼自己），使用默认值
                    if speaker == target:
                        appellation_data[str(speaker)][str(
                            target)] = "私"  # 默认第一人称
                    else:
                        appellation_data[str(speaker)][str(
                            target)] = str(target)  # 默认直接称呼名字
                else:
                    appellation_data[str(speaker)][str(
                        target)] = str(appellation)

        return appellation_data

    except Exception as e:
        raise ValueError(f"解析Excel文件失败: {str(e)}")


def _parse_json_appellation(file_content: bytes) -> Dict[str, Dict[str, str]]:
    """
    解析JSON格式的称呼表

    JSON格式: {"A":{"A":"第一人称","B":"A称呼B",...},"B":{...}...}

    Args:
        file_content: JSON文件的字节内容

    Returns:
        Dict[str, Dict[str, str]]: 称呼表数据
    """
    try:
        # 解析JSON内容
        json_str = file_content.decode('utf-8')
        appellation_data = json.loads(json_str)

        # 验证数据格式
        if not isinstance(appellation_data, dict):
            raise ValueError("JSON文件格式错误：根元素必须是对象")

        # 验证每个角色的数据格式
        for speaker, appellations in appellation_data.items():
            if not isinstance(appellations, dict):
                raise ValueError(f"JSON文件格式错误：角色 '{speaker}' 的称呼数据必须是对象")

            # 确保所有值都是字符串
            for target, appellation in appellations.items():
                appellation_data[speaker][target] = str(appellation)

        return appellation_data

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON文件格式错误: {str(e)}")
    except UnicodeDecodeError as e:
        raise ValueError(f"JSON文件编码错误: {str(e)}")
    except Exception as e:
        raise ValueError(f"解析JSON文件失败: {str(e)}")


def validate_appellation_data(appellation_data: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """
    验证称呼表数据的完整性和一致性

    Args:
        appellation_data: 称呼表数据

    Returns:
        Dict包含验证结果和统计信息
    """
    if not appellation_data:
        return {
            "valid": False,
            "errors": ["称呼表数据为空"],
            "characters": [],
            "total_appellations": 0
        }

    errors: List[str] = []
    warnings: List[str] = []
    characters = list(appellation_data.keys())
    total_appellations = 0

    # 检查每个角色的数据
    for speaker in characters:
        appellations = appellation_data[speaker]
        total_appellations += len(appellations)

        # 检查是否有自称（第一人称）
        if speaker not in appellations:
            warnings.append(f"角色 '{speaker}' 缺少第一人称设定")

        # 检查是否对所有其他角色都有称呼设定
        for other_character in characters:
            if other_character not in appellations:
                warnings.append(
                    f"角色 '{speaker}' 对角色 '{other_character}' 缺少称呼设定")

    # 检查数据的对称性（可选警告）
    for speaker in characters:
        for target in characters:
            if (speaker in appellation_data and
                target in appellation_data[speaker] and
                target in appellation_data and
                    speaker in appellation_data[target]):
                # 这里可以添加更多的一致性检查
                pass

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "characters": characters,
        "total_appellations": total_appellations,
        "character_count": len(characters)
    }


async def get_appellation_for_characters(
    s3_path: str,
    speaker: str,
    target: str
) -> Optional[str]:
    """
    从S3上的称呼表文件中获取特定的称呼

    Args:
        s3_path: 称呼表文件的S3路径
        speaker: 说话者角色名称
        target: 目标角色名称

    Returns:
        Optional[str]: 称呼内容，如果未找到则返回None
    """
    try:
        from aimage_supervision.clients.aws_s3 import \
            download_file_content_from_s3

        # 从S3下载文件内容
        file_content = await download_file_content_from_s3(s3_path)

        # 解析JSON数据
        appellation_data = json.loads(file_content.decode('utf-8'))

        # 获取称呼
        if speaker in appellation_data and target in appellation_data[speaker]:
            return appellation_data[speaker][target]

        return None

    except Exception as e:
        print(f"获取称呼失败: {str(e)}")
        return None


async def get_all_appellations_from_s3(s3_url: str) -> Optional[Dict[str, Dict[str, str]]]:
    """
    从S3获取完整的称呼表数据

    Args:
        s3_url: 称呼表文件的S3 URL (格式: s3://bucket/path)

    Returns:
        Optional[Dict]: 完整的称呼表数据，如果失败则返回None
    """
    try:
        from aimage_supervision.clients.aws_s3 import \
            download_file_content_from_s3

        # 从S3 URL中提取路径部分
        if s3_url.startswith('s3://'):
            # 移除s3://bucket/前缀，只保留路径部分
            path_parts = s3_url.replace('s3://', '').split('/', 1)
            if len(path_parts) >= 2:
                s3_path = path_parts[1]  # 获取路径部分
            else:
                raise ValueError(f"无效的S3 URL格式: {s3_url}")
        else:
            # 如果不是完整URL，假设它已经是路径
            s3_path = s3_url

        # 从S3下载文件内容
        file_content = await download_file_content_from_s3(s3_path)

        # 解析JSON数据
        appellation_data = json.loads(file_content.decode('utf-8'))

        return appellation_data

    except Exception as e:
        print(f"获取称呼表失败: Error downloading file from S3: {str(e)}")
        return None
