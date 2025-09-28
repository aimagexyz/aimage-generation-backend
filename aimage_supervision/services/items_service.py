import asyncio
import io
import json
import os
from typing import List, Optional

from google import genai
from google.genai import types
from PIL import Image
from sqlalchemy import text

from aimage_supervision.clients.aws_s3 import (
    download_file_content_from_s3, download_image_from_url_or_s3_path)
from aimage_supervision.clients.multimodal_embedding import (
    get_multimodal_embedding_image, get_text_embedding)
from aimage_supervision.models import Item
from aimage_supervision.settings import logger, vecs_client, MAX_CONCURRENT


class ItemsVectorService:
    """Service for managing Items vector embeddings"""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.image_collection = vecs_client.get_or_create_collection(
            name=f"items_images_project_{project_id}",
            dimension=512
        )
        self.text_collection = vecs_client.get_or_create_collection(
            name=f"items_text_project_{project_id}",
            dimension=768
        )
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)  # 限制并发数
        self.client = vecs_client

    def get_embedding_ids(self, collection_name: str) -> List[str]:
        """
        获取指定集合中的所有ID

        Args:
            collection_name: 集合名称

        Returns:
            List[str]: ID列表
        """
        try:
            with self.client.Session() as session:
                # 使用SQL直接查询所有ID，避免collection.query的1000条限制
                # 使用双引号包围表名以处理特殊字符（如连字符）
                query = f'SELECT id FROM vecs."{collection_name}"'
                logger.info(f"执行SQL查询: {query}")
                result = session.execute(text(query))
                ids = [str(row[0]) for row in result.fetchall()]
                logger.info(f"集合({collection_name})中共有 {len(ids)} 个 ID")
                return ids

        except Exception as e:
            logger.error(f"获取集合({collection_name}) ID 失败: {e}")
            logger.error(f"执行的SQL: SELECT id FROM vecs.\"{collection_name}\"")

            # 尝试列出vecs schema下的所有表，帮助调试
            try:
                with self.client.Session() as debug_session:
                    tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'vecs'"
                    tables_result = debug_session.execute(text(tables_query))
                    available_tables = [row[0]
                                        for row in tables_result.fetchall()]
                    logger.error(f"vecs schema中可用的表: {available_tables}")
            except Exception as debug_e:
                logger.error(f"无法列出vecs表: {debug_e}")

            return []

    async def process_items_embeddings(
        self,
        item_ids: Optional[List[str]] = None,
        max_workers: int = 4
    ) -> dict:
        """
        批量处理items的embeddings

        Args:
            item_ids: 要处理的item ID列表，如果为None则处理所有项目的items
            max_workers: 最大并发worker数

        Returns:
            dict: 处理结果统计
        """
        logger.info(
            f"Starting embeddings processing for project {self.project_id}")

        # 构建查询条件
        query = Item.filter(project_id=self.project_id)
        if item_ids:
            query = query.filter(id__in=item_ids)

        items = await query.prefetch_related('project', 'uploaded_by').all()

        if not items:
            logger.warning(f"No items found for project {self.project_id}")
            return {"processed": 0, "failed": 0, "skipped": 0}

        logger.info(f"Found {len(items)} items to process")

        # 并发处理items
        tasks = []
        for item in items:
            task = asyncio.create_task(self._process_single_item_wrapper(item))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 统计结果
        successful_results = []
        failed_count = 0
        skipped_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error processing item {items[i].id}: {result}", exc_info=result)
                failed_count += 1
            elif result is None:
                skipped_count += 1
            elif isinstance(result, dict):
                successful_results.append(result)

        # 批量插入embeddings
        await self._batch_upsert_embeddings(successful_results)

        processed_count = len(successful_results)
        logger.info(
            f"Completed processing for project {self.project_id}. "
            f"Processed: {processed_count}, Failed: {failed_count}, Skipped: {skipped_count}"
        )

        return {
            "processed": processed_count,
            "failed": failed_count,
            "skipped": skipped_count
        }

    async def _process_single_item_wrapper(self, item: Item):
        """带信号量的单个item处理包装器"""
        try:
            async with self.semaphore:
                return await self._process_single_item(item)
        except Exception as e:
            logger.error(
                f"Exception in wrapper for item {item.id}: {e}", exc_info=e)
            raise

    async def _process_single_item(self, item: Item) -> Optional[dict]:
        """
        处理单个item的embeddings

        Args:
            item: Item对象

        Returns:
            dict: 包含embeddings数据，如果跳过则返回None
        """
        try:
            # 检查是否已存在embeddings
            existing_image_records = self.image_collection.query(
                data=None,
                filters={"item_id": str(item.id)},
                limit=1
            )

            if existing_image_records:
                logger.info(f"Item {item.id} already has embeddings, skipping")
                return None

            # 获取图片embedding
            image_embedding = None
            if item.s3_path:
                try:
                    # 从S3获取图片数据
                    image_data = await download_file_content_from_s3(item.s3_path)
                    if image_data:
                        # 将bytes转换为file-like对象，并添加name属性
                        image_file = io.BytesIO(image_data)
                        image_file.name = item.filename or f"item_{item.id}.jpg"
                        image_embedding = await get_multimodal_embedding_image(image_file)
                        logger.debug(
                            f"Generated image embedding for item {item.id}")
                except Exception as e:
                    logger.error(
                        f"Error getting image embedding for item {item.id}: {e}")

            # 获取文本embedding（组合标签和描述）
            text_embedding = None
            text_content = self._build_text_content(item)
            if text_content:
                try:
                    text_embedding = await get_text_embedding(text_content)
                    logger.debug(
                        f"Generated text embedding for item {item.id}")
                except Exception as e:
                    logger.error(
                        f"Error getting text embedding for item {item.id}: {e}")

            if image_embedding is None and text_embedding is None:
                logger.warning(f"No embeddings generated for item {item.id}")
                return None

            return {
                "item_id": str(item.id),
                "image_embedding": image_embedding,
                "text_embedding": text_embedding,
                "metadata": {
                    "item_id": str(item.id),
                    "filename": item.filename,
                    "content_type": item.content_type,
                    "tags": item.tags or [],
                    "s3_path": item.s3_path,
                    "description": item.description or "",
                    "project_id": str(item.project.id) if item.project else None,
                    "uploaded_by": str(item.uploaded_by.id),
                    "created_at": item.created_at.isoformat() if item.created_at else None,
                }
            }

        except Exception as e:
            logger.error(f"Error processing item {item.id}: {e}", exc_info=e)
            raise

    def _build_text_content(self, item: Item) -> str:
        """构建用于embedding的文本内容"""
        text_parts = []

        # 添加文件名（去掉扩展名）
        if item.filename:
            name_without_ext = item.filename.rsplit('.', 1)[0]
            text_parts.append(f"文件名: {name_without_ext}")

        # 添加标签
        if item.tags:
            tags_text = ", ".join(item.tags)
            text_parts.append(f"标签: {tags_text}")

        # 添加描述
        if item.description and item.description.strip():
            text_parts.append(f"描述: {item.description}")

        return " | ".join(text_parts)

    async def _batch_upsert_embeddings(self, results: List[dict]):
        """批量插入embeddings到vector database"""
        image_records = []
        text_records = []

        for data in results:
            item_id = data["item_id"]
            metadata = data["metadata"]

            if data["image_embedding"] is not None:
                image_records.append((
                    item_id,
                    data["image_embedding"],
                    metadata
                ))

            if data["text_embedding"] is not None:
                text_records.append((
                    item_id,
                    data["text_embedding"],
                    metadata
                ))

        # 批量插入图片embeddings
        if image_records:
            try:
                logger.info(
                    f"Batch upserting {len(image_records)} image embeddings...")
                self.image_collection.upsert(records=image_records)
                logger.info("Image embeddings batch upsert complete.")
            except Exception as e:
                logger.error(
                    f"Error during batch upsert for image embeddings: {e}", exc_info=e)

        # 批量插入文本embeddings
        if text_records:
            try:
                logger.info(
                    f"Batch upserting {len(text_records)} text embeddings...")
                self.text_collection.upsert(records=text_records)
                logger.info("Text embeddings batch upsert complete.")
            except Exception as e:
                logger.error(
                    f"Error during batch upsert for text embeddings: {e}", exc_info=e)

        # 创建索引
        try:
            logger.info("Creating vector indexes if they don't exist...")
            self.image_collection.create_index()
            self.text_collection.create_index()
            logger.info("Vector index creation check completed.")
        except Exception as e:
            logger.error(f"Error during vector index creation: {e}")

    async def search_similar_images(
        self,
        query_embedding: List[float],
        limit: int = 10,
        include_metadata: bool = True,
        filters: Optional[dict] = None
    ) -> List[dict]:
        """
        搜索相似图片

        Args:
            query_embedding: 查询向量
            limit: 返回结果数量限制
            include_metadata: 是否包含metadata

        Returns:
            List[dict]: 搜索结果
        """
        try:
            results = self.image_collection.query(
                data=query_embedding,
                limit=limit,
                filters=filters,
                include_metadata=True,
                include_value=True,
                ef_search=limit,
            )

            # 确保结果可序列化
            serializable_results = []
            for result in results:
                serializable_result = self._make_result_serializable(result)
                serializable_results.append(serializable_result)

            return serializable_results
        except Exception as e:
            logger.error(f"Error searching similar images: {e}")
            return []

    async def search_similar_text(
        self,
        query_text: str,
        limit: int = 10,
        include_metadata: bool = True,
        filters: Optional[dict] = None
    ) -> List[dict]:
        """
        根据文本搜索相似items

        Args:
            query_text: 查询文本
            limit: 返回结果数量限制
            include_metadata: 是否包含metadata
            filters: 搜索过滤条件

        Returns:
            List[dict]: 搜索结果
        """
        try:
            # 生成查询文本的embedding
            query_embedding = await get_text_embedding(query_text)
            if not query_embedding:
                return []

            results = self.text_collection.query(
                data=query_embedding,
                limit=limit,
                filters=filters,
                include_metadata=True,
                include_value=True,
                ef_search=limit,
            )

            # 确保结果可序列化
            serializable_results = []
            for result in results:
                serializable_result = self._make_result_serializable(result)
                serializable_results.append(serializable_result)

            return serializable_results
        except Exception as e:
            logger.error(f"Error searching similar text: {e}")
            return []

    def _make_result_serializable(self, result) -> dict:
        """
        将搜索结果转换为可序列化的字典格式

        Args:
            result: 向量搜索的原始结果

        Returns:
            dict: 可序列化的结果字典
        """
        try:
            # 处理不同类型的结果对象
            if hasattr(result, '_asdict'):
                # 如果是namedtuple，转换为字典
                result_dict = result._asdict()
            elif hasattr(result, '__dict__'):
                # 如果有__dict__属性，使用它
                result_dict = result.__dict__.copy()
            elif isinstance(result, dict):
                # 如果已经是字典，直接使用
                result_dict = result.copy()
            else:
                # 尝试将对象转换为字典
                result_dict = dict(result) if hasattr(
                    result, '__iter__') else {"data": str(result)}

            # 递归处理嵌套的不可序列化对象
            serializable_dict = {}
            for key, value in result_dict.items():
                serializable_dict[key] = self._serialize_value(value)

            return serializable_dict

        except Exception as e:
            logger.warning(
                f"Error serializing result, falling back to string representation: {e}")
            return {"error": "serialization_failed", "data": str(result)}

    def _serialize_value(self, value):
        """
        序列化单个值

        Args:
            value: 要序列化的值

        Returns:
            可序列化的值
        """
        try:
            # 基础类型直接返回
            if value is None or isinstance(value, (int, float, str, bool)):
                return value

            # 列表和元组
            elif isinstance(value, (list, tuple)):
                return [self._serialize_value(item) for item in value]

            # 字典
            elif isinstance(value, dict):
                return {k: self._serialize_value(v) for k, v in value.items()}

            # SQLAlchemy Row对象
            elif hasattr(value, '_asdict'):
                return value._asdict()

            # 其他对象尝试转换为字典
            elif hasattr(value, '__dict__'):
                return {k: self._serialize_value(v) for k, v in value.__dict__.items() if not k.startswith('_')}

            # 最后转换为字符串
            else:
                return str(value)

        except Exception:
            # 如果所有方法都失败，返回字符串表示
            return str(value)

    async def delete_item_embeddings(self, item_id: str):
        """删除指定item的embeddings"""
        try:
            # 从图片collection中删除
            self.image_collection.delete(filters={"item_id": item_id})
            # 从文本collection中删除
            self.text_collection.delete(filters={"item_id": item_id})
            logger.info(f"Deleted embeddings for item {item_id}")
        except Exception as e:
            logger.error(f"Error deleting embeddings for item {item_id}: {e}")

    def compare_arrays(self, A: List[str], B: List[str]):
        set_A = set(A)
        set_B = set(B)

        # 判断A是否等于B
        is_equal = set_A == set_B

        # 找出A中缺少的B中的元素
        missing_elements = set_B - set_A

        return is_equal, list(missing_elements)

    async def check_missing_embeddings(self, item_ids: Optional[List[str]] = None) -> dict:
        """
        检测项目中哪些items缺少embeddings

        Args:
            item_ids: 要检查的item ID列表，如果为None则检查所有项目的items

        Returns:
            dict: 包含缺少embeddings的items信息
        """
        logger.info(
            f"Checking missing embeddings for project {self.project_id}")

        # 构建查询条件
        query = Item.filter(project_id=self.project_id)
        if item_ids:
            query = query.filter(id__in=item_ids)

        items = await query.prefetch_related('project', 'uploaded_by').all()

        # if not items:
        #     logger.warning(f"No items found for project {self.project_id}")
        #     return {
        #         "total_items": 0,
        #         "items_missing_embeddings": 0,
        #         "missing_item_ids": []
        #     }

        logger.info(f"Found {len(items_ids)} items to check")

        # 检查每个item是否有embeddings
        # items_missing_embeddings: List[dict] = []

        try:

            # 获取所有embedding记录的ID (只需要ID，不需要完整记录)
            image_embedding_ids = self.get_embedding_ids(
                self.image_collection.name)
            text_embedding_ids = self.get_embedding_ids(
                self.text_collection.name)

            logger.info(
                f"Found {len(image_embedding_ids)} image embedding records")
            logger.info(
                f"Found {len(text_embedding_ids)} text embedding records")

            # 获取所有items的ID列表
            # item_ids = [str(item.id) for item in items]
            logger.info(f"Checking {len(item_ids)} items")

            # 比较哪些items缺少embeddings
            # compare_arrays(A, B) 返回 B中有但A中没有的元素
            # 所以我们传入 (embedding_ids, item_ids) 来找出缺少embeddings的items
            is_equal, image_missing_elements = self.compare_arrays(
                image_embedding_ids, item_ids)
            logger.info(
                f"Image embeddings - Is equal: {is_equal}, Missing: {len(image_missing_elements)}")

            is_equal, text_missing_elements = self.compare_arrays(
                text_embedding_ids, item_ids)
            logger.info(
                f"Text embeddings - Is equal: {is_equal}, Missing: {len(text_missing_elements)}")

            # 合并缺少embeddings的item IDs (如果image或text任一缺少就算缺少)
            missing_item_ids = set(image_missing_elements) | set(
                text_missing_elements)
            logger.info(
                f"Total items missing embeddings: {len(missing_item_ids)}")

            # 只返回统计信息和ID列表，避免详细信息造成的性能开销
            result = {
                "total_items": len(items),
                "items_missing_embeddings": len(missing_item_ids),
                "missing_item_ids": list(missing_item_ids),  # 只返回ID列表
            }

            logger.info(
                f"Embedding check completed for project {self.project_id}. "
                f"Total: {result['total_items']}, "
                f"Missing embeddings: {result['items_missing_embeddings']}"
            )

            return result
        except Exception as e:
            logger.error(f"Error getting embedding records: {e}")
            return {
                "total_items": 0,
                "items_missing_embeddings": 0,
                "missing_item_ids": []
            }


async def process_project_items_embeddings(project_id: str, item_ids: Optional[List[str]] = None) -> dict:
    """
    处理项目的items embeddings

    Args:
        project_id: 项目ID
        item_ids: 要处理的item ID列表，如果为None则处理所有

    Returns:
        dict: 处理结果
    """
    service = ItemsVectorService(project_id)
    return await service.process_items_embeddings(item_ids)


async def search_project_items_by_text(project_id: str, query_text: str, limit: int = 10) -> List[dict]:
    """
    在项目中根据文本搜索items

    Args:
        project_id: 项目ID
        query_text: 查询文本
        limit: 结果数量限制

    Returns:
        List[dict]: 搜索结果
    """
    service = ItemsVectorService(project_id)
    return await service.search_similar_text(query_text, limit)


async def search_project_items_by_image(project_id: str, image_url: str, limit: int = 10, filters: Optional[dict] = None, crop: Optional[dict] = None) -> List[dict]:
    """
    在项目中根据图片embedding搜索相似items

    Args:
        project_id: 项目ID
        image_url: 图片的HTTP URL (presigned URL) 或 S3路径
        limit: 结果数量限制
        filters: 搜索过滤条件
        crop: 裁剪区域信息 {'x': float, 'y': float, 'width': float, 'height': float}

    Returns:
        List[dict]: 搜索结果
    """
    service = ItemsVectorService(project_id)
    # 智能下载图片：支持HTTP URL和S3路径
    try:
        logger.info(f"Downloading image from: {image_url}")
        image_file = await download_image_from_url_or_s3_path(image_url)

        # 如果有裁剪参数，先裁剪图片
        if crop:
            image_file = await crop_image(image_file, crop)
            logger.info(f"Image cropped with parameters: {crop}")

        # 获取图片embedding
        image_embedding = await get_multimodal_embedding_image(image_file)
        return await service.search_similar_images(image_embedding, limit, True, filters)
    except Exception as e:
        logger.error(f"Error searching similar images: {e}")
        return []


async def crop_image(image_file, crop_info: dict):
    """
    裁剪图片

    Args:
        image_file: 图片文件对象
        crop_info: 裁剪信息 {'x': float, 'y': float, 'width': float, 'height': float}

    Returns:
        裁剪后的图片文件对象
    """
    try:
        # 确保文件指针在开始位置
        image_file.seek(0)

        # 使用PIL打开图片
        with Image.open(image_file) as img:
            # 获取原图尺寸
            img_width, img_height = img.size

            # 计算裁剪区域的像素坐标
            left = int(crop_info['x'] * img_width)
            top = int(crop_info['y'] * img_height)
            right = int((crop_info['x'] + crop_info['width']) * img_width)
            bottom = int((crop_info['y'] + crop_info['height']) * img_height)

            # 确保坐标在有效范围内
            left = max(0, min(left, img_width))
            top = max(0, min(top, img_height))
            right = max(left, min(right, img_width))
            bottom = max(top, min(bottom, img_height))

            # 裁剪图片
            cropped_img = img.crop((left, top, right, bottom))

            # 保存到内存
            output = io.BytesIO()
            # 保持原图格式，如果是RGBA则转换为RGB
            if cropped_img.mode == 'RGBA':
                cropped_img = cropped_img.convert('RGB')
            cropped_img.save(output, format='JPEG', quality=95)
            output.seek(0)

            # 添加name属性以保持兼容性
            if hasattr(image_file, 'name'):
                output.name = f"cropped_{image_file.name}"
            else:
                output.name = "cropped_image.jpg"

            return output

    except Exception as e:
        logger.error(f"Error cropping image: {e}")
        raise ValueError(f"Failed to crop image: {e}")


def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            # Remove everything before "```json"
            json_output = "\n".join(lines[i+1:])
            # Remove everything after the closing "```"
            json_output = json_output.split("```")[0]
            break  # Exit the loop once "```json" is found

    # 解析JSON
    try:
        parsed_data = json.loads(json_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    # 如果不是列表，直接返回
    if not isinstance(parsed_data, list):
        return parsed_data

    # 处理每个字典项，统一键名
    for item in parsed_data:
        if not isinstance(item, dict):
            continue

        # 一次性处理所有键名统一
        for key, value in item.items():
            # 处理bounding box：四位数字列表 -> box_2d
            if (key != "box_2d" and
                isinstance(value, list) and
                len(value) == 4 and
                    all(isinstance(x, (int, float)) for x in value)):
                item["box_2d"] = item.pop(key)

            # 处理label：非空字符串且不是box_2d -> label
            elif (key != "label" and key != "box_2d" and
                  isinstance(value, str) and
                  value.strip()):
                item["label"] = item.pop(key)

    return parsed_data


async def list_up_items_in_image(project_id: str, image_url: str, limit: int = 10) -> List[dict]:
    """
    在图片中搜索相似items
    """
    try:
        logger.info(f"Downloading image from: {image_url}")
        image_file = await download_image_from_url_or_s3_path(image_url)

        task_gemini_client = genai.Client(api_key=os.getenv(
            'GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None
        if not task_gemini_client:
            print("Error: GEMINI_API_KEY not set. Cannot generate NG review findings.")
            return []
        bounding_box_system_instructions = """
            Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
            If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
            """
        prompt = "Detect the 2d bounding boxes of the possible items in the image. Not characters"  # @param {type:"string"}

        prompt_parts = [
            prompt,
            "Image:",
            genai.types.Part.from_bytes(
                data=image_file.read(), mime_type='image/jpeg'),
        ]
        safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]

        generation_config = types.GenerateContentConfig(
            system_instruction=bounding_box_system_instructions,
            temperature=0.5,
            safety_settings=safety_settings,
        )

        # 重试机制：最多重试3次
        max_retries = 3
        bounding_boxes = []

        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (总共4次尝试)
            try:
                response = task_gemini_client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt_parts,
                    config=generation_config,
                )

                if response.text:
                    try:
                        bounding_boxes = parse_json(response.text)

                        logger.info(
                            f"Successfully parsed JSON on attempt {attempt + 1}")
                        break
                    except Exception as e:
                        logger.error(
                            f"Attempt {attempt + 1}: Error parsing JSON: {e}")
                        logger.error(f"Response text: {response.text}")
                        if attempt == max_retries:
                            # 最后一次尝试失败，返回空列表
                            logger.error(
                                f"Failed to parse JSON after {max_retries + 1} attempts")
                            bounding_boxes = []
                        else:
                            # 还有重试机会，继续下一次尝试
                            logger.info(
                                f"Retrying... ({attempt + 1}/{max_retries + 1})")
                            continue
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}: No response text received")
                    if attempt == max_retries:
                        bounding_boxes = []
                    else:
                        continue

            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1}: Error generating content: {e}")
                if attempt == max_retries:
                    bounding_boxes = []
                else:
                    continue

        return bounding_boxes

    except Exception as e:
        logger.error(f"Error searching similar images: {e}")
        return []
