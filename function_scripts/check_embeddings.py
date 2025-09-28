import asyncio
import io
import sys
from pathlib import Path
from typing import List, Optional

from sqlalchemy import text

from aimage_supervision.clients.aws_s3 import download_file_content_from_s3
from aimage_supervision.clients.multimodal_embedding import (
    get_multimodal_embedding_image, get_text_embedding)
from aimage_supervision.models import Item
from aimage_supervision.settings import logger, vecs_client, MAX_CONCURRENT

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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

        # 只获取item的ID，不加载完整对象
        item_ids_result = await query.values_list('id', flat=True)

        if not item_ids_result:
            logger.warning(f"No items found for project {self.project_id}")
            return {"processed": 0, "failed": 0, "skipped": 0}

        logger.info(f"Found {len(item_ids_result)} items to process")

        # 将UUID转换为字符串
        item_ids_to_process = [str(item_id) for item_id in item_ids_result]

        # 并发处理items
        tasks = []
        for item_id in item_ids_to_process:
            task = asyncio.create_task(
                self._process_single_item_wrapper(item_id))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 统计结果
        successful_results = []
        failed_count = 0
        skipped_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error processing item {item_ids_to_process[i]}: {result}", exc_info=result)
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

    async def _process_single_item_wrapper(self, item_id: str):
        """带信号量的单个item处理包装器"""
        try:
            async with self.semaphore:
                return await self._process_single_item(item_id)
        except Exception as e:
            logger.error(
                f"Exception in wrapper for item {item_id}: {e}", exc_info=e)
            raise

    async def _process_single_item(self, item_id: str) -> Optional[dict]:
        """
        处理单个item的embeddings

        Args:
            item_id: Item的ID字符串

        Returns:
            dict: 包含embeddings数据，如果跳过则返回None
        """
        try:

            # 只获取处理embeddings所需的字段
            item_values = await Item.filter(id=item_id).values(
                'id', 'filename', 'content_type', 'tags', 's3_path',
                'description', 'project_id', 'uploaded_by_id', 'created_at'
            )

            if not item_values:
                logger.warning(f"Item {item_id} not found, skipping")
                return None

            item_data = item_values[0]  # 获取第一个（也是唯一的）结果

            # 获取图片embedding
            image_embedding = None
            if item_data['s3_path']:
                try:
                    # 从S3获取图片数据
                    image_data = await download_file_content_from_s3(item_data['s3_path'])
                    if image_data:
                        # 将bytes转换为file-like对象，并添加name属性
                        image_file = io.BytesIO(image_data)
                        image_file.name = item_data['filename'] or f"item_{item_data['id']}.jpg"
                        image_embedding = await get_multimodal_embedding_image(image_file)
                        logger.debug(
                            f"Generated image embedding for item {item_data['id']}")
                except Exception as e:
                    logger.error(
                        f"Error getting image embedding for item {item_data['id']}: {e}")

            # 获取文本embedding（组合标签和描述）
            text_embedding = None
            text_content = self._build_text_content_from_data(item_data)
            if text_content:
                try:
                    text_embedding = await get_text_embedding(text_content)
                    logger.debug(
                        f"Generated text embedding for item {item_data['id']}")
                except Exception as e:
                    logger.error(
                        f"Error getting text embedding for item {item_data['id']}: {e}")

            if image_embedding is None and text_embedding is None:
                logger.warning(
                    f"No embeddings generated for item {item_data['id']}")
                return None

            return {
                "item_id": str(item_data['id']),
                "image_embedding": image_embedding,
                "text_embedding": text_embedding,
                "metadata": {
                    "item_id": str(item_data['id']),
                    "filename": item_data['filename'],
                    "content_type": item_data['content_type'],
                    "tags": item_data['tags'] or [],
                    "s3_path": item_data['s3_path'],
                    "description": item_data['description'] or "",
                    "project_id": str(item_data['project_id']) if item_data['project_id'] else None,
                    "uploaded_by": str(item_data['uploaded_by_id']),
                    "created_at": item_data['created_at'].isoformat() if item_data['created_at'] else None,
                }
            }

        except Exception as e:
            logger.error(f"Error processing item {item_id}: {e}", exc_info=e)
            raise

    def _build_text_content(self, item: Item) -> str:
        """构建用于embedding的文本内容（兼容旧代码）"""
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

    def _build_text_content_from_data(self, item_data: dict) -> str:
        """从字典数据构建用于embedding的文本内容"""
        text_parts = []

        # 添加文件名（去掉扩展名）
        if item_data.get('filename'):
            name_without_ext = item_data['filename'].rsplit('.', 1)[0]
            text_parts.append(f"文件名: {name_without_ext}")

        # 添加标签
        if item_data.get('tags'):
            tags_text = ", ".join(item_data['tags'])
            text_parts.append(f"标签: {tags_text}")

        # 添加描述
        if item_data.get('description') and item_data['description'].strip():
            text_parts.append(f"描述: {item_data['description']}")

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

        # 只获取item的ID，不加载完整对象
        item_ids_result = await query.values_list('id', flat=True)

        if not item_ids_result:
            logger.warning(f"No items found for project {self.project_id}")
            return {
                "total_items": 0,
                "items_missing_embeddings": 0,
                "missing_item_ids": []
            }

        logger.info(f"Found {len(item_ids_result)} items to check")

        # 将UUID转换为字符串
        item_ids_list = [str(item_id) for item_id in item_ids_result]

        # 检查每个item是否有embeddings
        items_missing_embeddings: List[dict] = []

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

            logger.info(f"Checking {len(item_ids_list)} items")

            # 比较哪些items缺少embeddings
            # compare_arrays(A, B) 返回 B中有但A中没有的元素
            # 所以我们传入 (embedding_ids, item_ids_list) 来找出缺少embeddings的items
            is_equal, image_missing_elements = self.compare_arrays(
                image_embedding_ids, item_ids_list)
            logger.info(
                f"Image embeddings - Is equal: {is_equal}, Missing: {len(image_missing_elements)}")

            is_equal, text_missing_elements = self.compare_arrays(
                text_embedding_ids, item_ids_list)
            logger.info(
                f"Text embeddings - Is equal: {is_equal}, Missing: {len(text_missing_elements)}")

            # 合并缺少embeddings的item IDs (如果image或text任一缺少就算缺少)
            missing_item_ids = set(image_missing_elements) | set(
                text_missing_elements)
            logger.info(
                f"Total items missing embeddings: {len(missing_item_ids)}")

            # 只返回统计信息和ID列表，避免详细信息造成的性能开销
            result = {
                "total_items": len(item_ids_list),
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


# 便捷函数
async def check_project_items_embeddings(project_id: str, item_ids: Optional[List[str]] = None) -> dict:
    """
    检测项目中哪些items缺少embeddings

    Args:
        project_id: 项目ID
        item_ids: 要检查的item ID列表，如果为None则检查所有

    Returns:
        dict: 检查结果
    """
    service = ItemsVectorService(project_id)
    return await service.check_missing_embeddings(item_ids)


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
