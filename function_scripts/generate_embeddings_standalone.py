#!/usr/bin/env python3
"""
独立的Python脚本用于检测和生成项目中items的embeddings

使用方法:
    python generate_embeddings_standalone.py --project-id <PROJECT_ID> [--database-url <DATABASE_URL>]

环境变量:
    DATABASE_URL: 数据库连接URL
    GEMINI_API_KEY: Google Gemini API密钥
    TEXT_EMBEDDING_MODEL: 文本embedding模型名称（默认: text-multilingual-embedding-002）
    GOOGLE_CREDS: Google Cloud认证信息（JSON格式）
    VERTEX_AI_PROJECT: Vertex AI项目ID
    VERTEX_AI_LOCATION: Vertex AI位置（默认: us-central1）
"""

import argparse
import asyncio
from aimage_supervision.settings import logger
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from tortoise import Tortoise

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# 加载环境变量
load_dotenv()


class EmbeddingsGenerator:
    """独立的embeddings生成器"""

    def __init__(self, project_id: str, database_url: str):
        self.project_id = project_id
        self.database_url = database_url
        self.vecs_client = None
        self.items_service = None

    async def initialize(self):
        """初始化数据库连接和服务"""
        try:
            # 初始化Tortoise ORM
            await Tortoise.init(
                db_url=self.database_url,
                modules={'models': ['aimage_supervision.models']}
            )
            logger.info("数据库连接初始化成功")

            # 初始化vecs客户端
            from aimage_supervision.clients.vecs import create_vecs_client
            self.vecs_client = create_vecs_client(self.database_url)
            logger.info("Vecs客户端初始化成功")

            # 初始化ItemsVectorService
            from function_scripts.check_embeddings import ItemsVectorService
            self.items_service = ItemsVectorService(self.project_id)
            logger.info(f"ItemsVectorService初始化成功 (项目ID: {self.project_id})")

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise

    async def check_missing_embeddings(self, item_ids: Optional[List[str]] = None) -> dict:
        """检测缺少embeddings的items"""
        if not self.items_service:
            raise RuntimeError("ItemsVectorService未初始化，请先调用initialize()方法")

        try:
            logger.info("开始检测缺少embeddings的items...")
            result = await self.items_service.check_missing_embeddings(item_ids)

            logger.info("检测结果:")
            logger.info(f"  - 总计items: {result.get('total_items', 0)}")
            logger.info(
                f"  - 缺少embeddings的items: {result.get('items_missing_embeddings', 0)}")
            logger.info(
                f"  - 缺少图片embeddings: {result.get('items_missing_image_embeddings', 0)}")
            logger.info(
                f"  - 缺少文本embeddings: {result.get('items_missing_text_embeddings', 0)}")

            return result

        except Exception as e:
            logger.error(f"检测缺少embeddings时出错: {e}")
            raise

    async def generate_embeddings(self, item_ids: Optional[List[str]] = None) -> dict:
        """生成embeddings"""
        if not self.items_service:
            raise RuntimeError("ItemsVectorService未初始化，请先调用initialize()方法")

        try:
            logger.info("开始生成embeddings...")
            result = await self.items_service.process_items_embeddings(item_ids)

            logger.info("生成结果:")
            logger.info(f"  - 处理的items总数: {result.get('processed_items', 0)}")
            logger.info(
                f"  - 成功生成图片embeddings: {result.get('image_embeddings_created', 0)}")
            logger.info(
                f"  - 成功生成文本embeddings: {result.get('text_embeddings_created', 0)}")
            logger.info(f"  - 失败的items: {result.get('failed_items', 0)}")

            if result.get('errors'):
                logger.warning("生成过程中出现错误:")
                for error in result['errors']:
                    logger.warning(f"  - {error}")

            return result

        except Exception as e:
            logger.error(f"生成embeddings时出错: {e}")
            raise

    async def run_check_and_generate(self, item_ids: Optional[List[str]] = None):
        """检测并生成embeddings的完整流程"""
        try:
            # 首先检测缺少embeddings的items
            check_result = await self.check_missing_embeddings(item_ids)

            # 如果有缺少embeddings的items，则生成它们
            if check_result.get('items_missing_embeddings', 0) > 0:
                missing_item_ids = check_result.get('missing_item_ids', [])
                logger.info(
                    f"发现 {len(missing_item_ids)} 个items缺少embeddings，开始生成...")

                generate_result = await self.generate_embeddings(missing_item_ids)

                logger.info("✅ Embeddings检测和生成完成!")
                return {
                    'check_result': check_result,
                    'generate_result': generate_result
                }
            else:
                logger.info("✅ 所有items都已有embeddings，无需生成!")
                return {
                    'check_result': check_result,
                    'generate_result': None
                }

        except Exception as e:
            logger.error(f"执行检测和生成流程时出错: {e}")
            raise

    async def cleanup(self):
        """清理资源"""
        try:
            await Tortoise.close_connections()
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")


def validate_environment():
    """验证所需的环境变量"""
    required_vars = [
        'GEMINI_API_KEY',
        'GOOGLE_CREDS',
        'VERTEX_AI_PROJECT',
    ]

    # AWS S3相关环境变量（如果需要处理S3文件）
    s3_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_REGION',
        'AWS_BUCKET_NAME',
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    missing_s3_vars = []
    for var in s3_vars:
        if not os.getenv(var):
            missing_s3_vars.append(var)

    if missing_vars:
        logger.error(f"缺少必需的环境变量: {', '.join(missing_vars)}")
        logger.error("请设置以下环境变量:")
        logger.error("  - GEMINI_API_KEY: Google Gemini API密钥")
        logger.error("  - GOOGLE_CREDS: Google Cloud认证信息（JSON格式）")
        logger.error("  - VERTEX_AI_PROJECT: Vertex AI项目ID")
        logger.error("  - VERTEX_AI_LOCATION: Vertex AI位置（可选，默认: us-central1）")
        return False

    if missing_s3_vars:
        logger.warning(f"缺少AWS S3环境变量: {', '.join(missing_s3_vars)}")
        logger.warning("如果需要处理S3文件，请设置以下环境变量:")
        logger.warning("  - AWS_ACCESS_KEY_ID: AWS访问密钥ID")
        logger.warning("  - AWS_SECRET_ACCESS_KEY: AWS秘密访问密钥")
        logger.warning("  - AWS_REGION: AWS区域")
        logger.warning("  - AWS_BUCKET_NAME: S3存储桶名称")
        logger.warning("如果不处理S3文件，可以忽略此警告")

    return True


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='为项目生成items的embeddings')
    parser.add_argument('--project-id', required=True, help='项目ID')
    parser.add_argument(
        '--database-url', help='数据库连接URL（也可通过DATABASE_URL环境变量设置）')
    parser.add_argument('--item-ids', nargs='*', help='要处理的特定item ID列表（可选）')
    parser.add_argument('--check-only', action='store_true',
                        help='仅检测缺少embeddings的items，不生成')

    args = parser.parse_args()

    # 获取数据库URL
    database_url = args.database_url or os.getenv('DATABASE_URL')
    if not database_url:
        logger.error("必须提供数据库连接URL（通过--database-url参数或DATABASE_URL环境变量）")
        return 1

    # 验证环境变量
    if not validate_environment():
        return 1

    # 初始化生成器
    generator = EmbeddingsGenerator(args.project_id, database_url)

    try:
        # 初始化
        await generator.initialize()

        if args.check_only:
            # 仅检测模式
            logger.info("🔍 仅检测模式")
            await generator.check_missing_embeddings(args.item_ids)
        else:
            # 检测并生成模式
            logger.info("🚀 开始检测并生成embeddings...")
            await generator.run_check_and_generate(args.item_ids)

        return 0

    except Exception as e:
        logger.error(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        await generator.cleanup()


if __name__ == '__main__':
    # 运行主函数
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
