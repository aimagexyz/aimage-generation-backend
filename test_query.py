#!/usr/bin/env python3
"""
测试查询逻辑的脚本
"""

import asyncio
import sys
from uuid import UUID

from tortoise import Tortoise

from aimage_supervision.enums import AiReviewProcessingStatus
from aimage_supervision.models import AiReview, Subtask
from aimage_supervision.settings import DATABASE_URL


async def test_query(subtask_id_str: str):
    """测试查询逻辑"""
    try:
        # 初始化数据库连接
        await Tortoise.init(
            db_url=DATABASE_URL,
            modules={'models': ['aimage_supervision.models']}
        )

        subtask_id = UUID(subtask_id_str)
        print(f"🔍 测试子任务 {subtask_id} 的查询逻辑...")

        # 1. 获取subtask对象
        subtask = await Subtask.get(id=subtask_id)
        print(f"✅ 找到subtask: {subtask.name}")

        # 2. 查询所有版本
        all_reviews = await AiReview.filter(subtask=subtask).order_by('-version')
        print(f"📋 找到 {len(all_reviews)} 个版本:")
        for review in all_reviews:
            print(
                f"  版本 {review.version}: {review.processing_status} (is_latest={review.is_latest})")

        if not all_reviews:
            print("❌ 没有找到任何版本")
            return

        # 3. 模拟查找成功版本的逻辑
        failed_version = all_reviews[0].version  # 假设最新版本失败了
        print(f"\n🎯 假设版本 {failed_version} 失败，查找之前的成功版本...")

        # 查找所有之前的版本
        previous_reviews = await AiReview.filter(
            subtask=subtask,
            version__lt=failed_version
        ).order_by('-version')

        print(f"📋 找到 {len(previous_reviews)} 个之前的版本:")
        for review in previous_reviews:
            print(f"  版本 {review.version}: {review.processing_status}")

        # 查找成功的版本
        successful_reviews = await AiReview.filter(
            subtask=subtask,
            version__lt=failed_version,
            processing_status=AiReviewProcessingStatus.COMPLETED.value
        ).order_by('-version')

        print(f"✅ 找到 {len(successful_reviews)} 个成功的版本:")
        for review in successful_reviews:
            print(f"  版本 {review.version}: {review.processing_status}")

        # 获取最新的成功版本
        latest_successful = await AiReview.filter(
            subtask=subtask,
            version__lt=failed_version,
            processing_status=AiReviewProcessingStatus.COMPLETED.value
        ).order_by('-version').first()

        if latest_successful:
            print(f"🎯 最新成功版本: {latest_successful.version}")
        else:
            print("⚠️  没有找到成功的版本")

        print(f"\n✅ 查询测试完成!")

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await Tortoise.close_connections()


async def main():
    if len(sys.argv) != 2:
        print("使用方法: python test_query.py <subtask_id>")
        print("例如: python test_query.py 123e4567-e89b-12d3-a456-426614174000")
        sys.exit(1)

    subtask_id_str = sys.argv[1]
    await test_query(subtask_id_str)


if __name__ == "__main__":
    asyncio.run(main())
