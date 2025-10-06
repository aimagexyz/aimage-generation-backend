#!/usr/bin/env python3
"""
调试AI监修回退功能的脚本
"""

import asyncio
import sys
from uuid import UUID

from tortoise import Tortoise

from aimage_supervision.enums import AiReviewProcessingStatus
from aimage_supervision.models import AiReview
from aimage_supervision.services.ai_review_service import (
    _rollback_failed_review, get_latest_ai_review_for_subtask)
from aimage_supervision.settings import DATABASE_URL


async def debug_rollback(subtask_id_str: str):
    """调试回退功能"""
    try:
        # 初始化数据库连接
        await Tortoise.init(
            db_url=DATABASE_URL,
            modules={'models': ['aimage_supervision.models']}
        )

        subtask_id = UUID(subtask_id_str)
        print(f"🔍 调试子任务 {subtask_id} 的回退功能...")

        # 1. 查看所有版本
        print("\n📋 当前所有版本:")
        all_reviews = await AiReview.filter(subtask_id=subtask_id).order_by('-version')
        if not all_reviews:
            print("❌ 没有找到任何AI review版本")
            return

        for review in all_reviews:
            status_icon = "✅" if review.processing_status == AiReviewProcessingStatus.COMPLETED.value else "❌" if review.processing_status == AiReviewProcessingStatus.FAILED.value else "🔄"
            latest_icon = "🌟" if review.is_latest else "  "
            print(
                f"  {latest_icon}{status_icon} 版本 {review.version}: {review.processing_status} (is_latest={review.is_latest})")

        # 2. 找到最新版本
        latest_review_orm = await AiReview.filter(subtask_id=subtask_id, is_latest=True).first()
        if not latest_review_orm:
            print("❌ 没有找到标记为latest的版本")
            return

        print(
            f"\n🎯 当前latest版本: {latest_review_orm.version} (状态: {latest_review_orm.processing_status})")

        # 3. 如果当前版本是成功的，先将其标记为失败来测试回退
        if latest_review_orm.processing_status == AiReviewProcessingStatus.COMPLETED.value:
            print("⚠️  当前版本是成功的，将其标记为失败来测试回退...")
            latest_review_orm.processing_status = AiReviewProcessingStatus.FAILED
            latest_review_orm.error_message = "Test failure for rollback debugging"
            await latest_review_orm.save()
            print("✅ 已标记为失败")

        # 4. 执行回退
        print(f"\n🔄 执行回退...")
        await _rollback_failed_review(latest_review_orm)

        # 5. 检查回退结果
        print(f"\n📊 回退后的状态:")
        all_reviews_after = await AiReview.filter(subtask_id=subtask_id).order_by('-version')
        for review in all_reviews_after:
            status_icon = "✅" if review.processing_status == AiReviewProcessingStatus.COMPLETED.value else "❌" if review.processing_status == AiReviewProcessingStatus.FAILED.value else "🔄"
            latest_icon = "🌟" if review.is_latest else "  "
            print(
                f"  {latest_icon}{status_icon} 版本 {review.version}: {review.processing_status} (is_latest={review.is_latest})")

        # 6. 测试前端API
        print(f"\n🌐 测试前端API结果:")
        latest_from_api = await get_latest_ai_review_for_subtask(subtask_id)
        if latest_from_api:
            print(
                f"  ✅ API返回版本 {latest_from_api.version} (状态: {latest_from_api.processing_status})")
        else:
            print("  ℹ️  API返回None (无监修状态)")

        print(f"\n✅ 调试完成!")

    except Exception as e:
        print(f"❌ 调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await Tortoise.close_connections()


async def main():
    if len(sys.argv) != 2:
        print("使用方法: python debug_rollback.py <subtask_id>")
        print("例如: python debug_rollback.py 123e4567-e89b-12d3-a456-426614174000")
        sys.exit(1)

    subtask_id_str = sys.argv[1]
    await debug_rollback(subtask_id_str)


if __name__ == "__main__":
    asyncio.run(main())
