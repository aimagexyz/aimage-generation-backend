import asyncio
import os
from typing import Any, List

import dotenv
from google import genai
from pydantic import BaseModel, Field

from aimage_supervision.schemas import ReviewPointDefinitionVersionInDB
from aimage_supervision.settings import MAX_CONCURRENT


class YesNoResult(BaseModel):
    yes_no: str = Field(..., description="yes or no")
    word: str = Field(..., description="the word that describes the object")


async def _filter_single_rpd(
    rpd_version: ReviewPointDefinitionVersionInDB,
    prompt_parts: List,
    task_gemini_client: genai.Client,
    semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    """过滤单个RPD版本

    Args:
        rpd_version: 要过滤的RPD版本
        prompt_parts: 图片提示内容
        task_gemini_client: Gemini客户端
        semaphore: 并发控制信号量

    Returns:
        ReviewPointDefinitionVersionInDB | None: 如果应该保留则返回RPD版本，否则返回None
    """
    async with semaphore:
        try:
            SYSTEM_PROMPT = f"""我想监修这些图片，但是我首先需要判断图片中是否有相应的监修项目，比如，我想监修鞋底的图案，但是如果图片本身没有鞋底，那么就不需要监修。
                我要监修的内容是：{rpd_version.description_for_ai}
                请判断所给的图片是否包含{rpd_version.title}，
                如果是请回答yes，并用一个常见的英文单词概括性描述{rpd_version.title}，比如鞋底可以概括成shoes, 如果本身就是比较常见的单词比如眼睛，那就直接用这个单词的英文比如eyes。
                否则回答no,也不需要返回英文单词,返回空字符串，
                如果无法判断，请按照yes回答
                
                for example:
                {{
                    "yes_no": "yes",
                    "word": "shoes"
                }},
                {{
                    "yes_no": "yes",
                    "word": "eyes"
                }},
                {{
                    "yes_no": "no",
                    "word": ""
                }}"""

            # 使用 asyncio.to_thread 在线程池中执行同步调用
            result = await asyncio.to_thread(
                task_gemini_client.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt_parts,
                config=genai.types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=YesNoResult,
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                )
            )

            if hasattr(result, 'parsed') and result.parsed and hasattr(result.parsed, 'yes_no'):
                return {
                    "yes_no": result.parsed.yes_no,
                    "rpd_version": rpd_version,
                    "word": result.parsed.word
                }
            else:
                # 如果解析失败，保守地保留RPD
                print(f"无法解析RPD {rpd_version.title} 的结果")
                return {
                    "yes_no": "yes",
                    "rpd_version": rpd_version,
                    "word": rpd_version.title
                }

        except Exception as e:
            print(f"处理RPD {rpd_version.title} 时出错: {e}")
            # 发生错误时保守地保留RPD
            return {
                "yes_no": "yes",
                "rpd_version": rpd_version,
                "word": rpd_version.title
            }


async def filter_rpd_async(
    image_bytes: bytes,
    rpd_list: List[ReviewPointDefinitionVersionInDB],
    max_concurrent: int = MAX_CONCURRENT,
    cancellation_check_callback=None
) -> List[dict[str, Any]]:
    """异步过滤不存在的RPD内容

    Args:
        image_bytes (bytes): 图片字节数据
        rpd_list (List[ReviewPointDefinitionVersionInDB]): RPD版本列表
        max_concurrent (int): 最大并发数，默认为5
        cancellation_check_callback: 可选的取消检查回调函数

    Returns:
        List[dict[str, Any]]: 过滤后的RPD版本列表
    """
    task_gemini_client = genai.Client(api_key=os.getenv(
        'GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else None
    if not task_gemini_client:
        print("Error: GEMINI_API_KEY not set. Cannot do filter.")
        return [{
            "yes_no": "yes",
            "rpd_version": rpd_version,
            "word": rpd_version.title
        } for rpd_version in rpd_list]

    prompt_parts = [
        "Image:",
        genai.types.Part.from_bytes(
            data=image_bytes, mime_type='image/jpeg'),
    ]

    # 设置并发控制
    semaphore = asyncio.Semaphore(max_concurrent)

    print(f"开始并行过滤 {len(rpd_list)} 个RPD版本")

    # 检查取消信号
    if cancellation_check_callback:
        if await cancellation_check_callback():
            print("在 RPD 过滤开始前检测到中断信号")
            return []

    # 创建并行任务
    tasks = []
    for i, rpd_version in enumerate(rpd_list):
        task = asyncio.create_task(
            _filter_single_rpd(rpd_version, prompt_parts,
                               task_gemini_client, semaphore),
            name=f"rpd_filter_task_{i}"
        )
        tasks.append(task)

    # 等待所有任务完成，支持取消检查
    if cancellation_check_callback:
        # 使用可中断的等待机制
        done: List[asyncio.Task] = []
        pending = set(tasks)

        while pending:
            # 检查取消信号
            if await cancellation_check_callback():
                print("在 RPD 过滤过程中检测到中断信号，取消剩余任务")
                # 取消所有挂起的任务
                for task in pending:
                    task.cancel()
                # 等待被取消的任务完成
                try:
                    await asyncio.gather(*pending, return_exceptions=True)
                except Exception:
                    pass
                return []

            # 等待一些任务完成（超时 2 秒）
            try:
                finished, pending = await asyncio.wait(
                    pending,
                    timeout=2.0,
                    return_when=asyncio.FIRST_COMPLETED
                )
                done.extend(finished)
            except asyncio.TimeoutError:
                continue

        results: List[dict[str, Any]] = []
        for task in done:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                results.append({
                    "yes_no": "yes",
                    "rpd_version": rpd_list[i],
                    "word": rpd_list[i].title
                })
    else:
        # 原始的等待方式
        results_raw = await asyncio.gather(*tasks)
        results = list(results_raw)

    # 收集结果
    output: List[dict[str, Any]] = []
    for i, result in enumerate(results):
        if isinstance(result, dict):
            if result["yes_no"] == "yes":
                output.append(result)
            else:
                continue
        else:
            print(f"任务 {i} 执行异常: {result}")
            # 发生异常时保守地保留原始RPD
            if i < len(rpd_list):
                output.append({
                    "yes_no": "yes",
                    "rpd_version": rpd_list[i],
                    "word": rpd_list[i].title
                })
        # 如果result是None，说明该RPD被过滤掉了，不需要添加到输出中

    print(f"RPD过滤完成，从 {len(rpd_list)} 个减少到 {len(output)} 个")
    return output


def filter_rpd(image_bytes: bytes, rpd_list: List[ReviewPointDefinitionVersionInDB]) -> List[dict[str, Any]]:
    """同步包装函数，用于向后兼容

    Args:
        image_bytes (bytes): 图片字节数据
        rpd_list (List[ReviewPointDefinitionVersionInDB]): RPD版本列表

    Returns:
        List[ReviewPointDefinitionVersionInDB]: 过滤后的RPD版本列表
    """
    return asyncio.run(filter_rpd_async(image_bytes, rpd_list))
