#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF图片提取Pipeline
功能：从PDF中提取所有图片，并尝试给出有意义的文件名
"""

import argparse
import hashlib
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import fitz  # type: ignore
from PIL import Image, ImageOps


class PDFImageExtractor:
    """PDF图片提取器"""

    def __init__(self, output_dir: str = "extracted_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.extraction_log: List[Dict[str, Any]] = []
        self.duplicate_hashes: set[str] = set()

    def clean_filename(self, text: str, max_length: int = 50) -> str:
        """清理文件名中的非法字符，针对角色参考图优化"""
        if not text or not text.strip():
            return ""

        # 移除多余空白
        text = re.sub(r'\s+', ' ', text.strip())

        # 保留更多有用字符：中英文、数字、下划线、连字符、点号（用于扩展名）
        cleaned = re.sub(r'[^\w\u4e00-\u9fff\-\.\s]', '_', text)

        # 特殊处理：保留常见的文件扩展名分隔符
        cleaned = re.sub(r'_+', '_', cleaned)  # 合并多个下划线
        cleaned = cleaned.strip('_. ')  # 移除首尾的下划线、点号和空格

        # 如果包含文件扩展名，确保格式正确
        if '.' in cleaned:
            parts = cleaned.split('.')
            if len(parts) >= 2:
                name_part = parts[0]
                ext_part = parts[-1].lower()
                # 如果是常见扩展名，保留；否则替换为下划线
                if ext_part in ['psd', 'ai', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp']:
                    cleaned = f"{name_part}.{ext_part}"
                else:
                    cleaned = '_'.join(parts)

        # 限制长度
        if len(cleaned) > max_length:
            if '.' in cleaned:
                # 如果有扩展名，保留扩展名
                name, ext = cleaned.rsplit('.', 1)
                max_name_length = max_length - len(ext) - 1
                cleaned = f"{name[:max_name_length].rstrip('_')}.{ext}"
            else:
                cleaned = cleaned[:max_length].rstrip('_')

        return cleaned if cleaned else ""

    def get_image_hash(self, image_data: bytes) -> str:
        """获取图片的MD5哈希值用于去重"""
        return hashlib.md5(image_data).hexdigest()

    def find_nearby_text(self, page, img_rect, search_radius: int = 100) -> List[str]:
        """查找图片周围的文本"""
        try:
            # 扩展搜索区域
            search_rect = fitz.Rect(
                img_rect.x0 - search_radius,
                img_rect.y0 - search_radius,
                img_rect.x1 + search_radius,
                img_rect.y1 + search_radius
            )

            # 获取文本块
            text_blocks = page.get_text("dict")
            nearby_texts = []

            for block in text_blocks.get("blocks", []):
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    line_rect = fitz.Rect(line["bbox"])

                    # 检查文本块是否在搜索区域内
                    if search_rect.intersects(line_rect):
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span.get("text", "")

                        line_text = line_text.strip()
                        if line_text and len(line_text) > 2:
                            # 计算与图片的距离
                            distance = min(
                                abs(line_rect.y1 - img_rect.y0),  # 上方距离
                                abs(line_rect.y0 - img_rect.y1),  # 下方距离
                                abs(line_rect.x1 - img_rect.x0),  # 左侧距离
                                abs(line_rect.x0 - img_rect.x1)   # 右侧距离
                            )
                            nearby_texts.append((distance, line_text))

            # 按距离排序，返回最近的文本
            nearby_texts.sort(key=lambda x: x[0])
            return [text for _, text in nearby_texts[:3]]

        except Exception as e:
            print(f"获取周围文本时出错: {e}")
            return []

    def get_image_name_candidates(self, doc, xref: int, page_num: int,
                                  img_index: int, img_rect=None, page=None) -> List[str]:
        """获取图片名称候选列表"""
        candidates = []

        # 1. 尝试获取PDF内部名称
        try:
            img_name = doc.xref_get_key(xref, "Name")
            if img_name and img_name[1]:
                internal_name = img_name[1].strip('/')
                if internal_name:
                    candidates.append(f"internal_{internal_name}")
        except:
            pass

        # 2. 尝试获取对象字典中的其他信息
        try:
            obj_dict = doc.xref_get_keys(xref)
            for key in ['Title', 'Subject', 'Alt', 'ActualText']:
                if key in obj_dict:
                    value = doc.xref_get_key(xref, key)
                    if value and value[1]:
                        candidates.append(f"attr_{value[1]}")
        except:
            pass

        # 3. 基于周围文本生成名称
        if img_rect and page:
            nearby_texts = self.find_nearby_text(page, img_rect)
            if nearby_texts:
                # 组合最相关的文本
                combined_text = "_".join(nearby_texts[:2])
                if combined_text:
                    candidates.append(f"context_{combined_text}")

        # 4. 基于页面内容寻找文件名模式
        if page:
            try:
                # 获取页面中的所有文本，寻找可能的文件名
                page_text = page.get_text()

                # 寻找常见的文件名模式（包括设计文件和图片文件）
                filename_patterns = [
                    # 文件名.扩展名
                    r'([a-zA-Z0-9_\-\u4e00-\u9fff]+\.(?:psd|ai|png|jpg|jpeg|gif|bmp|tiff|webp))',
                    # 提取文件名部分
                    r'([a-zA-Z0-9_\-\u4e00-\u9fff]{3,20})\.(?:psd|ai|png|jpg|jpeg)',
                    # 标签后的名称
                    r'(?:文件名|filename|name)[:：\s]*([a-zA-Z0-9_\-\u4e00-\u9fff]{3,30})',
                    # 参考图命名
                    r'([a-zA-Z0-9_\-\u4e00-\u9fff]{3,20})(?:\s*参考|_ref|_reference)',
                    # 角色名称
                    r'(?:角色|character|人物)[:：\s]*([a-zA-Z0-9_\-\u4e00-\u9fff]{2,20})',
                ]

                for pattern in filename_patterns:
                    matches = re.finditer(pattern, page_text, re.IGNORECASE)
                    for match in matches:
                        filename_candidate = match.group(1).strip()
                        if filename_candidate and len(filename_candidate) >= 2:
                            candidates.append(f"file_{filename_candidate}")

            except:
                pass

        return candidates

    def generate_final_filename(self, candidates: List[str], img_ext: str,
                                page_num: int, img_index: int, img_hash: str) -> str:
        """从候选名称中选择最佳文件名"""

        # 清理和评分候选名称
        scored_candidates = []

        for candidate in candidates:
            cleaned = self.clean_filename(candidate)
            if cleaned and len(cleaned) >= 3:  # 最短3个字符
                score = len(cleaned)  # 基础分数：长度

                # 加分项 - 针对角色参考图优化
                if 'file_' in candidate:  # 文件名模式匹配
                    score += 25
                if 'internal_' in candidate:  # PDF内部名称
                    score += 20
                if any(ext in cleaned.lower() for ext in ['psd', 'ai', 'png', 'jpg', 'jpeg']):
                    score += 15  # 包含文件扩展名
                if any(keyword in cleaned.lower() for keyword in ['角色', 'character', '人物', 'ref', 'reference']):
                    score += 12  # 包含角色相关关键词
                if any(char.isdigit() for char in cleaned):  # 包含数字（版本号等）
                    score += 8

                # 减分项
                if 'context_' in candidate and len(cleaned) < 10:
                    score -= 5
                if any(word in cleaned.lower() for word in ['figure', 'table', 'chart', '图', '表', '图表']):
                    score -= 10  # 降低传统图表命名的优先级

                scored_candidates.append((score, cleaned))

        # 选择最高分的候选名称
        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_name = scored_candidates[0][1]

            # 避免文件名冲突
            base_name = best_name
            counter = 1
            final_path = self.output_dir / f"{base_name}.{img_ext}"

            while final_path.exists():
                final_path = self.output_dir / \
                    f"{base_name}_{counter}.{img_ext}"
                counter += 1

            return final_path.name

        # 回退到默认命名
        return f"page_{page_num:03d}_img_{img_index:02d}_{img_hash[:8]}.{img_ext}"

    def generate_thumbnail(self, image_data: bytes, max_size: int = 300) -> bytes:
        """生成缩略图"""
        try:
            # 打开图片
            image = Image.open(io.BytesIO(image_data))

            # 转换为RGB模式（如果需要）
            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')

            # 生成缩略图，保持宽高比
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # 保存为JPEG格式
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=85, optimize=True)
            return output.getvalue()

        except Exception as e:
            print(f"生成缩略图失败: {e}")
            # 如果缩略图生成失败，返回原图数据（可能会比较大）
            return image_data

    async def extract_images_for_preview(self, pdf_data: BinaryIO,
                                         skip_duplicates: bool = True,
                                         min_size: int = 1000,
                                         thumbnail_size: int = 300) -> Dict[str, Any]:
        """从PDF提取图片并生成预览数据（用于临时存储）"""

        # 重置重复检测状态
        self.duplicate_hashes.clear()

        try:
            # 读取PDF数据
            pdf_data.seek(0)
            pdf_bytes = pdf_data.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise Exception(f"无法打开PDF文件: {e}")

        extraction_result: Dict[str, Any] = {
            "total_pages": len(doc),
            "total_images_found": 0,
            "images_extracted": 0,
            "duplicates_skipped": 0,
            "small_images_skipped": 0,
            "errors": [],
            "extracted_images": []  # 包含图片数据的列表
        }

        print(f"PDF共有 {len(doc)} 页")

        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)

                extraction_result["total_images_found"] += len(image_list)

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]

                        # 获取图片数据
                        img_dict = doc.extract_image(xref)
                        img_data = img_dict["image"]
                        img_ext = img_dict["ext"]

                        # 检查图片大小
                        if len(img_data) < min_size:
                            extraction_result["small_images_skipped"] += 1
                            continue

                        # 检查重复
                        img_hash = self.get_image_hash(img_data)
                        if skip_duplicates and img_hash in self.duplicate_hashes:
                            extraction_result["duplicates_skipped"] += 1
                            print(
                                f"  ⏩ 跳过重复图片: {img_hash[:8]}... (第{page_num+1}页)")
                            continue

                        self.duplicate_hashes.add(img_hash)
                        print(f"  ✓ 新图片哈希: {img_hash[:8]}... (第{page_num+1}页)")

                        # 获取图片位置信息
                        img_rect = None
                        try:
                            img_rects = page.get_image_rects(xref)
                            if img_rects:
                                img_rect = img_rects[0]
                        except:
                            pass

                        # 生成文件名候选
                        candidates = self.get_image_name_candidates(
                            doc, xref, page_num, img_index, img_rect, page
                        )

                        # 选择最终文件名
                        filename = self.generate_final_filename(
                            candidates, img_ext, page_num, img_index, img_hash
                        )

                        # 生成缩略图
                        thumbnail_data = self.generate_thumbnail(
                            img_data, thumbnail_size)

                        # 记录提取信息
                        extract_info = {
                            "filename": filename,
                            "original_filename": filename,  # 保持一致
                            "page": page_num + 1,
                            "index": img_index,
                            "size_bytes": len(img_data),
                            "format": img_ext,
                            "hash": img_hash,
                            "candidates_tried": candidates,
                            "dimensions": f"{img_dict.get('width', 'unknown')}x{img_dict.get('height', 'unknown')}",
                            "original_data": img_data,  # 原始图片数据
                            "thumbnail_data": thumbnail_data,  # 缩略图数据
                            "thumbnail_size_bytes": len(thumbnail_data)
                        }

                        if img_rect:
                            extract_info["position"] = {
                                "x": round(img_rect.x0, 2),
                                "y": round(img_rect.y0, 2),
                                "width": round(img_rect.width, 2),
                                "height": round(img_rect.height, 2)
                            }

                        extraction_result["extracted_images"].append(
                            extract_info)
                        extraction_result["images_extracted"] += 1

                        print(
                            f"  ✓ 提取: {filename} (原图: {len(img_data)} bytes, 缩略图: {len(thumbnail_data)} bytes)")

                    except Exception as e:
                        error_msg = f"第{page_num+1}页图片{img_index}提取失败: {e}"
                        extraction_result["errors"].append(error_msg)
                        print(f"  ✗ {error_msg}")

            except Exception as e:
                error_msg = f"处理第{page_num+1}页时出错: {e}"
                extraction_result["errors"].append(error_msg)
                print(f"✗ {error_msg}")

        doc.close()
        return extraction_result

    def extract_images_from_pdf(self, pdf_path: str,
                                skip_duplicates: bool = True,
                                min_size: int = 1000) -> Dict[str, Any]:
        """从PDF提取所有图片"""

        # 重置重复检测状态
        self.duplicate_hashes.clear()

        print(f"开始处理PDF文件: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise Exception(f"无法打开PDF文件: {e}")

        extraction_stats: Dict[str, Any] = {
            "total_pages": len(doc),
            "total_images_found": 0,
            "images_extracted": 0,
            "duplicates_skipped": 0,
            "small_images_skipped": 0,
            "errors": []
        }

        print(f"PDF共有 {len(doc)} 页")

        for page_num in range(len(doc)):
            print(f"处理第 {page_num + 1}/{len(doc)} 页...")

            try:
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)

                extraction_stats["total_images_found"] += len(image_list)

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]

                        # 获取图片数据
                        img_dict = doc.extract_image(xref)
                        img_data = img_dict["image"]
                        img_ext = img_dict["ext"]

                        # 检查图片大小
                        if len(img_data) < min_size:
                            extraction_stats["small_images_skipped"] += 1
                            continue

                        # 检查重复
                        img_hash = self.get_image_hash(img_data)
                        if skip_duplicates and img_hash in self.duplicate_hashes:
                            extraction_stats["duplicates_skipped"] += 1
                            print(
                                f"  ⏩ 跳过重复图片: {img_hash[:8]}... (第{page_num+1}页)")
                            continue

                        self.duplicate_hashes.add(img_hash)
                        print(f"  ✓ 新图片哈希: {img_hash[:8]}... (第{page_num+1}页)")

                        # 获取图片位置信息
                        img_rect = None
                        try:
                            img_rects = page.get_image_rects(xref)
                            if img_rects:
                                img_rect = img_rects[0]
                        except:
                            pass

                        # 生成文件名候选
                        candidates = self.get_image_name_candidates(
                            doc, xref, page_num, img_index, img_rect, page
                        )

                        # 选择最终文件名
                        filename = self.generate_final_filename(
                            candidates, img_ext, page_num, img_index, img_hash
                        )

                        # 保存图片
                        output_path = self.output_dir / filename
                        with open(output_path, "wb") as f:
                            f.write(img_data)

                        # 记录提取信息
                        extract_info = {
                            "filename": filename,
                            "page": page_num + 1,
                            "index": img_index,
                            "size_bytes": len(img_data),
                            "format": img_ext,
                            "hash": img_hash,
                            "candidates_tried": candidates,
                            "dimensions": f"{img_dict.get('width', 'unknown')}x{img_dict.get('height', 'unknown')}"
                        }

                        if img_rect:
                            extract_info["position"] = {
                                "x": round(img_rect.x0, 2),
                                "y": round(img_rect.y0, 2),
                                "width": round(img_rect.width, 2),
                                "height": round(img_rect.height, 2)
                            }

                        self.extraction_log.append(extract_info)
                        extraction_stats["images_extracted"] += 1

                        print(f"  ✓ 提取: {filename} ({len(img_data)} bytes)")

                    except Exception as e:
                        error_msg = f"第{page_num+1}页图片{img_index}提取失败: {e}"
                        extraction_stats["errors"].append(error_msg)
                        print(f"  ✗ {error_msg}")

            except Exception as e:
                error_msg = f"处理第{page_num+1}页时出错: {e}"
                extraction_stats["errors"].append(error_msg)
                print(f"✗ {error_msg}")

        doc.close()
        return extraction_stats

    def save_extraction_report(self, pdf_path: str, stats: Dict):
        """保存提取报告"""
        report = {
            "extraction_time": datetime.now().isoformat(),
            "source_pdf": os.path.basename(pdf_path),
            "output_directory": str(self.output_dir),
            "statistics": stats,
            "extracted_images": self.extraction_log
        }

        report_path = self.output_dir / "extraction_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n📊 提取报告已保存: {report_path}")

        # 打印摘要
        print(f"""
📈 提取完成！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 源文件: {os.path.basename(pdf_path)}
📁 输出目录: {self.output_dir}
🔍 发现图片: {stats['total_images_found']} 个
✅ 成功提取: {stats['images_extracted']} 个
🔄 跳过重复: {stats['duplicates_skipped']} 个
📏 跳过小图: {stats['small_images_skipped']} 个
❌ 提取失败: {len(stats['errors'])} 个
""")


def main():
    parser = argparse.ArgumentParser(description="PDF图片提取Pipeline")
    parser.add_argument("pdf_path", help="PDF文件路径")
    parser.add_argument("-o", "--output", default="extracted_images",
                        help="输出目录 (默认: extracted_images)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="不跳过重复图片")
    parser.add_argument("--min-size", type=int, default=1000,
                        help="最小图片大小(字节) (默认: 1000)")

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"❌ 文件不存在: {args.pdf_path}")
        return

    try:
        extractor = PDFImageExtractor(args.output)
        stats = extractor.extract_images_from_pdf(
            args.pdf_path,
            skip_duplicates=not args.no_dedup,
            min_size=args.min_size
        )
        extractor.save_extraction_report(args.pdf_path, stats)

    except Exception as e:
        print(f"❌ 处理失败: {e}")


if __name__ == "__main__":
    main()


# 使用示例：
"""
# 基本使用
python pdf_extractor.py document.pdf

# 指定输出目录
python pdf_extractor.py document.pdf -o my_images

# 不跳过重复图片
python pdf_extractor.py document.pdf --no-dedup

# 设置最小图片大小
python pdf_extractor.py document.pdf --min-size 5000

# 在代码中使用
extractor = PDFImageExtractor("output_folder")
stats = extractor.extract_images_from_pdf("example.pdf")
extractor.save_extraction_report("example.pdf", stats)
"""
