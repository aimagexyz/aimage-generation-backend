# Standard library imports
import asyncio
import io
from aimage_supervision.settings import logger
import os
import re
import tempfile
from typing import BinaryIO, Optional, Tuple

# Third-party imports
from PIL import Image
from pdf2image import convert_from_bytes
import cairosvg


# Availability flags for optional dependencies
ASPOSE_PSD_AVAILABLE = False  # will be detected lazily
PDF2IMAGE_AVAILABLE = True
CAIROSVG_AVAILABLE = True
DISABLE_ASPOSE_PSD = os.getenv(
    'DISABLE_ASPOSE_PSD', '').lower() in ('1', 'true', 'yes')


class ImageCompressionConfig:
    """图片压缩配置类"""

    def __init__(
        self,
        max_width: int = 1920,
        max_height: int = 1080,
        jpeg_quality: int = 85,
        png_optimize: bool = True,
        auto_format_convert: bool = True
    ):
        self.max_width = max_width
        self.max_height = max_height
        self.jpeg_quality = jpeg_quality
        self.png_optimize = png_optimize
        self.auto_format_convert = auto_format_convert


# デフォルト設定 - 参考画像用に最適化
DEFAULT_CONFIG = ImageCompressionConfig(
    max_width=1920,    # 一般的な参考用途に十分
    max_height=1080,   # フルHD解像度まで
    jpeg_quality=85,   # 品質と圧縮のバランス
    png_optimize=True,
    auto_format_convert=True
)


def _is_svg_file(content: bytes) -> bool:
    """
    Check if the content is an SVG file
    """
    # Check first 1024 bytes for SVG indicators
    header = content[:1024] if len(content) > 1024 else content

    # Check for XML declaration or SVG tag
    if header.startswith(b'<?xml') or header.startswith(b'<svg'):
        return True

    # Check for SVG tag within first 500 bytes (might have comments before)
    if b'<svg' in header[:500]:
        return True

    # Check if decoded text contains SVG namespace
    try:
        text_header = header.decode('utf-8', errors='ignore')
        if 'xmlns="http://www.w3.org/2000/svg"' in text_header or '<svg' in text_header:
            return True
    except:
        pass

    return False


def _is_psd_file(content: bytes) -> bool:
    """
    Check if the content is a PSD file (Adobe Photoshop)
    """
    # PSD files start with '8BPS'
    return len(content) >= 4 and content[:4] == b'8BPS'


def _convert_psd_to_png_aspose(file_content: bytes) -> Tuple[Optional[bytes], Optional[int]]:
    """
    Try converting PSD to PNG using Aspose.PSD. Returns (png_bytes, size) or (None, None) on failure.
    """
    if DISABLE_ASPOSE_PSD:
        return None, None

    try:
        import importlib

        psd_module = importlib.import_module('aspose.psd')
        imageoptions_module = importlib.import_module(
            'aspose.psd.imageoptions')

        PsdImage = psd_module.Image
        PngOptions = imageoptions_module.PngOptions

        # Create temporary file for PSD processing
        with tempfile.NamedTemporaryFile(suffix='.psd', delete=False) as tmp_psd:
            tmp_psd.write(file_content)
            tmp_psd_path = tmp_psd.name

        tmp_png_path = None
        try:
            # Load PSD with Aspose.PSD
            psd_image = PsdImage.load(tmp_psd_path)

            # Create PNG options
            png_options = PngOptions()
            # Use true color with alpha for best fidelity
            png_options.color_type = psd_module.fileformats.png.PngColorType.TRUECOLOR_WITH_ALPHA

            # Save to temporary PNG file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
                tmp_png_path = tmp_png.name

            psd_image.save(tmp_png_path, png_options)

            # Read the PNG back into memory
            with open(tmp_png_path, 'rb') as f:
                png_content = f.read()

            return png_content, len(png_content)

        finally:
            # Clean up temporary files
            if os.path.exists(tmp_psd_path):
                os.unlink(tmp_psd_path)
            if tmp_png_path and os.path.exists(tmp_png_path):
                os.unlink(tmp_png_path)

    except Exception as e:
        logger.warning(f"aspose-psd変換失敗、PILフォールバック: {e}")
        return None, None


def _is_ai_file(content: bytes) -> bool:
    """
    Check if the content is an AI file (Adobe Illustrator)
    """
    # AI files can be PDF-based (newer versions) or EPS-based (older versions)
    # Check for PDF header
    if content.startswith(b'%PDF'):
        # Further check if it might be an AI file saved as PDF
        header = content[:2048] if len(content) > 2048 else content
        try:
            header_str = header.decode('utf-8', errors='ignore')
            if 'Adobe Illustrator' in header_str or '/Creator (Adobe Illustrator' in header_str:
                return True
        except:
            pass

    # Check for EPS-based AI files
    if content.startswith(b'%!PS-Adobe'):
        header = content[:1024] if len(content) > 1024 else content
        if b'%%Creator: Adobe Illustrator' in header:
            return True

    return False


def _convert_svg_to_png(svg_content: bytes, max_width: int = 1920, max_height: int = 1080) -> Tuple[bytes, int]:
    """
    Convert SVG file to PNG format

    Args:
        svg_content: SVG file content as bytes
        max_width: Maximum width for the output PNG
        max_height: Maximum height for the output PNG

    Returns:
        Tuple[bytes, int]: (PNG content, file size) or (original content, original size) if conversion fails
    """
    if not CAIROSVG_AVAILABLE:
        logger.warning("cairosvg not available, cannot convert SVG to PNG")
        return svg_content, len(svg_content)

    try:
        # First try to get the SVG dimensions
        try:
            # Parse SVG to get dimensions (simplified approach)
            svg_str = svg_content.decode('utf-8')
            width_match = re.search(r'width="?(\d+)', svg_str)
            height_match = re.search(r'height="?(\d+)', svg_str)

            svg_width = int(width_match.group(1)) if width_match else None
            svg_height = int(height_match.group(1)) if height_match else None

            # Calculate output dimensions maintaining aspect ratio
            if svg_width and svg_height:
                ratio = min(max_width / svg_width,
                            max_height / svg_height, 1.0)
                output_width = int(svg_width * ratio)
                output_height = int(svg_height * ratio)
            else:
                output_width = max_width
                output_height = max_height

        except:
            # If parsing fails, use max dimensions
            output_width = max_width
            output_height = max_height

        # Convert SVG to PNG using cairosvg
        png_bytes = cairosvg.svg2png(
            bytestring=svg_content,
            output_width=output_width,
            output_height=output_height
        )

        # Further optimize with PIL if needed
        try:
            img = Image.open(io.BytesIO(png_bytes))
            output = io.BytesIO()
            img.save(output, format='PNG', optimize=True)
            optimized_bytes = output.getvalue()

            # Use optimized version if smaller
            if len(optimized_bytes) < len(png_bytes):
                return optimized_bytes, len(optimized_bytes)
        except:
            pass

        return png_bytes, len(png_bytes)

    except Exception as e:
        logger.warning(
            f"SVG to PNG conversion failed: {e}, returning original")
        return svg_content, len(svg_content)


def compress_image(
    image_file: BinaryIO,
    config: Optional[ImageCompressionConfig] = None
) -> Tuple[BinaryIO, str, int]:
    """
    画像ファイルを圧縮する

    Args:
        image_file: 圧縮する画像ファイル（BinaryIO）
        config: 圧縮設定（省略時はデフォルト設定）

    Returns:
        Tuple[BinaryIO, str, int]: (圧縮後のファイル, フォーマット, ファイルサイズ)

    Raises:
        ValueError: 画像処理に失敗した場合
    """
    if config is None:
        config = DEFAULT_CONFIG

    try:
        # ファイルポインタを先頭に移動
        image_file.seek(0)
        file_content = image_file.read()
        original_size = len(file_content)
        image_file.seek(0)

        # Check for SVG files - convert to PNG for better frontend compatibility
        if _is_svg_file(file_content):
            logger.info(
                f"SVGファイル検出: PNGに変換します (元サイズ: {original_size/1024:.1f}KB)")

            png_content, png_size = _convert_svg_to_png(file_content)

            # Check if conversion was successful
            if png_content != file_content:
                logger.info(f"SVG→PNG変換成功: {png_size/1024:.1f}KB")
                return io.BytesIO(png_content), 'PNG', png_size
            else:
                logger.info(f"SVG変換失敗: 元ファイルを使用")
                image_file.seek(0)
                return image_file, 'SVG', original_size

        # Check for PSD files - try to convert to PNG
        if _is_psd_file(file_content):
            logger.info(
                f"PSDファイル検出: PNGに変換を試みます (元サイズ: {original_size/1024:.1f}KB)")

            # Try aspose-psd first (lazily)
            png_bytes, png_size = _convert_psd_to_png_aspose(file_content)
            if png_bytes is not None:
                output = io.BytesIO(png_bytes)
                logger.info(f"PSD→PNG変換成功 (aspose-psd): {png_size/1024:.1f}KB")
                return output, 'PNG', png_size

            # Fallback to PIL
            try:
                image_file.seek(0)
                with Image.open(image_file) as img:
                    # Convert to RGB if necessary
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGBA')

                    # Save as PNG
                    output = io.BytesIO()
                    img.save(output, format='PNG', optimize=True)
                    output_size = output.tell()
                    output.seek(0)

                    logger.info(f"PSD→PNG変換成功 (PIL): {output_size/1024:.1f}KB")
                    return output, 'PNG', output_size
            except Exception as e:
                logger.warning(f"PSD変換失敗、元ファイルを使用: {e}")
                image_file.seek(0)
                return image_file, 'PSD', original_size

        # Check for AI files - try to convert to PNG
        if _is_ai_file(file_content):
            logger.info(
                f"AIファイル検出: PNGに変換を試みます (元サイズ: {original_size/1024:.1f}KB)")

            # Try pdf2image first if available (for PDF-based AI files)
            if PDF2IMAGE_AVAILABLE and file_content.startswith(b'%PDF'):
                try:
                    # Convert PDF to images (AI files are often single-page)
                    images = convert_from_bytes(
                        file_content, dpi=150, fmt='png')

                    if images:
                        # Take the first page/image
                        first_image = images[0]

                        # Save as PNG
                        output = io.BytesIO()
                        first_image.save(output, format='PNG', optimize=True)
                        output_size = output.tell()
                        output.seek(0)

                        logger.info(
                            f"AI→PNG変換成功 (pdf2image): {output_size/1024:.1f}KB")
                        return output, 'PNG', output_size

                except Exception as e:
                    logger.warning(f"pdf2image変換失敗、PILフォールバック: {e}")

            # Fallback to PIL
            try:
                image_file.seek(0)
                with Image.open(image_file) as img:
                    # Convert to RGB if necessary
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGBA')

                    # Save as PNG
                    output = io.BytesIO()
                    img.save(output, format='PNG', optimize=True)
                    output_size = output.tell()
                    output.seek(0)

                    logger.info(f"AI→PNG変換成功 (PIL): {output_size/1024:.1f}KB")
                    return output, 'PNG', output_size
            except Exception as e:
                logger.warning(f"AI変換失敗、元ファイルを使用: {e}")
                image_file.seek(0)
                return image_file, 'AI', original_size

        # 小文件直接返回（小于50KB且已经是JPEG格式）
        if original_size < 50 * 1024:  # 50KB以下
            try:
                with Image.open(image_file) as img:
                    if img.format == 'JPEG' and img.size[0] <= config.max_width and img.size[1] <= config.max_height:
                        logger.info(
                            f"小ファイル({original_size/1024:.1f}KB)でリサイズ不要: そのまま使用")
                        image_file.seek(0)
                        return image_file, 'JPEG', original_size
            except Exception:
                pass  # 如果检查失败，继续正常压缩流程

        # PIL Imageで画像を読み込み
        with Image.open(image_file) as img:
            # 元の画像情報をログ出力
            original_width, original_height = img.size
            original_format = img.format or 'UNKNOWN'

            logger.info(
                f"元画像: {original_width}x{original_height}, "
                f"フォーマット: {original_format}, サイズ: {original_size/1024:.1f}KB"
            )

            # サイズ調整が必要かチェック
            new_width, new_height = _calculate_new_dimensions(
                original_width, original_height,
                config.max_width, config.max_height
            )

            # 必要に応じてリサイズ
            if (new_width, new_height) != (original_width, original_height):
                img = img.resize((new_width, new_height),
                                 Image.Resampling.LANCZOS)
                logger.info(f"画像サイズを {new_width}x{new_height} に調整")

            # 出力フォーマットを決定
            output_format = _determine_output_format(img, config)

            # フォーマット変換（必要な場合）
            if output_format == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
                # JPEGは透明度をサポートしないため、白背景で合成
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])
                elif img.mode == 'LA':
                    # LA（Luminance + Alpha）の場合
                    rgb_img = img.convert('RGB')
                    background.paste(rgb_img, mask=img.split()[-1])
                else:
                    # Pモード（パレット）の場合
                    background.paste(img.convert('RGB'))
                img = background
                logger.info(f"フォーマットを {img.mode} から RGB に変換")

            # 圧縮して保存
            output = io.BytesIO()
            save_kwargs = _get_save_kwargs(output_format, config)
            img.save(output, **save_kwargs)

            # ファイルサイズとフォーマット情報
            compressed_size = output.tell()
            output.seek(0)

            # 圧縮結果をログ出力
            compression_ratio = (1 - compressed_size / original_size) * 100

            # 如果压缩后文件更大，使用原文件
            if compressed_size >= original_size:
                logger.info(
                    f"圧縮効果なし: 元ファイルを使用 "
                    f"(元: {original_size/1024:.1f}KB, 圧縮後: {compressed_size/1024:.1f}KB)"
                )
                # 返回原始文件
                image_file.seek(0)
                return image_file, original_format, original_size
            else:
                logger.info(
                    f"圧縮完了: {compressed_size/1024:.1f}KB "
                    f"(圧縮率: {compression_ratio:.1f}%)"
                )
                return output, output_format, compressed_size

    except Exception as e:
        logger.error(f"画像圧縮エラー: {str(e)}")
        raise ValueError(f"画像の圧縮に失敗しました: {str(e)}")


def _calculate_new_dimensions(
    original_width: int,
    original_height: int,
    max_width: int,
    max_height: int
) -> Tuple[int, int]:
    """
    アスペクト比を保持して新しいサイズを計算

    Args:
        original_width: 元の幅
        original_height: 元の高さ
        max_width: 最大幅
        max_height: 最大高さ

    Returns:
        Tuple[int, int]: (新しい幅, 新しい高さ)
    """
    # 最大サイズ以下の場合はそのまま
    if original_width <= max_width and original_height <= max_height:
        return original_width, original_height

    # アスペクト比を保持して縮小
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height

    # より制限的な比率を使用
    scale_ratio = min(width_ratio, height_ratio)

    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)

    return new_width, new_height


def _determine_output_format(img: Image.Image, config: ImageCompressionConfig) -> str:
    """
    出力フォーマットを決定

    Args:
        img: PIL画像オブジェクト
        config: 圧縮設定

    Returns:
        str: 出力フォーマット ('JPEG' または 'PNG')
    """
    if not config.auto_format_convert:
        # 自動変換しない場合は元のフォーマットを保持
        return img.format or 'JPEG'

    # 透明度がある場合はPNG、それ以外はJPEG
    if img.mode in ('RGBA', 'LA') and _has_transparency(img):
        return 'PNG'
    else:
        return 'JPEG'


def _has_transparency(img: Image.Image) -> bool:
    """
    画像に透明度があるかチェック

    Args:
        img: PIL画像オブジェクト

    Returns:
        bool: 透明度がある場合True
    """
    if img.mode == 'RGBA':
        # アルファチャンネルをチェック
        alpha = img.split()[-1]
        return alpha.getextrema()[0] < 255
    elif img.mode == 'LA':
        # Luminance + Alphaの場合
        alpha = img.split()[-1]
        return alpha.getextrema()[0] < 255
    elif img.mode == 'P' and 'transparency' in img.info:
        # パレットモードで透明色指定がある場合
        return True

    return False


def _get_save_kwargs(output_format: str, config: ImageCompressionConfig) -> dict:
    """
    保存時のパラメータを取得

    Args:
        output_format: 出力フォーマット
        config: 圧縮設定

    Returns:
        dict: 保存パラメータ
    """
    kwargs = {'format': output_format}

    if output_format == 'JPEG':
        kwargs.update({
            'quality': config.jpeg_quality,
            'optimize': True
        })
    elif output_format == 'PNG':
        kwargs.update({
            'optimize': config.png_optimize
        })

    return kwargs


async def compress_image_async(
    image_file: BinaryIO,
    config: Optional[ImageCompressionConfig] = None
) -> Tuple[BinaryIO, str, int]:
    """
    画像圧縮の非同期ラッパー

    Args:
        image_file: 圧縮する画像ファイル
        config: 圧縮設定

    Returns:
        Tuple[BinaryIO, str, int]: (圧縮後のファイル, フォーマット, ファイルサイズ)
    """
    # CPU集約的なタスクなので、executor で実行
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        compress_image,
        image_file,
        config
    )
