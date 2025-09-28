import asyncio
import io
import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

from fastapi import (APIRouter, File, Form, HTTPException, Security,
                     UploadFile, status)
from pydantic import BaseModel

from ..clients.aws_s3 import (cleanup_s3_prefix,
                              download_file_content_as_bytesio_from_s3,
                              get_s3_url_from_path, upload_file_to_s3,
                              upload_file_to_s3_with_ttl)
from ..middlewares.auth import get_current_user
from ..models import PDF, Project, User
from ..schemas import PDFResponse
from ..utils.pdf_services import PDFImageExtractor

# 定义S3 URL前缀
S3_URL_PREFIX = 's3://ai-mage-supervision/'


router = APIRouter(prefix='/pdf', tags=['PDF Processing'])


class PDFImageInfo(BaseModel):
    filename: str
    original_filename: str
    page: int
    index: int
    size_bytes: int
    format: str
    hash: str
    dimensions: str
    position: Optional[Dict[str, float]] = None
    thumbnail_url: str
    candidates_tried: List[str]
    thumbnail_size_bytes: int


class PDFExtractionPreviewResponse(BaseModel):
    session_id: str
    total_pages: int
    total_images_found: int
    images_extracted: int
    duplicates_skipped: int
    small_images_skipped: int
    errors: List[str]
    extracted_images: List[PDFImageInfo]


class ConfirmExtractionRequest(BaseModel):
    selected_images: List[str]  # 用户选中的图片文件名列表
    project_id: Optional[str] = None  # 目标项目ID（如果有）


class ConfirmExtractionResponse(BaseModel):
    moved_files: List[str]
    errors: List[str]
    pdf_id: str
    extracted_items_count: int


@router.post('/extract-preview')
async def extract_pdf_preview(
    user: Annotated[User, Security(get_current_user)],
    session_id: str = Form(...),
    file: UploadFile = File(...),
    thumbnail_size: int = Form(300),
    min_size: int = Form(1000),
    skip_duplicates: bool = Form(True)
) -> PDFExtractionPreviewResponse:
    """从PDF提取图片并生成预览，存储到临时S3路径"""

    # 验证文件类型
    if not file.content_type or file.content_type != 'application/pdf':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='文件必须是PDF格式'
        )

    # 验证会话ID格式
    if not session_id or not session_id.startswith('pdf-'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='无效的会话ID格式，必须以pdf-开头'
        )

    try:
        # 创建PDF提取器
        extractor = PDFImageExtractor()

        # 提取图片和生成缩略图
        extraction_result = await extractor.extract_images_for_preview(
            pdf_data=file.file,
            skip_duplicates=skip_duplicates,
            min_size=min_size,
            thumbnail_size=thumbnail_size
        )

        # 首先保存原始PDF文件到临时存储
        file.file.seek(0)  # 重置文件指针
        pdf_data = await file.read()
        pdf_s3_path = f'temp/pdf-sessions/{session_id}/original.pdf'

        pdf_upload_task = upload_file_to_s3_with_ttl(
            file_data=io.BytesIO(pdf_data),
            s3_path=pdf_s3_path,
            ttl_hours=2,
            content_type='application/pdf'
        )

        # 并发上传所有缩略图到S3临时路径
        upload_tasks = [pdf_upload_task]  # 包含PDF文件上传任务
        extracted_images_info = []

        for img_info in extraction_result["extracted_images"]:
            # 生成唯一的文件名，避免同名文件覆盖
            # 使用页码、索引和哈希前缀来确保唯一性
            original_filename = img_info["filename"]
            file_ext = original_filename.split(
                '.')[-1] if '.' in original_filename else 'png'
            unique_filename = f"p{img_info['page']}_i{img_info['index']}_{img_info['hash'][:8]}.{file_ext}"

            # 准备缩略图上传任务
            thumbnail_s3_path = f'temp/pdf-sessions/{session_id}/thumbnails/{unique_filename}'
            thumbnail_data = io.BytesIO(img_info["thumbnail_data"])

            upload_task = upload_file_to_s3_with_ttl(
                file_data=thumbnail_data,
                s3_path=thumbnail_s3_path,
                ttl_hours=2,
                content_type='image/jpeg'
            )
            upload_tasks.append(upload_task)

            # 同时准备原图上传任务（存储在不同路径）
            original_s3_path = f'temp/pdf-sessions/{session_id}/originals/{unique_filename}'
            original_data = io.BytesIO(img_info["original_data"])

            original_upload_task = upload_file_to_s3_with_ttl(
                file_data=original_data,
                s3_path=original_s3_path,
                ttl_hours=2,
                content_type=f'image/{img_info["format"]}'
            )
            upload_tasks.append(original_upload_task)

            # 暂时构建图片信息（缩略图URL稍后生成）
            image_info = {
                "filename": unique_filename,  # 使用唯一文件名
                "original_filename": img_info["original_filename"],
                "page": img_info["page"],
                "index": img_info["index"],
                "size_bytes": img_info["size_bytes"],
                "format": img_info["format"],
                "hash": img_info["hash"],
                "dimensions": img_info["dimensions"],
                "position": img_info.get("position"),
                "thumbnail_s3_path": thumbnail_s3_path,
                "candidates_tried": img_info["candidates_tried"],
                "thumbnail_size_bytes": img_info["thumbnail_size_bytes"]
            }
            extracted_images_info.append(image_info)

        # 等待所有上传完成
        await asyncio.gather(*upload_tasks)

        # 上传完成后，为每个缩略图生成预签名URL
        final_images_info = []
        for img_info in extracted_images_info:
            thumbnail_url = await get_s3_url_from_path(img_info["thumbnail_s3_path"])

            final_image_info = PDFImageInfo(
                filename=img_info["filename"],
                original_filename=img_info["original_filename"],
                page=img_info["page"],
                index=img_info["index"],
                size_bytes=img_info["size_bytes"],
                format=img_info["format"],
                hash=img_info["hash"],
                dimensions=img_info["dimensions"],
                position=img_info.get("position"),
                thumbnail_url=thumbnail_url,
                candidates_tried=img_info["candidates_tried"],
                thumbnail_size_bytes=img_info["thumbnail_size_bytes"]
            )
            final_images_info.append(final_image_info)

        # 保存提取元数据到S3
        metadata = {
            "session_id": session_id,
            "user_id": str(user.id),
            "pdf_name": file.filename,
            "pdf_file_size": len(pdf_data),
            "pdf_s3_path": pdf_s3_path,
            "extraction_time": extraction_result.get("extraction_time"),
            "statistics": {
                "total_pages": extraction_result["total_pages"],
                "total_images_found": extraction_result["total_images_found"],
                "images_extracted": extraction_result["images_extracted"],
                "duplicates_skipped": extraction_result["duplicates_skipped"],
                "small_images_skipped": extraction_result["small_images_skipped"],
                "errors": extraction_result["errors"]
            },
            "extracted_images": [img.dict() for img in final_images_info]
        }

        import json
        metadata_json = io.BytesIO(json.dumps(
            metadata, ensure_ascii=False).encode('utf-8'))
        metadata_s3_path = f'temp/pdf-sessions/{session_id}/metadata.json'
        await upload_file_to_s3_with_ttl(
            file_data=metadata_json,
            s3_path=metadata_s3_path,
            ttl_hours=2,
            content_type='application/json'
        )

        return PDFExtractionPreviewResponse(
            session_id=session_id,
            total_pages=extraction_result["total_pages"],
            total_images_found=extraction_result["total_images_found"],
            images_extracted=extraction_result["images_extracted"],
            duplicates_skipped=extraction_result["duplicates_skipped"],
            small_images_skipped=extraction_result["small_images_skipped"],
            errors=extraction_result["errors"],
            extracted_images=final_images_info
        )

    except Exception as e:
        print(f"PDF提取预览失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'PDF提取失败: {str(e)}'
        )


@router.post('/confirm-extraction/{session_id}')
async def confirm_pdf_extraction(
    user: Annotated[User, Security(get_current_user)],
    session_id: str,
    request: ConfirmExtractionRequest
) -> ConfirmExtractionResponse:
    """确认PDF图片提取，将选中的图片通过items流程保存"""

    # 验证会话ID格式
    if not session_id or not session_id.startswith('pdf-'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='无效的会话ID格式'
        )

    if not request.selected_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='必须选择至少一张图片'
        )

    try:
        # 获取PDF会话元数据
        metadata_path = f'temp/pdf-sessions/{session_id}/metadata.json'
        metadata_data = await download_file_content_as_bytesio_from_s3(metadata_path)
        import json
        session_metadata = json.loads(metadata_data.read().decode('utf-8'))

        # 验证项目权限
        project = None
        if request.project_id:
            project = await Project.of_user(user).get_or_none(id=request.project_id)
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail='项目不存在或无权限访问'
                )

        # 1. 首先保存原始PDF文件并创建PDF记录
        pdf_s3_path = f"projects/{request.project_id}/pdfs/{uuid.uuid4()}_{session_metadata['pdf_name']}"

        # 从临时存储获取原始PDF并保存到正式路径
        temp_pdf_path = session_metadata.get(
            'pdf_s3_path', f'temp/pdf-sessions/{session_id}/original.pdf')
        pdf_data = await download_file_content_as_bytesio_from_s3(temp_pdf_path)

        # 上传到正式的项目路径
        await upload_file_to_s3(
            file_data=pdf_data,
            cloud_file_path=pdf_s3_path
        )

        pdf_record = await PDF.create(
            filename=session_metadata.get('pdf_name', 'unknown.pdf'),
            s3_path=pdf_s3_path,
            file_size=session_metadata.get('pdf_file_size', 0),
            total_pages=session_metadata.get('total_pages', 0),
            extraction_session_id=session_id,
            extracted_at=datetime.now(),
            extraction_method='pymupdf',
            extraction_stats=session_metadata.get('statistics', {}),
            project=project,
            uploaded_by=user
        )

        # 2. 批量处理选中的图片，通过items流程创建Item记录
        from ..endpoints.items import _process_single_file
        from ..services.items_service import process_project_items_embeddings

        moved_files = []
        errors = []
        extracted_items_count = 0

        # 获取提取的图片信息
        extracted_images = session_metadata.get('extracted_images', [])

        # 创建UploadFile对象模拟上传
        class MockUploadFile:
            def __init__(self, data: bytes, filename: str, content_type: str):
                self.file = io.BytesIO(data)
                self.filename = filename
                self.content_type = content_type
                self.size = len(data)

            async def read(self):
                return self.file.read()

            async def seek(self, position: int):
                self.file.seek(position)

        # 准备批处理任务
        tasks = []
        image_infos = []

        for filename in request.selected_images:
            # 找到对应的图片元数据
            image_info = None
            for img in extracted_images:
                if img['filename'] == filename:
                    image_info = img
                    break

            if not image_info:
                errors.append(f'{filename}: 图片元数据未找到')
                continue

            image_infos.append(image_info)

            # 创建异步任务来获取图片数据和处理
            async def process_single_pdf_image(filename, image_info):
                try:
                    # 从临时存储获取原图数据
                    source_path = f'temp/pdf-sessions/{session_id}/originals/{filename}'
                    image_data = await download_file_content_as_bytesio_from_s3(source_path)

                    mock_file = MockUploadFile(
                        data=image_data.read(),
                        filename=filename,
                        content_type=f"image/{image_info['format']}"
                    )

                    # 调用统一的items处理流程
                    result = await _process_single_file(
                        file=mock_file,
                        project_id=request.project_id,
                        parsed_tags=["PDF抽出"],  # 自动添加标签
                        description=f"PDF「{session_metadata.get('pdf_name', 'unknown')}」から抽出 (ページ{image_info['page']})",
                        project=project,
                        user=user,
                        # PDF相关参数
                        source_type='pdf_extracted',
                        source_pdf_id=str(pdf_record.id),
                        pdf_page_number=image_info['page'],
                        pdf_image_index=image_info['index']
                    )

                    return result, filename
                except Exception as e:
                    return {'success': False, 'error': str(e), 'filename': filename}, filename

            # 添加到任务列表
            tasks.append(process_single_pdf_image(filename, image_info))

        # 并发执行所有任务
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            uploaded_items = []

            # 处理结果
            for result_data in results:
                if isinstance(result_data, Exception):
                    errors.append(f"处理异常: {str(result_data)}")
                    continue

                if not isinstance(result_data, tuple) or len(result_data) != 2:
                    errors.append(f"结果格式错误: {result_data}")
                    continue

                result, filename = result_data

                if isinstance(result, dict) and result.get('success'):
                    moved_files.append(filename)
                    extracted_items_count += 1
                    uploaded_items.append(result['item'])
                else:
                    error_msg = result.get('error', '未知错误') if isinstance(
                        result, dict) else str(result)
                    errors.append(f"{filename}: {error_msg}")

            # 如果有成功上传的items且有项目ID，则在后台生成embeddings
            if uploaded_items and request.project_id:
                uploaded_item_ids = [str(item.id) for item in uploaded_items]
                # 在后台任务中生成embeddings，不等待完成
                embedding_task = asyncio.create_task(process_project_items_embeddings(
                    request.project_id, uploaded_item_ids))
                # 保存task引用防止被垃圾回收
                embedding_task.add_done_callback(lambda t: None)

        # 清理临时会话文件
        try:
            prefix = f'temp/pdf-sessions/{session_id}/'
            await cleanup_s3_prefix(prefix)
        except Exception as e:
            print(f"清理临时会话文件失败: {e}")
            # 不影响主要流程，只记录错误

        return ConfirmExtractionResponse(
            moved_files=moved_files,
            errors=errors,
            pdf_id=str(pdf_record.id),
            extracted_items_count=extracted_items_count
        )

    except Exception as e:
        print(f"确认PDF提取失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'确认提取失败: {str(e)}'
        )


@router.delete('/sessions/{session_id}')
async def cleanup_pdf_session(
    user: Annotated[User, Security(get_current_user)],
    session_id: str,
) -> Dict[str, Any]:
    """清理PDF会话的临时文件"""

    # 验证会话ID格式
    if not session_id or not session_id.startswith('pdf-'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='无效的会话ID格式'
        )

    try:
        prefix = f'temp/pdf-sessions/{session_id}/'
        deleted_count, deleted_files = await cleanup_s3_prefix(prefix)

        return {
            "session_id": session_id,
            "deleted_count": deleted_count,
            "deleted_files": deleted_files,
            "message": f"已清理{deleted_count}个文件"
        }

    except Exception as e:
        print(f"清理PDF会话失败 {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'清理会话失败: {str(e)}'
        )


@router.get('/projects/{project_id}/pdfs')
async def list_project_pdfs(
    project_id: str,
    user: Annotated[User, Security(get_current_user)]
) -> List[PDFResponse]:
    """获取项目中的所有PDF文件"""

    # 验证项目权限
    project = await Project.of_user(user).get_or_none(id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='项目不存在或无权限访问'
        )

    pdfs = await PDF.filter(project_id=project_id).prefetch_related('project', 'uploaded_by').order_by('-created_at')

    return [
        PDFResponse(
            id=pdf.id,
            filename=pdf.filename,
            s3_path=pdf.s3_path,
            file_size=pdf.file_size,
            total_pages=pdf.total_pages,
            extraction_session_id=pdf.extraction_session_id,
            extracted_at=pdf.extracted_at,
            extraction_method=pdf.extraction_method,
            extraction_stats=pdf.extraction_stats,
            project_id=pdf.project.id,
            uploaded_by=pdf.uploaded_by.id,
            created_at=pdf.created_at,
            updated_at=pdf.updated_at
        ) for pdf in pdfs
    ]


@router.get('/pdfs/{pdf_id}')
async def get_pdf_details(
    pdf_id: str,
    user: Annotated[User, Security(get_current_user)]
) -> PDFResponse:
    """获取PDF详细信息"""

    pdf = await PDF.get_or_none(id=pdf_id).prefetch_related('project', 'uploaded_by')
    if not pdf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='PDF不存在'
        )

    # 验证权限
    project = await Project.of_user(user).get_or_none(id=pdf.project.id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='无权限访问此PDF'
        )

    return PDFResponse(
        id=pdf.id,
        filename=pdf.filename,
        s3_path=pdf.s3_path,
        file_size=pdf.file_size,
        total_pages=pdf.total_pages,
        extraction_session_id=pdf.extraction_session_id,
        extracted_at=pdf.extracted_at,
        extraction_method=pdf.extraction_method,
        extraction_stats=pdf.extraction_stats,
        project_id=pdf.project.id,
        uploaded_by=pdf.uploaded_by.id,
        created_at=pdf.created_at,
        updated_at=pdf.updated_at
    )


@router.get('/pdfs/{pdf_id}/items')
async def get_pdf_extracted_items(
    pdf_id: str,
    user: Annotated[User, Security(get_current_user)]
) -> Dict[str, Any]:
    """获取PDF提取的所有图片"""

    # 获取PDF信息
    pdf = await PDF.get_or_none(id=pdf_id).prefetch_related('project', 'uploaded_by')
    if not pdf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='PDF不存在'
        )

    # 验证权限
    project = await Project.of_user(user).get_or_none(id=pdf.project.id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='无权限访问此PDF'
        )

    # 获取提取的图片，按页码和图片索引排序
    from ..models import Item
    from ..schemas import ItemResponse

    items = await Item.filter(
        source_type='pdf_extracted',
        source_pdf_id=pdf_id
    ).prefetch_related('uploaded_by', 'project').order_by('pdf_page_number', 'pdf_image_index')

    # 转换为响应格式
    item_responses = []
    for item in items:
        await item.fetch_image_url()

        item_response = ItemResponse(
            id=item.id,
            filename=item.filename,
            s3_path=item.s3_path,
            s3_url=item.s3_url,
            image_url=item.image_url(),
            content_type=item.content_type,
            file_size=item.file_size,
            tags=item.tags,
            description=item.description,
            project_id=item.project.id if item.project else None,
            uploaded_by=item.uploaded_by.id,
            created_at=item.created_at,
            updated_at=item.updated_at,
            source_type=item.source_type,
            source_pdf_id=item.source_pdf_id,
            pdf_page_number=item.pdf_page_number,
            pdf_image_index=item.pdf_image_index
        )
        item_responses.append(item_response)

    return {
        "pdf": PDFResponse(
            id=pdf.id,
            filename=pdf.filename,
            s3_path=pdf.s3_path,
            file_size=pdf.file_size,
            total_pages=pdf.total_pages,
            extraction_session_id=pdf.extraction_session_id,
            extracted_at=pdf.extracted_at,
            extraction_method=pdf.extraction_method,
            extraction_stats=pdf.extraction_stats,
            project_id=pdf.project.id,
            uploaded_by=pdf.uploaded_by.id,
            created_at=pdf.created_at,
            updated_at=pdf.updated_at
        ),
        "extracted_items": item_responses,
        "total_extracted": len(item_responses)
    }
