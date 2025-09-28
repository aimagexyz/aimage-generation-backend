import json
from aimage_supervision.settings import logger
import os
import uuid
from datetime import UTC, datetime
from typing import Any, BinaryIO, Dict, Optional
from uuid import UUID

from dotenv import load_dotenv
from tortoise.transactions import atomic

from aimage_supervision.clients.aws_s3 import upload_file_to_s3
from aimage_supervision.enums import SubtaskStatus as SubtaskStatusEnum
from aimage_supervision.enums import SubtaskType
from aimage_supervision.models import (Project, Subtask, Task, TaskPriority,
                                       TaskStatus, User)
from aimage_supervision.schemas import SubtaskContent
from aimage_supervision.utils.temp_file import temp_file_manager

# 加载环境变量
load_dotenv()


class UploadPPTXProcessor:
    """处理直接上传的PPT文件的工具类"""

    def __init__(self):
        # S3存储桶名称
        self.S3_BUCKET = os.getenv('AWS_BUCKET_NAME')

    async def insert_task(self,
                          name: str,
                          description: str,
                          s3_path: str,  # This s3_path is for the main PPTX file
                          project_obj: Project,  # Changed from project_id to Project ORM object
                          assignee_user: User   # Changed from assignee_id to User ORM object
                          ) -> Task:  # Return the created Task ORM object
        """
        使用 Tortoise ORM 插入主任务并返回 Task 对象。

        Args:
            name: 任务名称
            description: 任务描述
            s3_path: 主PPTX文件在S3上的路径
            project_obj: 关联的 Project ORM 对象
            assignee_user: 分配的 User ORM 对象

        Returns:
            Task: 创建的 Task ORM 对象
        """
        try:
            # 1. 获取默认的 TaskStatus 和 TaskPriority
            # 遵循 task_service.py 的逻辑：先尝试全局名，然后项目特定，然后任意一个
            # 对于 PPTX 上传场景，我们可能不直接知道 org，所以全局/任意更合适

            default_task_status = await TaskStatus.filter(name='TODO').first()
            if not default_task_status:
                # Fallback: 尝试从项目可用的状态中获取，如果项目对象有此信息
                #  if hasattr(project_obj, 'status_candidates') and await project_obj.status_candidates.all().exists():
                #      default_task_status = await project_obj.status_candidates.all().first()
                #  else: # Absolute fallback if project has no candidates or attr not available
                #      default_task_status = await TaskStatus.all().first()
                # 简化：对于批量上传，通常使用更固定的全局默认值
                # Get any status, ordered by oid
                default_task_status = await TaskStatus.all().order_by('oid').first()

            default_task_priority = await TaskPriority.filter(name='LOW').first()
            if not default_task_priority:
                # Fallback similar to status
                # if hasattr(project_obj, 'priority_candidates') and await project_obj.priority_candidates.all().exists():
                #     default_task_priority = await project_obj.priority_candidates.all().first()
                # else:
                #     default_task_priority = await TaskPriority.all().first()
                default_task_priority = await TaskPriority.all().order_by('oid').first()  # Get any priority

            if not default_task_status:
                logger.error("No TaskStatus found in the database.")
                raise ValueError("タスクステータスが見つかりません")  # Task status not found
            if not default_task_priority:
                logger.error("No TaskPriority found in the database.")
                raise ValueError("タスクの優先度が見つかりません")  # Task priority not found

            # 2. 生成 tid (Timestamp-based Task ID)
            tid = datetime.now(UTC).strftime("T%Y%m%d%H%M%S")

            # 3. 创建 Task 实例
            # 确保任务名称长度符合模型限制 (Task.name max_length=255)
            task_name_truncated = name[:255]

            new_task = await Task.create(
                tid=tid,
                name=task_name_truncated,
                description=description,
                s3_path=s3_path,  # s3_path for the main task (PPTX file)
                project=project_obj,
                assignee=assignee_user,
                status=default_task_status,
                priority=default_task_priority
                # created_by is implicitly assignee_user in this context
            )

            logger.info(
                f"Created task with TID: {tid}, Name: {task_name_truncated}, ORM ID: {new_task.id}")
            return new_task
        except Exception as e:
            logger.error(f"Error inserting task '{name}' using ORM: {str(e)}")
            # It's better to let the caller handle the HTTPException or re-raise a specific one.
            raise

    async def insert_subtask(self,
                             parent_task: Task,  # Changed from task_id to Task ORM object
                             order_index: int,
                             name: str,
                             subtask_raw_content: Dict,  # Content extracted from PPTX slide
                             assignee_user: User  # User ORM object for assignee
                             ) -> Subtask:  # Return the created Subtask ORM object
        """
        使用 Tortoise ORM 插入子任务。

        Args:
            parent_task: 父 Task ORM 对象。
            order_index: 子任务的顺序索引 (oid)。
            name: 子任务名称。
            subtask_raw_content: 从 PPTX 提取的原始子任务内容字典。
                                 Expected keys: 's3_path', 'description', 'task_type', 'slide_page_number'.
            assignee_user: 分配的 User ORM 对象 (用于记录创建者或相关人，如果模型支持)

        Returns:
            Subtask: 创建的 Subtask ORM 对象。
        """
        try:
            # 1. 构建 SubtaskContent Pydantic 对象
            # 确保子任务名称长度符合模型限制 (Subtask.name max_length=255)
            subtask_name_truncated = name[:255]

            # 从原始字典中提取 task_type 字符串值
            task_type_str = subtask_raw_content.get('task_type')
            if not task_type_str or not isinstance(task_type_str, str):
                logger.warning(
                    f"Missing or invalid task_type in subtask_raw_content for {name}. Defaulting to PICTURE.")
                task_type_enum = SubtaskType.PICTURE
            else:
                try:
                    # 将字符串转换为 SubtaskType 枚举成员
                    task_type_enum = SubtaskType(task_type_str.lower())
                except ValueError:
                    logger.warning(
                        f"Invalid task_type value '{task_type_str}' for {name}. Defaulting to PICTURE.")
                    task_type_enum = SubtaskType.PICTURE

            # 构建 Pydantic 模型，注意 s3_path 在 subtask_raw_content 中，而主 task 的 s3_path 不同
            content_obj = SubtaskContent(
                title=subtask_name_truncated,  # Typically, subtask name can be the title
                # s3_path of the image for this slide
                s3_path=subtask_raw_content.get('s3_path', ''),
                compressed_s3_path=subtask_raw_content.get(
                    'compressed_s3_path'),  # 新增压缩图片路径
                description=subtask_raw_content.get('description', ''),
                # Use the enum's value (string)
                task_type=task_type_enum,
                author=str(assignee_user.id),  # Author as user ID string
                created_at=datetime.now(UTC).isoformat() + "Z",
                slide_page_number=subtask_raw_content.get('slide_page_number')
            )

            # 2. 创建 Subtask 实例
            new_subtask = await Subtask.create(
                oid=order_index,
                name=subtask_name_truncated,
                task_type=task_type_enum,  # Use the SubtaskType enum member here
                # Or a more specific description for the subtask itself
                description=content_obj.description,
                # Store Pydantic model as JSON
                content=content_obj.model_dump(exclude_none=True),
                task=parent_task,  # Link to parent Task ORM object
                status=SubtaskStatusEnum.PENDING,  # Default status for new subtasks
                slide_page_number=content_obj.slide_page_number,
                # As per task_service.py, Subtask.create can take assignee_id.
                # If Subtask model has an assignee FK field, it should be `assignee=assignee_user`
                # If it takes an id, then `assignee_id=assignee_user.id`
                # Given task_service.py uses assignee_id, we'll use that.
                assignee_id=assignee_user.id
            )
            logger.info(
                f"Created subtask '{subtask_name_truncated}' (OID: {order_index}) for Task ID: {parent_task.id}, Subtask ORM ID: {new_subtask.id}")
            return new_subtask
        except Exception as e:
            logger.error(
                f"Error inserting subtask '{name}' for task {parent_task.id if parent_task else 'Unknown'} using ORM: {str(e)}")
            raise

    @atomic()
    async def process_uploaded_pptx(self, file: BinaryIO, filename: str,
                                    project_id_str: str, user_id_str: str) -> Dict[str, Any]:
        """
        处理上传的PPT文件，提取内容并使用 ORM 创建任务和子任务。
        此操作现在是原子性的：要么所有数据库更改都成功，要么全部回滚。

        Args:
            file (BinaryIO): 上传的文件对象
            filename (str): 文件名
            project_id_str (str): 项目ID (UUID as string)
            user_id_str (str): 用户ID (UUID as string), 将作为任务的assignee和创建者

        Returns:
            Dict[str, Any]: 处理结果，包含task_id (string representation of UUID)
        """
        try:
            # 0. 获取 Project 和 User ORM 对象 (这些需要在事务之外，因为如果它们失败，我们不想开始事务)
            try:
                project_uuid = UUID(project_id_str)
                project_obj = await Project.get_or_none(id=project_uuid)
                if not project_obj:
                    logger.error(
                        f"Project with ID {project_id_str} not found.")
                    return {
                        'success': False,
                        # Project ID not found
                        'message': f"プロジェクトID '{project_id_str}' が見つかりません",
                        'task_id': None
                    }
            except ValueError:  # Invalid UUID format
                logger.error(f"Invalid Project ID format: {project_id_str}.")
                return {
                    'success': False,
                    # Invalid Project ID format
                    'message': f"無効なプロジェクトID形式です: '{project_id_str}'",
                    'task_id': None
                }

            try:
                user_uuid = UUID(user_id_str)
                user_obj = await User.get_or_none(id=user_uuid)
                if not user_obj:
                    logger.error(f"User with ID {user_id_str} not found.")
                    return {
                        'success': False,
                        # User ID not found
                        'message': f"ユーザーID '{user_id_str}' が見つかりません",
                        'task_id': None
                    }
            except ValueError:  # Invalid UUID format
                logger.error(f"Invalid User ID format: {user_id_str}.")
                return {
                    'success': False,
                    # Invalid User ID format
                    'message': f"無効なユーザーID形式です: '{user_id_str}'",
                    'task_id': None
                }

            # 移除 self.init_db() - Tortoise ORM handles connections

            async with temp_file_manager(prefix='pptx_upload_') as manager:
                temp_dir = manager.create_temp_dir()
                local_pptx_path = os.path.join(temp_dir, filename)
                os.makedirs(os.path.dirname(local_pptx_path), exist_ok=True)
                with open(local_pptx_path, 'wb') as f:
                    file.seek(0)
                    f.write(file.read())

                unique_file_id = str(uuid.uuid4())
                # Use project_obj.id (which is a UUID) then convert to string for path construction
                s3_folder = f"uploads/{str(project_obj.id)}/{unique_file_id}"
                s3_path_main_pptx = f"{s3_folder}/{filename}"

                with open(local_pptx_path, 'rb') as f_pptx:
                    await upload_file_to_s3(f_pptx, s3_path_main_pptx, bucket_name=self.S3_BUCKET)

                from aimage_supervision.utils.extract_pptx import PPTXProcessor
                pptx_processor = PPTXProcessor()
                # Remove init_db and close_db calls as PPTXProcessor no longer manages DB connections
                # if hasattr(pptx_processor, 'init_db'):  # Check if methods exist before calling
                #     await pptx_processor.init_db()

                try:
                    extraction_result = await pptx_processor.process_pptx_file(
                        local_pptx_path,
                        temp_dir,
                        s3_folder  # This s3_folder is for images extracted from PPTX, not the PPTX itself
                    )

                    if not extraction_result or not extraction_result.get('subtasks'):
                        logger.warning(
                            f"No content extracted from {filename} by PPTXProcessor.")
                        return {
                            'success': False,
                            # Could not extract content from file
                            'message': f"ファイル '{filename}' からコンテンツを抽出できませんでした",
                            'task_id': None
                        }

                    task_name = os.path.splitext(filename)[0]
                    # Create main task using the new ORM method
                    created_task = await self.insert_task(
                        name=task_name,
                        description=extraction_result.get(
                            'first_slide_description', ''),
                        s3_path=s3_path_main_pptx,  # Path to the uploaded PPTX file itself
                        project_obj=project_obj,
                        assignee_user=user_obj
                    )

                    created_subtask_count = 0
                    for idx, subtask_raw_data in enumerate(extraction_result['subtasks']):
                        slide_number = subtask_raw_data.get(
                            'slide_page_number', idx + 1)
                        subtask_name = f"{task_name}_slide{slide_number}"
                        await self.insert_subtask(
                            parent_task=created_task,
                            order_index=idx,  # oid for subtask
                            name=subtask_name,
                            subtask_raw_content=subtask_raw_data,
                            # Assignee for subtask context (e.g. author in content)
                            assignee_user=user_obj
                        )
                        created_subtask_count += 1

                    logger.info(
                        f"Successfully created task (ID: {created_task.id}) and {created_subtask_count} subtasks for {filename} using ORM.")

                    return {
                        'success': True,
                        'task_id': str(created_task.id),  # Return string UUID
                        'subtask_count': created_subtask_count
                    }
                finally:
                    # Remove init_db and close_db calls as PPTXProcessor no longer manages DB connections
                    # if hasattr(pptx_processor, 'close_db'):  # Check if methods exist
                    #     await pptx_processor.close_db()
                    pass  # Add pass to satisfy indentation for an empty finally block

        except Exception as e:
            logger.error(
                f"Error processing uploaded file {filename} (transaction boundary): {str(e)}", exc_info=True)
            return {
                'success': False,
                'message': f"ファイルの処理中にエラーが発生しました: {str(e)}",
                'task_id': None
            }
