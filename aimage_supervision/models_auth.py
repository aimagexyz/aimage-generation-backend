from enum import Enum
from locale import DAY_1
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from tortoise import Model, Tortoise, fields
from tortoise.contrib.pydantic.base import PydanticModel
from tortoise.contrib.pydantic.creator import pydantic_model_creator
from tortoise.expressions import Q

from aimage_supervision.clients.aws_s3 import get_s3_url_from_path
from aimage_supervision.enums import (AIClassificationStatus,
                                      AiReviewProcessingStatus, AssetStatus,
                                      BatchJobStatus, SubtaskStatus,
                                      SubtaskType, UserRole)
from aimage_supervision.schemas import (StatusHistoryEntry, SubtaskAnnotation,
                                        SubtaskContent, SubtaskStatusUser)
from aimage_supervision.settings import MAX_CONCURRENT


class TimeStampMixin:
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)


class UUIDMixin:
    id = fields.UUIDField(primary_key=True)


class User(UUIDMixin, TimeStampMixin, Model):
    email = fields.CharField(
        max_length=255,
        unique=True,
        db_db_index=True,
    )
    display_name = fields.CharField(max_length=255)
    role = fields.CharEnumField(
        UserRole,
        max_length=16,
        default=UserRole.USER,
    )

    # Relationships
    coop_projects: fields.ManyToManyRelation['Project']
    joined_orgs: fields.ManyToManyRelation['Organization']
    managed_orgs: fields.ManyToManyRelation['Organization']

    assigned_tasks: fields.ReverseRelation['Task']
    preferences: fields.ReverseRelation['UserPreferences']

    def avatar_url(self) -> str:
        return getattr(self, '_avatar_url', '')

    class _Meta:
        table = 'users'
        abstract = False
        ordering = ['created_at', 'email']

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False
        computed = ('avatar_url',)


class UserPreferences(UUIDMixin, TimeStampMixin, Model):
    """User preferences and settings (liked images moved to dedicated LikedImage model)"""
    user: fields.ForeignKeyRelation[User] = fields.ForeignKeyField(
        'auth.User',
        related_name='preferences',
        unique=True  # One preferences record per user
    )

    # User settings and configuration
    settings = fields.JSONField[dict](
        default=dict,
        field_type=dict,
        description="User settings and configuration preferences"
    )

    class _Meta:
        table = 'user_preferences'
        abstract = False
        ordering = ['created_at']

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


class LikedImage(UUIDMixin, TimeStampMixin, Model):
    """Liked images with proper S3 path storage and polymorphic source tracking"""

    # User who liked the image
    user: fields.ForeignKeyRelation[User] = fields.ForeignKeyField(
        'auth.User',
        related_name='liked_images_new',
        on_delete=fields.CASCADE,
        description="User who liked this image"
    )

    # S3 path (key) - not URL for fresh generation
    image_path = fields.CharField(
        max_length=1024,
        db_index=True,
        description="S3 path/key for the liked image"
    )

    # Polymorphic source tracking for referential integrity
    source_type = fields.CharField(
        max_length=50,
        db_index=True,
        description="Type of source object (character, item, generated_reference, etc.)"
    )
    source_id = fields.UUIDField(
        db_index=True,
        description="UUID of the source object that contains this image"
    )

    # Optional metadata for better organization
    display_name = fields.CharField(
        max_length=255,
        null=True,
        description="Optional display name for the liked image"
    )
    tags = fields.JSONField[List[str]](
        null=True,
        default=list,
        field_type=List[str],
        description="Optional tags for categorizing liked images"
    )

    # Ensure uniqueness: one user can't like the same image from the same source twice
    class _Meta:
        table = 'liked_images'
        abstract = False
        ordering = ['-created_at']
        unique_together = (('user', 'source_type', 'source_id', 'image_path'),)

    Meta = cast(type[Model.Meta], _Meta)

    def image_url(self) -> str:
        """Get the image URL. Depends on _image_url being filled by fetch_image_url"""
        return getattr(self, '_image_url', '')

    async def fetch_image_url(self, response: 'LikedImage | None' = None):
        """Fetch and set the presigned S3 URL for the image"""
        target_instance = response or self

        if not target_instance.image_path:
            setattr(target_instance, '_image_url', '')
            return

        try:
            s3_url = await get_s3_url_from_path(target_instance.image_path)
            setattr(target_instance, '_image_url', s3_url or '')
        except Exception as e:
            setattr(target_instance, '_image_url', '')
            print(
                f"Warning: Failed to generate image URL for liked image {target_instance.id}: {e}")

    class PydanticMeta:
        allow_cycles = False
        computed = ('image_url',)


class Organization(UUIDMixin, TimeStampMixin, Model):
    name = fields.CharField(max_length=255, unique=True, db_index=True)
    description = fields.TextField()
    domain = fields.CharField(max_length=255, null=True, db_index=True)

    members: fields.ManyToManyRelation[User] = fields.ManyToManyField(
        'auth.User',
        related_name='joined_orgs',
        through='organizations_members',
    )
    admins: fields.ManyToManyRelation[User] = fields.ManyToManyField(
        'auth.User',
        related_name='managed_orgs',
        through='organizations_admins',
    )

    # Relationships
    status_candidates: fields.ReverseRelation['TaskStatus']
    priority_candidates: fields.ReverseRelation['TaskPriority']
    owned_projects: fields.ReverseRelation['Project']

    coop_projects: fields.ManyToManyRelation['Project']

    async def is_user_in(self, user: 'User') -> bool:
        return await self.members.filter(id=user.id).exists()

    async def is_user_admin(self, user: 'User') -> bool:
        return await self.admins.filter(id=user.id).exists()

    async def add_member(self, user: 'User') -> None:
        await self.members.add(user)

    async def remove_member(self, user: 'User') -> None:
        if await self.is_user_admin(user):
            await self.remove_admin(user)
        await self.members.remove(user)

    async def add_admin(self, user: 'User') -> None:
        if not await self.is_user_in(user):
            await self.add_member(user)
        await self.admins.add(user)

    async def remove_admin(self, user: 'User') -> None:
        await self.admins.remove(user)

    class _Meta:
        table = 'organizations'
        abstract = False
        ordering = ['created_at',]

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


class TaskStatus(UUIDMixin, TimeStampMixin, Model):
    oid = fields.IntField(db_index=True)
    name = fields.CharField(max_length=100, db_index=True)
    owner_org: fields.ForeignKeyRelation[Organization] = fields.ForeignKeyField(
        'auth.Organization',
        related_name='status_candidates',
    )

    # Relationships
    tasks: fields.ReverseRelation['Task']
    projects: fields.ManyToManyRelation['Project']

    kanban_order: fields.ReverseRelation['TaskKanbanOrder']

    class _Meta:
        table = 'task_status_candidates'
        abstract = False
        ordering = ['oid',]

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


class TaskPriority(UUIDMixin, TimeStampMixin, Model):
    oid = fields.IntField(db_index=True)
    name = fields.CharField(max_length=100, db_index=True)
    owner_org: fields.ForeignKeyRelation[Organization] = fields.ForeignKeyField(
        'auth.Organization',
        related_name='priority_candidates',
    )

    # Relationships
    tasks: fields.ReverseRelation['Task']
    projects: fields.ManyToManyRelation['Project']

    class _Meta:
        table = 'task_priority_candidates'
        abstract = False
        ordering = ['oid',]

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


class Project(UUIDMixin, TimeStampMixin, Model):
    name = fields.CharField(max_length=255, db_index=True)
    description = fields.TextField()

    owner_org: fields.ForeignKeyRelation[Organization] = fields.ForeignKeyField(
        'auth.Organization',
        related_name='owned_projects',
    )
    coop_orgs: fields.ManyToManyRelation[Organization] = fields.ManyToManyField(
        'auth.Organization',
        related_name='coop_projects',
        through='projects_coop_orgs',
    )
    coop_members: fields.ManyToManyRelation[User] = fields.ManyToManyField(
        'auth.User',
        related_name='coop_projects',
        through='projects_coop_members',
    )

    status_candidates: fields.ManyToManyRelation[TaskStatus] = fields.ManyToManyField(
        'auth.TaskStatus',
        related_name='projects',
        through='projects_status_candidates',
    )
    priority_candidates: fields.ManyToManyRelation[TaskPriority] = fields.ManyToManyField(
        'auth.TaskPriority',
        related_name='projects',
        through='projects_priority_candidates',
    )

    # Relationships
    tasks: fields.ReverseRelation['Task']
    kanban_order: fields.ReverseRelation['TaskKanbanOrder']
    task_tags: fields.ReverseRelation['TaskTag']
    review_sets: fields.ReverseRelation['ReviewSet']

    async def is_user_owner(self, user: 'User') -> bool:
        return await self.owner_org.admins.filter(id=user.id).exists()

    async def is_user_in_coop_members(self, user: 'User') -> bool:
        return await self.coop_members.filter(id=user.id).exists()

    async def is_user_in_org(self, user: 'User') -> bool:
        owner_org = await self.owner_org
        return await owner_org.is_user_in(user)

    async def is_user_can_access(self, user: 'User') -> bool:
        return await self.is_user_in_coop_members(user) or await self.is_user_in_org(user)

    @classmethod
    def of_user(cls, user: 'User'):
        '''Get all projects of a user.'''
        return cls.filter((
            Q(owner_org__members=user) |
            Q(coop_orgs__members=user) |
            Q(coop_members=user))).distinct()

    @classmethod
    def of_user_simple(cls, user: 'User'):
        '''Get simple project data (id, name only) for a user - optimized for performance.'''
        return cls.filter((
            Q(owner_org__members=user) |
            Q(coop_orgs__members=user) |
            Q(coop_members=user))).distinct().only('id', 'name')

    class _Meta:
        table = 'projects'
        abstract = False
        ordering = ['name',]

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


class Task(UUIDMixin, TimeStampMixin, Model):
    tid = fields.CharField(max_length=255, db_index=True)
    name = fields.CharField(max_length=255, db_index=True)
    description = fields.TextField()
    s3_path = fields.CharField(max_length=1024, db_index=True)
    is_deleted = fields.BooleanField(
        default=False, db_index=True, description="Flag for soft-deleting the task")
    project: fields.ForeignKeyRelation[Project] = fields.ForeignKeyField(
        'auth.Project',
        related_name='tasks',
    )
    assignee: fields.ForeignKeyRelation[User] = fields.ForeignKeyField(
        'auth.User',
        related_name='assigned_tasks',
    )

    status: fields.ForeignKeyRelation[TaskStatus] = fields.ForeignKeyField(
        'auth.TaskStatus',
        related_name='tasks',
    )
    priority: fields.ForeignKeyRelation[TaskPriority] = fields.ForeignKeyField(
        'auth.TaskPriority',
        related_name='tasks',
    )

    # 截止日期字段
    due_date = fields.DatetimeField(
        null=True,
        description="Task due date"
    )

    # Relationships
    subtasks: fields.ReverseRelation['Subtask']
    illustration_doc: fields.ReverseRelation['IllustrationDoc']
    tags: fields.ManyToManyRelation['TaskTag'] = fields.ManyToManyField(
        'auth.TaskTag',
        related_name='tasks',
        through='tasks_tags',
    )

    async def set_assignee(self, user: User) -> None:
        if not await self.project.is_user_can_access(user):
            raise ValueError('User cannot access the project')
        self.assignee = user

    async def set_status(self, status: TaskStatus) -> None:
        if not await self.project.status_candidates.filter(id=status.id).exists():
            raise ValueError('Status is not allowed in the project')
        self.status = status

    async def set_priority(self, priority: TaskPriority) -> None:
        if not await self.project.priority_candidates.filter(id=priority.id).exists():
            raise ValueError('Priority is not allowed in the project')
        self.priority = priority

    class _Meta:
        table = 'tasks'
        abstract = False
        ordering = ['tid', 'created_at', 'name',]
        # 组合索引：常用过滤及排序字段，提升列表查询性能
        indexes = [
            ('project_id', 'status_id', 'priority_id', 'assignee_id', 'created_at'),
        ]

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


class TaskKanbanOrder(UUIDMixin, TimeStampMixin, Model):
    project: fields.ForeignKeyRelation[Project] = fields.ForeignKeyField(
        'auth.Project',
        related_name='kanban_order',
    )
    status: fields.ForeignKeyRelation[TaskStatus] = fields.ForeignKeyField(
        'auth.TaskStatus',
        related_name='kanban_order',
    )
    task_order = fields.JSONField[list[str]](
        null=True,
        field_type=list[str],
    )

    class _Meta:
        table = 'task_kanban_order'
        abstract = False
        ordering = ['status__oid',]

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


class Subtask(UUIDMixin, TimeStampMixin, Model):
    oid = fields.IntField(db_index=True)
    name = fields.CharField(max_length=255, db_index=True)
    task_type = fields.CharEnumField(SubtaskType, default=SubtaskType.PICTURE)
    description = fields.TextField(null=True)
    slide_page_number = fields.IntField(null=True)
    content = fields.JSONField[SubtaskContent](
        null=True,
        field_type=SubtaskContent,
    )
    history = fields.JSONField[list[SubtaskContent]](
        null=True,
        default=list,
        field_type=list[SubtaskContent],
    )
    annotations = fields.JSONField[list[SubtaskAnnotation]](
        null=True,
        default=list,
        field_type=list[SubtaskAnnotation],
    )
    status_history = fields.JSONField[list[StatusHistoryEntry]](
        null=True,
        default=list,
        field_type=list[StatusHistoryEntry],
    )

    character_ids = fields.JSONField[List[str]](
        null=True,
        default=list,
        field_type=List[str],
        description='A list of character UUIDs associated with this subtask.'
    )

    # 用户手动选择的角色ID列表，优先级高于AI预测的character_ids
    user_selected_character_ids = fields.JSONField[List[str]](
        null=True,
        default=list,
        field_type=List[str],
        description='A list of character UUIDs manually selected by user. Takes priority over AI predictions.'
    )

    task: fields.ForeignKeyRelation[Task] = fields.ForeignKeyField(
        'auth.Task',
        related_name='subtasks',
    )
    status = fields.CharEnumField(SubtaskStatus, default=SubtaskStatus.PENDING)

    # add ai detection json fields
    ai_detection = fields.JSONField[dict](
        null=True,
        default=dict,
        field_type=dict,
        description="AI detection result",
    )
    ai_classification_status = fields.CharEnumField(
        AIClassificationStatus,
        default=AIClassificationStatus.UNCLASSIFIED,
        description='Status of AI classification for the subtask',
    )
    # ai_review_output = fields.JSONField(  # DEPRECATED: Replaced by new AiReview model structure
    #     null=True,
    #     description="AI review results, including all viewpoints and findings. Stored as a dictionary mapping to AiReviewResult schema."
    # )

    # Relationships to new AI Review models
    ai_reviews: fields.ReverseRelation["AiReview"]

    def get_effective_character_ids(self) -> List[str]:
        """
        获取有效的角色ID列表
        优先返回用户手动选择的角色，如果用户没有选择则返回AI预测的角色
        """
        if self.user_selected_character_ids and len(self.user_selected_character_ids) > 0:
            return self.user_selected_character_ids
        return self.character_ids or []

    def version(self) -> int:
        if not hasattr(self, 'history'):
            return 1
        if not self.history:
            return 1
        return len(self.history) + 1

    async def update_status(self, new_status: SubtaskStatus, updated_by: User) -> None:
        '''Update status and record the change in history.'''
        old_status = self.status
        self.status = new_status

        if self.status_history is None:
            self.status_history = []

        self.status_history.append(StatusHistoryEntry(
            updated_at=str(self.updated_at),
            updated_by=SubtaskStatusUser(
                user_id=str(updated_by.id),
                user_name=updated_by.display_name
            ),
            status_from=old_status or None,
            status_to=new_status,
        ).model_dump())  # type: ignore
        await self.save()

    async def accept(self, updated_by: User) -> None:
        '''Set status to accepted.'''
        if self.status == SubtaskStatus.ACCEPTED:
            return
        await self.update_status(SubtaskStatus.ACCEPTED, updated_by)

    async def deny(self, updated_by: User) -> None:
        '''Set status to denied.'''
        if self.status == SubtaskStatus.DENIED:
            return
        await self.update_status(SubtaskStatus.DENIED, updated_by)

    async def reset(self, updated_by: User) -> None:
        '''Set status to pending.'''
        if self.status == SubtaskStatus.PENDING:
            return
        await self.update_status(SubtaskStatus.PENDING, updated_by)

    class _Meta:
        table = 'subtasks'
        abstract = False
        ordering = ['name',]

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False
        computed = ('version',)


class Asset(UUIDMixin, TimeStampMixin, Model):
    drive_file_id = fields.CharField(max_length=255, db_index=True)
    file_name = fields.CharField(max_length=255, db_index=True)
    s3_path = fields.CharField(max_length=1024, db_index=True)
    mime_type = fields.CharField(max_length=255, null=True)
    status = fields.CharEnumField(AssetStatus, default=AssetStatus.UPLOADING)
    author: fields.ForeignKeyRelation[User] = fields.ForeignKeyField(
        'auth.User',
        related_name='uploaded_assets',
    )
    project: fields.ForeignKeyRelation[Project] = fields.ForeignKeyField(
        'auth.Project',
        related_name='assets',
    )

    class _Meta:
        table = 'assets'
        abstract = False
        ordering = ['created_at',]

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


class Document(UUIDMixin, TimeStampMixin, Model):
    project: fields.ForeignKeyRelation[Project] = fields.ForeignKeyField(
        'auth.Project',
        related_name='documents',
    )
    file_path = fields.CharField(max_length=1024, db_index=True)
    file_name = fields.CharField(max_length=255, db_index=True)
    file_size = fields.IntField()
    file_type = fields.CharField(max_length=255, null=True)

    class _Meta:
        table = 'documents'
        abstract = False
        ordering = ['created_at']

    # get url from s3 path
    def file_url(self) -> str:
        return getattr(self, '_file_url', '')

    async def fetch_s3_url(self, response: 'Document | None' = None):
        if response is None:
            response = self
        file_url = await get_s3_url_from_path(self.file_path)

        setattr(response, '_file_url', file_url)

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False
        computed = ('file_url',)


class IP(UUIDMixin, TimeStampMixin, Model):
    """知识产権模型，如动漫、游戏等"""
    name = fields.CharField(max_length=255, db_index=True)
    description = fields.TextField(null=True)

    # Relationships
    projects: fields.ManyToManyRelation[Project] = fields.ManyToManyField(
        'auth.Project',
        related_name='related_ips',
        through='ip_projects',
    )
    characters: fields.ReverseRelation['Character']

    class _Meta:
        table = 'ips'
        abstract = False
        ordering = ['name',]

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


class Character(UUIDMixin, TimeStampMixin, Model):
    """角色模型，用于存储角色信息"""
    name = fields.CharField(max_length=255, db_index=True)
    alias = fields.CharField(max_length=255, null=True)
    description = fields.TextField(null=True)
    features = fields.TextField(null=True)
    # S3 key or relative local path like "uuid.jpg"
    image_path = fields.CharField(max_length=1024, null=True)

    # Multiple reference images for character gallery
    reference_images = fields.JSONField[list[str]](
        null=True,
        default=list,
        field_type=list[str],
        description="List of S3 paths for character reference images"
    )

    # Character concept art/design collection images
    concept_art_images = fields.JSONField[list[str]](
        null=True,
        default=list,
        field_type=list[str],
        description="List of S3 paths for character concept art and design collection images"
    )

    ip: fields.ForeignKeyNullableRelation[IP] = fields.ForeignKeyField(
        'auth.IP',
        related_name='characters',
        null=True,
        on_delete=fields.SET_NULL
    )

    project: fields.ForeignKeyRelation[Project] = fields.ForeignKeyField(
        'auth.Project',
        related_name='characters',
    )

    review_sets: fields.ManyToManyRelation['ReviewSet']

    # Reverse relation for RPD associations (defined in ReviewPointDefinition)
    associated_rpds: fields.ManyToManyRelation['ReviewPointDefinition']

    def image_url(self) -> str:
        """获取角色图片的URL。依赖于 _image_url 是否被 fetch_image_url 填充。"""
        return getattr(self, '_image_url', '')

    def gallery_image_urls(self) -> list[str]:
        """获取角色画廊图片的URL列表。依赖于 _gallery_image_urls 是否被 fetch_gallery_image_urls 填充。"""
        return getattr(self, '_gallery_image_urls', [])

    def concept_art_image_urls(self) -> list[str]:
        """获取角色设定集图片的URL列表。依赖于 _concept_art_image_urls 是否被 fetch_concept_art_image_urls 填充。"""
        return getattr(self, '_concept_art_image_urls', [])

    async def fetch_image_url(self, response: 'Character | None' = None):
        """构造图片的完整访问URL。如果是S3路径，则获取预签名URL；如果是本地相对路径，则构造成静态文件URL。"""
        target_instance = response or self

        if not target_instance.image_path:
            setattr(target_instance, '_image_url', '')
            return

        try:
            s3_url = await get_s3_url_from_path(target_instance.image_path)
            setattr(target_instance, '_image_url', s3_url or '')
        except Exception as e:
            # Log the error but don't raise it to avoid breaking the response
            # Set empty string as fallback
            setattr(target_instance, '_image_url', '')
            print(
                f"Warning: Failed to generate image URL for character {target_instance.id}: {e}")

    async def fetch_gallery_image_urls(self, response: 'Character | None' = None):
        """构造画廊图片的完整访问URL列表。"""
        target_instance = response or self
        if not target_instance.reference_images:
            setattr(target_instance, '_gallery_image_urls', [])
            return

        gallery_urls = []
        for image_path in target_instance.reference_images:
            if not image_path:
                continue

            # 检查是否是完整的URL
            if image_path.startswith('http://') or image_path.startswith('https://'):
                gallery_urls.append(image_path)
                continue

            # 检查是否可能是S3的key
            is_likely_s3_key = '/' not in image_path or not Path(
                image_path).is_absolute()
            potential_local_path = Path("uploads/characters") / image_path

            if not potential_local_path.is_file() and is_likely_s3_key:
                # 尝试作为S3路径处理
                try:
                    s3_url = await get_s3_url_from_path(image_path)
                    gallery_urls.append(s3_url)
                except Exception as e:
                    # 如果S3获取失败，跳过这张图片
                    continue
            elif potential_local_path.is_file():
                # 是本地文件, 构建静态URL
                static_url = f"/static/characters/{image_path}"
                gallery_urls.append(static_url)
            # 如果都不是，跳过这张图片

        setattr(target_instance, '_gallery_image_urls', gallery_urls)

    async def fetch_concept_art_image_urls(self, response: 'Character | None' = None):
        """构造设定集图片的完整访问URL列表。"""
        target_instance = response or self
        if not target_instance.concept_art_images:
            setattr(target_instance, '_concept_art_image_urls', [])
            return

        concept_art_urls = []
        for image_path in target_instance.concept_art_images:
            if not image_path:
                continue

            # 检查是否是完整的URL
            if image_path.startswith('http://') or image_path.startswith('https://'):
                concept_art_urls.append(image_path)
                continue

            # 检查是否可能是S3的key
            is_likely_s3_key = '/' not in image_path or not Path(
                image_path).is_absolute()
            potential_local_path = Path("uploads/characters") / image_path

            if not potential_local_path.is_file() and is_likely_s3_key:
                # 尝试作为S3路径处理
                try:
                    s3_url = await get_s3_url_from_path(image_path)
                    concept_art_urls.append(s3_url)
                except Exception as e:
                    # 如果S3获取失败，跳过这张图片
                    continue
            elif potential_local_path.is_file():
                # 是本地文件, 构建静态URL
                static_url = f"/static/characters/{image_path}"
                concept_art_urls.append(static_url)
            # 如果都不是，跳过这张图片

        setattr(target_instance, '_concept_art_image_urls', concept_art_urls)

    class _Meta:
        table = 'characters'
        abstract = False
        ordering = ['name',]

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False
        computed = ('image_url', 'gallery_image_urls',
                    'concept_art_image_urls')


class Prompt(UUIDMixin, TimeStampMixin, Model):
    project: fields.ForeignKeyRelation[Project] = fields.ForeignKeyField(
        'auth.Project',
        related_name='prompts',
    )
    name = fields.CharField(max_length=255, db_index=True)
    description = fields.TextField()
    content = fields.TextField()

    class _Meta:
        table = 'prompts'
        abstract = False
        ordering = ['created_at',]

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


# --- New AI Review System Models ---

class ReviewPointDefinition(UUIDMixin, TimeStampMixin, Model):
    key = fields.CharField(max_length=255,
                           description='e.g., "general_ng_review", "visual_review"')
    is_active = fields.BooleanField(default=True)
    is_deleted = fields.BooleanField(
        default=False, db_index=True, description="Flag for soft-deleting the RPD")
    # Add project relationship to make RPDs project-specific
    project: fields.ForeignKeyRelation[Project] = fields.ForeignKeyField(
        'auth.Project',
        related_name='review_point_definitions',
    )

    versions: fields.ReverseRelation["ReviewPointDefinitionVersion"]

    # Direct many-to-many relationship with Character
    characters: fields.ManyToManyRelation['Character'] = fields.ManyToManyField(
        'auth.Character',
        related_name='associated_rpds',
        through='rpd_characters'
    )

    class _Meta:
        table = 'review_point_definitions'
        ordering = ['key']
        # Remove unique constraint to allow multiple RPDs with same key
        # Users can now have multiple RPDs with the same key for different purposes

    Meta = cast(type[Model.Meta], _Meta)

    def __str__(self):
        return self.key


class ReviewPointDefinitionVersion(UUIDMixin, TimeStampMixin, Model):
    review_point_definition: fields.ForeignKeyRelation[ReviewPointDefinition] = fields.ForeignKeyField(
        "auth.ReviewPointDefinition", related_name="versions", on_delete=fields.CASCADE
    )
    version_number = fields.IntField()
    title = fields.CharField(max_length=255)
    user_instruction = fields.TextField(
        null=True, description="User instruction for AI review")
    description_for_ai = fields.TextField(
        description="generated prompt for AI review")
    # eng_description_for_ai: Deprecated, using description_for_ai instead
    # Logic to ensure only one active per definition needed in services
    is_active_version = fields.BooleanField(default=True)

    # Can be a User ID (UUID string) or a system identifier like "AI_refined"
    created_by = fields.CharField(max_length=255, null=True)

    # Reference images for AI review guidance
    reference_images = fields.JSONField[List[str]](
        null=True,
        default=list,
        field_type=List[str],
        description="List of S3 paths for reference images that provide visual context and guidance for AI reviews"
    )

    # Tag list for NG review
    tag_list = fields.JSONField[List[str]](
        null=True,
        default=list,
        field_type=List[str],
        description="List of tags for NG review filtering"
    )

    # Reference files for text_review (e.g., appellation table)
    reference_files = fields.JSONField[List[str]](
        null=True,
        default=list,
        field_type=List[str],
        description="List of S3 URLs for reference files (e.g., appellation table for text_review)"
    )

    # Special rules for text_review
    special_rules = fields.JSONField[List[Dict[str, Any]]](
        null=True,
        default=list,
        field_type=List[Dict[str, Any]],
        description="Special rules in JSON format: [{'speaker': '角色A', 'target': '角色B', 'alias': '特殊称呼', 'conditions': ['条件1', '条件2']}]"
    )

    # Subcategory for general_ng_review (具体形状 or 抽象类型)
    ng_subcategory = fields.CharField(
        max_length=50,
        null=True,
        description="Subcategory for general_ng_review: 'concrete_shape' or 'abstract_type'"
    )

    # RPD预处理相关字段
    rpd_type = fields.CharField(
        max_length=100,
        null=True,
        description="RPD classification type (e.g., 'right/wrong tasks', 'classification tasks', etc.)"
    )

    guidelines = fields.JSONField[dict](
        null=True,
        field_type=dict,
        description="AI-generated guidelines for the review"
    )

    constraints = fields.JSONField[dict](
        null=True,
        field_type=dict,
        description="AI-generated constraints for the review"
    )

    detector = fields.TextField(
        null=True,
        description="AI-generated detector prompt for the review"
    )

    assessor = fields.TextField(
        null=True,
        description="AI-generated assessor prompt for the review"
    )

    is_ready_for_ai_review = fields.BooleanField(
        default=True,
        db_index=True,
        description="Whether this RPD version has been preprocessed and is ready for AI review"
    )

    findings_via_version: fields.ReverseRelation["AiReviewFindingEntry"]

    @property
    def parent_key(self):
        # This assumes 'review_point_definition' is the ForeignKeyRelation field name
        # and it has been prefetched by the caller (e.g., get_active_review_point_versions).
        if hasattr(self, 'review_point_definition') and self.review_point_definition:
            return self.review_point_definition.key
        return None

    class _Meta:
        table = 'review_point_definition_versions'
        unique_together = (("review_point_definition", "version_number"),)
        ordering = ['review_point_definition__key', '-version_number']

    Meta = cast(type[Model.Meta], _Meta)

    def __str__(self):
        return f"{self.review_point_definition_id} v{self.version_number}"


class AiReview(UUIDMixin, TimeStampMixin, Model):
    subtask: fields.ForeignKeyRelation[Subtask] = fields.ForeignKeyField(
        "auth.Subtask", related_name="ai_reviews", on_delete=fields.CASCADE
    )
    version = fields.IntField()
    # Logic to ensure only one latest per subtask needed in services
    is_latest = fields.BooleanField(default=True)

    # For AiDetectedElements and other non-relational metadata from the review process
    ai_review_output_json = fields.JSONField[Optional[dict]](null=True)

    review_timestamp = fields.DatetimeField(auto_now_add=True)

    initiated_by_user: fields.ForeignKeyNullableRelation[User] = fields.ForeignKeyField(
        "auth.User", related_name="initiated_ai_reviews", null=True, on_delete=fields.SET_NULL
    )
    last_modified_by_user: fields.ForeignKeyNullableRelation[User] = fields.ForeignKeyField(
        "auth.User", related_name="modified_ai_reviews", null=True, on_delete=fields.SET_NULL
    )

    # 新增状态字段
    processing_status = fields.CharEnumField(
        AiReviewProcessingStatus,
        default=AiReviewProcessingStatus.PENDING
    )
    error_message = fields.TextField(null=True)
    processing_started_at = fields.DatetimeField(null=True)
    processing_completed_at = fields.DatetimeField(null=True)

    # 中断控制字段
    should_cancel = fields.BooleanField(default=False)

    findings: fields.ReverseRelation["AiReviewFindingEntry"]

    class _Meta:
        table = 'ai_reviews'
        unique_together = (("subtask", "version"),)
        ordering = ['subtask__oid', '-version']

    Meta = cast(type[Model.Meta], _Meta)

    def __str__(self):
        return f"Review for Subtask {self.subtask_id} v{self.version}"


class AiReviewFindingEntry(UUIDMixin, TimeStampMixin, Model):
    ai_review: fields.ForeignKeyRelation[AiReview] = fields.ForeignKeyField(
        "auth.AiReview", related_name="findings", on_delete=fields.CASCADE
    )
    review_point_definition_version: fields.ForeignKeyRelation[ReviewPointDefinitionVersion] = fields.ForeignKeyField(
        "auth.ReviewPointDefinitionVersion", related_name="findings_via_version", on_delete=fields.RESTRICT
        # RESTRICT to prevent deleting a version if findings are attached. Or SET_NULL if findings can be orphaned.
    )
    description = fields.TextField()
    severity = fields.CharField(max_length=50)  # "high", "medium", "low"
    suggestion = fields.TextField(null=True)

    area = fields.JSONField[Optional[dict]](null=True)

    reference_images = fields.JSONField[List[str]](
        null=True,
        default=list,
        field_type=List[str],
        description="List of reference images for the finding."
    )
    reference_source = fields.TextField(null=True)

    # Tag for categorizing the finding
    tag = fields.CharField(max_length=255, null=True,
                           description="Tag for categorizing the finding")

    is_ai_generated = fields.BooleanField()

    # 添加 is_fixed 字段来标记用户是否想保留这个发现
    is_fixed = fields.BooleanField(default=False, db_index=True,
                                   description="Whether this finding is marked as fixed/resolved by user")

    # Self-referential FK to link a human-modified finding to the original AI one
    original_ai_finding: fields.ForeignKeyNullableRelation["AiReviewFindingEntry"] = fields.ForeignKeyField(
        "auth.AiReviewFindingEntry", related_name="modifications", null=True, on_delete=fields.SET_NULL
    )
    modifications: fields.ReverseRelation["AiReviewFindingEntry"]

    # From schemas.FindingStatus Literal
    status = fields.CharField(max_length=100)

    # 内容类型，复用现有的 SubtaskType 枚举
    content_type = fields.CharEnumField(SubtaskType, default=SubtaskType.PICTURE,
                                        description="Type of content being reviewed")

    # 类型特定的元数据
    content_metadata = fields.JSONField[Optional[dict]](
        null=True,
        default=None,
        field_type=Optional[dict],
        description="Type-specific metadata (e.g., video timestamps, text positions)"
    )

    # Reverse relation to PromotedFinding (a finding entry can be promoted once)
    # Tortoise uses ReverseRelation for OneToOne backref
    promotion_details: fields.ReverseRelation["PromotedFinding"]

    class _Meta:
        table = 'ai_review_finding_entries'
        ordering = ['created_at']

    Meta = cast(type[Model.Meta], _Meta)

    def __str__(self):
        return f"Finding: {self.description[:50]}"


class PromotedFinding(UUIDMixin, TimeStampMixin, Model):
    # OneToOneField ensures a finding_entry is promoted at most once.
    finding_entry: fields.OneToOneRelation[AiReviewFindingEntry] = fields.OneToOneField(
        "auth.AiReviewFindingEntry", related_name="promotion_details_forward", on_delete=fields.CASCADE
        # Using a different related_name for the forward relation if 'promotion_details' is used for reverse by Tortoise
    )
    subtask_id_context = fields.UUIDField(
        description="Stores the UUID of the Subtask for context, not a direct FK to allow Subtask deletion without affecting KB.")

    promoted_by_user: fields.ForeignKeyNullableRelation[User] = fields.ForeignKeyField(
        "auth.User", related_name="promoted_findings", on_delete=fields.SET_NULL, null=True
    )
    promotion_timestamp = fields.DatetimeField(auto_now_add=True)
    notes = fields.TextField(null=True)
    tags = fields.JSONField[Optional[List[str]]](
        null=True)  # Example: ["tag1", "tag2"]
    sharing_scope = fields.CharField(max_length=100, null=True)

    class _Meta:
        table = 'promoted_findings'
        ordering = ['-promotion_timestamp']

    Meta = cast(type[Model.Meta], _Meta)

    def __str__(self):
        return f"Promoted Finding ID: {self.id} (Original: {self.finding_entry_id})"


# --- End New AI Review System Models ---


class IllustrationDoc(UUIDMixin, TimeStampMixin, Model):
    task: fields.OneToOneRelation['Task'] = fields.OneToOneField(
        'auth.Task',
        related_name='illustration_doc',
        on_delete=fields.CASCADE
    )

    # Relationships
    contents: fields.ReverseRelation['IllustrationDocContent']

    class _Meta:
        table = 'illustration_docs'
        ordering = ['created_at']

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


class IllustrationDocContent(UUIDMixin, TimeStampMixin, Model):
    doc: fields.ForeignKeyRelation[IllustrationDoc] = fields.ForeignKeyField(
        'auth.IllustrationDoc',
        related_name='contents',
        on_delete=fields.CASCADE
    )
    prompt = fields.TextField()
    tags = fields.JSONField[list[str]](
        null=True,
        default=list,
        field_type=list[str],
        description="Tags for the illustration"
    )
    image_s3_path = fields.CharField(
        max_length=1024, description="S3 path for the generated image")

    class _Meta:
        table = 'illustration_doc_contents'
        ordering = ['created_at']

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


class PDF(UUIDMixin, TimeStampMixin, Model):
    """PDF model for storing uploaded PDF files and their extraction metadata"""
    filename = fields.CharField(
        max_length=255, description="Original PDF filename")
    s3_path = fields.CharField(
        max_length=1024, db_index=True, description="S3 path where the PDF is stored")
    file_size = fields.BigIntField(description="PDF file size in bytes")
    total_pages = fields.IntField(description="Total number of pages in PDF")
    extraction_session_id = fields.CharField(
        max_length=255, null=True, description="Session ID used during extraction")
    extracted_at = fields.DatetimeField(
        description="When the PDF was processed for extraction")
    extraction_method = fields.CharField(
        max_length=50, default='pymupdf', description="Method used for extraction")
    extraction_stats = fields.JSONField[Optional[dict]](
        null=True, description="Statistics from the extraction process")

    # Relationship to project
    project: fields.ForeignKeyRelation[Project] = fields.ForeignKeyField(
        'auth.Project',
        related_name='pdfs',
        description="Project this PDF belongs to"
    )

    # User who uploaded the PDF
    uploaded_by: fields.ForeignKeyRelation[User] = fields.ForeignKeyField(
        'auth.User',
        related_name='uploaded_pdfs',
        description="User who uploaded this PDF"
    )

    # Reverse relation to extracted items
    extracted_items: fields.ReverseRelation['Item']

    async def get_pdf_url(self) -> str:
        """Get presigned URL for the PDF file"""
        from .clients.aws_s3 import get_s3_url_from_path
        return await get_s3_url_from_path(self.s3_path)

    class Meta:
        table = 'pdfs'

    class PydanticMeta:
        computed: list = []


class Item(UUIDMixin, TimeStampMixin, Model):
    """Item model for storing uploaded images and their metadata"""
    filename = fields.CharField(
        max_length=255, db_index=True, description="Original filename of the uploaded image")
    s3_path = fields.CharField(
        max_length=1024, db_index=True, description="S3 path where the image is stored")
    s3_url = fields.CharField(max_length=2048, null=True,
                              description="Pre-signed S3 URL for accessing the image")
    content_type = fields.CharField(
        max_length=100, null=True, description="MIME type of the uploaded file")
    file_size = fields.IntField(null=True, description="File size in bytes")

    # Optional metadata
    tags = fields.JSONField[list[str]](
        null=True,
        default=list,
        field_type=list[str],
        description="Tags associated with the item"
    )
    description = fields.TextField(
        null=True, description="Description of the item")

    # Relationship to project (optional)
    project: fields.ForeignKeyNullableRelation[Project] = fields.ForeignKeyField(
        'auth.Project',
        related_name='items',
        null=True,
        on_delete=fields.SET_NULL,
        description="Project this item belongs to"
    )

    # User who uploaded the item
    uploaded_by: fields.ForeignKeyRelation[User] = fields.ForeignKeyField(
        'auth.User',
        related_name='uploaded_items',
        description="User who uploaded this item"
    )

    # PDF-related fields
    source_type = fields.CharField(
        max_length=20,
        default='direct_upload',
        description="Source type: direct_upload | pdf_extracted"
    )
    source_pdf: fields.ForeignKeyNullableRelation['PDF'] = fields.ForeignKeyField(
        'auth.PDF',
        related_name='extracted_items',
        null=True,
        on_delete=fields.SET_NULL,
        description="Source PDF if this item was extracted from a PDF"
    )
    pdf_page_number = fields.IntField(
        null=True,
        description="PDF page number (1-based) if extracted from PDF"
    )
    pdf_image_index = fields.IntField(
        null=True,
        description="Image index within the PDF page (0-based) if extracted from PDF"
    )

    def image_url(self) -> str:
        """Get the image URL. Depends on _image_url being filled by fetch_image_url"""
        return getattr(self, '_image_url', '')

    async def fetch_image_url(self, response: 'Item | None' = None):
        """Fetch and set the presigned S3 URL for the image"""
        target_instance = response or self

        if not target_instance.s3_path:
            setattr(target_instance, '_image_url', '')
            return

        try:
            s3_url = await get_s3_url_from_path(target_instance.s3_path)
            setattr(target_instance, '_image_url', s3_url or '')
        except Exception as e:
            setattr(target_instance, '_image_url', '')
            print(
                f"Warning: Failed to generate image URL for item {target_instance.id}: {e}")

    async def get_pdf_info(self) -> 'PDF | None':
        """Get the associated PDF information if this item was extracted from a PDF"""
        if self.source_type == 'pdf_extracted' and self.source_pdf:
            await self.fetch_related('source_pdf')
            return self.source_pdf
        return None

    class _Meta:
        table = 'items'
        ordering = ['created_at']

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False
        computed = ('image_url',)


class BatchProcessJob(UUIDMixin, TimeStampMixin, Model):
    """批处理任务模型，用于记录和跟踪批量处理任务的状态"""
    batch_id = fields.CharField(max_length=255, db_index=True,
                                description="批次标识符，用于关联相关的处理任务")
    job_name = fields.CharField(max_length=255, db_index=True,
                                description="任务名称")
    job_type = fields.CharField(max_length=100, db_index=True,
                                description="任务类型，如ai_review_cr_check")
    status = fields.CharEnumField(BatchJobStatus, default=BatchJobStatus.PENDING,
                                  description="批处理任务状态")

    # 创建者信息
    created_by: fields.ForeignKeyNullableRelation[User] = fields.ForeignKeyField(
        'auth.User',
        related_name='created_batch_jobs',
        null=True,
        on_delete=fields.SET_NULL,
        description="创建该批处理任务的用户"
    )

    # 项目关联（可选）
    project: fields.ForeignKeyNullableRelation[Project] = fields.ForeignKeyField(
        'auth.Project',
        related_name='batch_jobs',
        null=True,
        on_delete=fields.SET_NULL,
        description="关联的项目"
    )

    # 任务统计信息
    total_items = fields.IntField(default=0, description="总任务数量")
    processed_items = fields.IntField(default=0, description="已处理任务数量")
    successful_items = fields.IntField(default=0, description="成功处理的任务数量")
    failed_items = fields.IntField(default=0, description="失败的任务数量")

    # 时间信息
    started_at = fields.DatetimeField(null=True, description="任务开始时间")
    completed_at = fields.DatetimeField(null=True, description="任务完成时间")

    # 配置信息
    max_concurrent = fields.IntField(
        default=MAX_CONCURRENT, description="最大并发数")

    # JSON字段存储复杂数据
    parameters = fields.JSONField[Optional[dict]](
        null=True,
        default=dict,
        field_type=dict,
        description="任务参数，以JSON格式存储"
    )

    results = fields.JSONField[Optional[dict]](
        null=True,
        default=dict,
        field_type=dict,
        description="任务结果，以JSON格式存储"
    )

    # 错误信息
    error_message = fields.TextField(null=True, description="错误信息")

    # 进度计算方法
    def progress_percentage(self) -> float:
        """计算任务进度百分比"""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

    def success_rate(self) -> float:
        """计算成功率"""
        if self.processed_items == 0:
            return 0.0
        return (self.successful_items / self.processed_items) * 100

    def is_completed(self) -> bool:
        """检查任务是否已完成"""
        return self.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED, BatchJobStatus.CANCELLED]

    def duration_seconds(self) -> Optional[float]:
        """计算任务持续时间（秒）"""
        if not self.started_at:
            return None
        end_time = self.completed_at or self.updated_at
        return (end_time - self.started_at).total_seconds()

    class _Meta:
        table = 'batch_process_jobs'
        abstract = False
        ordering = ['-created_at']

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False
        computed = ('progress_percentage', 'success_rate',
                    'is_completed', 'duration_seconds')


class GeneratedReference(UUIDMixin, TimeStampMixin, Model):
    """AI-generated reference images - MVP version"""

    # Core fields only
    base_prompt = fields.TextField()
    enhanced_prompt = fields.TextField()
    image_path = fields.CharField(max_length=1024)  # S3 key

    # Simple tags as JSON (no complex relationships)
    # {"style": "anime", "pose": "standing"}
    tags = fields.JSONField[dict](default=dict)

    # Basic relationships
    project: fields.ForeignKeyRelation[Project] = fields.ForeignKeyField(
        'auth.Project', related_name='generated_references'
    )
    created_by: fields.ForeignKeyRelation[User] = fields.ForeignKeyField(
        'auth.User', related_name='generated_references'
    )

    def image_url(self) -> str:
        return getattr(self, '_image_url', '')

    async def fetch_image_url(self):
        if self.image_path:
            self._image_url = await get_s3_url_from_path(self.image_path)
        else:
            self._image_url = ''

    class _Meta:
        table = 'generated_references'
        abstract = False
        ordering = ['-created_at']

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False
        computed = ('image_url',)


class TaskTag(UUIDMixin, TimeStampMixin, Model):
    name = fields.CharField(max_length=255, db_index=True)
    project: fields.ForeignKeyRelation[Project] = fields.ForeignKeyField(
        'auth.Project',
        related_name='task_tags',
    )

    # Relationships
    tasks: fields.ManyToManyRelation['Task']
    review_sets: fields.ManyToManyRelation['ReviewSet']

    class _Meta:
        table = 'task_tags'
        unique_together = (('name', 'project'),)
        ordering = ['name']

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False


class ReviewSet(UUIDMixin, TimeStampMixin, Model):
    name = fields.CharField(max_length=255)
    description = fields.TextField(null=True)
    project: fields.ForeignKeyRelation[Project] = fields.ForeignKeyField(
        'auth.Project',
        related_name='review_sets'
    )

    # Relationships
    rpds: fields.ManyToManyRelation['ReviewPointDefinition'] = fields.ManyToManyField(
        'auth.ReviewPointDefinition',
        related_name='review_sets',
        through='review_sets_rpds'
    )
    characters: fields.ManyToManyRelation['Character'] = fields.ManyToManyField(
        'auth.Character',
        related_name='review_sets',
        through='review_sets_characters'
    )
    task_tags: fields.ManyToManyRelation['TaskTag'] = fields.ManyToManyField(
        'auth.TaskTag',
        related_name='review_sets',
        through='review_sets_task_tags'
    )

    class _Meta:
        table = 'review_sets'
        unique_together = (("name", "project"),)
        ordering = ['name']

    Meta = cast(type[Model.Meta], _Meta)

    class PydanticMeta:
        allow_cycles = False

# --- End Review Set and Task Tag Models ---


Tortoise.init_models(['aimage_supervision.models_auth'], 'auth')


if TYPE_CHECKING:
    class UserIn(User, PydanticModel):  # type:ignore[misc]
        pass

    class UserOut(User, PydanticModel):  # type:ignore[misc]
        pass

    class OrganizationIn(Organization, PydanticModel):  # type:ignore[misc]
        pass

    class OrganizationOut(Organization, PydanticModel):  # type:ignore[misc]
        pass

    class TaskStatusIn(TaskStatus, PydanticModel):  # type:ignore[misc]
        pass

    class TaskStatusOut(TaskStatus, PydanticModel):  # type:ignore[misc]
        pass

    class TaskPriorityIn(TaskPriority, PydanticModel):  # type:ignore[misc]
        pass

    class TaskPriorityOut(TaskPriority, PydanticModel):  # type:ignore[misc]
        pass

    class ProjectIn(Project, PydanticModel):  # type:ignore[misc]
        pass

    class ProjectOut(Project, PydanticModel):  # type:ignore[misc]
        pass

    class ProjectSimpleOut(Project, PydanticModel):  # type:ignore[misc]
        pass

    class TaskIn(Task, PydanticModel):  # type:ignore[misc]
        pass

    class TaskOut(Task, PydanticModel):  # type:ignore[misc]
        pass

    class TaskKanbanOrderIn(TaskKanbanOrder, PydanticModel):  # type: ignore[misc] # noqa
        pass

    class TaskKanbanOrderOut(TaskKanbanOrder, PydanticModel):  # type: ignore[misc] # noqa
        pass

    class SubtaskIn(Subtask, PydanticModel):  # type:ignore[misc]
        pass

    class SubtaskOut(Subtask, PydanticModel):  # type:ignore[misc]
        pass

    class SubtaskDetail(Subtask, PydanticModel):  # type:ignore[misc]
        pass

    class AssetIn(Asset, PydanticModel):  # type:ignore[misc]
        pass

    class AssetOut(Asset, PydanticModel):  # type:ignore[misc]
        pass

    class DocumentIn(Document, PydanticModel):  # type:ignore[misc]
        pass

    class DocumentOut(Document, PydanticModel):  # type:ignore[misc]
        pass

    class IPIn(IP, PydanticModel):  # type:ignore[misc]
        pass

    class IPOut(IP, PydanticModel):  # type:ignore[misc]
        pass

    class CharacterIn(Character, PydanticModel):  # type:ignore[misc]
        pass

    class CharacterOut(Character, PydanticModel):  # type:ignore[misc]
        pass

    class CharacterDetail(Character, PydanticModel):  # type:ignore[misc]
        pass

    # type:ignore[misc]
    class IllustrationDocIn(IllustrationDoc, PydanticModel):
        pass

    # type:ignore[misc]
    class IllustrationDocOut(IllustrationDoc, PydanticModel):
        pass

    # type:ignore[misc]
    class IllustrationDocContentIn(IllustrationDocContent, PydanticModel):
        pass

    # type:ignore[misc]
    class IllustrationDocContentOut(IllustrationDocContent, PydanticModel):
        pass

    class ItemIn(Item, PydanticModel):  # type:ignore[misc]
        pass

    class ItemOut(Item, PydanticModel):  # type:ignore[misc]
        pass

    # type:ignore[misc]
    class BatchProcessJobIn(BatchProcessJob, PydanticModel):
        pass

    # type:ignore[misc]
    class BatchProcessJobOut(BatchProcessJob, PydanticModel):
        pass

    # type:ignore[misc]
    class GeneratedReferenceIn(GeneratedReference, PydanticModel):
        pass

    # type:ignore[misc]
    class GeneratedReferenceOut(GeneratedReference, PydanticModel):
        pass

    # type:ignore[misc]
    class UserPreferencesIn(UserPreferences, PydanticModel):
        pass

    # type:ignore[misc]
    class UserPreferencesOut(UserPreferences, PydanticModel):
        pass

    class LikedImageIn(LikedImage, PydanticModel):  # type:ignore[misc]
        pass

    class LikedImageOut(LikedImage, PydanticModel):  # type:ignore[misc]
        pass

else:
    UserIn = pydantic_model_creator(
        User,
        name='UserIn',
        include=('email', 'display_name'),
    )
    UserOut = pydantic_model_creator(
        User,
        name='UserOut',
        exclude=(
            'coop_projects',
            'joined_orgs',
            'managed_orgs',
            'assigned_tasks',
            'uploaded_assets',
            'created_batch_jobs',
            'generated_references',
            'initiated_ai_reviews',
            'liked_images_new',
            'modified_ai_reviews',
            'preferences',
            'promoted_findings',
            'uploaded_items',
        ),
    )

    OrganizationIn = pydantic_model_creator(
        Organization,
        name='OrganizationIn',
        exclude_readonly=True,
    )
    OrganizationOut = pydantic_model_creator(
        Organization,
        name='OrganizationOut',
        exclude=(
            'members',
            'admins',
            'status_candidates',
            'priority_candidates',
            'owned_projects',
            'coop_projects',
        ),
    )
    # Lightweight organization for list views: only id and name
    OrganizationSimpleOut = pydantic_model_creator(
        Organization,
        name='OrganizationSimpleOut',
        include=('id', 'name'),
    )

    TaskStatusIn = pydantic_model_creator(
        TaskStatus,
        name='TaskStatusIn',
        exclude_readonly=True,
    )
    TaskStatusOut = pydantic_model_creator(
        TaskStatus,
        name='TaskStatusOut',
        exclude=(
            'tasks',
            'projects',
            'owner_org',
            'kanban_order',
        ),
    )

    TaskPriorityIn = pydantic_model_creator(
        TaskPriority,
        name='TaskPriorityIn',
        exclude_readonly=True,
    )
    TaskPriorityOut = pydantic_model_creator(
        TaskPriority,
        name='TaskPriorityOut',
        exclude=(
            'tasks',
            'projects',
            'owner_org',
        ),
    )

    ProjectIn = pydantic_model_creator(
        Project,
        name='ProjectIn',
        exclude_readonly=True,
    )
    ProjectOut = pydantic_model_creator(
        Project,
        name='ProjectOut',
        exclude=(
            'owner_org.members',
            'owner_org.admins',
            'owner_org.priority_candidates',
            'owner_org.status_candidates',
            'coop_orgs',
            'coop_members',
            'status_candidates',
            'priority_candidates',
            'tasks',
            'kanban_order',
        ),
    )
    ProjectSimpleOut = pydantic_model_creator(
        Project,
        name='ProjectSimpleOut',
        include=('id', 'name'),
    )

    # Admin user list lightweight output, including joined_orgs in a slim form
    # We keep only id/email/display_name/role for user and shrink joined_orgs
    UserAdminListOut = pydantic_model_creator(
        User,
        name='UserAdminListOut',
        include=('id', 'email', 'display_name', 'role', 'joined_orgs'),
        exclude=(
            # Slim down joined_orgs to id & name only
            'joined_orgs.description',
            'joined_orgs.domain',
            'joined_orgs.members',
            'joined_orgs.admins',
            'joined_orgs.status_candidates',
            'joined_orgs.priority_candidates',
            'joined_orgs.owned_projects',
            'joined_orgs.coop_projects',
            'joined_orgs.created_at',
            'joined_orgs.updated_at',
        ),
    )

    TaskIn = pydantic_model_creator(
        Task,
        name='TaskIn',
        include=(
            'tid',
            'name',
            'description',
            'status_id',
            'priority_id',
            'assignee_id',
            'project_id',
        ),
    )
    TaskOut = pydantic_model_creator(
        Task,
        name='TaskOut',
        exclude=(
            'project',
            'assignee.created_at',
            'assignee.updated_at',
            'assignee.role',
            'assignee.joined_orgs',
            'assignee.managed_orgs',
            'assignee.coop_projects',
            'assignee.uploaded_assets',
            'assignee.assigned_tasks',
            'assignee.created_batch_jobs',
            'assignee.generated_references',
            'assignee.initiated_ai_reviews',
            'assignee.liked_images_new',
            'assignee.modified_ai_reviews',
            'assignee.preferences',
            'assignee.promoted_findings',
            'assignee.uploaded_items',
            'status',
            'priority',
            'subtasks',
        ),
    )

    TaskKanbanOrderIn = pydantic_model_creator(
        TaskKanbanOrder,
        name='TaskKanbanOrderIn',
        exclude_readonly=True,
        exclude=(
            'project_id',
            'status_id',
        ),
    )
    TaskKanbanOrderOut = pydantic_model_creator(
        TaskKanbanOrder,
        name='TaskKanbanOrderOut',
        exclude=(
            'project',
            'status',
        ),
    )

    SubtaskIn = pydantic_model_creator(
        Subtask,
        name='SubtaskIn',
        exclude_readonly=True,
    )
    SubtaskOut = pydantic_model_creator(
        Subtask,
        name='SubtaskOut',
        exclude=(
            'task',
            'history',
            'ai_reviews',  # 列表场景不需要AI review数据，避免性能问题
        ),
    )
    SubtaskDetail = pydantic_model_creator(
        Subtask,
        name='SubtaskDetail',
        exclude=(
            'task.project',
            'task.assignee.created_at',
            'task.assignee.updated_at',
            'task.assignee.role',
            'task.assignee.joined_orgs',
            'task.assignee.managed_orgs',
            'task.assignee.coop_projects',
            'task.assignee.uploaded_assets',
            'task.assignee.assigned_tasks',
            'task.assignee.created_batch_jobs',
            'task.assignee.generated_references',
            'task.assignee.initiated_ai_reviews',
            'task.assignee.liked_images_new',
            'task.assignee.modified_ai_reviews',
            'task.assignee.preferences',
            'task.assignee.promoted_findings',
            'task.assignee.uploaded_items',
            'task.status',
            'task.priority',
            # 只排除ai_reviews中用户的厚重字段，保留基本信息（id, email, display_name等）
            'ai_reviews.initiated_by_user.created_batch_jobs',
            'ai_reviews.initiated_by_user.generated_references',
            'ai_reviews.initiated_by_user.initiated_ai_reviews',
            'ai_reviews.initiated_by_user.liked_images_new',
            'ai_reviews.initiated_by_user.modified_ai_reviews',
            'ai_reviews.initiated_by_user.preferences',
            'ai_reviews.initiated_by_user.promoted_findings',
            'ai_reviews.initiated_by_user.uploaded_items',
            'ai_reviews.initiated_by_user.joined_orgs',
            'ai_reviews.initiated_by_user.managed_orgs',
            'ai_reviews.initiated_by_user.coop_projects',
            'ai_reviews.initiated_by_user.uploaded_assets',
            'ai_reviews.initiated_by_user.assigned_tasks',
            'ai_reviews.last_modified_by_user.created_batch_jobs',
            'ai_reviews.last_modified_by_user.generated_references',
            'ai_reviews.last_modified_by_user.initiated_ai_reviews',
            'ai_reviews.last_modified_by_user.liked_images_new',
            'ai_reviews.last_modified_by_user.modified_ai_reviews',
            'ai_reviews.last_modified_by_user.preferences',
            'ai_reviews.last_modified_by_user.promoted_findings',
            'ai_reviews.last_modified_by_user.uploaded_items',
            'ai_reviews.last_modified_by_user.joined_orgs',
            'ai_reviews.last_modified_by_user.managed_orgs',
            'ai_reviews.last_modified_by_user.coop_projects',
            'ai_reviews.last_modified_by_user.uploaded_assets',
            'ai_reviews.last_modified_by_user.assigned_tasks',
        ),
    )

    AssetIn = pydantic_model_creator(
        Asset,
        name='AssetIn',
        include=('drive_file_id', 'file_name', 'mime_type'),
    )
    AssetOut = pydantic_model_creator(
        Asset,
        name='AssetOut',
        exclude=(
            'author',
        ),
    )

    DocumentIn = pydantic_model_creator(
        Document,
        name='DocumentIn',
        include=('file_path', 'file_name', 'file_size', 'file_type'),
    )
    DocumentOut = pydantic_model_creator(
        Document,
        name='DocumentOut',
        exclude=(
            'project',
        ),
    )

    IPIn = pydantic_model_creator(
        IP,
        name='IPIn',
        exclude_readonly=True,
    )
    IPOut = pydantic_model_creator(
        IP,
        name='IPOut',
        exclude=(
            'projects',
            'characters',
        ),
    )

    CharacterIn = pydantic_model_creator(
        Character,
        name='CharacterIn',
        exclude_readonly=True,
    )
    CharacterOut = pydantic_model_creator(
        Character,
        name='CharacterOut',
        exclude=(
            'project',
        ),
    )
    CharacterDetail = pydantic_model_creator(
        Character,
        name='CharacterDetail',
        exclude=(
            'project',
        ),
    )

    IllustrationDocIn = pydantic_model_creator(
        IllustrationDoc,
        name='IllustrationDocIn',
        include=('task_id',),
    )
    IllustrationDocOut = pydantic_model_creator(
        IllustrationDoc,
        name='IllustrationDocOut',
        exclude=('contents',),
    )
    IllustrationDocContentIn = pydantic_model_creator(
        IllustrationDocContent,
        name='IllustrationDocContentIn',
        include=('doc_id', 'prompt', 'tags', 'image_s3_path'),
    )
    IllustrationDocContentOut = pydantic_model_creator(
        IllustrationDocContent,
        name='IllustrationDocContentOut',
        exclude=('doc',)
    )

    BatchProcessJobIn = pydantic_model_creator(
        BatchProcessJob,
        name='BatchProcessJobIn',
        include=(
            'batch_id', 'job_name', 'job_type', 'total_items',
            'max_concurrent', 'parameters', 'created_by_id', 'project_id'
        ),
    )
    BatchProcessJobOut = pydantic_model_creator(
        BatchProcessJob,
        name='BatchProcessJobOut',
        exclude=(
            'created_by',
            'project',
            'parameters',
        ),
    )

    GeneratedReferenceIn = pydantic_model_creator(
        GeneratedReference,
        name='GeneratedReferenceIn',
        exclude_readonly=True,
    )
    GeneratedReferenceOut = pydantic_model_creator(
        GeneratedReference,
        name='GeneratedReferenceOut',
        exclude=(
            'project',
            'created_by',
        ),
    )

    UserPreferencesIn = pydantic_model_creator(
        UserPreferences,
        name='UserPreferencesIn',
        exclude_readonly=True,
    )
    UserPreferencesOut = pydantic_model_creator(
        UserPreferences,
        name='UserPreferencesOut',
        exclude=(
            'user',
        ),
    )

    LikedImageIn = pydantic_model_creator(
        LikedImage,
        name='LikedImageIn',
        exclude_readonly=True,
    )
    LikedImageOut = pydantic_model_creator(
        LikedImage,
        name='LikedImageOut',
        exclude=(
            'user',
        ),
    )
