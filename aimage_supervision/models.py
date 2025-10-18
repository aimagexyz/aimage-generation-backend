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
    project_id = fields.UUIDField(db_index=True)
    created_by_user_id = fields.UUIDField(db_index=True)

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


Tortoise.init_models(['aimage_supervision.models'], 'models')


if TYPE_CHECKING:
    class GeneratedReferenceIn(GeneratedReference, PydanticModel):  # type:ignore[misc]
        pass

    class GeneratedReferenceOut(GeneratedReference, PydanticModel):  # type:ignore[misc]
        pass
else:
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
