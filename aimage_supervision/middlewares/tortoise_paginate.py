__all__ = ['tortoise_paginate']

import asyncio
from typing import Any, Sequence, Type, TypeVar

from fastapi import Query
from fastapi_pagination import Page as DefaultPage
from fastapi_pagination.api import apply_items_transformer, create_page
from fastapi_pagination.bases import AbstractParams
from fastapi_pagination.customization import (CustomizedPage, UseName,
                                              UseParamsFields)
from fastapi_pagination.ext.utils import generic_query_apply_params
from fastapi_pagination.types import AdditionalData, AsyncItemsTransformer
from fastapi_pagination.utils import verify_params
from tortoise.contrib.pydantic.base import PydanticModel, _get_fetch_fields
from tortoise.models import Model
from tortoise.query_utils import Prefetch
from tortoise.queryset import QuerySet

from aimage_supervision.clients.aws_s3 import get_path_with_s3_url

T = TypeVar('T')

Page = CustomizedPage[
    DefaultPage[T],
    UseName('Page'),
    UseParamsFields(size=Query(
        default=300, ge=1, le=1000,
        description='The number of items to return per page',
    )),
]

TModel = TypeVar('TModel', bound=Model)


def _generate_query(
    query: QuerySet[TModel],
    prefetch_related: bool | list[str | Prefetch],
) -> QuerySet[TModel]:
    if prefetch_related:
        if prefetch_related is True:
            prefetch_related = [*query.model._meta.fetch_fields]
        return query.prefetch_related(*prefetch_related)
    return query


async def tortoise_to_pydantic(cls: PydanticModel, obj: 'Model'):
    # Get fields needed to fetch
    try:
        # Try to get fetch fields for Tortoise-derived models
        fetch_fields = _get_fetch_fields(
            cls, cls.model_config['orig_model'])  # type: ignore
        # Fetch fields
        await obj.fetch_related(*fetch_fields)
    except (KeyError, AttributeError):
        # Handle cases where model_config doesn't have 'orig_model'
        # This can happen with simplified models that don't need related field fetching
        pass
    
    return cls.model_validate(obj)


async def add_s3_url_computed_fields(objs: Sequence[PydanticModel], model: Model):
    computed_fields: list[str] = []

    if hasattr(model, 'PydanticMeta') and hasattr(model.PydanticMeta, 'computed'):  # type: ignore
        # Get computed fields from PydanticMeta
        computed_fields = list(model.PydanticMeta.computed)  # type: ignore

    if not computed_fields:
        # No computed fields to fetch
        return

    # Workaround for Fetching S3 URLs
    # Fetch S3 URLs for computed fields
    s3_url_tasks = [
        get_path_with_s3_url(getattr(obj, field.replace('url', 'path'), None))
        for obj in objs
        for field in computed_fields
        if field.endswith('url')
    ]
    s3_urls = await asyncio.gather(*s3_url_tasks)
    s3_urls_map = {path: url for path, url in s3_urls}

    # Set S3 URLs to computed fields
    for obj in objs:
        for field in computed_fields:
            if field.endswith('url'):
                source_field = field.replace('url', 'path')
                s3_path = getattr(obj, source_field, None)
                if s3_path:
                    target_field = '_' + field
                    setattr(obj, target_field, s3_urls_map.get(s3_path))


async def tortoise_paginate(
    query: QuerySet[TModel] | Type[TModel],
    params: AbstractParams | None = None,
    prefetch_related: bool | list[str | Prefetch] = False,
    *,
    transformer: AsyncItemsTransformer | None = None,
    additional_data: AdditionalData | None = None,
    total: int | None = None,
    model: PydanticModel,
) -> Any:
    params, raw_params = verify_params(params, 'limit-offset')

    if not isinstance(query, QuerySet):
        query = query.all()

    if total is None and raw_params.include_total:
        total = await query.count()

    items = await generic_query_apply_params(_generate_query(query, prefetch_related), raw_params).all()

    t_items = await apply_items_transformer(items, transformer, async_=True)

    # Tortoise ORM Model to Pydantic Model
    if model:
        t_items = await asyncio.gather(*(tortoise_to_pydantic(model, item) for item in t_items))

    # Fetch S3 URLs for computed fields
    await add_s3_url_computed_fields(t_items, query.model)  # type: ignore

    return create_page(
        t_items,
        total=total,
        params=params,
        **(additional_data or {}),
    )
