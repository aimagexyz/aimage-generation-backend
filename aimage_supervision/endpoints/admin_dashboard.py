from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response, Security, status
from tortoise.exceptions import IntegrityError

from aimage_supervision.enums import UserRole
from aimage_supervision.middlewares.auth import get_current_admin_user
from aimage_supervision.middlewares.tortoise_paginate import (
    Page, tortoise_paginate)
from aimage_supervision.models import (
    Project,
    ProjectOut,
    User,
    UserIn,
    ProjectSimpleOut,
    UserOut,
    UserAdminListOut,
)

router = APIRouter(prefix='/admin', tags=['Admin Dashboard'])


@router.get('/users')
async def list_users(
    current_user: Annotated[User | None, Security(get_current_admin_user)],
    email: Annotated[str | None, Query(
        description='Contains the email of the user',
    )] = None,
    display_name: Annotated[str | None, Query(
        description='Contains the display name of the user',
    )] = None,
    role: Annotated[UserRole | None, Query(
        description='Contains the role of the user',
    )] = None,
) -> Page[UserAdminListOut]:
    users_queryset = User.all().prefetch_related('joined_orgs')
    if email:
        users_queryset = users_queryset.filter(email__icontains=email)
    if display_name:
        users_queryset = users_queryset.filter(
            display_name__icontains=display_name)
    if role:
        users_queryset = users_queryset.filter(role=role)

    response = await tortoise_paginate(
        users_queryset,
        model=UserAdminListOut,  # type: ignore
    )
    return response


@router.get('/projects')
async def list_projects(
    current_user: Annotated[User, Security(get_current_admin_user)],
    name: Annotated[str | None, Query(
        description='Contains the name of the project',
    )] = None,
) -> Page[ProjectSimpleOut]:
    # 使用优化的查询，只获取 id 和 name 字段
    projects_queryset = Project.of_user_simple(current_user)
    if name:
        projects_queryset = projects_queryset.filter(name__icontains=name)
    response = await tortoise_paginate(
        projects_queryset,
        model=ProjectSimpleOut,  # type: ignore
    )
    return response


@router.get('/users/{user_id}/projects')
async def get_user_projects(
    current_user: Annotated[User, Security(get_current_admin_user)],
    user_id: str,
) -> Page[ProjectOut]:
    user = await User.get_or_none(id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='User not found',
        )
    projects_queryset = Project.of_user(user)
    response = await tortoise_paginate(
        projects_queryset,
        model=ProjectOut,  # type: ignore
    )
    return response


@router.post('/users', status_code=status.HTTP_201_CREATED)
async def create_user(
    current_user: Annotated[User, Security(get_current_admin_user)],
    payload: UserIn,
) -> UserOut:
    user_exists = await User.filter(email=payload.email).exists()
    if user_exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='このメールアドレスのユーザーは既に存在します',
        )

    # 直接创建用户，不需要调用add_google_user_to_database
    # 因为那个函数已经会创建用户了
    try:
        user = await User.create(
            **payload.model_dump(),
        )
        response = await UserOut.from_tortoise_orm(user)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='ユーザーの作成に失敗しました',
        )


# @router.post('/users/{user_id}/projects/{project_id}', status_code=status.HTTP_201_CREATED)
# async def add_user_to_project(
#     current_user: Annotated[User, Security(get_current_admin_user)],
#     user_id: str,
#     project_id: str,
# ):
#     user = await User.get_or_none(id=user_id)
#     project = await Project.get_or_none(id=project_id)
#     if not user or not project:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail='User or Project not found',
#         )
#     try:
#         await project.members.add(user)
#         return Response(status_code=status.HTTP_201_CREATED)
#     except IntegrityError:
#         raise HTTPException(
#             status_code=status.HTTP_409_CONFLICT,
#             detail='User is already a member of this project',
#         )


# @router.delete('/users/{user_id}/projects/{project_id}', status_code=status.HTTP_204_NO_CONTENT)
# async def remove_user_from_project(
#     current_user: Annotated[User, Security(get_current_admin_user)],
#     user_id: str,
#     project_id: str,
# ):
#     user = await User.get_or_none(id=user_id)
#     project = await Project.get_or_none(id=project_id)
#     if not user or not project:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail='User or Project not found',
#         )
#     await project.members.remove(user)
#     return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.patch('/users/{user_id}')
async def update_user_role(
    current_user: Annotated[User, Security(get_current_admin_user)],
    user_id: str,
    role: UserRole,
) -> UserOut:
    user = await User.get_or_none(id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='User not found',
        )
    user.role = role
    await user.save()
    response = await UserOut.from_tortoise_orm(user)
    return response


@router.delete('/users/{user_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    current_user: Annotated[User, Security(get_current_admin_user)],
    user_id: str,
):
    '''删除用户（仅限管理员）'''
    # 防止管理员删除自己
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='自分自身を削除することはできません',
        )

    user = await User.get_or_none(id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='ユーザーが見つかりません',
        )

    try:
        await user.delete()
        return None
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='ユーザーの削除に失敗しました',
        )
