from typing import Annotated

from aiocache import cached
from fastapi import APIRouter, HTTPException, Security, status

from ..middlewares.auth import (get_current_admin_user, get_current_user,
                                get_current_user_id)
from ..models import User, UserIn, UserOut

router = APIRouter(prefix='/users', tags=['Users'])


@router.post('', status_code=status.HTTP_201_CREATED)
@cached(ttl=300)  # 缓存5分钟
async def auto_register_user(
    user_id: Annotated[str, Security(get_current_user_id)],
    payload: UserIn,
) -> UserOut:
    '''Auto register user if not exists.'''
    # 使用 get_or_create 模式，避免重复创建
    print(f'user_id: {user_id}')
    user = await User.get_or_none(id=user_id)
    if user:
        response = await UserOut.from_tortoise_orm(user)
        return response

    # 检查邮箱是否已存在
    user_exists = await User.filter(email=payload.email).exists()
    if user_exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='User with this email already exists',
        )

    # 创建新用户
    user = await User.create(
        id=user_id,
        **payload.model_dump(),
    )
    response = await UserOut.from_tortoise_orm(user)
    return response


@router.get('/me')
@cached(ttl=300)  # 缓存5分钟
async def retrieve_current_user(
    user: Annotated[User, Security(get_current_user)],
) -> UserOut:
    print(f'user: {user}')
    response = await UserOut.from_tortoise_orm(user)
    return response


@router.get('/{user_id}')
async def retrieve_user(
    current_user: Annotated[User, Security(get_current_user)],
    user_id: str,
) -> UserOut:
    user = await User.get_or_none(id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='User not found',
        )

    response = await UserOut.from_tortoise_orm(user)
    return response
