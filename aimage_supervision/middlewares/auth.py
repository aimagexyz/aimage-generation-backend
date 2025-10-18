import datetime
import uuid
from typing import Annotated, Optional

import google.auth.transport.requests
import google.oauth2.id_token
import jwt
from aiocache import cached
from fastapi import Cookie, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from aimage_supervision.enums import UserRole
from aimage_supervision.models_auth import User
from aimage_supervision.settings import get_google_client_id, get_jwt_secret

oauth2_scheme = HTTPBearer(
    scheme_name='Bearer',
    description='Add the token to your bearer authentication',
    auto_error=False,  # 不自动报错，以便可以回退到Cookie认证
)

# JWT 配置
JWT_SECRET = None
GOOGLE_CLIENT_ID = None


async def init_jwt_config():
    '''初始化 JWT 配置'''
    global JWT_SECRET, GOOGLE_CLIENT_ID
    if not JWT_SECRET:
        JWT_SECRET = get_jwt_secret()
    if not GOOGLE_CLIENT_ID:
        GOOGLE_CLIENT_ID = get_google_client_id()
    return JWT_SECRET, GOOGLE_CLIENT_ID


@cached(ttl=300)  # 缓存5分钟
async def get_uid_from_access_token(access_token: str) -> str:
    '''从 JWT token 获取用户 ID'''
    try:
        # 初始化 JWT 配置
        jwt_secret, _ = await init_jwt_config()

        # 解码 JWT token
        payload = jwt.decode(
            access_token,
            jwt_secret,
            algorithms=['HS256']
        )
        user_id = payload.get('user_id')
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='無効なトークンペイロード',
            )
        return user_id
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f'無効なアクセストークン: {str(e)}',
        )


async def verify_google_token(token: str) -> dict:
    '''验证谷歌 ID token 并返回用户信息'''
    try:
        # 初始化 JWT 配置
        _, google_client_id = await init_jwt_config()

        # 验证谷歌 ID token
        request = google.auth.transport.requests.Request()
        id_info = google.oauth2.id_token.verify_oauth2_token(
            token,
            request,
            google_client_id
        )

        # 检查 token 是否有效
        if id_info['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('不正な発行者です。')

        return id_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f'無効なGoogleトークン: {str(e)}',
        )


def generate_jwt_token(user_id: str) -> str:
    '''生成 JWT token'''
    payload = {
        'user_id': user_id,
        # 30天过期
        'exp': datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=30)
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm='HS256')


async def add_google_user_to_database(
    email: str,
    user_metadata: dict | None = None,
):
    '''将谷歌用户添加到数据库'''
    if user_metadata is None:
        user_metadata = {}

    try:
        # 生成一个新的 UUID
        user_id = str(uuid.uuid4())

        # 创建用户
        user = await User.create(
            id=user_id,
            email=email,
            display_name=user_metadata.get('name', email.split('@')[0]),
            role=UserRole.USER,
        )

        class UserResponse:
            def __init__(self, id):
                self.id = id

        class Response:
            def __init__(self, user):
                self.user = user

        return Response(UserResponse(user_id))
    except Exception as e:
        print(f'ユーザー作成エラー: {e}')
        return None


@cached(ttl=300)  # 缓存5分钟
async def get_current_user_id(
    bearer_key: Annotated[Optional[HTTPAuthorizationCredentials], Security(
        oauth2_scheme)] = None,
    access_token: Optional[str] = Cookie(None, alias="access_token")
) -> str:
    '''从 token 获取当前用户 ID，优先从Cookie获取，回退到Authorization header'''
    token = None

    # 优先从Cookie中获取token
    if access_token:
        token = access_token
    # 如果Cookie中没有，则尝试从Authorization header获取
    elif bearer_key:
        token = bearer_key.credentials

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='認証情報が提供されていません',
        )

    user_id = await get_uid_from_access_token(token)
    return user_id


@cached(ttl=300)  # 缓存5分钟
async def get_current_user(
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> User:
    '''从 token 获取当前用户'''
    user = await User.get_or_none(id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='資格情報を検証できませんでした',
        )
    return user


async def get_current_admin_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    '''获取管理员用户'''
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='このリソースにアクセスする権限がありません',
        )
    return current_user
