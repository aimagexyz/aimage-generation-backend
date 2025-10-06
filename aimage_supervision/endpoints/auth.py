from fastapi import APIRouter, Body, HTTPException, Response, status
from pydantic import BaseModel

from ..middlewares.auth import (add_google_user_to_database,
                                generate_jwt_token, verify_google_token)
from ..models import User
from ..settings import USE_SECURE_COOKIES

router = APIRouter(prefix='/auth', tags=['Authentication'])


class GoogleLoginRequest(BaseModel):
    token: str


class LoginResponse(BaseModel):
    user_id: str
    # 不在响应中包含token，因为我们使用Cookie


@router.post('/google', response_model=LoginResponse)
async def google_login(request: GoogleLoginRequest, response: Response):
    """
    使用谷歌 ID token 登录

    验证谷歌 ID token，并在数据库中创建或更新用户信息
    """
    try:
        # 验证谷歌 token
        user_info = await verify_google_token(request.token)

        # 获取用户邮箱
        email = user_info.get('email')
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Googleトークンからメールアドレスが見つかりません',
            )

        # 检查用户是否已存在
        user = await User.get_or_none(email=email)

        if user:
            # 用户已存在，生成 JWT token
            jwt_token = generate_jwt_token(str(user.id))

            # 将token设置为HTTP-only Cookie
            response.set_cookie(
                key="access_token",
                value=jwt_token,
                httponly=True,
                secure=USE_SECURE_COOKIES,  # 根据环境设置secure属性
                samesite="lax",  # 防止CSRF攻击
                max_age=60 * 60 * 24 * 7,  # 7天有效期
                path="/"  # Cookie适用于整个域
            )

            return LoginResponse(user_id=str(user.id))
        else:
            # 用户不存在，创建新用户
            user_metadata = {
                'name': user_info.get('name', email.split('@')[0]),
                'picture': user_info.get('picture'),
            }

            # 创建用户
            response_data = await add_google_user_to_database(email, user_metadata)

            if not response_data:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail='ユーザーの作成に失敗しました',
                )

            # 生成 JWT token
            jwt_token = generate_jwt_token(response_data.user.id)

            # 将token设置为HTTP-only Cookie
            response.set_cookie(
                key="access_token",
                value=jwt_token,
                httponly=True,
                secure=USE_SECURE_COOKIES,  # 根据环境设置secure属性
                samesite="lax",  # 防止CSRF攻击
                max_age=60 * 60 * 24 * 7,  # 7天有效期
                path="/"  # Cookie适用于整个域
            )

            return LoginResponse(user_id=response_data.user.id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'ログインに失敗しました: {str(e)}',
        )


@router.post('/logout')
async def logout(response: Response):
    """
    登出用户，清除Cookie中的token
    """
    response.delete_cookie(
        key="access_token",
        path="/",
        httponly=True,
        secure=USE_SECURE_COOKIES,
        samesite="lax"
    )
    return {"message": "正常にログアウトしました"}
