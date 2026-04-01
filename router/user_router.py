"""
用户相关路由：注册（邮箱验证码）、登录（用户名 / 邮箱）
"""

import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

from db_crud.user_crud import (
    create_user, authenticate_user, check_email_exists,
    generate_verify_code, store_verify_code, check_verify_code,
    send_verify_email,
)

router = APIRouter(prefix="/api", tags=["用户"])

# ======================== 请求体 ========================

class SendCodeRequest(BaseModel):
    email: EmailStr


class RegisterRequest(BaseModel):
    username: str
    password: str
    email: EmailStr
    code: str          # 邮箱验证码


class LoginRequest(BaseModel):
    account: str       # 用户名 或 邮箱
    password: str


# ======================== 路由 ========================

@router.post("/send_code")
async def send_code(req: SendCodeRequest):
    """发送邮箱验证码（注册前调用）"""
    # 检查邮箱是否已注册
    if await check_email_exists(req.email):
        raise HTTPException(status_code=409, detail="该邮箱已被注册")

    code = generate_verify_code()
    store_verify_code(req.email, code)

    ok = send_verify_email(req.email, code)
    if not ok:
        raise HTTPException(status_code=500, detail="验证码发送失败，请稍后重试")

    return {"message": "验证码已发送，请查收邮箱"}


@router.post("/register")
async def register(req: RegisterRequest):
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    if len(req.username) > 50 or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="用户名长度≤50，密码≥6位")

    # 校验邮箱验证码
    if not check_verify_code(req.email, req.code):
        raise HTTPException(status_code=400, detail="验证码错误或已过期")

    user_id = await create_user(req.username, req.password, email=req.email)
    if user_id is None:
        raise HTTPException(status_code=409, detail="用户名或邮箱已存在")
    return {"user_id": user_id, "message": "注册成功"}


@router.post("/login")
async def login(req: LoginRequest):
    """登录：支持用户名或邮箱"""
    user_id = await authenticate_user(req.account, req.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="用户名/邮箱或密码错误")
    return {"user_id": user_id, "message": "登录成功"}