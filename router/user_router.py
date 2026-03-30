"""
用户相关路由：注册、登录
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db_crud.user_crud import create_user, authenticate_user

router = APIRouter(prefix="/api", tags=["用户"])


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register(req: RegisterRequest):
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    if len(req.username) > 50 or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="用户名长度≤50，密码≥6位")

    user_id = await create_user(req.username, req.password)
    if user_id is None:
        raise HTTPException(status_code=409, detail="用户名已存在")
    return {"user_id": user_id, "message": "注册成功"}


@router.post("/login")
async def login(req: LoginRequest):
    user_id = await authenticate_user(req.username, req.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {"user_id": user_id, "message": "登录成功"}
