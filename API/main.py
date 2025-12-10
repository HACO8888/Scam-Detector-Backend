import os
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.templating import Jinja2Templates

from authlib.integrations.starlette_client import OAuth
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from .database import get_db
from .models import User
from scam_detector import ScamDetector
from config import db_config

load_dotenv()


# -------------------------------------------------
# JWT CONFIG
# -------------------------------------------------

SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120


# -------------------------------------------------
# FASTAPI 初始化
# -------------------------------------------------

app = FastAPI(
    title="Scam Detector API",
    description="詐騙偵測系統（Google 登入 + JWT + 使用者管理 + 三分類模型）",
    version="3.0.0",
)

origins = [
    "http://localhost:3000",
    "https://scam.haco.tw",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
)

templates = Jinja2Templates(directory="API/templates")


# -------------------------------------------------
# GOOGLE OAUTH
# -------------------------------------------------

oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# -------------------------------------------------
# GLOBAL MODEL
# -------------------------------------------------

detector: Optional[ScamDetector] = None


@app.on_event("startup")
def load_model():
    global detector
    detector = ScamDetector(db_config)
    detector.create_tables()
    detector.load_model("scam_detector_model.pkl")
    print("✓ 模型已載入")


# -------------------------------------------------
# JWT 功能
# -------------------------------------------------

class Token(BaseModel):
    access_token: str
    token_type: str


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_password(raw: str, hashed: str) -> bool:
    return raw == hashed


def hash_password(raw: str) -> str:
    return raw


def get_current_user_from_token(token: str, db: Session) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token 無效")

        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="使用者不存在或已停用")

        return user

    except JWTError:
        raise HTTPException(status_code=401, detail="Token 解析錯誤")


async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    access_token_cookie: Optional[str] = Cookie(default=None, alias="access_token"),
) -> User:

    auth_header = request.headers.get("Authorization")
    token = None

    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ")[1]
    elif access_token_cookie:
        token = access_token_cookie

    if not token:
        raise HTTPException(status_code=401, detail="未登入")

    return get_current_user_from_token(token, db)


async def get_current_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="需要管理員權限")
    return user


# -------------------------------------------------
# AUTH: 帳密登入
# -------------------------------------------------

@app.post("/auth/token", response_model=Token, summary="帳密登入取得 JWT")
async def login_with_password(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == form_data.username).first()

    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="帳號不存在或停用")

    if not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="密碼錯誤")

    token = create_access_token(
        {"user_id": user.id, "email": user.email, "is_admin": user.is_admin},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return Token(access_token=token, token_type="bearer")


# -------------------------------------------------
# GOOGLE LOGIN
# -------------------------------------------------

@app.get("/auth/google/login")
async def google_login(request: Request):
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/google/callback")
async def google_callback(
    request: Request,
    db: Session = Depends(get_db),
):
    token = await oauth.google.authorize_access_token(request)
    userinfo = (await oauth.google.get("userinfo", token=token)).json()

    email = userinfo.get("email")
    sub = userinfo.get("sub")
    full_name = userinfo.get("name")

    if not email:
        raise HTTPException(status_code=400, detail="無法取得 email")

    user = db.query(User).filter(User.email == email).first()

    # 第一次登入自動成為 admin
    make_admin = (db.query(User).count() == 0)

    if not user:
        user = User(
            email=email,
            google_id=sub,
            full_name=full_name,
            is_active=True,
            is_admin=make_admin,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            last_login_at=datetime.utcnow(),
        )
        db.add(user)
    else:
        user.last_login_at = datetime.utcnow()
        if make_admin:
            user.is_admin = True

    db.commit()
    db.refresh(user)

    # 產生 JWT
    jwt_token = create_access_token(
        {"user_id": user.id, "email": user.email, "is_admin": user.is_admin},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    resp = RedirectResponse(url="/admin")
    resp.set_cookie("access_token", jwt_token, httponly=True, max_age=7200)

    return resp


# -------------------------------------------------
# 後台頁面
# -------------------------------------------------

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(
    request: Request,
    user: User = Depends(get_current_admin),
):
    return templates.TemplateResponse(
        "admin.html",
        {"request": request, "user_email": user.email},
    )


# -------------------------------------------------
# 使用者 CRUD
# -------------------------------------------------

class UserBase(BaseModel):
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False


class UserCreate(UserBase):
    password: Optional[str] = None


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    password: Optional[str] = None


class UserOut(UserBase):
    id: int

    class Config:
        orm_mode = True


@app.get("/users", summary="列出所有使用者", response_model=List[UserOut])
async def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    return db.query(User).order_by(User.id).all()


@app.post("/users", summary="新增使用者", response_model=UserOut)
async def create_user(
    body: UserCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email 已存在")

    user = User(
        email=body.email,
        full_name=body.full_name,
        is_active=body.is_active,
        is_admin=body.is_admin,
        password_hash=hash_password(body.password) if body.password else None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.put("/users/{user_id}", summary="更新使用者", response_model=UserOut)
async def update_user(
    user_id: int,
    body: UserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="使用者不存在")

    if body.full_name is not None:
        user.full_name = body.full_name
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.is_admin is not None:
        user.is_admin = body.is_admin
    if body.password:
        user.password_hash = hash_password(body.password)

    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user


@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="使用者不存在")

    db.delete(user)
    db.commit()
    return {"detail": "已刪除"}


# -------------------------------------------------
# PREDICT API
# -------------------------------------------------

class PredictRequest(BaseModel):
    message: str
    save_to_db: bool = True


class PredictResult(BaseModel):
    is_scam: bool
    scam_probability: float
    confidence: float
    warning_level: str
    features: dict
    message: str


@app.post("/predict", summary="單筆偵測", response_model=PredictResult)
async def predict_single(
    body: PredictRequest,
    user: User = Depends(get_current_user),
):
    if detector is None:
        raise HTTPException(status_code=500, detail="模型未初始化")

    result = detector.predict(body.message, save_to_db=body.save_to_db)

    return PredictResult(
        is_scam=result["is_scam"],
        scam_probability=result["scam_probability"],
        confidence=result["confidence"],
        warning_level=result["warning_level"],
        features=result["features"],
        message=body.message,
    )


# -------------------------------------------------
# 批次偵測
# -------------------------------------------------

class BatchPredictRequest(BaseModel):
    messages: List[str]
    save_to_db: bool = True


class BatchPredictResult(BaseModel):
    results: List[PredictResult]


@app.post("/predict/batch", summary="批次偵測", response_model=BatchPredictResult)
async def predict_batch(
    body: BatchPredictRequest,
    user: User = Depends(get_current_user),
):
    if detector is None:
        raise HTTPException(status_code=500, detail="模型未初始化")

    output = []
    for msg in body.messages:
        r = detector.predict(msg, save_to_db=body.save_to_db)
        output.append(
            PredictResult(
                is_scam=r["is_scam"],
                scam_probability=r["scam_probability"],
                confidence=r["confidence"],
                warning_level=r["warning_level"],
                features=r["features"],
                message=msg,
            )
        )

    return BatchPredictResult(results=output)


# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": detector is not None}
