import os
from datetime import datetime, timedelta
from typing import List, Optional

from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware  # ← 新增這行

from .database import get_db
from .models import User
from scam_detector import ScamDetector
from config import db_config

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

app = FastAPI(
    title="Scam Detector API",
    description="詐騙偵測 API（Google 登入 + JWT + User 管理）",
    version="2.0.0",
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
)

templates = Jinja2Templates(directory="API/templates")


oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
    api_base_url="https://openidconnect.googleapis.com/v1/",  # ← 加這行
)

detector: Optional[ScamDetector] = None


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: int
    email: str
    is_admin: bool


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


class PredictRequest(BaseModel):
    message: str
    save_to_db: bool = True


class BatchPredictRequest(BaseModel):
    messages: List[str]
    save_to_db: bool = True


class PredictResult(BaseModel):
    is_scam: bool
    scam_probability: float
    confidence: float
    warning_level: str
    features: dict
    message: str


class BatchPredictResult(BaseModel):
    results: List[PredictResult]


def hash_password(raw: str) -> str:
    return raw


def verify_password(raw: str, hashed: str) -> bool:
    return raw == hashed


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user_from_token(token: str, db: Session) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="無法驗證憑證",
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        email = payload.get("email")
        is_admin = payload.get("is_admin")
        if user_id is None or email is None:
            raise credentials_exception
        TokenData(user_id=user_id, email=email, is_admin=is_admin)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise credentials_exception
    return user


async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    access_token_cookie: Optional[str] = Cookie(default=None, alias="access_token"),
) -> User:
    auth_header = request.headers.get("Authorization")
    token = None
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1]
    elif access_token_cookie:
        token = access_token_cookie
    if not token:
        raise HTTPException(status_code=401, detail="未登入")
    return get_current_user_from_token(token, db)


async def get_current_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="需要管理員權限")
    return user


@app.on_event("startup")
def startup_event() -> None:
    global detector
    detector = ScamDetector(db_config)
    detector.load_model("scam_detector_model.pkl")


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "detector_loaded": detector is not None}


@app.post("/auth/token", response_model=Token, summary="帳密登入取得 JWT")
async def login_with_password(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> Token:
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="帳號不存在或已停用")
    if not user.password_hash or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="密碼錯誤")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        {
            "user_id": user.id,
            "email": user.email,
            "is_admin": user.is_admin,
        },
        expires_delta=access_token_expires,
    )
    return Token(access_token=token, token_type="bearer")


@app.get("/auth/google/login", summary="Google 登入")
async def google_login(request: Request):
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI") or str(
        request.url_for("google_callback")
    )
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/google/callback", summary="Google 登入 Callback")
async def google_callback(
    request: Request,
    db: Session = Depends(get_db),
):
    token = await oauth.google.authorize_access_token(request)
    resp = await oauth.google.get("userinfo", token=token)
    userinfo = resp.json()

    email = userinfo.get("email")
    sub = userinfo.get("sub")
    full_name = userinfo.get("name")

    if not email:
        raise HTTPException(status_code=400, detail="無法取得 email")

    initial_admin_email = os.getenv("INITIAL_ADMIN_EMAIL")

    total_users = db.query(User).count()
    is_first_user = total_users == 0
    make_admin = bool(initial_admin_email and email == initial_admin_email) or is_first_user

    user = db.query(User).filter(User.email == email).first()
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
        user.google_id = user.google_id or sub
        # user.full_name = full_name or user.full_name
        user.last_login_at = datetime.utcnow()
        if make_admin and not user.is_admin:
            user.is_admin = True

    db.commit()
    db.refresh(user)

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    jwt_token = create_access_token(
        {
            "user_id": user.id,
            "email": user.email,
            "is_admin": user.is_admin,
        },
        expires_delta=access_token_expires,
    )

    response = RedirectResponse(url="/admin")
    response.set_cookie(
        "access_token",
        jwt_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
    return response



@app.get("/admin", response_class=HTMLResponse, summary="後台管理頁")
async def admin_page(
    request: Request,
    user: User = Depends(get_current_admin),
):
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "user_email": user.email,
            "user_is_admin": user.is_admin,
        },
    )


@app.get("/users", response_model=List[UserOut], summary="列出所有使用者")
async def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> List[UserOut]:
    users = db.query(User).order_by(User.id).all()
    return users


@app.post("/users", response_model=UserOut, summary="新增使用者")
async def create_user(
    body: UserCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> UserOut:
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
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


@app.get("/users/{user_id}", response_model=UserOut, summary="取得使用者資料")
async def get_user_detail(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> UserOut:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="使用者不存在")
    return user


@app.put("/users/{user_id}", response_model=UserOut, summary="更新使用者")
async def update_user(
    user_id: int,
    body: UserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> UserOut:
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


@app.delete("/users/{user_id}", summary="刪除使用者")
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
    return JSONResponse({"detail": "已刪除"})


@app.post("/predict", response_model=PredictResult, summary="單筆訊息詐騙偵測")
async def predict_single(
    body: PredictRequest,
    current_user: User = Depends(get_current_user),
) -> PredictResult:
    if detector is None:
        raise HTTPException(status_code=500, detail="模型尚未初始化")
    try:
        result = detector.predict(body.message, save_to_db=body.save_to_db)
        return PredictResult(
            is_scam=result["is_scam"],
            scam_probability=result["scam_probability"],
            confidence=result["confidence"],
            warning_level=result["warning_level"],
            features=result["features"],
            message=body.message,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測失敗: {e}")


@app.post("/predict/batch", response_model=BatchPredictResult, summary="多筆訊息批次偵測")
async def predict_batch(
    body: BatchPredictRequest,
    current_user: User = Depends(get_current_user),
) -> BatchPredictResult:
    if detector is None:
        raise HTTPException(status_code=500, detail="模型尚未初始化")
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages 不可為空")
    results: List[PredictResult] = []
    for msg in body.messages:
        try:
            result = detector.predict(msg, save_to_db=body.save_to_db)
            results.append(
                PredictResult(
                    is_scam=result["is_scam"],
                    scam_probability=result["scam_probability"],
                    confidence=result["confidence"],
                    warning_level=result["warning_level"],
                    features=result["features"],
                    message=msg,
                )
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"預測失敗: {e}")
    return BatchPredictResult(results=results)
