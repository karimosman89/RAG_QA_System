"""
Authentication Routes

Basic authentication system with:
- JWT token-based auth
- User registration and login
- API key authentication
- Session management
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import jwt
from passlib.context import CryptContext

from ...core.config import settings


router = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token handler
security = HTTPBearer()


class UserCreate(BaseModel):
    """User creation model."""
    email: EmailStr
    password: str
    full_name: str


class UserLogin(BaseModel):
    """User login model."""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_in: int


class User(BaseModel):
    """User model."""
    id: int
    email: str
    full_name: str
    is_active: bool = True


# Mock user database (replace with real database)
fake_users_db = {
    "demo@example.com": {
        "id": 1,
        "email": "demo@example.com",
        "full_name": "Demo User",
        "hashed_password": pwd_context.hash("demo123"),
        "is_active": True
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def authenticate_user(email: str, password: str) -> Optional[dict]:
    """Authenticate a user."""
    user = fake_users_db.get(email)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.security.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.security.secret_key, 
        algorithm=settings.security.algorithm
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT token."""
    try:
        payload = jwt.decode(
            token, 
            settings.security.secret_key, 
            algorithms=[settings.security.algorithm]
        )
        email: str = payload.get("sub")
        if email is None:
            return None
        return payload
    except jwt.PyJWTError:
        return None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise credentials_exception
    
    email: str = payload.get("sub")
    user = fake_users_db.get(email)
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    """Get the current active user."""
    if not current_user.get("is_active", False):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@router.post("/register", response_model=User, summary="Register new user")
async def register(user_data: UserCreate):
    """Register a new user."""
    # Check if user already exists
    if user_data.email in fake_users_db:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create new user
    user_id = len(fake_users_db) + 1
    hashed_password = get_password_hash(user_data.password)
    
    fake_users_db[user_data.email] = {
        "id": user_id,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "hashed_password": hashed_password,
        "is_active": True
    }
    
    return User(
        id=user_id,
        email=user_data.email,
        full_name=user_data.full_name,
        is_active=True
    )


@router.post("/login", response_model=Token, summary="Login user")
async def login(user_credentials: UserLogin):
    """Login a user and return access token."""
    user = authenticate_user(user_credentials.email, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.security.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user["email"]}, 
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.security.access_token_expire_minutes * 60
    )


@router.get("/me", response_model=User, summary="Get current user")
async def read_users_me(current_user: dict = Depends(get_current_active_user)):
    """Get current user information."""
    return User(
        id=current_user["id"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        is_active=current_user["is_active"]
    )


@router.post("/logout", summary="Logout user")
async def logout(current_user: dict = Depends(get_current_active_user)):
    """Logout current user."""
    # In a real implementation, you'd invalidate the token
    return {"message": "Successfully logged out"}


@router.get("/verify-token", summary="Verify token")
async def verify_token(current_user: dict = Depends(get_current_active_user)):
    """Verify if the current token is valid."""
    return {
        "valid": True,
        "user": {
            "email": current_user["email"],
            "full_name": current_user["full_name"]
        }
    }