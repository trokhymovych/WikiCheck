from datetime import datetime, timedelta, timezone
from typing import Optional, Union
from typing_extensions import Annotated

import jwt

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext  # type: ignore

from modules.constants import ALGORITHM, SECRET_KEY
from modules.entities import TokenData, User, UserInDB
from modules.users.users import fake_users_db


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def get_user(db: dict, username: str) -> Optional[UserInDB]:
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    else:
        return None


def update_user_credits(username: str, credits: int, users_db: dict = fake_users_db) -> None:
    if username in users_db:
        user_dict = users_db[username]
        user_dict["credits"] += credits
        users_db[username] = user_dict


def authenticate_user(username: str, password: str, users_db: dict = fake_users_db) -> Union[UserInDB, bool]:
    user = get_user(users_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)  # type: ignore
    if user is None:
        raise credentials_exception
    return user


async def validate_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    if current_user.credits < 1:
        raise HTTPException(status_code=402, detail="Insufficient credits")
    return current_user
