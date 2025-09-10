from enum import Enum
from uuid import uuid4, UUID
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import re

class User(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    phone_number: str = Field(
        ..., description="Номер телефона в международном формате, начинающийся с '+'"
    )
    first_name: str = Field(
        ..., min_length=3, max_length=50, description="Имя, от 3 до 50 символов"
    )
    last_name: str = Field(
        ..., min_length=3, max_length=50, description="Фамилия, от 3 до 50 символов"
    )

    @field_validator('phone_number')
    @classmethod
    def validate_phone_number(cls, v):
        if not re.match(r'^\+[1-9]\d{1,14}$', v):
            raise ValueError('Неверный формат номера телефона')
        return v

class UserRegister(BaseModel):
    phone_number: str = Field(
        ..., description="Номер телефона в международном формате"
    )
    password: str = Field(
        ..., min_length=5, max_length=100, description="Пароль, от 5 до 100 знаков"
    )
    first_name: str = Field(
        ..., min_length=3, max_length=50, description="Имя"
    )
    last_name: str = Field(
        ..., min_length=3, max_length=50, description="Фамилия"
    )

    @field_validator('phone_number')
    @classmethod
    def validate_phone_number(cls, v):
        if not re.match(r'^\+[1-9]\d{1,14}$', v):
            raise ValueError('Неверный формат номера телефона')
        return v

class UserLogin(BaseModel):
    phone_number: str
    password: str

class UserResponse(BaseModel):
    uuid: UUID
    phone_number: str
    first_name: str
    last_name: str
    registration_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    phone_number: Optional[str] = None
