from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"


class ReviewCreate(BaseModel):
    source: str = Field(default="API", description="Источник отзыва")
    text: str = Field(..., min_length=1, description="Текст отзыва")
    rating: Optional[SentimentType] = Field(None, description="Тональность отзыва")
    product: Optional[str] = Field(None, description="Название продукта")
    gender: Optional[Gender] = Field(None, description="Пол написавшего")
    city: Optional[str] = Field(None, description="Город")
    datetime_review: Optional[datetime] = Field(None, description="Дата и время создания отзыва")


class ReviewResponse(BaseModel):
    uuid: UUID = Field(..., description="Уникальный идентификатор отзыва")
    source: str = Field(..., description="Источник отзыва")
    text: str = Field(..., description="Текст отзыва")
    rating: Optional[SentimentType] = Field(None, description="Тональность отзыва")
    product: Optional[str] = Field(None, description="Название продукта")
    gender: Optional[Gender] = Field(None, description="Пол написавшего")
    city: Optional[str] = Field(None, description="Город")
    datetime_review: datetime = Field(..., description="Дата и время создания отзыва")
    created_at: datetime = Field(..., description="Время добавления в систему")

    class Config:
        from_attributes = True
        use_enum_values = True
