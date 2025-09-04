from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ReviewCreate(BaseModel):
    source_id: int = Field(..., description="ID источника отзыва", gt=0)
    text: str = Field(..., min_length=1, description="Текст отзыва")
    rating: Optional[SentimentType] = Field(None, description="Рейтинг (число)")
    product: Optional[str] = Field(None, description="Название продукта")
    datetime_review:Optional[datetime]  = Field(None, description="Дата и время создания отзыва")

class ReviewResponse(BaseModel):
    uuid: str = Field(..., description="Уникальный идентификатор отзыва (UUID)")
    source_id: int = Field(..., description="ID источника отзыва")
    text: str = Field(..., description="Текст отзыва")
    rating: Optional[int] = Field(None, description="Рейтинг (число)")
    datetime_review: datetime = Field(..., description="Временная метка создания")
    product: Optional[str] = Field(None, description="Название продукта")

    class Config:
        from_attributes = True
