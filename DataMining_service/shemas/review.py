# shemas/review.py
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ReviewCreate(BaseModel):
    """Схема для создания сырого отзыва (raw_reviews)"""

    source: str = Field(
        default="API",
        description="Источник отзыва (banki.ru, sravni.ru, API и т.д.)"
    )

    text: str = Field(
        ...,
        min_length=1,
        description="Текст отзыва"
    )

    gender: Optional[str] = Field(
        None,
        description="Пол автора: М (мужчина) или Ж (женщина)"
    )

    datetime_review: Optional[datetime] = Field(
        None,
        description="Дата и время написания отзыва"
    )

    city: Optional[str] = Field(
        None,
        max_length=100,
        description="Город автора отзыва"
    )

    region: Optional[str] = Field(
        None,
        max_length=100,
        description="Регион"
    )

    region_code: Optional[str] = Field(
        None,
        max_length=10,
        description="Код региона (например: RU-MOW для Москвы)"
    )



class RawReviewResponse(BaseModel):
    """Схема ответа для сырого отзыва"""

    uuid: UUID
    source: str
    text: str
    gender: Optional[str]
    city: Optional[str]
    region: Optional[str]
    region_code: Optional[str]
    datetime_review: datetime
    created_at: datetime

    class Config:
        from_attributes = True
