from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from uuid import UUID


class ReviewCreate(BaseModel):
    source: str = Field(default="API", description="Источник отзыва")
    text: str = Field(..., min_length=1, description="Текст отзыва")
    gender: Optional[str] = Field(None, description="Пол написавшего")
    city: Optional[str] = Field(None, description="Город")
    region_code: Optional[str] = Field(None, description="Код региона")
    datetime_review: Optional[datetime] = Field(
        None, description="Дата и время создания отзыва"
    )


class ReviewResponse(BaseModel):
    uuid: UUID = Field(..., description="Уникальный идентификатор отзыва")
    source: str = Field(..., description="Источник отзыва")
    text: str = Field(..., description="Текст отзыва")
    rating: Optional[str] = Field(None, description="Тональность отзыва")
    product: Optional[str] = Field(None, description="Название продукта")
    gender: Optional[str] = Field(None, description="Пол написавшего")
    city: Optional[str] = Field(None, description="Город")
    region_code: Optional[str] = Field(None, description="Код региона")
    datetime_review: datetime = Field(..., description="Дата и время создания отзыва")
    created_at: datetime = Field(..., description="Время добавления в систему")

    class Config:
        from_attributes = True


class ReviewUpdate(BaseModel):
    text: Optional[str] = Field(None, min_length=1, description="Текст отзыва")
    rating: Optional[str] = Field(None, description="Тональность отзыва")
    product: Optional[str] = Field(None, description="Название продукта")
    gender: Optional[str] = Field(None, description="Пол написавшего")
    city: Optional[str] = Field(None, description="Город")
    region_code: Optional[str] = Field(None, description="Код региона")


class ReviewFilter(BaseModel):
    source: Optional[str] = Field(None, description="Фильтр по источнику")
    rating: Optional[str] = Field(None, description="Фильтр по тональности")
    gender: Optional[str] = Field(None, description="Фильтр по полу")
    city: Optional[str] = Field(None, description="Фильтр по городу")
    region_code: Optional[str] = Field(None, description="Фильтр по региону")
    date_from: Optional[datetime] = Field(None, description="Дата начала периода")
    date_to: Optional[datetime] = Field(None, description="Дата окончания периода")
