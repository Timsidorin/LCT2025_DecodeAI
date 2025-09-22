# schemas/review_schemas.py
from datetime import datetime
from typing import Optional, List, Literal, Dict, Any
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, validator, field_validator


# ========== БАЗОВЫЕ СХЕМЫ ==========

class ReviewBase(BaseModel):
    """Базовая схема отзыва"""
    text: str = Field(..., min_length=1, description="Текст отзыва")
    source: str = Field(default="API", max_length=100, description="Источник отзыва")
    rating: Optional[str] = Field(None, description="Тональность отзыва")
    product: Optional[str] = Field(None, max_length=255, description="Название продукта")
    gender: Optional[str] = Field(None, description="Пол автора")
    city: Optional[str] = Field(None, max_length=100, description="Город")
    region_code: Optional[str] = Field(None, max_length=20, description="Код региона")
    datetime_review: datetime = Field(..., description="Дата отзыва")


class ReviewCreate(ReviewBase):
    """Схема для создания отзыва"""
    pass



class ReviewResponse(ReviewBase):
    """Схема ответа с отзывом"""
    model_config = ConfigDict(from_attributes=True)

    uuid: UUID
    created_at: datetime


# ========== ФИЛЬТРАЦИЯ ==========

class SortOrder(str, Enum):
    """Порядок сортировки"""
    ASC = "asc"
    DESC = "desc"


class SortField(str, Enum):
    """Поля для сортировки"""
    CREATED_AT = "created_at"
    DATETIME_REVIEW = "datetime_review"
    TEXT = "text"
    RATING = "rating"
    CITY = "city"
    PRODUCT = "product"


class GenderFilter(str, Enum):
    """Фильтр по полу"""
    MALE = "М"
    FEMALE = "Ж"
    UNKNOWN = ""


class RatingFilter(str, Enum):
    """Фильтр по тональности"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewFilters(BaseModel):
    """Модель фильтров для отзывов"""
    rating: Optional[RatingFilter] = Field(None, description="Фильтр по тональности")
    product: Optional[str] = Field(None, description="Фильтр по продукту")
    gender: Optional[GenderFilter] = Field(None, description="Фильтр по полу")
    city: Optional[str] = Field(None, description="Фильтр по городу")
    region_code: Optional[str] = Field(None, description="Фильтр по коду региона")

    # Фильтры по датам
    date_from: Optional[datetime] = Field(None, description="Дата отзыва с")
    date_to: Optional[datetime] = Field(None, description="Дата отзыва по")
    created_from: Optional[datetime] = Field(None, description="Дата создания с")
    created_to: Optional[datetime] = Field(None, description="Дата создания по")

    # Фильтры по спискам
    sources: Optional[List[str]] = Field(None, description="Список источников")
    ratings: Optional[List[RatingFilter]] = Field(None, description="Список тональностей")
    products: Optional[List[str]] = Field(None, description="Список продуктов")
    cities: Optional[List[str]] = Field(None, description="Список городов")
    region_codes: Optional[List[str]] = Field(None, description="Список кодов регионов")

    @field_validator('date_to')
    def validate_date_range(cls, v, values):
        if v and 'date_from' in values and values['date_from']:
            if v < values['date_from']:
                raise ValueError('date_to должна быть больше date_from')
        return v

    @field_validator('created_to')
    def validate_created_range(cls, v, values):
        if v and 'created_from' in values and values['created_from']:
            if v < values['created_from']:
                raise ValueError('created_to должна быть больше created_from')
        return v


# ========== ПАГИНАЦИЯ ==========

class PaginationParams(BaseModel):
    """Параметры пагинации"""
    page: int = Field(default=1, ge=1, description="Номер страницы")
    size: int = Field(default=50, ge=1, le=1000, description="Размер страницы")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class SortParams(BaseModel):
    """Параметры сортировки"""
    sort_by: SortField = Field(default=SortField.CREATED_AT, description="Поле сортировки")
    sort_order: SortOrder = Field(default=SortOrder.DESC, description="Порядок сортировки")


class QueryParams(BaseModel):
    """Объединенные параметры запроса"""
    filters: ReviewFilters = Field(default_factory=ReviewFilters)
    pagination: PaginationParams = Field(default_factory=PaginationParams)
    sort: SortParams = Field(default_factory=SortParams)


# ========== ОТВЕТЫ ==========

class PaginationMeta(BaseModel):
    """Метаданные пагинации"""
    page: int = Field(..., description="Текущая страница")
    size: int = Field(..., description="Размер страницы")
    total: int = Field(..., description="Общее количество записей")
    pages: int = Field(..., description="Общее количество страниц")
    has_next: bool = Field(..., description="Есть ли следующая страница")
    has_prev: bool = Field(..., description="Есть ли предыдущая страница")


class ReviewListResponse(BaseModel):
    """Ответ со списком отзывов и пагинацией"""
    items: List[ReviewResponse] = Field(..., description="Список отзывов")
    meta: PaginationMeta = Field(..., description="Метаданные пагинации")


# ========== АНАЛИТИКА ==========

class ReviewStats(BaseModel):
    """Статистика по отзывам"""
    total_reviews: int = Field(..., description="Общее количество отзывов")
    reviews_by_rating: Dict[str, int] = Field(..., description="Распределение по тональности")
    reviews_by_gender: Dict[str, int] = Field(..., description="Распределение по полу")
    top_cities: List[Dict[str, Any]] = Field(..., description="Топ городов")
    top_products: List[Dict[str, Any]] = Field(..., description="Топ продуктов")
    date_range: Dict[str, datetime] = Field(..., description="Диапазон дат")


class ReviewTrends(BaseModel):
    """Тренды отзывов по времени"""
    daily_stats: List[Dict[str, Any]] = Field(..., description="Статистика по дням")
    weekly_stats: List[Dict[str, Any]] = Field(..., description="Статистика по неделям")
    monthly_stats: List[Dict[str, Any]] = Field(..., description="Статистика по месяцам")


class DashboardSummary(BaseModel):
    """Основная сводка дашборда"""
    total_reviews: int = Field(..., description="Общее количество отзывов")
    sentiment_distribution: Dict[str, int] = Field(..., description="Распределение по тональности")
    gender_distribution: Dict[str, int] = Field(..., description="Распределение по полу")
    source_distribution: Dict[str, int] = Field(..., description="Распределение по источникам")
    recent_activity: List[Dict[str, Any]] = Field(..., description="Недавняя активность")
    growth_metrics: Dict[str, Any] = Field(..., description="Метрики роста")
    last_updated: datetime = Field(..., description="Время последнего обновления")

class RegionalDashboard(BaseModel):
    """Региональная аналитика"""
    regional_stats: List[Dict[str, Any]] = Field(..., description="Статистика по регионам")
    city_stats: List[Dict[str, Any]] = Field(..., description="Статистика по городам")
    region_sentiment: Dict[str, int] = Field(None, description="Тональность по региону")
    region_trends: List[Dict[str, Any]] = Field(None, description="Тренды региона")
    last_updated: datetime = Field(..., description="Время последнего обновления")

class RealTimeMetrics(BaseModel):
    """Метрики реального времени"""
    recent_reviews: List[Dict[str, Any]] = Field(..., description="Недавние отзывы")
    current_timestamp: datetime = Field(..., description="Текущее время")
    period_minutes: int = Field(..., description="Период в минутах")