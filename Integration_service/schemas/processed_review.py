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
    product: Optional[str] = Field(
        None, max_length=255, description="Название продукта"
    )
    gender: Optional[str] = Field(None, description="Пол автора")
    city: Optional[str] = Field(None, max_length=100, description="Город")
    region: Optional[str] = Field(None, max_length=100, description="Регион")
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
    ratings: Optional[List[RatingFilter]] = Field(
        None, description="Список тональностей"
    )
    products: Optional[List[str]] = Field(None, description="Список продуктов")
    cities: Optional[List[str]] = Field(None, description="Список городов")
    region_codes: Optional[List[str]] = Field(None, description="Список кодов регионов")

    @field_validator("date_to")
    def validate_date_range(cls, v, info):
        if v and info.data.get("date_from"):
            if v < info.data["date_from"]:
                raise ValueError("date_to должна быть больше date_from")
        return v

    @field_validator("created_to")
    def validate_created_range(cls, v, info):
        if v and info.data.get("created_from"):
            if v < info.data["created_from"]:
                raise ValueError("created_to должна быть больше created_from")
        return v


class SourceAnalyticsFilters(BaseModel):
    """Фильтры для анализа статистики по источникам отзывов"""


    date_from: Optional[datetime] = Field(None, description="Дата начала периода анализа")
    date_to: Optional[datetime] = Field(None, description="Дата окончания периода анализа")
    region_code: Optional[str] = Field(None, description="Код региона для анализа")

    # Дополнительные фильтры
    city: Optional[str] = Field(None, description="Город для дополнительной фильтрации")
    product: Optional[str] = Field(None, description="Продукт для фильтрации")

    class Config:
        json_schema_extra = {
            "example": {
                "date_from": "2024-01-01T00:00:00",
                "date_to": "2024-12-31T23:59:59",
                "region_code": "MSK",
                "city": "Москва",
                "product": "Кредитная карта",
                "min_reviews": 10
            }
        }

    @field_validator("date_to")
    def validate_date_range(cls, v, info):
        if v and info.data.get("date_from"):
            if v < info.data["date_from"]:
                raise ValueError("date_to должна быть больше date_from")
        return v

    def to_review_filters(self) -> ReviewFilters:
        """Конвертировать в стандартный ReviewFilters для совместимости"""
        return ReviewFilters(
            region_code=self.region_code,
            city=self.city,
            product=self.product,
            date_from=self.date_from,
            date_to=self.date_to
        )


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

    sort_by: SortField = Field(
        default=SortField.CREATED_AT, description="Поле сортировки"
    )
    sort_order: SortOrder = Field(
        default=SortOrder.DESC, description="Порядок сортировки"
    )


class QueryParams(BaseModel):
    """Объединенные параметры запроса"""

    filters: ReviewFilters = Field(default_factory=ReviewFilters)
    pagination: PaginationParams = Field(default_factory=PaginationParams)
    sort: SortParams = Field(default_factory=SortParams)


# ========== ОТВЕТЫ ==========


class PaginationMeta(BaseModel):
    page: int = Field(..., description="Текущая страница")
    size: int = Field(..., description="Размер страницы")
    total: int = Field(..., description="Общее количество записей")
    pages: int = Field(..., description="Общее количество страниц")
    has_next: bool = Field(..., description="Есть ли следующая страница")
    has_prev: bool = Field(..., description="Есть ли предыдущая страница")


class ReviewListResponse(BaseModel):
    items: List[ReviewResponse] = Field(..., description="Список отзывов")
    meta: PaginationMeta = Field(..., description="Метаданные пагинации")


# ========== АНАЛИТИКА ==========


class ReviewStats(BaseModel):
    """Статистика по отзывам"""

    total_reviews: int = Field(..., description="Общее количество отзывов")
    reviews_by_rating: Dict[str, int] = Field(
        ..., description="Распределение по тональности"
    )
    reviews_by_gender: Dict[str, int] = Field(..., description="Распределение по полу")
    top_cities: List[Dict[str, Any]] = Field(..., description="Топ городов")
    top_products: List[Dict[str, Any]] = Field(..., description="Топ продуктов")
    date_range: Dict[str, datetime] = Field(..., description="Диапазон дат")


class ReviewTrends(BaseModel):
    """Тренды отзывов по времени"""

    daily_stats: List[Dict[str, Any]] = Field(..., description="Статистика по дням")
    weekly_stats: List[Dict[str, Any]] = Field(..., description="Статистика по неделям")
    monthly_stats: List[Dict[str, Any]] = Field(
        ..., description="Статистика по месяцам"
    )


class DashboardSummary(BaseModel):
    """Основная сводка дашборда"""

    total_reviews: int = Field(..., description="Общее количество отзывов")
    sentiment_distribution: Dict[str, int] = Field(
        ..., description="Распределение по тональности"
    )
    gender_distribution: Dict[str, int] = Field(
        ..., description="Распределение по полу"
    )
    source_distribution: Dict[str, int] = Field(
        ..., description="Распределение по источникам"
    )
    recent_activity: List[Dict[str, Any]] = Field(
        ..., description="Недавняя активность"
    )
    growth_metrics: Dict[str, Any] = Field(..., description="Метрики роста")
    last_updated: datetime = Field(..., description="Время последнего обновления")


class RegionalDashboard(BaseModel):
    """Региональная аналитика"""

    regional_stats: List[Dict[str, Any]] = Field(
        ..., description="Статистика по регионам"
    )
    city_stats: List[Dict[str, Any]] = Field(..., description="Статистика по городам")
    region_sentiment: Dict[str, int] = Field(None, description="Тональность по региону")
    region_trends: List[Dict[str, Any]] = Field(None, description="Тренды региона")
    last_updated: datetime = Field(..., description="Время последнего обновления")


class RealTimeMetrics(BaseModel):
    """Метрики реального времени"""

    recent_reviews: List[Dict[str, Any]] = Field(..., description="Недавние отзывы")
    current_timestamp: datetime = Field(..., description="Текущее время")
    period_minutes: int = Field(..., description="Период в минутах")


# ========== ИСТОЧНИКИ ОТЗЫВОВ ==========


class SourceStatistics(BaseModel):
    """Статистика по одному источнику"""

    source: str = Field(..., description="Название источника")
    total_reviews: int = Field(..., description="Общее количество отзывов")
    positive_reviews: int = Field(..., description="Количество положительных отзывов")
    negative_reviews: int = Field(..., description="Количество отрицательных отзывов")
    neutral_reviews: int = Field(..., description="Количество нейтральных отзывов")
    positive_percentage: float = Field(..., description="Процент положительных отзывов")
    negative_percentage: float = Field(..., description="Процент отрицательных отзывов")
    neutral_percentage: float = Field(..., description="Процент нейтральных отзывов")
    sentiment_score: float = Field(..., description="Общий sentiment score от -1 до 1")
    quality_rating: str = Field(..., description="Оценка качества источника")
    dominance_rank: str = Field(..., description="Доминирующий тип отзывов")
    filtered_by_region: str = Field(..., description="Регион фильтрации")
    filtered_by_date_range: str = Field(..., description="Период фильтрации")
    period_days: int = Field(..., description="Количество дней в периоде")


class SourceAnalyticsSummary(BaseModel):
    """Сводка по анализу источников"""

    region_code: str = Field(..., description="Код региона")
    period_from: str = Field(..., description="Начало периода")
    period_to: str = Field(..., description="Конец периода")
    period_days: int = Field(..., description="Длительность периода в днях")
    total_sources: int = Field(..., description="Общее количество источников")
    total_reviews: int = Field(..., description="Общее количество отзывов")
    overall_positive_percentage: float = Field(..., description="Общий процент позитивных отзывов")
    overall_negative_percentage: float = Field(..., description="Общий процент негативных отзывов")
    overall_neutral_percentage: float = Field(..., description="Общий процент нейтральных отзывов")
    overall_sentiment_score: float = Field(..., description="Общий sentiment score")
    additional_filters: Dict[str, Any] = Field(..., description="Дополнительные примененные фильтры")
    period_analysis: Dict[str, Any] = Field(..., description="Анализ по периоду")


class SourceAnalyticsResponse(BaseModel):
    """Полный ответ анализа источников"""

    sources: List[SourceStatistics] = Field(..., description="Статистика по каждому источнику")
    summary: SourceAnalyticsSummary = Field(..., description="Общая сводка")
    analytics: Dict[str, Any] = Field(..., description="Дополнительная аналитика")
    filters_applied: Dict[str, Any] = Field(..., description="Примененные фильтры")
    timestamp: str = Field(..., description="Время генерации отчета")


# ========== РЕГИОНАЛЬНАЯ СТАТИСТИКА ПО ПРОДУКТАМ ==========


class RegionProductStatistics(BaseModel):
    """Статистика по региону и продукту"""
    region: str = Field(..., description="Название региона")
    region_code: str = Field(..., description="Код региона")
    product: str = Field(..., description="Название продукта")
    total_reviews: int = Field(..., description="Общее количество отзывов")
    positive_reviews: int = Field(..., description="Количество положительных отзывов")
    negative_reviews: int = Field(..., description="Количество отрицательных отзывов")
    neutral_reviews: int = Field(..., description="Количество нейтральных отзывов")
    positive_percentage: float = Field(..., description="Процент положительных отзывов")
    negative_percentage: float = Field(..., description="Процент отрицательных отзывов")
    neutral_percentage: float = Field(..., description="Процент нейтральных отзывов")


class RegionProductSummary(BaseModel):
    """Сводка по региональной статистике продуктов"""

    total_region_product_combinations: int = Field(..., description="Общее количество комбинаций регион-продукт")
    total_reviews: int = Field(..., description="Общее количество отзывов")
    unique_regions: int = Field(..., description="Количество уникальных регионов")
    unique_products: int = Field(..., description="Количество уникальных продуктов")
    overall_positive_percentage: float = Field(..., description="Общий процент позитивных отзывов")
    overall_negative_percentage: float = Field(..., description="Общий процент негативных отзывов")
    overall_neutral_percentage: float = Field(..., description="Общий процент нейтральных отзывов")
    overall_sentiment_score: float = Field(..., description="Общий sentiment score")
    min_reviews_threshold: int = Field(..., description="Минимальный порог отзывов")


class RegionProductAnalyticsResponse(BaseModel):
    """Полный ответ региональной аналитики по продуктам"""
    regions_products: List[RegionProductStatistics] = Field(..., description="Статистика по комбинациям регион-продукт")


# ========== ТЕПЛОВАЯ КАРТА SENTIMENT ==========


class SentimentType(str, Enum):
    """Типы sentiment для тепловой карты"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class RegionSentimentHeatmap(BaseModel):
    """Данные региона для тепловой карты sentiment"""

    region_name: str = Field(..., description="Название региона")
    region_code: str = Field(..., description="Код региона")
    total_reviews: int = Field(..., description="Общее количество отзывов")
    target_sentiment_count: int = Field(..., description="Количество отзывов целевого типа")
    target_sentiment_percentage: float = Field(..., description="Процент отзывов целевого типа")
    positive_reviews: int = Field(..., description="Количество положительных отзывов")
    negative_reviews: int = Field(..., description="Количество отрицательных отзывов")
    neutral_reviews: int = Field(..., description="Количество нейтральных отзывов")
    color: str = Field(..., description="RGB цвет в формате (r g b)")
    sentiment_filter: str = Field(..., description="Примененный фильтр sentiment")


class SentimentHeatmapResponse(BaseModel):
    """Ответ для тепловой карты sentiment"""
    regions: List[RegionSentimentHeatmap] = Field(..., description="Данные регионов")
    total_regions: int = Field(..., description="Общее количество регионов")
    sentiment_filter: SentimentType = Field(..., description="Тип sentiment фильтра")
    color_scheme: Dict[str, str] = Field(..., description="Информация о цветовой схеме")