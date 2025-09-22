# api/dashboard_routes.py
from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from Integration_service.core.database import get_async_session
from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.schemas.processed_review import ReviewFilters, RatingFilter, GenderFilter
from Integration_service.schemas.processed_review import DashboardSummary, RegionalDashboard, RealTimeMetrics
from Integration_service.services.DashBoardService import DashboardService

router = APIRouter(prefix="/api/dashboard", tags=["Дашборд Аналитика"])


# ========== DEPENDENCY INJECTION ==========

async def get_analytics_repo() -> ReviewAnalyticsRepository:
    """Dependency для получения репозитория аналитики"""
    async with get_async_session() as session:
        yield ReviewAnalyticsRepository(session)


async def get_dashboard_service(
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
) -> DashboardService:
    """Dependency для получения сервиса дашборда"""
    return DashboardService(analytics_repo)


# ========== ОСНОВНЫЕ ЭНДПОИНТЫ ДАШБОРДА ==========

@router.get("/summary", response_model=Dict[str, Any], summary="Основная сводка дашборда")
async def get_dashboard_summary(
        rating: Optional[RatingFilter] = Query(None, description="Фильтр по тональности"),
        gender: Optional[GenderFilter] = Query(None, description="Фильтр по полу"),
        region_code: Optional[str] = Query(None, description="Код региона"),
        city: Optional[str] = Query(None, description="Город"),
        product: Optional[str] = Query(None, description="Продукт"),
        source: Optional[str] = Query(None, description="Источник"),
        date_from: Optional[datetime] = Query(None, description="Дата с"),
        date_to: Optional[datetime] = Query(None, description="Дата по"),
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    """
   Получить основную сводку дашборда с возможностью фильтрации

   Возвращает:
   - Общее количество отзывов
   - Распределение по тональности
   - Распределение по полу
   - Недавнюю активность
   - Метрики роста
   """
    try:
        filters = ReviewFilters(
            rating=rating,
            gender=gender,
            region_code=region_code,
            city=city,
            product=product,
            sources=[source] if source else None,
            date_from=date_from,
            date_to=date_to
        )

        summary = await dashboard_service.get_dashboard_summary(filters)
        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения сводки: {str(e)}")


@router.get("/regional", response_model=Dict[str, Any], summary="Региональная аналитика")
async def get_regional_dashboard(
        region_code: Optional[str] = Query(None, description="Конкретный регион для детального анализа"),
        limit: int = Query(20, ge=1, le=100, description="Количество регионов в топе"),
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    """
   Региональная аналитика отзывов

   Возвращает:
   - Топ регионов по количеству отзывов
   - Статистику по городам
   - Распределение тональности по регионам
   """
    try:
        regional_data = await dashboard_service.get_regional_dashboard(region_code, limit)
        return regional_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка региональной аналитики: {str(e)}")


@router.get("/realtime", response_model=Dict[str, Any], summary="Данные реального времени")
async def get_realtime_metrics(
        minutes_back: int = Query(5, ge=1, le=60, description="Период в минутах"),
        limit: int = Query(20, ge=1, le=100, description="Количество недавних отзывов"),
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    """
   Метрики реального времени

   Возвращает недавние отзывы за указанный период
   """
    try:
        realtime_data = await dashboard_service.get_realtime_metrics(minutes_back, limit)
        return realtime_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения данных реального времени: {str(e)}")


# ========== СТАТИСТИЧЕСКИЕ ЭНДПОИНТЫ ==========

@router.get("/stats/sentiment", summary="Статистика по тональности")
async def get_sentiment_stats(
        region_code: Optional[str] = Query(None, description="Код региона"),
        city: Optional[str] = Query(None, description="Город"),
        product: Optional[str] = Query(None, description="Продукт"),
        date_from: Optional[datetime] = Query(None, description="Дата с"),
        date_to: Optional[datetime] = Query(None, description="Дата по"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    """Детальная статистика по тональности с фильтрами"""
    try:
        filters = ReviewFilters(
            region_code=region_code,
            city=city,
            product=product,
            date_from=date_from,
            date_to=date_to
        )

        sentiment_data = await analytics_repo.get_sentiment_distribution(filters)
        total = sum(sentiment_data.values())

        sentiment_with_percentages = {}
        for sentiment, count in sentiment_data.items():
            percentage = (count / total * 100) if total > 0 else 0
            sentiment_with_percentages[sentiment] = {
                'count': count,
                'percentage': round(percentage, 2)
            }

        return {
            'sentiment_distribution': sentiment_with_percentages,
            'total_reviews': total,
            'filters_applied': filters.model_dump(exclude_none=True)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики тональности: {str(e)}")


@router.get("/stats/regions", summary="Статистика по регионам")
async def get_regions_stats(
        limit: int = Query(10, ge=1, le=50, description="Количество регионов"),
        rating: Optional[RatingFilter] = Query(None, description="Фильтр по тональности"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    """Топ регионов с детальной статистикой"""
    try:
        filters = ReviewFilters(rating=rating) if rating else None
        regional_stats = await analytics_repo.get_regional_stats(filters, limit)

        return {
            'regional_stats': regional_stats,
            'total_regions': len(regional_stats),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики регионов: {str(e)}")


@router.get("/stats/cities", summary="Статистика по городам")
async def get_cities_stats(
        region_code: Optional[str] = Query(None, description="Код региона"),
        limit: int = Query(15, ge=1, le=50, description="Количество городов"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    """Топ городов с возможностью фильтрации по региону"""
    try:
        city_stats = await analytics_repo.get_city_stats(region_code, limit)

        return {
            'city_stats': city_stats,
            'region_filter': region_code,
            'total_cities': len(city_stats),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики городов: {str(e)}")


@router.get("/stats/products", summary="Анализ продуктов")
async def get_products_sentiment_analysis(
        limit: int = Query(10, ge=1, le=50, description="Количество продуктов"),
        region_code: Optional[str] = Query(None, description="Код региона"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    """Анализ тональности по продуктам"""
    try:
        filters = ReviewFilters(region_code=region_code) if region_code else None
        products_analysis = await analytics_repo.get_product_sentiment_analysis(limit, filters)

        return {
            'products_analysis': products_analysis,
            'region_filter': region_code,
            'total_products': len(products_analysis),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа продуктов: {str(e)}")


# ========== ВРЕМЕННЫЕ ТРЕНДЫ ==========

@router.get("/trends/daily", summary="Дневные тренды")
async def get_daily_trends(
        days_back: int = Query(30, ge=1, le=365, description="Количество дней назад"),
        region_code: Optional[str] = Query(None, description="Код региона"),
        rating: Optional[RatingFilter] = Query(None, description="Фильтр по тональности"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    """Тренды отзывов по дням"""
    try:
        filters = ReviewFilters(region_code=region_code, rating=rating)
        daily_trends = await analytics_repo.get_daily_trends(days_back, filters)

        return {
            'daily_trends': daily_trends,
            'period_days': days_back,
            'filters_applied': filters.model_dump(exclude_none=True),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения дневных трендов: {str(e)}")


@router.get("/trends/hourly", summary="Почасовое распределение")
async def get_hourly_distribution(
        region_code: Optional[str] = Query(None, description="Код региона"),
        date_from: Optional[datetime] = Query(None, description="Дата с"),
        date_to: Optional[datetime] = Query(None, description="Дата по"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    """Распределение отзывов по часам дня"""
    try:
        filters = ReviewFilters(
            region_code=region_code,
            date_from=date_from,
            date_to=date_to
        )

        hourly_data = await analytics_repo.get_hourly_distribution(filters)

        return {
            'hourly_distribution': hourly_data,
            'filters_applied': filters.model_dump(exclude_none=True),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения почасового распределения: {str(e)}")


# ========== МЕТРИКИ РОСТА ==========

@router.get("/growth/metrics", summary="Метрики роста")
async def get_growth_metrics(
        period_hours: int = Query(24, ge=1, le=168, description="Период в часах (max 7 дней)"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    """Метрики роста за указанный период"""
    try:
        growth_data = await analytics_repo.get_growth_metrics(period_hours)

        return {
            'growth_metrics': growth_data,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения метрик роста: {str(e)}")


@router.get("/growth/weekly", summary="Недельное сравнение")
async def get_weekly_comparison(
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    """Сравнение текущей и предыдущей недели"""
    try:
        weekly_data = await analytics_repo.get_weekly_comparison()

        return {
            'weekly_comparison': weekly_data,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения недельного сравнения: {str(e)}")


# ========== ДОПОЛНИТЕЛЬНАЯ АНАЛИТИКА ==========

@router.get("/stats/text-analysis", summary="Анализ текста")
async def get_text_analysis(
        region_code: Optional[str] = Query(None, description="Код региона"),
        rating: Optional[RatingFilter] = Query(None, description="Тональность"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    """Статистика по длине и характеристикам текста отзывов"""
    try:
        filters = ReviewFilters(region_code=region_code, rating=rating)
        text_stats = await analytics_repo.get_text_length_stats(filters)

        return {
            'text_statistics': text_stats,
            'filters_applied': filters.model_dump(exclude_none=True),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа текста: {str(e)}")


@router.get("/recent/activity", summary="Недавняя активность")
async def get_recent_activity(
        minutes_back: int = Query(60, ge=1, le=1440, description="Период в минутах (max 24 часа)"),
        limit: int = Query(50, ge=1, le=200, description="Количество отзывов"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    """Недавние отзывы с детальной информацией"""
    try:
        recent_reviews = await analytics_repo.get_recent_activity(minutes_back, limit)

        return {
            'recent_activity': recent_reviews,
            'period_minutes': minutes_back,
            'total_returned': len(recent_reviews),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения недавней активности: {str(e)}")



@router.get("/regions", summary="Все встречающиеся регионы в отзывах")
async def get_recent_activity(
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения недавней активности: {str(e)}")



@router.get("/region/citys", summary="Все города в регионе")
async def get_recent_activity(
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения недавней активности: {str(e)}")