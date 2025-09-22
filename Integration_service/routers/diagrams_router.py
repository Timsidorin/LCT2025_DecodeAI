# api/dashboard_routes.py
from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from Integration_service.core.database import get_async_session
from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.services.DashBoardService import DashboardService
from Integration_service.schemas.processed_review import ReviewFilters, RatingFilter, GenderFilter

router = APIRouter(prefix="/api/dashboard", tags=["Дашборд аналитика"])



async def get_analytics_repo(
    session: AsyncSession = Depends(get_async_session)
) -> ReviewAnalyticsRepository:
    return ReviewAnalyticsRepository(session)


async def get_dashboard_service(
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
) -> DashboardService:
    return DashboardService(analytics_repo)


# ========== ОСНОВНЫЕ ДАШБОРД ЭНДПОИНТЫ ==========

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
    try:
        regional_data = await dashboard_service.get_regional_dashboard(region_code, limit)
        return regional_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка региональной аналитики: {str(e)}")



# ========== РЕГИОНЫ И ГОРОДА ==========

@router.get("/regions/sentiment-map", summary="Регионы с sentiment анализом для карты")
async def get_regions_sentiment_map(
        min_reviews: int = Query(0, ge=0, description="Минимальное количество отзывов"),
        include_cities: bool = Query(False, description="Включить города (пока не реализовано)"),
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    """
    Получить регионы с sentiment анализом и готовыми цветами для визуализации на карте.

    Возвращает:
    - region_name: название региона
    - region_code: код региона
    - reviews_count: количество отзывов
    - positive/negative_percentage: процентное соотношение
    - sentiment_score: оценка от -1 до 1
    - color: HEX цвет для карты (#228B22, #FA8072, #E2B007)
    - color_intensity: прозрачность от 0 до 1
    """
    try:
        return await dashboard_service.get_regions_sentiment_map(
            min_reviews=min_reviews,
            include_cities=include_cities
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения карты sentiment регионов: {str(e)}"
        )


@router.get("/regions", summary="Все встречающиеся регионы в отзывах")
async def get_regions(
        include_cities: bool = Query(False, description="Включить города для каждого региона"),
        min_reviews: int = Query(0, ge=0, description="Минимальное количество отзывов"),
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    try:
        return await dashboard_service.get_all_regions(include_cities, min_reviews)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения регионов: {str(e)}")


@router.get("/regions/codes", summary="Коды регионов")
async def get_region_codes(
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    try:
        return await dashboard_service.get_region_codes_only()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения кодов регионов: {str(e)}")


@router.get("/regions/{region_code}", summary="Детали конкретного региона")
async def get_region_details(
        region_code: str,
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    try:
        return await dashboard_service.get_region_details(region_code)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения деталей региона: {str(e)}")


@router.get("/cities", summary="Все уникальные города")
async def get_cities(
        region_code: Optional[str] = Query(None, description="Фильтр по коду региона"),
        min_reviews: int = Query(1, ge=0, description="Минимальное количество отзывов"),
        limit: int = Query(100, ge=1, le=500, description="Лимит результатов"),
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    try:
        return await dashboard_service.get_cities_by_region(region_code, min_reviews, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения городов: {str(e)}")



@router.get("/stats/sentiment", summary="Статистика по тональности")
async def get_sentiment_stats(
        region_code: Optional[str] = Query(None, description="Код региона"),
        city: Optional[str] = Query(None, description="Город"),
        product: Optional[str] = Query(None, description="Продукт"),
        date_from: Optional[datetime] = Query(None, description="Дата с"),
        date_to: Optional[datetime] = Query(None, description="Дата по"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
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


