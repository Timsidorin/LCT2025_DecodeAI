from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, Dict, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from Integration_service.core.database import get_async_session
from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.services.DashBoardService import DashboardService
from Integration_service.schemas.processed_review import ReviewFilters, RatingFilter, RegionProductAnalyticsResponse

router = APIRouter(prefix="/api/dashboard/regions", tags=["Региональная аналитика"])

async def get_analytics_repo(
        session: AsyncSession = Depends(get_async_session),
) -> ReviewAnalyticsRepository:
    return ReviewAnalyticsRepository(session)


async def get_dashboard_service(
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
) -> DashboardService:
    return DashboardService(analytics_repo)



@router.get("/", summary="Все встречающиеся регионы в отзывах")  # ← ИЗМЕНЕНО С "/all" НА "/"
async def get_regions(
        include_cities: bool = Query(False, description="Включить города для каждого региона"),
        dashboard_service: DashboardService = Depends(get_dashboard_service),
):
    """ОРИГИНАЛЬНЫЙ ENDPOINT - НЕ ТРОГАТЬ!"""
    try:
        return await dashboard_service.get_all_regions(include_cities)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения регионов: {str(e)}")

@router.get("/dashboard", summary="Региональная аналитика") 
async def get_regional_dashboard(
        region_code: Optional[str] = Query(None, description="Конкретный регион для детального анализа"),
        limit: int = Query(20, ge=1, le=100, description="Количество регионов в топе"),
        dashboard_service: DashboardService = Depends(get_dashboard_service),
):
    try:
        regional_data = await dashboard_service.get_regional_dashboard(region_code, limit)
        return regional_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка региональной аналитики: {str(e)}")

@router.get("/all", summary="Все встречающиеся регионы в отзывах")
async def get_regions(
        include_cities: bool = Query(False, description="Включить города для каждого региона"),
        dashboard_service: DashboardService = Depends(get_dashboard_service),
):
    try:
        return await dashboard_service.get_all_regions(include_cities)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения регионов: {str(e)}")

@router.get("/codes", summary="Коды регионов")
async def get_region_codes(dashboard_service: DashboardService = Depends(get_dashboard_service)):
    try:
        return await dashboard_service.get_region_codes_only()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения кодов регионов: {str(e)}")

@router.get("/sentiment-map", summary="Регионы с sentiment анализом для карты")
async def get_regions_sentiment_map(
        include_cities: bool = Query(False, description="Включить города"),
        dashboard_service: DashboardService = Depends(get_dashboard_service),
):
    try:
        return await dashboard_service.get_regions_sentiment_map(include_cities=include_cities)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения карты sentiment регионов: {str(e)}")

@router.get("/statistics", summary="Статистика по регионам")
async def get_regions_stats(
        limit: int = Query(10, ge=1, le=50, description="Количество регионов"),
        rating: Optional[RatingFilter] = Query(None, description="Фильтр по тональности"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
):
    try:
        filters = ReviewFilters(rating=rating) if rating else None
        regional_stats = await analytics_repo.get_regional_stats(filters, limit)
        return {
            "regional_stats": regional_stats,
            "total_regions": len(regional_stats),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики регионов: {str(e)}")

@router.get("/{region_code}", summary="Детали конкретного региона")
async def get_region_details(
        region_code: str,
        dashboard_service: DashboardService = Depends(get_dashboard_service),
):
    try:
        return await dashboard_service.get_region_details(region_code)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения деталей региона: {str(e)}")

@router.get("/products/statistics", summary="Статистика по регионам и продуктам", response_model=RegionProductAnalyticsResponse)
async def get_regions_products_statistics(
        region_code: Optional[str] = Query(None, description="Код региона для фильтрации"),
        product: Optional[str] = Query(None, description="Продукт для фильтрации"),
        city: Optional[str] = Query(None, description="Город для фильтрации"),
        date_from: Optional[datetime] = Query(None, description="Дата начала периода"),
        date_to: Optional[datetime] = Query(None, description="Дата окончания периода"),
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    try:
        filters = ReviewFilters(region_code=region_code, product=product, city=city, date_from=date_from, date_to=date_to)
        regions_products_stats = await dashboard_service.get_regions_products_dashboard_statistics(filters=filters)
        return regions_products_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики по регионам и продуктам: {str(e)}")
