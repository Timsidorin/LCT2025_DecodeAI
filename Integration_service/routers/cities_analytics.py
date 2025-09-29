from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from Integration_service.core.database import get_async_session
from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.services.DashBoardService import DashboardService

router = APIRouter(prefix="/api/dashboard/cities", tags=["Анализ по городам"])

async def get_analytics_repo(
        session: AsyncSession = Depends(get_async_session),
) -> ReviewAnalyticsRepository:
    return ReviewAnalyticsRepository(session)


async def get_dashboard_service(
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
) -> DashboardService:
    return DashboardService(analytics_repo)


@router.get("/", summary="Все уникальные города")
async def get_cities(
        region_code: Optional[str] = Query(None, description="Фильтр по коду региона"),
        limit: int = Query(100, ge=1, le=500, description="Лимит результатов"),
        dashboard_service: DashboardService = Depends(get_dashboard_service),
):
    try:
        return await dashboard_service.get_cities_by_region(region_code, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения городов: {str(e)}")

@router.get("/statistics", summary="Статистика по городам")
async def get_cities_stats(
        region_code: Optional[str] = Query(None, description="Код региона"),
        limit: int = Query(15, ge=1, le=50, description="Количество городов"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
):
    try:
        city_stats = await analytics_repo.get_city_stats(region_code, limit)
        return {
            "city_stats": city_stats,
            "region_filter": region_code,
            "total_cities": len(city_stats),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики городов: {str(e)}")
