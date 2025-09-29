from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from Integration_service.core.database import get_async_session
from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.services.DashBoardService import DashboardService
from Integration_service.schemas.processed_review import SourceAnalyticsFilters

router = APIRouter(prefix="/api/dashboard/sources", tags=["Анализ источников"])

async def get_analytics_repo(
        session: AsyncSession = Depends(get_async_session),
) -> ReviewAnalyticsRepository:
    return ReviewAnalyticsRepository(session)


async def get_dashboard_service(
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
) -> DashboardService:
    return DashboardService(analytics_repo)

@router.get("/statistics", summary="Статистика по источникам отзывов")
async def get_sources_statistics(
        date_from: Optional[datetime] = Query(None, description="Дата начала периода"),
        date_to: Optional[datetime] = Query(None, description="Дата окончания периода"),
        region_code: Optional[str] = Query(None, description="Код региона"),
        city: Optional[str] = Query(None, description="Город для фильтрации"),
        product: Optional[str] = Query(None, description="Продукт для фильтрации"),
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    try:
        filters = SourceAnalyticsFilters(
            date_from=date_from, date_to=date_to, region_code=region_code, city=city, product=product
        )
        sources_data = await dashboard_service.get_sources_dashboard_statistics(filters)
        return {"sources": sources_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики по источникам: {str(e)}")
