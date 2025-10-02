from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from Integration_service.core.database import get_async_session
from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.services.DashBoardService import DashboardService
from Integration_service.schemas.processed_review import ReviewFilters, RatingFilter, GenderFilter

router = APIRouter(prefix="/api/dashboard", tags=["Основной дашборд"])


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
        session: AsyncSession = Depends(get_async_session),  # Прямая зависимость от сессии
):
    try:
        analytics_repo = ReviewAnalyticsRepository(session)
        dashboard_service = DashboardService(analytics_repo)

        filters = ReviewFilters(
            rating=rating,
            gender=gender,
            region_code=region_code,
            city=city,
            product=product,
            sources=[source] if source else None,
            date_from=date_from,
            date_to=date_to,
        )

        summary = await dashboard_service.get_dashboard_summary(filters)
        return summary

    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Ошибка получения сводки: {str(e)}")
