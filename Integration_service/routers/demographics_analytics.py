from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from Integration_service.core.database import get_async_session
from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.services.DashBoardService import DashboardService
from Integration_service.schemas.processed_review import ReviewFilters

router = APIRouter(prefix="/api/dashboard/demographics", tags=["Демографическая аналитика"])

async def get_analytics_repo(
        session: AsyncSession = Depends(get_async_session),
) -> ReviewAnalyticsRepository:
    return ReviewAnalyticsRepository(session)


async def get_dashboard_service(
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
) -> DashboardService:
    return DashboardService(analytics_repo)

@router.get("/gender-analysis", summary="Анализ отзывов по полу")
async def get_gender_analysis(
        region_code: Optional[str] = Query(None, description="Код региона"),
        city: Optional[str] = Query(None, description="Название города"),
        product: Optional[str] = Query(None, description="Название продукта"),
        date_from: Optional[datetime] = Query(None, description="Дата начала периода"),
        date_to: Optional[datetime] = Query(None, description="Дата окончания периода"),
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    try:
        analysis_result = await dashboard_service.get_gender_analysis_dashboard(
            region_code=region_code, city=city, product=product, date_from=date_from, date_to=date_to
        )
        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения анализа по полу: {str(e)}")

@router.get("/gender-distribution", summary="Распределение отзывов по полу в процентах")
async def get_gender_distribution(
        region_code: Optional[str] = Query(None, description="Код региона"),
        city: Optional[str] = Query(None, description="Название города"),
        product: Optional[str] = Query(None, description="Название продукта"),
        date_from: Optional[datetime] = Query(None, description="Дата начала периода"),
        date_to: Optional[datetime] = Query(None, description="Дата окончания периода"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    try:
        filters = ReviewFilters(region_code=region_code, city=city, product=product, date_from=date_from, date_to=date_to)
        gender_data = await analytics_repo.get_gender_distribution(filters)
        total_reviews = sum(gender_data.values())

        gender_percentages = {}
        for gender, count in gender_data.items():
            percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
            gender_percentages[gender] = {"count": count, "percentage": round(percentage, 1)}

        return {
            "gender_distribution": gender_percentages,
            "total_reviews": total_reviews,
            "filters_applied": filters.model_dump(exclude_none=True),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения распределения по полу: {str(e)}")
