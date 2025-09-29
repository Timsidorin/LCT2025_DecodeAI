from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from Integration_service.core.database import get_async_session
from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.services.DashBoardService import DashboardService

router = APIRouter(prefix="/api/dashboard/trends", tags=["Анализ трендов"])

async def get_analytics_repo(
        session: AsyncSession = Depends(get_async_session),
) -> ReviewAnalyticsRepository:
    return ReviewAnalyticsRepository(session)


async def get_dashboard_service(
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
) -> DashboardService:
    return DashboardService(analytics_repo)

@router.get("/echarts-data", summary="Данные для графика в формате ECharts")
async def get_echarts_trends_data(
        region_code: Optional[str] = Query(None, description="Код региона"),
        city: Optional[str] = Query(None, description="Город"),
        product: Optional[str] = Query(None, description="Продукт"),
        date_from: Optional[datetime] = Query(None, description="Дата начала"),
        date_to: Optional[datetime] = Query(None, description="Дата окончания"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    try:
        chart_data = await analytics_repo.get_reviews_trends_data_echarts_format(
            region_code=region_code, city=city, product=product, date_from=date_from, date_to=date_to
        )
        return chart_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения данных: {str(e)}")

@router.get("/detailed-data", summary="Детальные данные трендов с разбивкой")
async def get_detailed_trends_data(
        region_code: Optional[str] = Query(None, description="Код региона"),
        city: Optional[str] = Query(None, description="Город"),
        product: Optional[str] = Query(None, description="Продукт"),
        date_from: Optional[datetime] = Query(None, description="Дата начала"),
        date_to: Optional[datetime] = Query(None, description="Дата окончания"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    try:
        detailed_data = await analytics_repo.get_reviews_trends_data(
            region_code=region_code, city=city, product=product, date_from=date_from, date_to=date_to
        )
        return {
            "data": detailed_data,
            "filters": {
                "region_code": region_code, "city": city, "product": product,
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None
            },
            "format": "echarts_compatible",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения детальных данных: {str(e)}")
