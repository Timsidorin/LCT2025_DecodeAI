from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from Integration_service.core.database import get_async_session
from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.services.DashBoardService import DashboardService
from Integration_service.schemas.processed_review import ReviewFilters, ProductsAnalysisFilters

router = APIRouter(prefix="/api/dashboard/products", tags=["Анализ продуктов"])


async def get_analytics_repo(
        session: AsyncSession = Depends(get_async_session),
) -> ReviewAnalyticsRepository:
    return ReviewAnalyticsRepository(session)


async def get_dashboard_service(
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
) -> DashboardService:
    return DashboardService(analytics_repo)


@router.get("/statistics", summary="Анализ продуктов")
async def get_products_sentiment_analysis(
        limit: int = Query(10, ge=1, le=50, description="Количество продуктов"),
        region_code: Optional[str] = Query(None, description="Код региона"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
):
    try:
        filters = ReviewFilters(region_code=region_code) if region_code else None
        products_analysis = await analytics_repo.get_product_sentiment_analysis(limit, filters)
        return {
            "products_analysis": products_analysis,
            "region_filter": region_code,
            "total_products": len(products_analysis),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа продуктов: {str(e)}")


@router.post("/sentiment-analysis", summary="Анализ продуктов по типам отзывов")
async def get_products_sentiment_analysis_endpoint(
        filters: ProductsAnalysisFilters,
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    try:
        if not filters.products:
            raise HTTPException(status_code=400, detail="Необходимо указать минимум один продукт для анализа")
        if len(filters.products) > 10:
            raise HTTPException(status_code=400, detail="Максимальное количество продуктов для анализа: 10")

        analysis_result = await dashboard_service.get_products_sentiment_analysis(filters)
        return analysis_result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа продуктов: {str(e)}")


@router.post("/trends-data", summary="Данные трендов продуктов для ECharts")
async def get_products_sentiment_trends_data_endpoint(
        filters: ProductsAnalysisFilters,
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    try:
        products_filters = [{"name": product.name, "type": product.type} for product in filters.products]
        chart_data = await analytics_repo.get_products_sentiment_trends_data(
            products_filters=products_filters, date_from=filters.date_from, date_to=filters.date_to,
            region_code=filters.region_code, city=filters.city
        )
        return chart_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения данных трендов: {str(e)}")
