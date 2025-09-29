from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, Literal
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from Integration_service.core.database import get_async_session
from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.services.DashBoardService import DashboardService
from Integration_service.schemas.processed_review import ReviewFilters, RatingFilter, SentimentHeatmapResponse

router = APIRouter(prefix="/api/dashboard/sentiment", tags=["Анализ настроений"])


async def get_analytics_repo(
        session: AsyncSession = Depends(get_async_session),
) -> ReviewAnalyticsRepository:
    return ReviewAnalyticsRepository(session)


async def get_dashboard_service(
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
) -> DashboardService:
    return DashboardService(analytics_repo)


@router.get("/statistics", summary="Статистика по тональности")
async def get_sentiment_stats(
        region_code: Optional[str] = Query(None, description="Код региона"),
        city: Optional[str] = Query(None, description="Город"),
        product: Optional[str] = Query(None, description="Продукт"),
        date_from: Optional[datetime] = Query(None, description="Дата с"),
        date_to: Optional[datetime] = Query(None, description="Дата по"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
):
    try:
        filters = ReviewFilters(region_code=region_code, city=city, product=product, date_from=date_from,
                                date_to=date_to)
        sentiment_data = await analytics_repo.get_sentiment_distribution(filters)
        total = sum(sentiment_data.values())

        sentiment_with_percentages = {}
        for sentiment, count in sentiment_data.items():
            percentage = (count / total * 100) if total > 0 else 0
            sentiment_with_percentages[sentiment] = {"count": count, "percentage": round(percentage, 2)}

        return {
            "sentiment_distribution": sentiment_with_percentages,
            "total_reviews": total,
            "filters_applied": filters.model_dump(exclude_none=True),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики тональности: {str(e)}")


@router.get("/heatmap", summary="Тепловая карта регионов по настроениям", response_model=SentimentHeatmapResponse)
async def get_regions_sentiment_heatmap(
        sentiment_filter: Literal["positive", "negative", "neutral"] = Query(...,
                                                                             description="Тип отзывов для отображения"),
        dashboard_service: DashboardService = Depends(get_dashboard_service),
):
    try:
        heatmap_data = await dashboard_service.get_regions_sentiment_heatmap_filtered(sentiment_filter=sentiment_filter)
        return heatmap_data
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Ошибка получения тепловой карты по {sentiment_filter} отзывам: {str(e)}")
