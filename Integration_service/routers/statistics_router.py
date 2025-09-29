from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
from datetime import datetime
from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.services.DashBoardService import DashboardService
from Integration_service.schemas.processed_review import ReviewFilters, RatingFilter
from Integration_service.core.database import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/dashboard/stats", tags=["Статистика"])

async def get_analytics_repo(session: AsyncSession = Depends(get_async_session)) -> ReviewAnalyticsRepository:
    return ReviewAnalyticsRepository(session)

async def get_dashboard_service(analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)) -> DashboardService:
    return DashboardService(analytics_repo)

@router.get("/sentiment", summary="Статистика по тональности")
async def get_sentiment_stats(
        region_code: Optional[str] = Query(None, description="Код региона"),
        city: Optional[str] = Query(None, description="Город"),
        product: Optional[str] = Query(None, description="Продукт"),
        date_from: Optional[datetime] = Query(None, description="Дата с"),
        date_to: Optional[datetime] = Query(None, description="Дата по"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
):
    try:
        filters = ReviewFilters(
            region_code=region_code,
            city=city,
            product=product,
            date_from=date_from,
            date_to=date_to,
        )

        sentiment_data = await analytics_repo.get_sentiment_distribution(filters)
        total = sum(sentiment_data.values())
        sentiment_with_percentages = {}
        for sentiment, count in sentiment_data.items():
            percentage = (count / total * 100) if total > 0 else 0
            sentiment_with_percentages[sentiment] = {
                "count": count,
                "percentage": round(percentage, 2),
            }

        return {
            "sentiment_distribution": sentiment_with_percentages,
            "total_reviews": total,
            "filters_applied": filters.model_dump(exclude_none=True),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка получения статистики тональности: {str(e)}"
        )

@router.get("/regions", summary="Статистика по регионам")
async def get_regions_stats(
        limit: int = Query(10, ge=1, le=50, description="Количество регионов"),
        rating: Optional[RatingFilter] = Query(None, description="Фильтр по тональности"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
):
    """Топ регионов с детальной статистикой"""
    try:
        filters = ReviewFilters(rating=rating) if rating else None
        regional_stats = await analytics_repo.get_regional_stats(filters, limit)

        return {
            "regional_stats": regional_stats,
            "total_regions": len(regional_stats),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка получения статистики регионов: {str(e)}"
        )

@router.get("/cities", summary="Статистика по городам")
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
        raise HTTPException(
            status_code=500, detail=f"Ошибка получения статистики городов: {str(e)}"
        )

@router.get("/products", summary="Анализ продуктов")  # ← ВОТ ЭТОТ ENDPOINT!
async def get_products_sentiment_analysis(
        limit: int = Query(10, ge=1, le=50, description="Количество продуктов"),
        region_code: Optional[str] = Query(None, description="Код региона"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo),
):
    """Анализ тональности по продуктам"""
    try:
        filters = ReviewFilters(region_code=region_code) if region_code else None
        products_analysis = await analytics_repo.get_product_sentiment_analysis(
            limit, filters
        )

        return {
            "products_analysis": products_analysis,
            "region_filter": region_code,
            "total_products": len(products_analysis),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ошибка анализа продуктов: {str(e)}"
        )
