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


@router.get("/gender-product-preferences", summary="Предпочтения по продуктам у мужчин и женщин")
async def get_gender_product_preferences(
        analysis_type: Optional[str] = Query("preferences", description="Тип анализа: preferences или comparison"),
        region_code: Optional[str] = Query(None, description="Код региона"),
        city: Optional[str] = Query(None, description="Название города"),
        date_from: Optional[datetime] = Query(None, description="Дата начала периода"),
        date_to: Optional[datetime] = Query(None, description="Дата окончания периода"),
        dashboard_service: DashboardService = Depends(get_dashboard_service)
):
    """
    API для анализа предпочтений по продуктам у мужчин и женщин.

    Параметры:
    - analysis_type: "preferences" (детальный анализ) или "comparison" (сравнительный анализ)
    - Стандартные фильтры по региону, городу и датам

    Возвращает:
    - Топ продукты для каждого пола
    - Уровень удовлетворенности по полу и продуктам
    - Сравнительную статистику
    - Инсайты и рекомендации
    """
    try:
        analysis_result = await dashboard_service.get_gender_product_preferences_dashboard(
            region_code=region_code,
            city=city,
            date_from=date_from,
            date_to=date_to,
            analysis_type=analysis_type
        )
        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа предпочтений по продуктам: {str(e)}")


@router.get("/gender-product-matrix", summary="Матрица предпочтений: пол vs продукты")
async def get_gender_product_matrix(
        region_code: Optional[str] = Query(None, description="Код региона"),
        city: Optional[str] = Query(None, description="Название города"),
        date_from: Optional[datetime] = Query(None, description="Дата начала периода"),
        date_to: Optional[datetime] = Query(None, description="Дата окончания периода"),
        min_reviews: int = Query(10, ge=1, le=100, description="Минимальное количество отзывов"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    """
    API для получения матрицы данных: пол vs продукты.
    Подходит для построения тепловых карт и матричных диаграмм.
    """
    try:
        preferences_data = await analytics_repo.get_gender_product_preferences(
            region_code=region_code,
            city=city,
            date_from=date_from,
            date_to=date_to,
            min_reviews=min_reviews
        )

        # Преобразуем в матричный формат для визуализации
        matrix_data = {}
        products = set()

        for item in preferences_data:
            gender = item["gender"]
            product = item["product"]
            products.add(product)

            if gender not in matrix_data:
                matrix_data[gender] = {}

            matrix_data[gender][product] = {
                "reviews": item["total_reviews"],
                "satisfaction": item["satisfaction_score"],
                "positive_ratio": item["positive_ratio"]
            }

        # Формируем данные для матрицы
        matrix_result = []
        for gender in ["Мужской", "Женский"]:
            if gender in matrix_data:
                for product in products:
                    if product in matrix_data[gender]:
                        data = matrix_data[gender][product]
                        matrix_result.append([
                            gender,
                            product,
                            data["reviews"],
                            data["satisfaction"],
                            data["positive_ratio"]
                        ])

        return {
            "matrix_data": [
                               ["Gender", "Product", "Reviews", "Satisfaction", "PositiveRatio"]
                           ] + matrix_result,
            "metadata": {
                "total_combinations": len(matrix_result),
                "unique_products": len(products),
                "genders": list(matrix_data.keys()),
                "min_reviews_threshold": min_reviews
            },
            "filters_applied": {
                "region_code": region_code,
                "city": city,
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения матричных данных: {str(e)}")
