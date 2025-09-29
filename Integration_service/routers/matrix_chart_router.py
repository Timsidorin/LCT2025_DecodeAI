# routers/matrix_chart_router.py - ФИНАЛЬНАЯ ВЕРСИЯ
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from Integration_service.core.database import get_async_session
from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.schemas.processed_review import ReviewFilters

router = APIRouter(prefix="/api/matrix", tags=["Matrix Charts"])


async def get_analytics_repo(session: AsyncSession = Depends(get_async_session)) -> ReviewAnalyticsRepository:
    return ReviewAnalyticsRepository(session)


@router.get("/product-sentiment-matrix", summary="Матричный график продуктов по месяцам с sentiment")
async def get_product_sentiment_matrix(
        months_back: int = Query(6, ge=3, le=12, description="Количество месяцев назад от текущего"),
        limit_products: int = Query(8, ge=3, le=15, description="Лимит продуктов для отображения"),
        region_code: Optional[str] = Query(None, description="Фильтр по региону"),
        min_reviews_per_month: int = Query(3, ge=1, description="Минимум отзывов за месяц для отображения"),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
) -> Dict[str, Any]:
    """
    Возвращает данные для матричного графика ECharts:
    - X-ось (столбцы): Месяцы
    - Y-ось (строки): Продукты
    - Каждая ячейка: круговая диаграмма с процентами положительных/отрицательных отзывов
    """
    try:
        # Создаем фильтры для последних N месяцев
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * months_back)

        filters = ReviewFilters(
            date_from=start_date,
            date_to=end_date,
            region_code=region_code
        )

        # Используем новый метод репозитория
        matrix_data = await analytics_repo.get_product_sentiment_by_months(
            filters=filters,
            limit_products=limit_products,
            min_reviews_per_month=min_reviews_per_month
        )

        # Формируем структуру для ECharts Matrix
        result = format_matrix_for_echarts(matrix_data, months_back)

        return {
            "matrix_data": result["matrix_data"],
            "x_axis": result["x_axis"],  # Месяцы
            "y_axis": result["y_axis"],  # Продукты
            "series": result["series"],  # Данные для pie charts
            "total_products": len(result["y_axis"]),
            "total_months": len(result["x_axis"]),
            "filters_applied": {
                "months_back": months_back,
                "region_code": region_code,
                "min_reviews_per_month": min_reviews_per_month,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения матричных данных: {str(e)}")


def format_matrix_for_echarts(raw_data: List[Dict], months_back: int) -> Dict[str, Any]:
    """Форматирование данных для ECharts Matrix"""

    # Генерируем список месяцев
    months = []
    for i in range(months_back):
        date = datetime.now() - timedelta(days=30 * i)
        month_str = date.strftime("%Y-%m")
        month_name_str = date.strftime("%b %Y")  # Mar 2024
        months.append({
            "value": month_str,
            "name": month_name_str
        })
    months.reverse()  # От старых к новым

    # Получаем уникальные продукты из данных
    products = list(set(item["product"] for item in raw_data))
    products.sort()

    # Создаем series для каждой ячейки матрицы
    series = []
    matrix_data = {}

    for prod_idx, product in enumerate(products):
        matrix_data[product] = {}

        for month_idx, month in enumerate(months):
            month_value = month["value"]

            # Ищем данные для этой комбинации продукт-месяц
            cell_data = next(
                (item for item in raw_data
                 if item["product"] == product and item["month"] == month_value),
                None
            )

            if cell_data and cell_data["total_reviews"] >= 1:
                positive_pct = cell_data.get("positive_percentage", 0)
                negative_pct = cell_data.get("negative_percentage", 0)
                neutral_pct = cell_data.get("neutral_percentage", 0)
                total_reviews = cell_data.get("total_reviews", 0)

                # Создаем данные только если есть позитивные или негативные отзывы
                chart_data = []
                if positive_pct > 0:
                    chart_data.append({
                        "value": positive_pct,
                        "name": "Позитивные",
                        "itemStyle": {"color": "#5cb85c"}
                    })
                if negative_pct > 0:
                    chart_data.append({
                        "value": negative_pct,
                        "name": "Негативные",
                        "itemStyle": {"color": "#d9534f"}
                    })
                if neutral_pct > 0:
                    chart_data.append({
                        "value": neutral_pct,
                        "name": "Нейтральные",
                        "itemStyle": {"color": "#f0ad4e"}
                    })

                if chart_data:  # Только если есть данные для отображения
                    # Создаем pie chart для этой ячейки
                    series.append({
                        "type": "pie",
                        "coordinateSystem": "matrix",
                        "center": [month_value, product],
                        "radius": 15,
                        "data": chart_data,
                        "label": {"show": False},
                        "emphasis": {
                            "label": {
                                "show": True,
                                "formatter": f"{total_reviews} отз."
                            }
                        }
                    })

                    matrix_data[product][month_value] = {
                        "positive_pct": positive_pct,
                        "negative_pct": negative_pct,
                        "neutral_pct": neutral_pct,
                        "total_reviews": total_reviews
                    }
            else:
                # Пустая ячейка
                matrix_data[product][month_value] = None

    return {
        "matrix_data": matrix_data,
        "x_axis": [m["name"] for m in months],
        "y_axis": products,
        "series": series,
        "months_raw": [m["value"] for m in months]
    }


# Дополнительный эндпоинт для получения сырых данных матрицы
@router.get("/matrix-raw-data", summary="Сырые данные матрицы для отладки")
async def get_matrix_raw_data(
        months_back: int = Query(6, ge=3, le=12),
        limit_products: int = Query(8, ge=3, le=15),
        region_code: Optional[str] = Query(None),
        analytics_repo: ReviewAnalyticsRepository = Depends(get_analytics_repo)
):
    """Получить сырые данные матрицы для отладки"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * months_back)

        filters = ReviewFilters(
            date_from=start_date,
            date_to=end_date,
            region_code=region_code
        )

        matrix_data = await analytics_repo.get_product_sentiment_by_months(
            filters=filters,
            limit_products=limit_products,
            min_reviews_per_month=1
        )

        return {
            "raw_data": matrix_data,
            "total_rows": len(matrix_data),
            "unique_products": list(set(item["product"] for item in matrix_data)),
            "unique_months": list(set(item["month"] for item in matrix_data)),
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")
