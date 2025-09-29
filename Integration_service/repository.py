from sqlalchemy import func, text, desc, asc, case, extract, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from models.processed_reviews import Review
from schemas.processed_review import ReviewFilters, GenderFilter, RatingFilter, SourceAnalyticsFilters


class ReviewAnalyticsRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ========== ОСНОВНАЯ СТАТИСТИКА ==========

    async def get_total_reviews_count(self, filters: Optional[ReviewFilters] = None) -> int:
        """Общее количество отзывов"""
        query = select(func.count(Review.uuid))
        if filters:
            query = self._apply_filters(query, filters)
        result = await self.session.execute(query)
        return result.scalar() or 0

    async def get_sentiment_distribution(self, filters: Optional[ReviewFilters] = None) -> Dict[str, int]:
        """Распределение по тональности с правильными значениями"""
        query = select(
            Review.rating,
            func.count(Review.uuid).label('count')
        ).group_by(Review.rating)

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)

        sentiment_data = {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0}
        for row in result:
            rating = row.rating or "unknown"
            if rating in sentiment_data:
                sentiment_data[rating] = row.count
            else:
                sentiment_data["unknown"] += row.count

        return sentiment_data

    async def get_gender_distribution(self, filters: Optional[ReviewFilters] = None) -> Dict[str, int]:
        """Распределение по полу с маппингом male/female/unknown"""
        query = select(
            Review.gender,
            func.count(Review.uuid).label('count')
        ).group_by(Review.gender)

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)

        gender_data = {"male": 0, "female": 0, "unknown": 0}
        for row in result:
            gender = row.gender or "unknown"
            if gender in ['М', 'м', 'male']:
                gender_data["male"] += row.count
            elif gender in ['Ж', 'ж', 'female']:
                gender_data["female"] += row.count
            else:
                gender_data["unknown"] += row.count

        return gender_data

    async def get_reviews_trends_data(
            self,
            region_code: Optional[str] = None,
            city: Optional[str] = None,
            product: Optional[str] = None,
            date_from: Optional[datetime] = None,
            date_to: Optional[datetime] = None
    ) -> List[List]:
        """Получить данные для графика трендов по отзывам"""
        query = select(
            extract('year', Review.datetime_review).label('year'),
            extract('month', Review.datetime_review).label('month'),
            extract('day', Review.datetime_review).label('day'),
            func.count(Review.uuid).label('total_reviews'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_count'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_count'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_count'),
            Review.region_code,
            Review.city,
            Review.product
        ).group_by(
            extract('year', Review.datetime_review),
            extract('month', Review.datetime_review),
            extract('day', Review.datetime_review),
            Review.region_code,
            Review.city,
            Review.product
        ).order_by(
            extract('year', Review.datetime_review),
            extract('month', Review.datetime_review),
            extract('day', Review.datetime_review)
        )

        if region_code:
            query = query.where(Review.region_code == region_code)

        if city:
            query = query.where(Review.city.ilike(f"%{city}%"))

        if product:
            query = query.where(Review.product.ilike(f"%{product}%"))

        if date_from:
            query = query.where(Review.datetime_review >= date_from)

        if date_to:
            query = query.where(Review.datetime_review <= date_to)

        result = await self.session.execute(query)

        chart_data = [
            [
                "Positive_Reviews",
                "Negative_Reviews",
                "Neutral_Reviews",
                "Total_Reviews",
                "Sentiment_Score",
                "Region_Code",
                "City",
                "Product",
                "Date"
            ]
        ]

        for row in result:
            date_str = f"{int(row.year)}-{int(row.month):02d}-{int(row.day):02d}"

            total = row.total_reviews
            sentiment_score = 0
            if total > 0:
                sentiment_score = (row.positive_count - row.negative_count) / total

            chart_data.append([
                row.positive_count,
                row.negative_count,
                row.neutral_count,
                row.total_reviews,
                round(sentiment_score, 3),
                row.region_code or "",
                row.city or "",
                row.product or "",
                date_str
            ])

        return chart_data

    async def get_products_sentiment_trends_data(
            self,
            products_filters: List[Dict[str, str]],
            date_from: datetime,
            date_to: datetime,
            region_code: Optional[str] = None,
            city: Optional[str] = None
    ) -> List[List]:
        """Получить данные трендов по продуктам"""
        from sqlalchemy import literal_column, union_all
        month_truncated = func.date_trunc('month', Review.datetime_review).label('month_period')
        all_data = {}
        all_months = set()
        all_series = []

        for product_filter in products_filters:
            product_name = product_filter['name']
            sentiment_type = product_filter['type']
            series_name = f"{product_name} ({sentiment_type})"
            all_series.append(series_name)
            query = select(
                month_truncated,
                func.count(Review.uuid).label('review_count')
            ).where(
                and_(
                    Review.product.ilike(f"%{product_name}%"),
                    Review.rating == sentiment_type,
                    Review.datetime_review >= date_from,
                    Review.datetime_review <= date_to
                )
            ).group_by(
                month_truncated
            ).order_by(
                month_truncated
            )
            if region_code:
                query = query.where(Review.region_code == region_code)

            if city:
                query = query.where(Review.city.ilike(f"%{city}%"))

            result = await self.session.execute(query)

            for row in result:
                month_str = row.month_period.strftime('%Y-%m') if row.month_period else "2024-01"
                all_months.add(month_str)

                if month_str not in all_data:
                    all_data[month_str] = {}

                all_data[month_str][series_name] = row.review_count

        all_months = sorted(list(all_months))

        header = ['Month'] + all_series
        chart_data = [header]
        for month in all_months:
            row_data = [month]

            for series in all_series:
                value = all_data.get(month, {}).get(series, 0)
                row_data.append(value)

            chart_data.append(row_data)

        return chart_data

    async def get_reviews_trends_aggregated(
            self,
            region_code: Optional[str] = None,
            city: Optional[str] = None,
            product: Optional[str] = None,
            date_from: Optional[datetime] = None,
            date_to: Optional[datetime] = None,
            group_by: str = "day"
    ) -> List[List]:
        """Получить агрегированные данные трендов с разной группировкой"""
        if group_by == "week":
            date_part = func.date_trunc('week', Review.datetime_review)
        elif group_by == "month":
            date_part = func.date_trunc('month', Review.datetime_review)
        else:  # day
            date_part = func.date_trunc('day', Review.datetime_review)

        query = select(
            date_part.label('period'),
            func.count(Review.uuid).label('total_reviews'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_count'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_count'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_count')
        ).group_by(
            date_part
        ).order_by(
            date_part
        )
        if region_code:
            query = query.where(Review.region_code == region_code)

        if city:
            query = query.where(Review.city.ilike(f"%{city}%"))

        if product:
            query = query.where(Review.product.ilike(f"%{product}%"))

        if date_from:
            query = query.where(Review.datetime_review >= date_from)

        if date_to:
            query = query.where(Review.datetime_review <= date_to)

        result = await self.session.execute(query)
        chart_data = [
            [
                "Positive_Reviews",
                "Negative_Reviews",
                "Neutral_Reviews",
                "Total_Reviews",
                "Sentiment_Score",
                "Date"
            ]
        ]

        for row in result:
            period_str = row.period.strftime('%Y-%m-%d')
            total = row.total_reviews
            sentiment_score = 0

            if total > 0:
                sentiment_score = (row.positive_count - row.negative_count) / total

            chart_data.append([
                row.positive_count,
                row.negative_count,
                row.neutral_count,
                row.total_reviews,
                round(sentiment_score, 3),
                period_str
            ])

        return chart_data


    async def get_product_sentiment_by_months(
            self,
            filters: Optional[ReviewFilters] = None,
            limit_products: int = 10,
            min_reviews_per_month: int = 5
    ) -> List[Dict[str, Any]]:
        """Получить sentiment по продуктам и месяцам для матричного графика"""

        month_alias = func.to_char(Review.datetime_review, 'YYYY-MM').label('month')

        query = select(
            Review.product,
            month_alias,
            func.count(Review.uuid).label('total_reviews'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_count'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_count'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_count')
        ).group_by(
            Review.product,
            month_alias
        ).having(
            func.count(Review.uuid) >= min_reviews_per_month
        ).order_by(
            Review.product,
            month_alias
        )

        if filters:
            query = self._apply_filters(query, filters)

        top_products_query = select(Review.product).group_by(Review.product).order_by(
            func.count(Review.uuid).desc()
        ).limit(limit_products)

        if filters:
            top_products_query = self._apply_filters(top_products_query, filters)

        top_products_result = await self.session.execute(top_products_query)
        top_products = [row[0] for row in top_products_result.fetchall()]

        if not top_products:
            return []

        query = query.where(Review.product.in_(top_products))

        result = await self.session.execute(query)
        rows = result.fetchall()

        data = []
        for row in rows:
            total = row.total_reviews or 0
            positive = row.positive_count or 0
            negative = row.negative_count or 0
            neutral = row.neutral_count or 0

            if total > 0:
                data.append({
                    "product": row.product,
                    "month": row.month,
                    "total_reviews": total,
                    "positive_count": positive,
                    "negative_count": negative,
                    "neutral_count": neutral,
                    "positive_percentage": round((positive / total) * 100, 1),
                    "negative_percentage": round((negative / total) * 100, 1),
                    "neutral_percentage": round((neutral / total) * 100, 1)
                })

        return data

    async def get_reviews_trends_data_echarts_format(
            self,
            region_code: Optional[str] = None,
            city: Optional[str] = None,
            product: Optional[str] = None,
            date_from: Optional[datetime] = None,
            date_to: Optional[datetime] = None
    ) -> List[List]:
        """Получить агрегированные данные по месяцам для графика трендов"""

        month_truncated = func.date_trunc('month', Review.datetime_review).label('month_period')

        query = select(
            month_truncated,
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_reviews'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_reviews'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_reviews'),
            func.count(Review.uuid).label('total_reviews')
        )

        if region_code:
            query = query.where(Review.region_code == region_code)
        if city:
            query = query.where(Review.city.ilike(f"%{city}%"))
        if product:
            query = query.where(Review.product.ilike(f"%{product}%"))

        query = query.group_by(month_truncated).order_by(month_truncated)

        result = await self.session.execute(query)

        chart_data = [
            ["Positive_Reviews", "Negative_Reviews", "Neutral_Reviews", "Total_Reviews", "Month"]
        ]

        filter_from_year_month = None
        filter_to_year_month = None

        if date_from:
            filter_from_year_month = (date_from.year, date_from.month)

        if date_to:
            filter_to_year_month = (date_to.year, date_to.month)

        for row in result:
            month_date = row.month_period
            row_year_month = (month_date.year, month_date.month)

            # Сравнение по году и месяцу
            if filter_from_year_month:
                if row_year_month < filter_from_year_month:
                    continue

            if filter_to_year_month:
                if row_year_month > filter_to_year_month:
                    continue

            month_str = month_date.strftime('%Y-%m')
            chart_data.append([
                row.positive_reviews,
                row.negative_reviews,
                row.neutral_reviews,
                row.total_reviews,
                month_str
            ])

        return chart_data

    async def get_gender_sentiment_analysis(
            self,
            region_code: Optional[str] = None,
            city: Optional[str] = None,
            product: Optional[str] = None,
            date_from: Optional[datetime] = None,
            date_to: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Анализ отзывов по полу с детализацией по настроениям"""

        query = select(
            Review.gender.label('gender'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_reviews'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_reviews'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_reviews'),
            func.count(Review.uuid).label('total_reviews'),
            func.avg(
                case(
                    (Review.rating == 'positive', 5),
                    (Review.rating == 'neutral', 3),
                    (Review.rating == 'negative', 1),
                    else_=3
                )
            ).label('avg_rating')
        )


        if region_code:
            query = query.where(Review.region_code == region_code)
        if city:
            query = query.where(Review.city.ilike(f"%{city}%"))
        if product:
            query = query.where(Review.product.ilike(f"%{product}%"))
        if date_from:
            date_from_naive = date_from.replace(tzinfo=None) if date_from.tzinfo else date_from
            query = query.where(Review.datetime_review >= date_from_naive)
        if date_to:
            date_to_naive = date_to.replace(tzinfo=None) if date_to.tzinfo else date_to
            date_to_end = date_to_naive.replace(hour=23, minute=59, second=59, microsecond=999999)
            query = query.where(Review.datetime_review <= date_to_end)

        query = query.where(Review.gender.in_(['Ж', 'М']))
        query = query.group_by(Review.gender)
        query = query.order_by(Review.gender)

        result = await self.session.execute(query)

        gender_data = []
        for row in result:
            if row.gender == 'М':
                gender_display = 'Мужской'
                gender_code = 'male'
            elif row.gender == 'Ж':
                gender_display = 'Женский'
                gender_code = 'female'
            else:
                gender_display = row.gender
                gender_code = 'unknown'

            positive_ratio = (row.positive_reviews / row.total_reviews * 100) if row.total_reviews > 0 else 0
            negative_ratio = (row.negative_reviews / row.total_reviews * 100) if row.total_reviews > 0 else 0
            neutral_ratio = (row.neutral_reviews / row.total_reviews * 100) if row.total_reviews > 0 else 0

            gender_data.append({
                "gender": gender_display,
                "gender_code": gender_code,
                "gender_raw": row.gender,
                "positive_reviews": row.positive_reviews,
                "negative_reviews": row.negative_reviews,
                "neutral_reviews": row.neutral_reviews,
                "total_reviews": row.total_reviews,
                "avg_rating": round(float(row.avg_rating), 2) if row.avg_rating else 0.0,
                "positive_ratio": round(positive_ratio, 1),
                "negative_ratio": round(negative_ratio, 1),
                "neutral_ratio": round(neutral_ratio, 1),
                "sentiment_score": round(
                    (row.positive_reviews - row.negative_reviews) / row.total_reviews if row.total_reviews > 0 else 0,
                    3)
            })
        return gender_data

    async def get_source_distribution(self, filters: Optional[ReviewFilters] = None) -> Dict[str, int]:
        query = select(
            Review.source,
            func.count(Review.uuid).label('count')
        ).group_by(Review.source)

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)
        return {(row.source or "unknown"): row.count for row in result}

    async def get_growth_metrics(self, period_hours: int = 24) -> Dict[str, Any]:
        period_start = datetime.now() - timedelta(hours=period_hours)

        current_query = select(func.count(Review.uuid)).where(Review.created_at >= period_start)
        total_query = select(func.count(Review.uuid))

        current_result = await self.session.execute(current_query)
        total_result = await self.session.execute(total_query)

        current_count = current_result.scalar() or 0
        total_count = total_result.scalar() or 0

        return {
            "period_hours": period_hours,
            "new_reviews": current_count,
            "total_reviews": total_count,
            "growth_rate": round((current_count / total_count * 100) if total_count > 0 else 0, 2),
        }

    async def get_recent_activity(self, minutes_back: int = 60, limit: int = 10) -> List[Dict[str, Any]]:
        time_threshold = datetime.now() - timedelta(minutes=minutes_back)

        query = select(
            Review.uuid, Review.text, Review.rating, Review.region, Review.city, Review.created_at
        ).where(
            Review.created_at >= time_threshold
        ).order_by(
            desc(Review.created_at)
        ).limit(limit)

        result = await self.session.execute(query)

        return [{
            'uuid': str(row.uuid),
            'text': row.text[:100] + '...' if len(row.text) > 100 else row.text,
            'rating': row.rating,
            'region': row.region,
            'city': row.city,
            'created_at': row.created_at.isoformat() if row.created_at else None
        } for row in result]

    # ========== РЕГИОНАЛЬНАЯ СТАТИСТИКА ==========

    async def get_unique_regions(self) -> List[Dict[str, Any]]:
        query = select(
            Review.region,
            Review.region_code,
            func.count(Review.uuid).label('reviews_count')
        ).where(
            and_(Review.region.isnot(None), Review.region_code.isnot(None))
        ).group_by(
            Review.region, Review.region_code
        ).order_by(
            desc('reviews_count'), Review.region
        )

        result = await self.session.execute(query)

        regions = []
        for row in result:
            regions.append({
                "label": row.region,
                "value": row.region_code,
                "reviews_count": row.reviews_count
            })

        return regions

    async def get_unique_region_codes(self) -> List[Dict[str, Any]]:
        query = select(
            Review.region_code,
            func.count(Review.uuid).label('reviews_count')
        ).where(
            Review.region_code.isnot(None)
        ).group_by(
            Review.region_code
        ).order_by(
            desc('reviews_count'), Review.region_code
        )

        result = await self.session.execute(query)
        return [{"region_code": row.region_code, "reviews_count": row.reviews_count} for row in result]

    async def get_regions_hierarchy(self) -> Dict[str, Any]:
        query = select(
            Review.region,
            Review.region_code,
            Review.city,
            func.count(Review.uuid).label('reviews_count')
        ).where(
            and_(Review.region.isnot(None), Review.region_code.isnot(None))
        ).group_by(
            Review.region, Review.region_code, Review.city
        ).order_by(
            Review.region, desc('reviews_count')
        )

        result = await self.session.execute(query)

        regions_hierarchy = {}
        for row in result:
            region_key = f"{row.region}_{row.region_code}"
            if region_key not in regions_hierarchy:
                regions_hierarchy[region_key] = {
                    "label": row.region,
                    "value": row.region_code,
                    "total_reviews": 0,
                    "cities": []
                }

            regions_hierarchy[region_key]["total_reviews"] += row.reviews_count
            if row.city:
                regions_hierarchy[region_key]["cities"].append({
                    "city_name": row.city,
                    "reviews_count": row.reviews_count
                })

        return {
            "regions": list(regions_hierarchy.values()),
            "total_regions": len(regions_hierarchy)
        }

    async def get_regional_stats(self, filters: Optional[ReviewFilters] = None, limit: int = 10) -> List[
        Dict[str, Any]]:
        query = select(
            Review.region_code,
            Review.region,
            func.count(Review.uuid).label('total_count'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_count'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_count'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_count')
        ).where(
            Review.region_code.isnot(None)
        ).group_by(
            Review.region_code, Review.region
        ).order_by(
            desc('total_count')
        ).limit(limit)

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)

        regional_stats = []
        for row in result:
            positive_rate = (row.positive_count / row.total_count * 100) if row.total_count > 0 else 0
            negative_rate = (row.negative_count / row.total_count * 100) if row.total_count > 0 else 0

            regional_stats.append({
                "region_name": row.region,
                "region_code": row.region_code,
                "total_count": row.total_count,
                "positive_count": row.positive_count,
                "negative_count": row.negative_count,
                "neutral_count": row.neutral_count,
                "positive_rate": round(positive_rate, 2),
                "negative_rate": round(negative_rate, 2),
                "sentiment_score": round(
                    (row.positive_count - row.negative_count) / row.total_count if row.total_count > 0 else 0, 3)
            })

        return regional_stats

    async def get_city_stats(self, region_code: Optional[str] = None, limit: int = 15) -> List[Dict[str, Any]]:
        query = select(
            Review.city,
            Review.region,
            Review.region_code,
            func.count(Review.uuid).label('count'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_count'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_count')
        ).where(
            Review.city.isnot(None)
        ).group_by(
            Review.city, Review.region, Review.region_code
        ).order_by(
            desc('count')
        ).limit(limit)

        if region_code:
            query = query.where(Review.region_code == region_code)

        result = await self.session.execute(query)

        return [{
            "city": row.city,
            "region_name": row.region,
            "region_code": row.region_code,
            "count": row.count,
            "positive_count": row.positive_count,
            "negative_count": row.negative_count,
            "sentiment_score": round((row.positive_count - row.negative_count) / row.count if row.count > 0 else 0, 3)
        } for row in result]

    async def get_product_sentiment_analysis(self, limit: int = 10, filters: Optional[ReviewFilters] = None) -> List[
        Dict[str, Any]]:
        query = select(
            Review.product,
            func.count(Review.uuid).label('total_reviews'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_count'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_count'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_count')
        ).where(
            Review.product.isnot(None)
        ).group_by(
            Review.product
        ).order_by(
            desc('total_reviews')
        ).limit(limit)

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)

        products_analysis = []
        for row in result:
            if row.total_reviews > 0:
                positive_rate = (row.positive_count / row.total_reviews * 100)
                negative_rate = (row.negative_count / row.total_reviews * 100)
                sentiment_score = (row.positive_count - row.negative_count) / row.total_reviews
            else:
                positive_rate = negative_rate = sentiment_score = 0

            products_analysis.append({
                "product": row.product,
                "total_reviews": row.total_reviews,
                "positive_count": row.positive_count,
                "negative_count": row.negative_count,
                "neutral_count": row.neutral_count,
                "positive_rate": round(positive_rate, 2),
                "negative_rate": round(negative_rate, 2),
                "sentiment_score": round(sentiment_score, 3)
            })

        return products_analysis

    # ==========  МЕТОДЫ ДЛЯ ИСТОЧНИКОВ ==========

    async def get_sources_sentiment_statistics(
            self,
            filters: SourceAnalyticsFilters
    ) -> List[Dict[str, Any]]:
        """Получить детальную статистику по источникам отзывов с разбивкой по sentiment"""

        query = select(
            Review.source,
            func.count(Review.uuid).label('total_reviews'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_reviews'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_reviews'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_reviews')
        ).where(
            Review.source.isnot(None)
        ).group_by(
            Review.source
        ).order_by(
            desc('total_reviews'),
            desc(func.sum(case((Review.rating == 'positive', 1), else_=0)))
        )

        if filters.region_code:
            query = query.where(Review.region_code == filters.region_code)

        if filters.date_from:
            query = query.where(Review.datetime_review >= filters.date_from)

        if filters.date_to:
            query = query.where(Review.datetime_review <= filters.date_to)

        if filters.city:
            query = query.where(Review.city.ilike(f"%{filters.city}%"))

        if filters.product:
            query = query.where(Review.product.ilike(f"%{filters.product}%"))

        result = await self.session.execute(query)

        sources_stats = []
        for row in result:
            total = row.total_reviews
            positive = row.positive_reviews
            negative = row.negative_reviews
            neutral = row.neutral_reviews


            positive_percentage = round((positive / total * 100) if total > 0 else 0, 2)
            negative_percentage = round((negative / total * 100) if total > 0 else 0, 2)
            neutral_percentage = round((neutral / total * 100) if total > 0 else 0, 2)

            sources_stats.append({
                'source': row.source,
                'total_reviews': total,
                'positive_reviews': positive,
                'negative_reviews': negative,
                'neutral_reviews': neutral,
                'positive_percentage': positive_percentage,
                'negative_percentage': negative_percentage,
                'neutral_percentage': neutral_percentage
            })

        return sources_stats


    def _calculate_source_quality(self, total_reviews: int, sentiment_score: float) -> str:
        """Определить качество источника на основе количества отзывов и sentiment score"""
        if total_reviews >= 1000:
            if sentiment_score >= 0.3:
                return "excellent"
            elif sentiment_score >= 0:
                return "good"
            else:
                return "poor"
        elif total_reviews >= 100:
            if sentiment_score >= 0.2:
                return "good"
            elif sentiment_score >= -0.1:
                return "average"
            else:
                return "poor"
        else:
            return "limited_data"

    def _calculate_dominance_rank(self, positive: int, negative: int, neutral: int) -> str:
        """Определить доминирующий тип отзывов"""
        max_count = max(positive, negative, neutral)
        if max_count == positive:
            return "positive_dominant"
        elif max_count == negative:
            return "negative_dominant"
        else:
            return "neutral_dominant"

    # ========== НОВЫЕ МЕТОДЫ ДЛЯ РЕГИОНОВ И ПРОДУКТОВ ==========

    async def get_regions_products_sentiment_statistics(
            self,
            filters: Optional[ReviewFilters] = None,
    ) -> List[Dict[str, Any]]:
        """Получить статистику по регионам и продуктам с разбивкой по sentiment"""

        query = select(
            Review.region,
            Review.region_code,
            Review.product,
            func.count(Review.uuid).label('total_reviews'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_reviews'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_reviews'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_reviews')
        ).where(
            and_(
                Review.region.isnot(None),
                Review.region_code.isnot(None),
                Review.product.isnot(None)
            )
        ).group_by(
            Review.region,
            Review.region_code,
            Review.product

        ).order_by(
            Review.region,
            desc('total_reviews'),
            desc(func.sum(case((Review.rating == 'positive', 1), else_=0)))
        )

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)

        regions_products_stats = []
        for row in result:
            total = row.total_reviews
            positive = row.positive_reviews
            negative = row.negative_reviews
            neutral = row.neutral_reviews

            # Вычисляем проценты и sentiment score
            positive_percentage = round((positive / total * 100) if total > 0 else 0, 2)
            negative_percentage = round((negative / total * 100) if total > 0 else 0, 2)
            neutral_percentage = round((neutral / total * 100) if total > 0 else 0, 2)
            sentiment_score = round((positive - negative) / total if total > 0 else 0, 3)

            regions_products_stats.append({
                'region': row.region,
                'region_code': row.region_code,
                'product': row.product,
                'total_reviews': total,
                'positive_reviews': positive,
                'negative_reviews': negative,
                'neutral_reviews': neutral,
                'positive_percentage': positive_percentage,
                'negative_percentage': negative_percentage,
                'neutral_percentage': neutral_percentage,
            })

        return regions_products_stats

    def _evaluate_product_performance(self, sentiment_score: float, total_reviews: int) -> str:
        """Оценить производительность продукта на основе sentiment и количества отзывов"""
        if total_reviews >= 100:
            if sentiment_score >= 0.4:
                return "excellent"
            elif sentiment_score >= 0.2:
                return "good"
            elif sentiment_score >= 0:
                return "average"
            else:
                return "poor"
        elif total_reviews >= 20:
            if sentiment_score >= 0.3:
                return "good"
            elif sentiment_score >= 0:
                return "average"
            else:
                return "poor"
        else:
            return "limited_data"

    # ==========МЕТОДЫ ДЛЯ ТЕПЛОВОЙ КАРТЫ ==========

    async def get_regions_with_filtered_sentiment_heatmap(
            self,
            sentiment_filter: str,

    ) -> List[Dict[str, Any]]:
        """Получение регионов с фильтром по типу sentiment для тепловой карты"""

        query = select(
            Review.region,
            Review.region_code,
            func.count(Review.uuid).label('total_reviews'),
            func.sum(case((Review.rating == sentiment_filter, 1), else_=0)).label('target_sentiment_count'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_reviews'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_reviews'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_reviews')
        ).where(
            and_(
                Review.region.isnot(None),
                Review.region_code.isnot(None)
            )
        ).group_by(
            Review.region,
            Review.region_code

        ).order_by(
            desc('target_sentiment_count')
        )

        result = await self.session.execute(query)

        all_data = []
        for row in result:
            total = row.total_reviews
            target_count = row.target_sentiment_count

            target_percentage = (target_count / total * 100) if total > 0 else 0

            all_data.append({
                'row': row,
                'total': total,
                'target_count': target_count,
                'target_percentage': target_percentage
            })
        all_percentages = [item['target_percentage'] for item in all_data]

        regions_with_colors = []
        for item in all_data:
            rgb_color = self._get_single_sentiment_color_scheme(
                sentiment_filter, item['target_percentage'], all_percentages
            )

            regions_with_colors.append({
                'region_name': item['row'].region,
                'region_code': item['row'].region_code,
                'total_reviews': item['total'],
                'target_sentiment_count': item['target_count'],
                'target_sentiment_percentage': round(item['target_percentage'], 2),
                'positive_reviews': item['row'].positive_reviews,
                'negative_reviews': item['row'].negative_reviews,
                'neutral_reviews': item['row'].neutral_reviews,
                'color': rgb_color,  # Теперь в формате "(r g b)"
                'sentiment_filter': sentiment_filter
            })

        return regions_with_colors

    def _get_single_sentiment_color_scheme(
            self,
            sentiment_type: str,
            percentage: float,
            all_percentages: List[float]
    ) -> str:
        """Определяет RGB цвет для одного типа sentiment в оттенках выбранного цвета"""

        BASE_COLORS = {
            'positive': (34, 139, 34),  # Зеленый RGB
            'negative': (220, 20, 60),  # Красный RGB
            'neutral': (255, 215, 0)  # Желтый RGB
        }

        base_rgb = BASE_COLORS.get(sentiment_type, (128, 128, 128))

        # Определяем интенсивность на основе процентиля
        sorted_percentages = sorted(all_percentages)
        total_regions = len(sorted_percentages)

        if total_regions == 0:
            intensity = 0.3
        else:
            try:
                position = sorted_percentages.index(percentage)
            except ValueError:
                position = 0
                for i, pct in enumerate(sorted_percentages):
                    if pct >= percentage:
                        position = i
                        break

            percentile = position / total_regions if total_regions > 0 else 0

            # Интенсивность от 0.2 до 0.9
            if percentile >= 0.8:
                intensity = 0.9
            elif percentile >= 0.6:
                intensity = 0.7
            elif percentile >= 0.4:
                intensity = 0.5
            elif percentile >= 0.2:
                intensity = 0.4
            else:
                intensity = 0.2


        r = int(base_rgb[0] * intensity + 255 * (1 - intensity))
        g = int(base_rgb[1] * intensity + 255 * (1 - intensity))
        b = int(base_rgb[2] * intensity + 255 * (1 - intensity))

        return f"({r} {g} {b})"

    # ========== СТАРЫЕ МЕТОДЫ ТЕПЛОВОЙ КАРТЫ ==========

    async def get_regions_with_sentiment_colors(self) -> List[Dict[str, Any]]:
        """Получение всех регионов с sentiment анализом и цветами для карты"""

        query = select(
            Review.region,
            Review.region_code,
            func.count(Review.uuid).label('reviews_count'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_reviews'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_reviews'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_reviews')
        ).where(
            and_(Review.region.isnot(None), Review.region_code.isnot(None))
        ).group_by(
            Review.region, Review.region_code
        ).order_by(
            desc('reviews_count')
        )

        result = await self.session.execute(query)

        all_data = []
        for row in result:
            total = row.reviews_count
            positive_count = row.positive_reviews
            negative_count = row.negative_reviews

            positive_percentage = (positive_count / total * 100) if total > 0 else 0
            negative_percentage = (negative_count / total * 100) if total > 0 else 0
            sentiment_score = (positive_count - negative_count) / total if total > 0 else 0

            all_data.append({
                'row': row,
                'total': total,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'positive_percentage': positive_percentage,
                'negative_percentage': negative_percentage,
                'sentiment_score': sentiment_score
            })

        all_scores = [item['sentiment_score'] for item in all_data]

        regions_with_colors = []
        for item in all_data:
            color, intensity = self._get_sentiment_color_scheme_relative(item['sentiment_score'], all_scores)

            regions_with_colors.append({
                'region_name': item['row'].region,
                'region_code': item['row'].region_code,
                'reviews_count': item['total'],
                'positive_reviews': item['positive_count'],
                'negative_reviews': item['negative_count'],
                'neutral_reviews': item['row'].neutral_reviews,
                'positive_percentage': round(item['positive_percentage'], 2),
                'negative_percentage': round(item['negative_percentage'], 2),
                'sentiment_score': round(item['sentiment_score'], 3),
                'color': color,
                'color_intensity': intensity
            })

        return regions_with_colors

    def _get_sentiment_color_scheme_relative(self, sentiment_score: float, all_scores: List[float]) -> tuple[
        str, float]:
        """Определяет цвет и интенсивность на основе относительной позиции среди всех регионов"""

        GREEN = "#00B050"  # Для положительных
        GOLD = "#FFC000"  # Для нейтральных
        SALMON = "#e03c32"  # Для отрицательных

        sorted_scores = sorted(all_scores)
        total_regions = len(sorted_scores)

        try:
            position = sorted_scores.index(sentiment_score)
        except ValueError:
            position = 0
            for i, score in enumerate(sorted_scores):
                if score >= sentiment_score:
                    position = i
                    break

        percentile = position / total_regions if total_regions > 0 else 0

        if percentile >= 0.8:  # Топ 20%
            return GREEN, 0.8
        elif percentile >= 0.6:  # 60-80%
            return GREEN, 0.5
        elif percentile >= 0.4:  # 40-60%
            return GOLD, 0.6
        elif percentile >= 0.2:  # 20-40%
            return SALMON, 0.5
        else:  # Низшие 20%
            return SALMON, 0.8

    # ========== УТИЛИТЫ ==========

    def _apply_filters(self, query, filters: ReviewFilters):
        if filters.rating:
            query = query.where(Review.rating == filters.rating.value)

        if filters.gender:
            gender_map = {"М": "male", "Ж": "female", "": "unknown"}
            db_gender = gender_map.get(filters.gender.value, filters.gender.value)

            if db_gender == "unknown":
                query = query.where(or_(Review.gender == None, Review.gender == "", Review.gender == "unknown"))
            else:
                query = query.where(Review.gender == db_gender)

        if filters.city:
            query = query.where(Review.city.ilike(f"%{filters.city}%"))

        if filters.region_code:
            query = query.where(Review.region_code == filters.region_code)

        if filters.product:
            query = query.where(Review.product.ilike(f"%{filters.product}%"))

        if filters.date_from:
            query = query.where(Review.datetime_review >= filters.date_from)

        if filters.date_to:
            query = query.where(Review.datetime_review <= filters.date_to)

        if filters.created_from:
            query = query.where(Review.created_at >= filters.created_from)

        if filters.created_to:
            query = query.where(Review.created_at <= filters.created_to)

        if filters.sources:
            query = query.where(Review.source.in_(filters.sources))

        if filters.ratings:
            query = query.where(Review.rating.in_([r.value for r in filters.ratings]))

        if filters.cities:
            query = query.where(Review.city.in_(filters.cities))

        if filters.region_codes:
            query = query.where(Review.region_code.in_(filters.region_codes))

        return query

    async def close(self):
        await self.session.close()
