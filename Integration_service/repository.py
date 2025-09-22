# repository/review_analytics_repository.py
from sqlalchemy import func, text, desc, asc, case, extract, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from models.processed_reviews import Review
from schemas.processed_review import ReviewFilters, GenderFilter, RatingFilter


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

        sentiment_data = {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'unknown': 0
        }

        for row in result:
            rating = row.rating or 'unknown'
            if rating in sentiment_data:
                sentiment_data[rating] = row.count
            else:
                sentiment_data['unknown'] += row.count

        return sentiment_data

    async def get_gender_distribution(self, filters: Optional[ReviewFilters] = None) -> Dict[str, int]:
        """Распределение по полу (male/female/unknown)"""
        query = select(
            Review.gender,
            func.count(Review.uuid).label('count')
        ).group_by(Review.gender)

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)

        gender_data = {
            'male': 0,
            'female': 0,
            'unknown': 0
        }

        for row in result:
            gender = row.gender or 'unknown'
            if gender in gender_data:
                gender_data[gender] = row.count
            else:
                gender_data['unknown'] += row.count

        return gender_data

    async def get_source_distribution(self, filters: Optional[ReviewFilters] = None) -> Dict[str, int]:
        """Распределение по источникам"""
        query = select(
            Review.source,
            func.count(Review.uuid).label('count')
        ).group_by(Review.source).order_by(desc('count'))

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)
        return {row.source: row.count for row in result}

    # ========== РЕГИОНАЛЬНАЯ АНАЛИТИКА ==========

    async def get_regional_stats(self, filters: Optional[ReviewFilters] = None, limit: int = 10) -> List[
        Dict[str, Any]]:
        """Топ регионов по количеству отзывов"""
        query = select(
            Review.region_code,
            func.count(Review.uuid).label('total_count'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_count'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_count'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_count')
        ).where(Review.region_code.isnot(None)) \
            .group_by(Review.region_code) \
            .order_by(desc('total_count')) \
            .limit(limit)

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)

        regional_stats = []
        for row in result:
            positive_rate = (row.positive_count / row.total_count * 100) if row.total_count > 0 else 0
            negative_rate = (row.negative_count / row.total_count * 100) if row.total_count > 0 else 0

            regional_stats.append({
                'region_code': row.region_code,
                'total_count': row.total_count,
                'positive_count': row.positive_count,
                'negative_count': row.negative_count,
                'neutral_count': row.neutral_count,
                'positive_rate': round(positive_rate, 2),
                'negative_rate': round(negative_rate, 2),
                'sentiment_score': round(
                    (row.positive_count - row.negative_count) / row.total_count if row.total_count > 0 else 0, 3)
            })

        return regional_stats

    async def get_city_stats(self, region_code: Optional[str] = None, limit: int = 15) -> List[Dict[str, Any]]:
        """Топ городов по количеству отзывов"""
        query = select(
            Review.city,
            Review.region_code,
            func.count(Review.uuid).label('count'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_count'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_count')
        ).where(Review.city.isnot(None)) \
            .group_by(Review.city, Review.region_code) \
            .order_by(desc('count')) \
            .limit(limit)

        if region_code:
            query = query.where(Review.region_code == region_code)

        result = await self.session.execute(query)
        return [
            {
                'city': row.city,
                'region_code': row.region_code,
                'count': row.count,
                'positive_count': row.positive_count,
                'negative_count': row.negative_count,
                'sentiment_score': round((row.positive_count - row.negative_count) / row.count if row.count > 0 else 0,
                                         3)
            }
            for row in result
        ]

    # ========== ВРЕМЕННАЯ АНАЛИТИКА ==========

    async def get_daily_trends(self, days_back: int = 30, filters: Optional[ReviewFilters] = None) -> List[
        Dict[str, Any]]:
        """Тренды по дням за последние N дней"""
        start_date = datetime.now() - timedelta(days=days_back)

        query = select(
            func.date(Review.datetime_review).label('date'),
            func.count(Review.uuid).label('total'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral')
        ).where(Review.datetime_review >= start_date) \
            .group_by(func.date(Review.datetime_review)) \
            .order_by(func.date(Review.datetime_review))

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)
        return [
            {
                'date': row.date.isoformat(),
                'total': row.total,
                'positive': row.positive,
                'negative': row.negative,
                'neutral': row.neutral,
                'sentiment_ratio': round((row.positive - row.negative) / row.total if row.total > 0 else 0, 3)
            }
            for row in result
        ]

    async def get_hourly_distribution(self, filters: Optional[ReviewFilters] = None) -> List[Dict[str, Any]]:
        """Распределение отзывов по часам дня"""
        query = select(
            extract('hour', Review.datetime_review).label('hour'),
            func.count(Review.uuid).label('count'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative')
        ).group_by(extract('hour', Review.datetime_review)) \
            .order_by('hour')

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)
        return [
            {
                'hour': int(row.hour),
                'count': row.count,
                'positive': row.positive,
                'negative': row.negative,
                'sentiment_score': round((row.positive - row.negative) / row.count if row.count > 0 else 0, 3)
            }
            for row in result
        ]

    # ========== ПРОДУКТОВАЯ АНАЛИТИКА ==========

    async def get_product_sentiment_analysis(self, limit: int = 10, filters: Optional[ReviewFilters] = None) -> List[
        Dict[str, Any]]:
        """Анализ тональности по продуктам"""
        query = select(
            Review.product,
            func.count(Review.uuid).label('total_reviews'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_count'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_count'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_count')
        ).where(Review.product.isnot(None)) \
            .group_by(Review.product) \
            .order_by(desc('total_reviews')) \
            .limit(limit)

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)

        products_analysis = []
        for row in result:
            if row.total_reviews > 0:
                positive_rate = row.positive_count / row.total_reviews * 100
                negative_rate = row.negative_count / row.total_reviews * 100
                sentiment_score = (row.positive_count - row.negative_count) / row.total_reviews
            else:
                positive_rate = negative_rate = sentiment_score = 0

            products_analysis.append({
                'product': row.product,
                'total_reviews': row.total_reviews,
                'positive_count': row.positive_count,
                'negative_count': row.negative_count,
                'neutral_count': row.neutral_count,
                'positive_rate': round(positive_rate, 2),
                'negative_rate': round(negative_rate, 2),
                'sentiment_score': round(sentiment_score, 3)
            })

        return products_analysis

    # ========== РЕАЛЬНОЕ ВРЕМЯ ==========

    async def get_recent_activity(self, minutes_back: int = 60, limit: int = 50) -> List[Dict[str, Any]]:
        """Недавняя активность за последние N минут"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes_back)

        query = select(Review) \
            .where(Review.created_at >= cutoff_time) \
            .order_by(desc(Review.created_at)) \
            .limit(limit)

        result = await self.session.execute(query)
        reviews = result.scalars().all()

        return [
            {
                'uuid': str(review.uuid),
                'text': review.text[:100] + '...' if len(review.text) > 100 else review.text,
                'rating': review.rating,
                'source': review.source,
                'city': review.city,
                'region_code': review.region_code,
                'gender': review.gender,
                'created_at': review.created_at.isoformat(),
                'datetime_review': review.datetime_review.isoformat(),
                'minutes_ago': int((datetime.now() - review.created_at).total_seconds() / 60)
            }
            for review in reviews
        ]

    async def get_growth_metrics(self, period_hours: int = 24) -> Dict[str, Any]:
        """Метрики роста за период"""
        current_time = datetime.now()
        period_start = current_time - timedelta(hours=period_hours)
        previous_period_start = period_start - timedelta(hours=period_hours)

        # Текущий период
        current_query = select(
            func.count(Review.uuid).label('total'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative')
        ).where(Review.created_at >= period_start)

        # Предыдущий период
        previous_query = select(
            func.count(Review.uuid).label('total'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative')
        ).where(
            and_(
                Review.created_at >= previous_period_start,
                Review.created_at < period_start
            )
        )

        current_result = await self.session.execute(current_query)
        previous_result = await self.session.execute(previous_query)

        current = current_result.first()
        previous = previous_result.first()

        def calculate_growth(current_val, previous_val):
            if previous_val == 0:
                return 100.0 if current_val > 0 else 0.0
            return ((current_val - previous_val) / previous_val) * 100

        return {
            'current_period': {
                'total': current.total,
                'positive': current.positive,
                'negative': current.negative,
                'neutral': current.total - current.positive - current.negative
            },
            'previous_period': {
                'total': previous.total,
                'positive': previous.positive,
                'negative': previous.negative,
                'neutral': previous.total - previous.positive - previous.negative
            },
            'growth': {
                'total_growth': round(calculate_growth(current.total, previous.total), 2),
                'positive_growth': round(calculate_growth(current.positive, previous.positive), 2),
                'negative_growth': round(calculate_growth(current.negative, previous.negative), 2)
            },
            'period_hours': period_hours
        }

    # ========== ДОПОЛНИТЕЛЬНАЯ АНАЛИТИКА ==========

    async def get_text_length_stats(self, filters: Optional[ReviewFilters] = None) -> Dict[str, Any]:
        """Статистика по длине текста отзывов"""
        query = select(
            func.avg(func.length(Review.text)).label('avg_length'),
            func.min(func.length(Review.text)).label('min_length'),
            func.max(func.length(Review.text)).label('max_length'),
            func.count(Review.uuid).label('total_reviews')
        )

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)
        row = result.first()

        return {
            'average_length': round(row.avg_length or 0, 2),
            'min_length': row.min_length or 0,
            'max_length': row.max_length or 0,
            'total_reviews': row.total_reviews or 0
        }

    async def get_weekly_comparison(self) -> Dict[str, Any]:
        """Сравнение текущей недели с предыдущей"""
        current_week_start = datetime.now() - timedelta(days=7)
        previous_week_start = current_week_start - timedelta(days=7)

        # Текущая неделя
        current_week_query = select(
            func.count(Review.uuid).label('total'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative')
        ).where(Review.created_at >= current_week_start)

        # Предыдущая неделя
        previous_week_query = select(
            func.count(Review.uuid).label('total'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative')
        ).where(
            and_(
                Review.created_at >= previous_week_start,
                Review.created_at < current_week_start
            )
        )

        current_result = await self.session.execute(current_week_query)
        previous_result = await self.session.execute(previous_week_query)

        current = current_result.first()
        previous = previous_result.first()

        return {
            'current_week': {
                'total': current.total,
                'positive': current.positive,
                'negative': current.negative,
                'positive_rate': round(current.positive / current.total * 100 if current.total > 0 else 0, 2)
            },
            'previous_week': {
                'total': previous.total,
                'positive': previous.positive,
                'negative': previous.negative,
                'positive_rate': round(previous.positive / previous.total * 100 if previous.total > 0 else 0, 2)
            },
            'week_growth': round(((current.total - previous.total) / previous.total * 100) if previous.total > 0 else 0,
                                 2)
        }

    # ========== УТИЛИТЫ ==========

    def _apply_filters(self, query, filters: ReviewFilters):
        """Применение фильтров к запросу с учетом вашей модели"""
        if filters.rating:
            query = query.where(Review.rating == filters.rating.value)
        if filters.gender:
            # Маппинг enum значений на строки БД
            gender_map = {'М': 'male', 'Ж': 'female', '': 'unknown'}
            db_gender = gender_map.get(filters.gender.value, filters.gender.value)
            if db_gender == 'unknown':
                query = query.where(or_(Review.gender == None, Review.gender == '', Review.gender == 'unknown'))
            else:
                query = query.where(Review.gender == db_gender)

        if filters.city:
            query = query.where(Review.city.ilike(f"%{filters.city}%"))
        if filters.region_code:
            query = query.where(Review.region_code == filters.region_code)
        if filters.product:
            query = query.where(Review.product.ilike(f"%{filters.product}%"))

        # Фильтры по датам
        if filters.date_from:
            query = query.where(Review.datetime_review >= filters.date_from)
        if filters.date_to:
            query = query.where(Review.datetime_review <= filters.date_to)
        if filters.created_from:
            query = query.where(Review.created_at >= filters.created_from)
        if filters.created_to:
            query = query.where(Review.created_at <= filters.created_to)

        # Фильтры по спискам
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
