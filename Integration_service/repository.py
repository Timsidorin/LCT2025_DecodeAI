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
        ).group_by(Review.source)

        if filters:
            query = self._apply_filters(query, filters)

        result = await self.session.execute(query)
        return {row.source or 'unknown': row.count for row in result}

    async def get_growth_metrics(self, period_hours: int = 24) -> Dict[str, Any]:
        """Метрики роста за указанный период"""
        period_start = datetime.now() - timedelta(hours=period_hours)

        current_query = select(func.count(Review.uuid)).where(Review.created_at >= period_start)
        total_query = select(func.count(Review.uuid))

        current_result = await self.session.execute(current_query)
        total_result = await self.session.execute(total_query)

        current_count = current_result.scalar() or 0
        total_count = total_result.scalar() or 0

        return {
            'period_hours': period_hours,
            'new_reviews': current_count,
            'total_reviews': total_count,
            'growth_rate': round((current_count / total_count * 100) if total_count > 0 else 0, 2)
        }

    async def get_recent_activity(self, minutes_back: int = 60, limit: int = 10) -> List[Dict[str, Any]]:
        """Недавняя активность"""
        time_threshold = datetime.now() - timedelta(minutes=minutes_back)

        query = select(
            Review.uuid,
            Review.text,
            Review.rating,
            Review.region,
            Review.city,
            Review.created_at
        ).where(
            Review.created_at >= time_threshold
        ).order_by(desc(Review.created_at)).limit(limit)

        result = await self.session.execute(query)

        return [
            {
                'uuid': str(row.uuid),
                'text': row.text[:100] + "..." if len(row.text) > 100 else row.text,
                'rating': row.rating,
                'region': row.region,
                'city': row.city,
                'created_at': row.created_at.isoformat() if row.created_at else None
            }
            for row in result
        ]

    # ========== РЕГИОНАЛЬНАЯ АНАЛИТИКА ==========

    async def get_unique_regions(self) -> List[Dict[str, Any]]:
        """Получение всех уникальных регионов с их кодами"""
        query = select(
            Review.region,
            Review.region_code,
            func.count(Review.uuid).label('reviews_count')
        ).where(
            and_(
                Review.region.isnot(None),
                Review.region_code.isnot(None)
            )
        ).group_by(
            Review.region,
            Review.region_code
        ).order_by(
            desc('reviews_count'),
            Review.region
        )

        result = await self.session.execute(query)

        regions = []
        for row in result:
            regions.append({
                'region_name': row.region,
                'region_code': row.region_code,
                'reviews_count': row.reviews_count
            })

        return regions

    async def get_unique_region_codes(self) -> List[Dict[str, Any]]:
        """Получение только уникальных кодов регионов"""
        query = select(
            Review.region_code,
            func.count(Review.uuid).label('reviews_count')
        ).where(
            Review.region_code.isnot(None)
        ).group_by(
            Review.region_code
        ).order_by(
            desc('reviews_count'),
            Review.region_code
        )

        result = await self.session.execute(query)

        return [
            {
                'region_code': row.region_code,
                'reviews_count': row.reviews_count
            }
            for row in result
        ]

    async def get_regions_hierarchy(self) -> Dict[str, Any]:
        """Получение иерархии регионов с городами"""
        query = select(
            Review.region,
            Review.region_code,
            Review.city,
            func.count(Review.uuid).label('reviews_count')
        ).where(
            and_(
                Review.region.isnot(None),
                Review.region_code.isnot(None)
            )
        ).group_by(
            Review.region,
            Review.region_code,
            Review.city
        ).order_by(
            Review.region,
            desc('reviews_count')
        )

        result = await self.session.execute(query)
        regions_hierarchy = {}

        for row in result:
            region_key = f"{row.region} ({row.region_code})"

            if region_key not in regions_hierarchy:
                regions_hierarchy[region_key] = {
                    'region_name': row.region,
                    'region_code': row.region_code,
                    'total_reviews': 0,
                    'cities': []
                }

            regions_hierarchy[region_key]['total_reviews'] += row.reviews_count

            if row.city:
                regions_hierarchy[region_key]['cities'].append({
                    'city_name': row.city,
                    'reviews_count': row.reviews_count
                })

        return {
            'regions': list(regions_hierarchy.values()),
            'total_regions': len(regions_hierarchy)
        }

    async def get_regional_stats(self, filters: Optional[ReviewFilters] = None, limit: int = 10) -> List[
        Dict[str, Any]]:
        """Топ регионов по количеству отзывов"""
        query = select(
            Review.region_code,
            Review.region,
            func.count(Review.uuid).label('total_count'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_count'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_count'),
            func.sum(case((Review.rating == 'neutral', 1), else_=0)).label('neutral_count')
        ).where(Review.region_code.isnot(None)) \
            .group_by(Review.region_code, Review.region) \
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
                'region_name': row.region,
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
            Review.region,
            Review.region_code,
            func.count(Review.uuid).label('count'),
            func.sum(case((Review.rating == 'positive', 1), else_=0)).label('positive_count'),
            func.sum(case((Review.rating == 'negative', 1), else_=0)).label('negative_count')
        ).where(Review.city.isnot(None)) \
            .group_by(Review.city, Review.region, Review.region_code) \
            .order_by(desc('count')) \
            .limit(limit)

        if region_code:
            query = query.where(Review.region_code == region_code)

        result = await self.session.execute(query)
        return [
            {
                'city': row.city,
                'region_name': row.region,
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

    async def get_region_basic_info(self, region_code: str) -> Optional[Dict[str, Any]]:
        """Получить базовую информацию о регионе"""
        query = select(
            Review.region,
            Review.region_code,
            func.count(Review.uuid).label('total_reviews'),
            func.count(func.distinct(Review.city)).label('unique_cities')
        ).where(Review.region_code == region_code).group_by(
            Review.region, Review.region_code
        )

        result = await self.session.execute(query)
        region_info = result.first()

        if region_info:
            return {
                'region_name': region_info.region,
                'region_code': region_info.region_code,
                'total_reviews': region_info.total_reviews,
                'unique_cities': region_info.unique_cities
            }
        return None

    async def get_cities_by_region_query(
            self,
            region_code: Optional[str] = None,
            min_reviews: int = 1,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Получить города с фильтрацией по региону"""
        query = select(
            Review.city,
            Review.region,
            Review.region_code,
            func.count(Review.uuid).label('reviews_count')
        ).where(
            Review.city.isnot(None)
        ).group_by(
            Review.city,
            Review.region,
            Review.region_code
        ).having(
            func.count(Review.uuid) >= min_reviews
        ).order_by(
            desc('reviews_count'),
            Review.city
        ).limit(limit)

        if region_code:
            query = query.where(Review.region_code == region_code)

        result = await self.session.execute(query)

        return [
            {
                'city_name': row.city,
                'region_name': row.region,
                'region_code': row.region_code,
                'reviews_count': row.reviews_count
            }
            for row in result
        ]

    # ========== SENTIMENT MAP METHODS ==========

    async def get_regions_with_sentiment_colors(self, min_reviews: int = 0) -> List[Dict[str, Any]]:
        """Получение регионов с sentiment анализом и цветами для карты"""
        query = select(
            Review.region,
            Review.region_code,
            func.count(Review.uuid).label('reviews_count'),
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
        ).having(
            func.count(Review.uuid) >= min_reviews
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
            sentiment_score = ((positive_count - negative_count) / total) if total > 0 else 0

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
            color, intensity = self._get_sentiment_color_scheme_relative(
                item['sentiment_score'], all_scores
            )

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
        """Определяет цвет относительно всех регионов"""
        GREEN = "#228B22"  # Лучшие регионы
        GOLD = "#E2B007"  # Средние регионы
        SALMON = "#FA8072"  # Худшие регионы

        # Сортируем все scores
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

        if percentile >= 0.8:  # Топ 20% - зеленый
            return GREEN, 0.8
        elif percentile >= 0.6:  # 60-80% - светло-зеленый
            return GREEN, 0.5
        elif percentile >= 0.4:  # 40-60% - желтый
            return GOLD, 0.6
        elif percentile >= 0.2:  # 20-40% - оранжевый
            return SALMON, 0.5
        else:  # Нижние 20% - красный
            return SALMON, 0.8


    def _apply_filters(self, query, filters: ReviewFilters):
        """Применение фильтров к запросу с учетом вашей модели"""
        if filters.rating:
            query = query.where(Review.rating == filters.rating.value)
        if filters.gender:
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
