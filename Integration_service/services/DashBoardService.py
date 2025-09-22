# services/dashboard_service.py
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from ..repository import  ReviewAnalyticsRepository
from ..schemas.processed_review import ReviewFilters


class DashboardService:
    def __init__(self, analytics_repo: ReviewAnalyticsRepository):
        self.analytics_repo = analytics_repo

    async def get_dashboard_summary(self, filters: Optional[ReviewFilters] = None) -> Dict[str, Any]:
        """Основная сводка для дашборда с параллельным выполнением запросов"""

        tasks = [
            self.analytics_repo.get_total_reviews_count(filters),
            self.analytics_repo.get_sentiment_distribution(filters),
            self.analytics_repo.get_gender_distribution(filters),
            self.analytics_repo.get_source_distribution(filters),
            self.analytics_repo.get_recent_activity(minutes_back=60, limit=10),
            self.analytics_repo.get_growth_metrics(period_hours=24)
        ]

        results = await asyncio.gather(*tasks)

        return {
            'total_reviews': results[0],
            'sentiment_distribution': results[1],
            'gender_distribution': results[2],
            'source_distribution': results[3],
            'recent_activity': results[4],
            'growth_metrics': results[5],
            'filters_applied': filters.model_dump(exclude_none=True) if filters else {},
            'last_updated': datetime.now().isoformat()
        }

    async def get_regional_dashboard(self, region_code: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """Региональный дашборд с детальной аналитикой"""

        tasks = [
            self.analytics_repo.get_regional_stats(limit=limit),
            self.analytics_repo.get_city_stats(region_code=region_code, limit=15)
        ]

        if region_code:
            region_filters = ReviewFilters(region_code=region_code)
            tasks.extend([
                self.analytics_repo.get_sentiment_distribution(region_filters),
                self.analytics_repo.get_daily_trends(days_back=14, filters=region_filters)
            ])

        results = await asyncio.gather(*tasks)

        dashboard_data = {
            'regional_stats': results[0],
            'city_stats': results[1],
            'focused_region': region_code,
            'last_updated': datetime.now().isoformat()
        }

        if region_code and len(results) > 2:
            dashboard_data.update({
                'region_sentiment': results[2],
                'region_trends': results[3]
            })

        return dashboard_data

    async def get_realtime_metrics(self, minutes_back: int = 5, limit: int = 20) -> Dict[str, Any]:
        """Метрики реального времени"""

        recent_reviews = await self.analytics_repo.get_recent_activity(minutes_back, limit)

        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'unknown': 0}
        for review in recent_reviews:
            rating = review.get('rating', 'unknown')
            if rating in sentiment_counts:
                sentiment_counts[rating] += 1
            else:
                sentiment_counts['unknown'] += 1

        return {
            'recent_reviews': recent_reviews,
            'period_minutes': minutes_back,
            'total_recent': len(recent_reviews),
            'recent_sentiment_distribution': sentiment_counts,
            'current_timestamp': datetime.now().isoformat()
        }
