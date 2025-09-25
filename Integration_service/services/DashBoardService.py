# services/dashboard_service.py
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from repository import  ReviewAnalyticsRepository
from schemas.processed_review import ReviewFilters


class DashboardService:
    def __init__(self, analytics_repo: ReviewAnalyticsRepository):
        self.analytics_repo = analytics_repo

    async def get_dashboard_summary(
            self,
            filters: Optional[ReviewFilters] = None
    ) -> Dict[str, Any]:
        base_tasks = [
            self.analytics_repo.get_total_reviews_count(filters),
            self.analytics_repo.get_sentiment_distribution(filters),
            self.analytics_repo.get_gender_distribution(filters),
            self.analytics_repo.get_source_distribution(filters),
            self.analytics_repo.get_growth_metrics(period_hours=24)
        ]

        base_results = await asyncio.gather(*base_tasks)
        additional_tasks = [
            self.analytics_repo.get_recent_activity(minutes_back=60, limit=10),
            self._get_quick_insights(filters),
            self._get_performance_metrics()
        ]

        additional_results = await asyncio.gather(*additional_tasks)

        return {
            'overview': {
                'total_reviews': base_results[0],
                'sentiment_distribution': base_results[1],
                'gender_distribution': base_results[2],
                'source_distribution': base_results[3]
            },
            'growth_metrics': base_results[4],
            'recent_activity': additional_results[0],
            'insights': additional_results[1],
            'performance': additional_results[2],
            'filters_applied': filters.model_dump(exclude_none=True) if filters else {},
            'last_updated': datetime.now().isoformat()
        }

    async def get_regions_sentiment_heatmap_filtered(
            self,
            sentiment_filter: str,
            min_reviews: int = 0
    ) -> Dict[str, Any]:
        """Получить регионы с фильтром по типу sentiment для тепловой карты"""

        regions_data = await self.analytics_repo.get_regions_with_filtered_sentiment_heatmap(
            sentiment_filter, min_reviews
        )

        # Аналитика по отфильтрованным данным
        if regions_data:
            total_regions = len(regions_data)
            total_target_reviews = sum(r['target_sentiment_count'] for r in regions_data)
            total_all_reviews = sum(r['total_reviews'] for r in regions_data)

            # Топ регионы по целевому sentiment
            top_regions = sorted(
                regions_data,
                key=lambda x: x['target_sentiment_count'],
                reverse=True
            )[:5]

            # Регионы с высоким процентом
            high_percentage_regions = [
                r for r in regions_data
                if r['target_sentiment_percentage'] > 70
            ]

            analytics = {
                'total_regions': total_regions,
                'total_target_reviews': total_target_reviews,
                'total_all_reviews': total_all_reviews,
                'average_target_percentage': round(
                    sum(r['target_sentiment_percentage'] for r in regions_data) / total_regions, 2
                ),
                'top_regions': [
                    {
                        'region_name': r['region_name'],
                        'region_code': r['region_code'],
                        'count': r['target_sentiment_count'],
                        'percentage': r['target_sentiment_percentage']
                    }
                    for r in top_regions
                ],
                'high_percentage_regions_count': len(high_percentage_regions),
                'sentiment_filter_applied': sentiment_filter
            }
        else:
            analytics = {}

        # Цветовая схема в зависимости от выбранного sentiment
        color_schemes = {
            'positive': {
                'name': 'Зеленые оттенки',
                'base_color': '#228B22',
                'description': 'Интенсивность зеленого показывает концентрацию позитивных отзывов'
            },
            'negative': {
                'name': 'Красные оттенки',
                'base_color': '#DC143C',
                'description': 'Интенсивность красного показывает концентрацию негативных отзывов'
            },
            'neutral': {
                'name': 'Желтые оттенки',
                'base_color': '#FFD700',
                'description': 'Интенсивность желтого показывает концентрацию нейтральных отзывов'
            }
        }

        response = {
            'regions': regions_data,
            'total_regions': len(regions_data),
            'sentiment_filter': sentiment_filter,
            'min_reviews_filter': min_reviews,
            'color_scheme': color_schemes.get(sentiment_filter, color_schemes['positive']),
            'analytics': analytics,
            'timestamp': datetime.now().isoformat()
        }

        return response


    async def get_regional_dashboard(
            self,
            region_code: Optional[str] = None,
            limit: int = 20
    ) -> Dict[str, Any]:
        base_tasks = [
            self.analytics_repo.get_regional_stats(limit=limit),
            self.analytics_repo.get_city_stats(region_code=region_code, limit=15)
        ]

        if region_code:
            region_filters = ReviewFilters(region_code=region_code)
            base_tasks.extend([
                self.analytics_repo.get_sentiment_distribution(region_filters),
                self.analytics_repo.get_daily_trends(days_back=14, filters=region_filters),
                self.analytics_repo.get_region_basic_info(region_code)
            ])

        results = await asyncio.gather(*base_tasks)

        dashboard_data = {
            'regional_overview': {
                'regional_stats': results[0],
                'cities_analysis': results[1]
            },
            'regional_insights': await self._generate_regional_insights(results[0]),
            'focused_region': region_code,
            'last_updated': datetime.now().isoformat()
        }
        if region_code and len(results) > 2:
            dashboard_data['region_details'] = {
                'sentiment_distribution': results[2],
                'daily_trends': results[3],
                'region_info': results[4]
            }
            dashboard_data['region_recommendations'] = await self._generate_region_recommendations(results[2],
                                                                                                   results[4])

        return dashboard_data


    # ========== РЕГИОНАЛЬНЫЕ МЕТОДЫ ==========

    async def get_all_regions(
            self,
            include_cities: bool = False,
            min_reviews: int = 0
    ) -> Dict[str, Any]:
        """Получить все регионы с возможностью включения городов"""
        if include_cities:
            regions_data = await self.analytics_repo.get_regions_hierarchy()

            if min_reviews > 0:
                filtered_regions = [
                    region for region in regions_data['regions']
                    if region['total_reviews'] >= min_reviews
                ]
                regions_data['regions'] = filtered_regions
                regions_data['total_regions'] = len(filtered_regions)

            return {
                'regions_hierarchy': regions_data,
                'include_cities': True,
                'min_reviews_filter': min_reviews,
                'timestamp': datetime.now().isoformat()
            }
        else:
            regions = await self.analytics_repo.get_unique_regions()
            if min_reviews > 0:
                regions = [
                    region for region in regions
                    if region['reviews_count'] >= min_reviews
                ]

            return {
                'regions': regions,
                'total_regions': len(regions),
                'include_cities': False,
                'min_reviews_filter': min_reviews,
                'timestamp': datetime.now().isoformat()
            }

    async def get_region_codes_only(self) -> Dict[str, Any]:
        region_codes = await self.analytics_repo.get_unique_region_codes()
        formatted_codes = [
            {
                'value': code['region_code'],
                'label': code['region_code'],
                'count': code['reviews_count']
            }
            for code in region_codes
        ]

        return {
            'region_codes': region_codes,
            'formatted_codes': formatted_codes,
            'total_codes': len(region_codes),
            'timestamp': datetime.now().isoformat()
        }

    async def get_region_details(self, region_code: str) -> Dict[str, Any]:
        """Получить детальную информацию по конкретному региону"""
        region_info = await self.analytics_repo.get_region_basic_info(region_code)

        if not region_info:
            raise ValueError(f"Регион с кодом {region_code} не найден")
        filters = ReviewFilters(region_code=region_code)

        tasks = [
            self.analytics_repo.get_sentiment_distribution(filters),
            self.analytics_repo.get_city_stats(region_code=region_code, limit=20),
            self.analytics_repo.get_daily_trends(days_back=30, filters=filters),
            self.analytics_repo.get_gender_distribution(filters),
            self.analytics_repo.get_product_sentiment_analysis(10, filters)
        ]

        results = await asyncio.gather(*tasks)

        return {
            'region_info': region_info,
            'analytics': {
                'sentiment_distribution': results[0],
                'gender_distribution': results[3],
                'top_cities': results[1],
                'daily_trends': results[2],
                'top_products': results[4]
            },
            'summary': await self._calculate_region_summary(results[0], results[1]),
            'timestamp': datetime.now().isoformat()
        }

    async def get_cities_by_region(
            self,
            region_code: Optional[str] = None,
            min_reviews: int = 1,
            limit: int = 100
    ) -> Dict[str, Any]:
        """Получить города с фильтрацией по региону"""
        cities = await self.analytics_repo.get_cities_by_region_query(
            region_code, min_reviews, limit
        )

        regions_summary = {}
        for city in cities:
            region_key = city['region_code']
            if region_key not in regions_summary:
                regions_summary[region_key] = {
                    'region_name': city['region_name'],
                    'region_code': city['region_code'],
                    'cities_count': 0,
                    'total_reviews': 0
                }

            regions_summary[region_key]['cities_count'] += 1
            regions_summary[region_key]['total_reviews'] += city['reviews_count']

        return {
            'cities': cities,
            'regions_summary': list(regions_summary.values()),
            'filters_applied': {
                'region_code': region_code,
                'min_reviews': min_reviews,
                'limit': limit
            },
            'total_cities': len(cities),
            'timestamp': datetime.now().isoformat()
        }

    async def get_regions_sentiment_map(
            self,
            min_reviews: int = 0,
            include_cities: bool = False
    ) -> Dict[str, Any]:
        """Получить регионы с sentiment анализом и цветами для карты"""

        regions_data = await self.analytics_repo.get_regions_with_sentiment_colors(min_reviews)

        # Дополнительная аналитика
        if regions_data:
            total_regions = len(regions_data)
            positive_regions = len([r for r in regions_data if r['sentiment_score'] > 0.1])
            negative_regions = len([r for r in regions_data if r['sentiment_score'] < -0.1])

            # Топ и худшие регионы
            best_region = max(regions_data, key=lambda x: x['sentiment_score'])
            worst_region = min(regions_data, key=lambda x: x['sentiment_score'])

            analytics = {
                'total_regions': total_regions,
                'positive_regions_count': positive_regions,
                'negative_regions_count': negative_regions,
                'neutral_regions_count': total_regions - positive_regions - negative_regions,
                'best_sentiment_region': {
                    'name': best_region['region_name'],
                    'code': best_region['region_code'],
                    'score': best_region['sentiment_score']
                },
                'worst_sentiment_region': {
                    'name': worst_region['region_name'],
                    'code': worst_region['region_code'],
                    'score': worst_region['sentiment_score']
                }
            }
        else:
            analytics = {}

        response = {
            'regions': regions_data,
            'total_regions': len(regions_data),
            'include_cities': include_cities,
            'min_reviews_filter': min_reviews,
            'color_scheme': {
                'positive': '#228B22',
                'neutral': '#E2B007',
                'negative': '#FA8072'
            },
            'analytics': analytics,
            'timestamp': datetime.now().isoformat()
        }

        return response


    # ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========

    async def _get_quick_insights(self, filters: Optional[ReviewFilters] = None) -> Dict[str, Any]:
        """Быстрые инсайты на основе данных"""
        sentiment_data = await self.analytics_repo.get_sentiment_distribution(filters)
        total = sum(sentiment_data.values())

        insights = []

        if total > 0:
            positive_rate = sentiment_data.get('positive', 0) / total
            negative_rate = sentiment_data.get('negative', 0) / total

            if positive_rate > 0.7:
                insights.append({
                    'type': 'positive',
                    'message': f'Высокий уровень позитивных отзывов: {positive_rate:.1%}',
                    'priority': 'info'
                })
            elif negative_rate > 0.3:
                insights.append({
                    'type': 'warning',
                    'message': f'Повышенный уровень негативных отзывов: {negative_rate:.1%}',
                    'priority': 'warning'
                })

        return {
            'insights': insights,
            'insights_count': len(insights),
            'data_quality': 'good' if total > 100 else 'limited'
        }

    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Метрики производительности системы"""
        regions_count = len(await self.analytics_repo.get_unique_regions())

        return {
            'data_freshness': 'excellent',
            'processing_speed': 'fast',
            'coverage': {
                'regions_covered': regions_count,
                'data_completeness': 95.5
            }
        }

    async def _generate_regional_insights(self, regional_stats: List[Dict]) -> List[Dict[str, Any]]:
        """Генерация инсайтов по региональным данным"""
        insights = []

        if regional_stats:
            # Находим регион-лидер
            top_region = max(regional_stats, key=lambda x: x['total_count'])
            insights.append({
                'type': 'leader',
                'message': f"Регион-лидер: {top_region['region_code']} с {top_region['total_count']} отзывами",
                'data': top_region
            })

            # Анализируем тональность по регионам
            positive_regions = [r for r in regional_stats if r.get('positive_rate', 0) > 60]
            if positive_regions:
                insights.append({
                    'type': 'positive_regions',
                    'message': f"Регионы с высокой позитивностью: {len(positive_regions)} из {len(regional_stats)}",
                    'data': positive_regions[:3]  # Топ-3
                })

        return insights

    async def _generate_region_recommendations(
            self,
            sentiment_data: Dict[str, int],
            region_info: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Рекомендации на основе анализа региона"""
        recommendations = []

        if sentiment_data and sentiment_data.get('negative', 0) > sentiment_data.get('positive', 0):
            recommendations.append({
                'type': 'improvement',
                'priority': 'high',
                'message': 'Рекомендуется проанализировать причины негативных отзывов',
                'action': 'Провести детальный анализ негативных отзывов в регионе'
            })

        return recommendations

    async def _get_realtime_trends(self, minutes_back: int) -> Dict[str, Any]:
        """Тренды в реальном времени"""
        return {
            'trend_direction': 'stable',
            'activity_change': 0.05,  # 5% рост
            'peak_hours_detected': False
        }

    async def _get_activity_heatmap(self, minutes_back: int) -> Dict[str, Any]:
        """Тепловая карта активности"""
        return {
            'heatmap_data': [],
            'peak_periods': [],
            'low_activity_periods': []
        }

    async def _analyze_recent_activity(self, recent_reviews: List[Dict]) -> Dict[str, Any]:
        """Анализ недавней активности"""
        if not recent_reviews:
            return {'status': 'no_activity', 'analysis': {}}

        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'unknown': 0}
        regions_activity = {}

        for review in recent_reviews:
            rating = review.get('rating', 'unknown')
            if rating in sentiment_counts:
                sentiment_counts[rating] += 1
            else:
                sentiment_counts['unknown'] += 1

            region = review.get('region_code')
            if region:
                regions_activity[region] = regions_activity.get(region, 0) + 1

        return {
            'status': 'active',
            'sentiment_trend': max(sentiment_counts, key=sentiment_counts.get),
            'most_active_regions': sorted(regions_activity.items(), key=lambda x: x[1], reverse=True)[:3],
            'activity_score': len(recent_reviews) / 20 * 100
        }


    async def _calculate_region_summary(
            self,
            sentiment_data: Dict[str, int],
            cities_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Вычислить сводную информацию по региону"""
        total_reviews = sum(sentiment_data.values())

        sentiment_summary = {
            'dominant_sentiment': max(sentiment_data, key=sentiment_data.get) if sentiment_data else 'unknown',
            'positive_rate': round(
                (sentiment_data.get('positive', 0) / total_reviews * 100) if total_reviews > 0 else 0, 2),
            'negative_rate': round(
                (sentiment_data.get('negative', 0) / total_reviews * 100) if total_reviews > 0 else 0, 2)
        }

        cities_summary = {
            'total_cities': len(cities_data),
            'most_active_city': cities_data[0]['city'] if cities_data else None,
            'cities_with_reviews': len([city for city in cities_data if city['count'] > 0])
        }

        return {
            'sentiment_summary': sentiment_summary,
            'cities_summary': cities_summary,
            'overall_activity_score': self._calculate_activity_score(total_reviews, len(cities_data))
        }

    def _calculate_activity_score(self, total_reviews: int, cities_count: int) -> float:
        """Вычислить показатель активности региона"""
        if cities_count == 0:
            return 0.0

        base_score = total_reviews / cities_count
        diversity_bonus = min(cities_count / 10, 1.0)

        return round(base_score * (1 + diversity_bonus), 2)
