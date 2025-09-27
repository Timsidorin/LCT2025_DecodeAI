import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.schemas.processed_review import ReviewFilters, SourceAnalyticsFilters


class DashboardService:
    def __init__(self, analytics_repo: ReviewAnalyticsRepository):
        self.analytics_repo = analytics_repo

    async def get_dashboard_summary(
            self,
            filters: Optional[ReviewFilters] = None
    ) -> Dict[str, Any]:
        """Получить основную сводку дашборда"""

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
            "overview": {
                "total_reviews": base_results[0],
                "sentiment_distribution": base_results[1],
                "gender_distribution": base_results[2],
                "source_distribution": base_results[3],
                "growth_metrics": base_results[4],
                "recent_activity": additional_results[0],
                "insights": additional_results[1],
                "performance": additional_results[2]
            },
            "filters_applied": filters.model_dump(exclude_none=True) if filters else {},
            "last_updated": datetime.now().isoformat()
        }

    # ==========  МЕТОДЫ ДЛЯ ИСТОЧНИКОВ ==========

    async def get_sources_dashboard_statistics(
            self,
            filters: SourceAnalyticsFilters
    ) -> Dict[str, Any]:
        """Получить упрощенную статистику по источникам для дашборда"""

        sources_data = await self.analytics_repo.get_sources_sentiment_statistics(filters)

        return {
            "sources": sources_data
        }

    def _classify_period_length(self, days: int) -> str:
        """Классифицировать длительность периода"""
        if days <= 7:
            return "weekly"
        elif days <= 31:
            return "monthly"
        elif days <= 92:
            return "quarterly"
        elif days <= 366:
            return "yearly"
        else:
            return "multi_year"

    def _generate_regional_source_insights(self, sources_data: List[Dict], region_code: str) -> List[str]:
        """Генерировать инсайты по источникам в регионе"""
        if not sources_data:
            return []

        insights = []
        total_reviews = sum(s['total_reviews'] for s in sources_data)

        # Доминирующий источник
        top_source = max(sources_data, key=lambda x: x['total_reviews'])
        top_share = (top_source['total_reviews'] / total_reviews * 100)

        if top_share > 50:
            insights.append(
                f"Источник '{top_source['source']}' доминирует в регионе {region_code} с долей {top_share:.1f}%")

        return insights

    # ========== НОВЫЕ МЕТОДЫ ДЛЯ РЕГИОНОВ И ПРОДУКТОВ ==========

    async def get_regions_products_dashboard_statistics(
            self,
            filters: Optional[ReviewFilters] = None,
    ) -> Dict[str, Any]:
        """Получить полную статистику по регионам и продуктам для дашборда"""

        regions_products_data = await self.analytics_repo.get_regions_products_sentiment_statistics(
            filters )

        if not regions_products_data:
            return {
                'regions_products': [],
                'summary': {},
                'analytics': {},
                'timestamp': datetime.now().isoformat()
            }

        regions_dict = {}
        for item in regions_products_data:
            region_key = item['region_code']
            if region_key not in regions_dict:
                regions_dict[region_key] = []
            regions_dict[region_key].append(item)

        # Присваиваем ранги
        for region_products in regions_dict.values():
            sorted_products = sorted(region_products, key=lambda x: x['total_reviews'], reverse=True)
            for i, product in enumerate(sorted_products, 1):
                product['regional_product_rank'] = i

        # Общая статистика
        total_combinations = len(regions_products_data)
        total_reviews = sum(item['total_reviews'] for item in regions_products_data)
        total_positive = sum(item['positive_reviews'] for item in regions_products_data)
        total_negative = sum(item['negative_reviews'] for item in regions_products_data)
        total_neutral = sum(item['neutral_reviews'] for item in regions_products_data)


        return {
            'regions_products': regions_products_data,
        }

    # ========== НОВЫЕ МЕТОДЫ ДЛЯ ТЕПЛОВОЙ КАРТЫ ==========

    async def get_regions_sentiment_heatmap_filtered(
            self,
            sentiment_filter: str,
    ) -> Dict[str, Any]:
        """Получить регионы с фильтром по типу sentiment для тепловой карты"""

        regions_data = await self.analytics_repo.get_regions_with_filtered_sentiment_heatmap(
            sentiment_filter
        )

        # Цветовая схема в зависимости от выбранного sentiment
        color_schemes = {
            'positive': {
                'name': 'Зеленые оттенки',
                'base_color': '(34 139 34)',
                'description': 'Интенсивность зеленого показывает концентрацию позитивных отзывов'
            },
            'negative': {
                'name': 'Красные оттенки',
                'base_color': '(220 20 60)',
                'description': 'Интенсивность красного показывает концентрацию негативных отзывов'
            },
            'neutral': {
                'name': 'Желтые оттенки',
                'base_color': '(255 215 0)',
                'description': 'Интенсивность желтого показывает концентрацию нейтральных отзывов'
            }
        }
        response = {
            'regions': regions_data,
            'total_regions': len(regions_data),
            'sentiment_filter': sentiment_filter,
            'color_scheme': color_schemes.get(sentiment_filter, color_schemes['positive']),
        }

        return response

    # ========== СТАРЫЕ МЕТОДЫ (СОХРАНЕНЫ ДЛЯ СОВМЕСТИМОСТИ) ==========

    async def get_regional_dashboard(
            self,
            region_code: Optional[str] = None,
            limit: int = 20
    ) -> Dict[str, Any]:
        """Получить региональную аналитику"""

        base_tasks = [
            self.analytics_repo.get_regional_stats(limit=limit),
            self.analytics_repo.get_city_stats(region_code=region_code, limit=15)
        ]

        if region_code:
            region_filters = ReviewFilters(region_code=region_code)
            base_tasks.extend([
                self.analytics_repo.get_sentiment_distribution(region_filters),
                self.analytics_repo.get_product_sentiment_analysis(10, region_filters)
            ])

        results = await asyncio.gather(*base_tasks)

        dashboard_data = {
            "regional_overview": {
                "regional_stats": results[0],
                "cities_analysis": results[1]
            },
            "regional_insights": await self._generate_regional_insights(results[0]),
            "focused_region": region_code,
            "last_updated": datetime.now().isoformat()
        }

        if region_code and len(results) > 2:
            dashboard_data["region_details"] = {
                "sentiment_distribution": results[2],
                "top_products": results[3]
            }

        return dashboard_data

    async def get_all_regions(self, include_cities: bool = False, ) -> Dict[str, Any]:
        """Получить все регионы"""
        if include_cities:
            regions_data = await self.analytics_repo.get_regions_hierarchy()
            filtered_regions = [region for region in regions_data["regions"]]
            regions_data["regions"] = filtered_regions
            regions_data["total_regions"] = len(filtered_regions)

            return {
                "region_hierarchy": regions_data,
                "include_cities": True,
            }
        else:
            regions = await self.analytics_repo.get_unique_regions()
            regions = [region for region in regions ]

            return {
                "regions": regions,
                "total_regions": len(regions),
                "include_cities": False,
            }

    async def get_region_codes_only(self) -> Dict[str, Any]:
        """Получить только коды регионов"""
        region_codes = await self.analytics_repo.get_unique_region_codes()
        formatted_codes = [{"value": code["region_code"], "label": code["region_code"], "count": code["reviews_count"]}
                           for code in region_codes]

        return {
            "region_codes": region_codes,
            "formatted_codes": formatted_codes,
            "total_codes": len(region_codes),
            "timestamp": datetime.now().isoformat()
        }

    async def get_regions_sentiment_map(self, include_cities: bool = False) -> Dict[str, Any]:
        """Получить регионы с sentiment анализом для карты"""

        regions_data = await self.analytics_repo.get_regions_with_sentiment_colors()

        if regions_data:
            total_regions = len(regions_data)
            positive_regions = len([r for r in regions_data if r["sentiment_score"] > 0.1])
            negative_regions = len([r for r in regions_data if r["sentiment_score"] < -0.1])

            best_region = max(regions_data, key=lambda x: x["sentiment_score"])
            worst_region = min(regions_data, key=lambda x: x["sentiment_score"])
        else:
            analytics = {}

        response = {
            "regions": regions_data,
            "total_regions": len(regions_data),
            "include_cities": include_cities,
            "color_scheme": {"positive": "#228B22", "neutral": "#E2B007", "negative": "#FA8072"},
        }

        return response

    # ========== УТИЛИТЫ ==========

    async def _get_quick_insights(self, filters: Optional[ReviewFilters] = None) -> Dict[str, Any]:
        """Получить быстрые инсайты"""
        sentiment_data = await self.analytics_repo.get_sentiment_distribution(filters)
        total = sum(sentiment_data.values())
        insights = []

        if total > 0:
            positive_rate = sentiment_data.get("positive", 0) / total
            negative_rate = sentiment_data.get("negative", 0) / total

            if positive_rate > 0.7:
                insights.append({"type": "positive",
                                 "message": f"Высокий процент положительных отзывов: {positive_rate * 100:.1f}%",
                                 "priority": "info"})
            elif negative_rate > 0.3:
                insights.append({"type": "warning", "message": f"Много негативных отзывов: {negative_rate * 100:.1f}%",
                                 "priority": "warning"})

        return {
            "insights": insights,
            "insights_count": len(insights),
            "data_quality": "good" if total > 100 else "limited"
        }

    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Получить метрики производительности"""
        regions_count = len(await self.analytics_repo.get_unique_regions())

        return {
            "data_freshness": "excellent",
            "processing_speed": "fast",
            "coverage": {"regions_covered": regions_count},
            "data_completeness": 95.5
        }

    async def _generate_regional_insights(self, regional_stats: List[Dict]) -> List[Dict[str, Any]]:
        """Генерировать региональные инсайты"""
        insights = []

        if regional_stats:
            top_region = max(regional_stats, key=lambda x: x["total_count"])
            insights.append({
                "type": "leader",
                "message": f"Лидер по отзывам: {top_region['region_code']} ({top_region['total_count']} отзывов)",
                "data": top_region
            })

            positive_regions = [r for r in regional_stats if r.get("positive_rate", 0) > 60]
            if positive_regions:
                insights.append({
                    "type": "positive_regions",
                    "message": f"Регионов с высокой положительной оценкой: {len(positive_regions)} из {len(regional_stats)}",
                    "data": positive_regions[:3]
                })

        return insights
