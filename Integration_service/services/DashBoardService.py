import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.schemas.processed_review import ReviewFilters, SourceAnalyticsFilters, ProductsAnalysisFilters


class DashboardService:
    def __init__(self, analytics_repo: ReviewAnalyticsRepository):
        self.analytics_repo = analytics_repo

    async def get_dashboard_summary(
            self,
            filters: Optional[ReviewFilters] = None
    ) -> Dict[str, Any]:

        try:
            total_reviews = await self.analytics_repo.get_total_reviews_count(filters)
            sentiment_distribution = await self.analytics_repo.get_sentiment_distribution(filters)
            gender_distribution = await self.analytics_repo.get_gender_distribution(filters)
            source_distribution = await self.analytics_repo.get_source_distribution(filters)
            growth_metrics = await self.analytics_repo.get_growth_metrics(period_hours=24)

            recent_activity = await self.analytics_repo.get_recent_activity(minutes_back=60, limit=10)
            insights = await self._get_quick_insights(filters)
            performance = await self._get_performance_metrics()

            return {
                "overview": {
                    "total_reviews": total_reviews,
                    "sentiment_distribution": sentiment_distribution,
                    "gender_distribution": gender_distribution,
                    "source_distribution": source_distribution,
                    "growth_metrics": growth_metrics,
                    "recent_activity": recent_activity,
                    "insights": insights,
                    "performance": performance
                },
                "filters_applied": filters.model_dump(exclude_none=True) if filters else {},
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "overview": {
                    "total_reviews": 0,
                    "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                    "gender_distribution": {"male": 0, "female": 0},
                    "source_distribution": {},
                    "growth_metrics": {"growth_24h": 0},
                    "recent_activity": [],
                    "insights": {"insights": [], "insights_count": 0},
                    "performance": {"data_freshness": "limited"}
                },
                "filters_applied": filters.model_dump(exclude_none=True) if filters else {},
                "last_updated": datetime.now().isoformat(),
                "error": "Данные временно недоступны"
            }

    # ==========  МЕТОДЫ ДЛЯ ИСТОЧНИКОВ ==========

    async def get_sources_dashboard_statistics(
            self,
            filters: SourceAnalyticsFilters
    ) -> Dict[str, Any]:
        """Получить упрощенную статистику по источникам для дашборда"""
        try:
            sources_data = await self.analytics_repo.get_sources_sentiment_statistics(filters)
            return {
                "sources": sources_data
            }
        except Exception as e:
            print(f"Ошибка в get_sources_dashboard_statistics: {str(e)}")
            return {"sources": []}

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
        try:
            regions_products_data = await self.analytics_repo.get_regions_products_sentiment_statistics(filters)

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

            return {
                'regions_products': regions_products_data,
            }
        except Exception as e:
            print(f"Ошибка в get_regions_products_dashboard_statistics: {str(e)}")
            return {
                'regions_products': [],
                'timestamp': datetime.now().isoformat(),
                'error': 'Данные недоступны'
            }

    # ========== НОВЫЕ МЕТОДЫ ДЛЯ ТЕПЛОВОЙ КАРТЫ ==========

    async def get_regions_sentiment_heatmap_filtered(
            self,
            sentiment_filter: str,
    ) -> Dict[str, Any]:
        """Получить регионы с фильтром по типу sentiment для тепловой карты"""
        try:
            regions_data = await self.analytics_repo.get_regions_with_filtered_sentiment_heatmap(sentiment_filter)

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

            return {
                'regions': regions_data,
                'total_regions': len(regions_data),
                'sentiment_filter': sentiment_filter,
                'color_scheme': color_schemes.get(sentiment_filter, color_schemes['positive']),
            }
        except Exception as e:
            print(f"Ошибка в get_regions_sentiment_heatmap_filtered: {str(e)}")
            return {
                'regions': [],
                'total_regions': 0,
                'sentiment_filter': sentiment_filter,
                'error': 'Данные недоступны'
            }

    # ========== ИСПРАВЛЕННЫЕ МЕТОДЫ ==========

    async def get_regional_dashboard(
            self,
            region_code: Optional[str] = None,
            limit: int = 20
    ) -> Dict[str, Any]:
        """ИСПРАВЛЕННАЯ версия - последовательные вызовы"""
        try:
            # Заменяем gather() на последовательные вызовы
            regional_stats = await self.analytics_repo.get_regional_stats(limit=limit)
            cities_analysis = await self.analytics_repo.get_city_stats(region_code=region_code, limit=15)

            dashboard_data = {
                "regional_overview": {
                    "regional_stats": regional_stats,
                    "cities_analysis": cities_analysis
                },
                "regional_insights": await self._generate_regional_insights(regional_stats),
                "focused_region": region_code,
                "last_updated": datetime.now().isoformat()
            }

            if region_code:
                region_filters = ReviewFilters(region_code=region_code)
                sentiment_distribution = await self.analytics_repo.get_sentiment_distribution(region_filters)
                product_analysis = await self.analytics_repo.get_product_sentiment_analysis(10, region_filters)

                dashboard_data["region_details"] = {
                    "sentiment_distribution": sentiment_distribution,
                    "top_products": product_analysis
                }

            return dashboard_data

        except Exception as e:
            print(f"Ошибка в get_regional_dashboard: {str(e)}")
            return {
                "regional_overview": {"regional_stats": [], "cities_analysis": []},
                "regional_insights": [],
                "focused_region": region_code,
                "last_updated": datetime.now().isoformat(),
                "error": "Данные недоступны"
            }

    async def get_gender_analysis_dashboard(
            self,
            region_code: Optional[str] = None,
            city: Optional[str] = None,
            product: Optional[str] = None,
            date_from: Optional[datetime] = None,
            date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Дашборд анализа по гендеру"""
        try:
            # Получаем данные по гендеру
            gender_data = await self.analytics_repo.get_gender_sentiment_analysis(
                region_code=region_code,
                city=city,
                product=product,
                date_from=date_from,
                date_to=date_to
            )

            if not gender_data:
                return {
                    "gender_analysis": [],
                    "summary": {
                        "total_genders": 0,
                        "total_reviews": 0,
                        "dominant_gender": None,
                        "gender_balance": "no_data"
                    },
                    "insights": [],
                    "timestamp": datetime.now().isoformat()
                }

            # Вычисляем общую статистику
            total_reviews = sum(item["total_reviews"] for item in gender_data)
            dominant_gender = max(gender_data, key=lambda x: x["total_reviews"])["gender"] if gender_data else None

            # Определяем баланс по гендеру
            if len(gender_data) == 2:
                male_count = next((item["total_reviews"] for item in gender_data if item["gender_raw"] == "М"), 0)
                female_count = next((item["total_reviews"] for item in gender_data if item["gender_raw"] == "Ж"), 0)

                if abs(male_count - female_count) / total_reviews <= 0.1:  # Разница менее 10%
                    gender_balance = "balanced"
                elif male_count > female_count:
                    gender_balance = "male_dominant"
                else:
                    gender_balance = "female_dominant"
            else:
                gender_balance = "single_gender"

            # Генерируем инсайты
            insights = self._generate_gender_insights(gender_data, total_reviews)

            return {
                "gender_analysis": gender_data,
                "summary": {
                    "total_genders": len(gender_data),
                    "total_reviews": total_reviews,
                    "dominant_gender": dominant_gender,
                    "gender_balance": gender_balance
                },
                "insights": insights,
                "filters_applied": {
                    "region_code": region_code,
                    "city": city,
                    "product": product,
                    "date_from": date_from.isoformat() if date_from else None,
                    "date_to": date_to.isoformat() if date_to else None
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Ошибка в get_gender_analysis_dashboard: {str(e)}")
            return {
                "gender_analysis": [],
                "summary": {"total_genders": 0, "total_reviews": 0, "dominant_gender": None,
                            "gender_balance": "no_data"},
                "insights": [],
                "timestamp": datetime.now().isoformat(),
                "error": "Данные недоступны"
            }

    def _generate_gender_insights(self, gender_data: List[Dict], total_reviews: int) -> List[Dict[str, str]]:
        """Генерация инсайтов по гендерному анализу"""
        insights = []

        if not gender_data:
            return insights

        # Наиболее активный пол
        most_active = max(gender_data, key=lambda x: x["total_reviews"])
        insights.append({
            "type": "activity",
            "message": f"Наиболее активный пол: {most_active['gender']} ({most_active['total_reviews']} отзывов, {most_active['total_reviews'] / total_reviews * 100:.1f}%)",
            "priority": "info"
        })

        # Анализ настроений по полу
        for item in gender_data:
            if item["positive_ratio"] > 70:
                insights.append({
                    "type": "positive",
                    "message": f"{item['gender']}: высокий уровень позитивных отзывов ({item['positive_ratio']:.1f}%)",
                    "priority": "success"
                })
            elif item["negative_ratio"] > 50:
                insights.append({
                    "type": "negative",
                    "message": f"{item['gender']}: преобладают негативные отзывы ({item['negative_ratio']:.1f}%)",
                    "priority": "warning"
                })

        # Сравнение между полами, если есть оба
        if len(gender_data) == 2:
            male_data = next((item for item in gender_data if item["gender_raw"] == "М"), None)
            female_data = next((item for item in gender_data if item["gender_raw"] == "Ж"), None)

            if male_data and female_data:
                if male_data["positive_ratio"] > female_data["positive_ratio"] + 10:
                    insights.append({
                        "type": "comparison",
                        "message": f"Мужчины более позитивно настроены ({male_data['positive_ratio']:.1f}% vs {female_data['positive_ratio']:.1f}%)",
                        "priority": "info"
                    })
                elif female_data["positive_ratio"] > male_data["positive_ratio"] + 10:
                    insights.append({
                        "type": "comparison",
                        "message": f"Женщины более позитивно настроены ({female_data['positive_ratio']:.1f}% vs {male_data['positive_ratio']:.1f}%)",
                        "priority": "info"
                    })

        return insights

    # Остальные методы остаются без изменений...
    # (продолжение следует с остальными методами)

    async def get_products_sentiment_analysis(
            self,
            filters: ProductsAnalysisFilters
    ) -> Dict[str, Any]:
        """Получить анализ продуктов по типам отзывов"""
        try:
            # Конвертируем Pydantic модели в словари (без mapping)
            products_filters = [
                {"name": product.name, "type": product.type}  # type уже positive/negative/neutral
                for product in filters.products
            ]

            # Получаем данные из репозитория
            chart_data = await self.analytics_repo.get_products_sentiment_trends_data(
                products_filters=products_filters,
                date_from=filters.date_from,
                date_to=filters.date_to,
                region_code=filters.region_code,
                city=filters.city
            )

            # Дополнительная аналитика
            total_products = len(filters.products)
            date_range_days = (filters.date_to - filters.date_from).days

            # Определяем преобладающий тип анализа
            sentiment_types = [product.type for product in filters.products]
            most_common_sentiment = max(set(sentiment_types), key=sentiment_types.count)

            return {
                "chart_data": chart_data,
                "analysis_info": {
                    "total_products_analyzed": total_products,
                    "date_range_days": date_range_days,
                    "period": f"{filters.date_from.strftime('%d.%m.%Y')} - {filters.date_to.strftime('%d.%m.%Y')}",
                    "most_analyzed_sentiment": most_common_sentiment,
                    "products_breakdown": [
                        {
                            "product": product.name,
                            "analysis_type": product.type
                        }
                        for product in filters.products
                    ]
                },
                "filters_applied": {
                    "region_code": filters.region_code,
                    "city": filters.city,
                    "products_count": total_products
                },
                "chart_config": {
                    "title": f"Анализ продуктов: {', '.join([p.name for p in filters.products[:3]])}{'...' if total_products > 3 else ''}",
                    "subtitle": f"Период: {filters.date_from.strftime('%m.%Y')} - {filters.date_to.strftime('%m.%Y')}",
                    "data_points": len(chart_data) - 1 if chart_data else 0
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Ошибка в get_products_sentiment_analysis: {str(e)}")
            return {
                "chart_data": [],
                "analysis_info": {"total_products_analyzed": 0, "date_range_days": 0, "period": "",
                                  "most_analyzed_sentiment": "", "products_breakdown": []},
                "filters_applied": {"region_code": filters.region_code, "city": filters.city, "products_count": 0},
                "chart_config": {"title": "", "subtitle": "", "data_points": 0},
                "timestamp": datetime.now().isoformat(),
                "error": "Данные недоступны"
            }

    async def get_reviews_trends_chart_data(
            self,
            region_code: Optional[str] = None,
            city: Optional[str] = None,
            product: Optional[str] = None,
            date_from: Optional[datetime] = None,
            date_to: Optional[datetime] = None,
            group_by: str = "day"
    ) -> Dict[str, Any]:
        """Получить данные для графика трендов отзывов"""
        try:
            chart_data = await self.analytics_repo.get_reviews_trends_aggregated(
                region_code=region_code,
                city=city,
                product=product,
                date_from=date_from,
                date_to=date_to,
                group_by=group_by
            )

            period_text = self._get_period_text(date_from, date_to)

            active_filters = []
            if region_code:
                active_filters.append(f"Регион: {region_code}")
            if city:
                active_filters.append(f"Город: {city}")
            if product:
                active_filters.append(f"Продукт: {product}")

            filters_text = " | ".join(active_filters) if active_filters else "Все данные"

            analytics = self._analyze_trends_data(chart_data)

            return {
                "chart_data": chart_data,
                "chart_config": {
                    "title": f"Динамика отзывов{' - ' + period_text if period_text else ''}",
                    "subtitle": filters_text,
                    "group_by": group_by,
                    "data_points": len(chart_data) - 1,
                },
                "analytics": analytics,
                "filters_applied": {
                    "region_code": region_code,
                    "city": city,
                    "product": product,
                    "date_from": date_from.isoformat() if date_from else None,
                    "date_to": date_to.isoformat() if date_to else None,
                    "group_by": group_by
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Ошибка в get_reviews_trends_chart_data: {str(e)}")
            return {
                "chart_data": [],
                "chart_config": {"title": "", "subtitle": "", "group_by": group_by, "data_points": 0},
                "analytics": {"status": "no_data"},
                "filters_applied": {"region_code": region_code, "city": city, "product": product,
                                    "date_from": date_from.isoformat() if date_from else None,
                                    "date_to": date_to.isoformat() if date_to else None, "group_by": group_by},
                "timestamp": datetime.now().isoformat(),
                "error": "Данные недоступны"
            }

    def _get_period_text(self, date_from: Optional[datetime], date_to: Optional[datetime]) -> str:
        """Создать читаемый текст периода"""
        if not date_from and not date_to:
            return ""

        if date_from and date_to:
            return f"{date_from.strftime('%d.%m.%Y')} - {date_to.strftime('%d.%m.%Y')}"
        elif date_from:
            return f"с {date_from.strftime('%d.%m.%Y')}"
        elif date_to:
            return f"до {date_to.strftime('%d.%m.%Y')}"

        return ""

    def _analyze_trends_data(self, chart_data: List[List]) -> Dict[str, Any]:
        """Анализ данных трендов для получения инсайтов"""
        if len(chart_data) <= 1:
            return {"status": "no_data"}

        data_rows = chart_data[1:]

        # Общая статистика
        total_reviews = sum(row[3] for row in data_rows)  # Total_Reviews
        total_positive = sum(row[0] for row in data_rows)  # Positive_Reviews
        total_negative = sum(row[1] for row in data_rows)  # Negative_Reviews

        if len(data_rows) >= 7:
            first_week_avg = sum(row[4] for row in data_rows[:7]) / 7  # Sentiment_Score
            last_week_avg = sum(row[4] for row in data_rows[-7:]) / 7
            trend_direction = "improving" if last_week_avg > first_week_avg else "declining"
            trend_change = abs(last_week_avg - first_week_avg)
        else:
            trend_direction = "stable"
            trend_change = 0

        return {
            "status": "success",
            "total_reviews": total_reviews,
            "total_positive": total_positive,
            "total_negative": total_negative,
            "positive_rate": round((total_positive / total_reviews * 100) if total_reviews > 0 else 0, 2),
            "negative_rate": round((total_negative / total_reviews * 100) if total_reviews > 0 else 0, 2),
            "trend_direction": trend_direction,
            "trend_change": round(trend_change, 3),
            "data_quality": "good" if len(data_rows) >= 30 else "limited"
        }

    async def get_all_regions(self, include_cities: bool = False, ) -> Dict[str, Any]:
        """Получить все регионы"""
        try:
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
                regions = [region for region in regions]

                return {
                    "regions": regions,
                    "total_regions": len(regions),
                    "include_cities": False,
                }
        except Exception as e:
            print(f"Ошибка в get_all_regions: {str(e)}")
            return {
                "regions": [],
                "total_regions": 0,
                "include_cities": include_cities,
                "error": "Данные недоступны"
            }

    async def get_region_codes_only(self) -> Dict[str, Any]:
        """Получить только коды регионов"""
        try:
            region_codes = await self.analytics_repo.get_unique_region_codes()
            formatted_codes = [
                {"value": code["region_code"], "label": code["region_code"], "count": code["reviews_count"]}
                for code in region_codes]

            return {
                "region_codes": region_codes,
                "formatted_codes": formatted_codes,
                "total_codes": len(region_codes),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Ошибка в get_region_codes_only: {str(e)}")
            return {
                "region_codes": [],
                "formatted_codes": [],
                "total_codes": 0,
                "timestamp": datetime.now().isoformat(),
                "error": "Данные недоступны"
            }

    async def get_regions_sentiment_map(self, include_cities: bool = False) -> Dict[str, Any]:
        """Получить регионы с sentiment анализом для карты"""
        try:
            regions_data = await self.analytics_repo.get_regions_with_sentiment_colors()

            response = {
                "regions": regions_data,
                "total_regions": len(regions_data),
                "include_cities": include_cities,
                "color_scheme": {"positive": "#228B22", "neutral": "#E2B007", "negative": "#FA8072"},
            }

            return response
        except Exception as e:
            print(f"Ошибка в get_regions_sentiment_map: {str(e)}")
            return {
                "regions": [],
                "total_regions": 0,
                "include_cities": include_cities,
                "color_scheme": {"positive": "#228B22", "neutral": "#E2B007", "negative": "#FA8072"},
                "error": "Данные недоступны"
            }

    # ========== УТИЛИТЫ ==========

    async def _get_quick_insights(self, filters: Optional[ReviewFilters] = None) -> Dict[str, Any]:
        """Получить быстрые инсайты"""
        try:
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
                    insights.append(
                        {"type": "warning", "message": f"Много негативных отзывов: {negative_rate * 100:.1f}%",
                         "priority": "warning"})

            return {
                "insights": insights,
                "insights_count": len(insights),
                "data_quality": "good" if total > 100 else "limited"
            }
        except Exception as e:
            print(f"Ошибка в _get_quick_insights: {str(e)}")
            return {
                "insights": [],
                "insights_count": 0,
                "data_quality": "limited"
            }

    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Получить метрики производительности"""
        try:
            regions_count = len(await self.analytics_repo.get_unique_regions())

            return {
                "data_freshness": "excellent",
                "processing_speed": "fast",
                "coverage": {"regions_covered": regions_count},
                "data_completeness": 95.5
            }
        except Exception as e:
            print(f"Ошибка в _get_performance_metrics: {str(e)}")
            return {
                "data_freshness": "limited",
                "processing_speed": "unknown",
                "coverage": {"regions_covered": 0},
                "data_completeness": 0
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
