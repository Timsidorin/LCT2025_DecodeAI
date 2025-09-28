# # services/monitoring_service.py
# import asyncio
# import logging
# from datetime import datetime, timedelta
# from typing import List, Dict, Any, Optional
# import httpx
# from apscheduler.schedulers.asyncio import AsyncIOScheduler
# from apscheduler.triggers.interval import IntervalTrigger
# from apscheduler.triggers.cron import CronTrigger
#
# from repository import ReviewAnalyticsRepository
# from core.database import get_async_session
# from schemas.review_schemas import ReviewFilters
# from core.config import configs
#
# logger = logging.getLogger(__name__)
#
#
# class ReviewMonitoringService:
#     def __init__(self,
#                  telegram_bot_url: str = None,
#                  notification_threshold: int = 3,
#                  check_interval_minutes: int = 10):
#
#         self.telegram_bot_url = telegram_bot_url or getattr(configs, 'TELEGRAM_BOT_WEBHOOK_URL', None)
#         self.notification_threshold = notification_threshold
#         self.check_interval_minutes = check_interval_minutes
#         self.scheduler = AsyncIOScheduler()
#
#         # Кэш для предотвращения дублирования
#         self.last_notification_time = None
#         self.notification_cooldown_minutes = 30
#         self.is_monitoring_enabled = bool(self.telegram_bot_url)
#
#     async def start_monitoring(self):
#         """Запустить планировщик мониторинга"""
#         if not self.is_monitoring_enabled:
#             logger.warning("Мониторинг отключен: не указан TELEGRAM_BOT_WEBHOOK_URL")
#             return
#
#         logger.info("Запуск службы мониторинга отзывов")
#
#         # Проверка негативных трендов каждые N минут
#         self.scheduler.add_job(
#             func=self.check_recent_reviews,
#             trigger=IntervalTrigger(minutes=self.check_interval_minutes),
#             id='review_monitoring',
#             name='Мониторинг последних отзывов',
#             replace_existing=True,
#             max_instances=1
#         )
#
#         # Ежедневный отчет в 9:00
#         self.scheduler.add_job(
#             func=self.send_daily_report,
#             trigger=CronTrigger(hour=9, minute=0),
#             id='daily_report',
#             name='Ежедневный отчет по отзывам',
#             replace_existing=True
#         )
#
#         self.scheduler.start()
#         logger.info(f"Планировщик запущен. Проверка каждые {self.check_interval_minutes} минут")
#
#     async def stop_monitoring(self):
#         """Остановить планировщик"""
#         if self.scheduler.running:
#             self.scheduler.shutdown(wait=False)
#             logger.info("Планировщик остановлен")
#
#     async def check_recent_reviews(self):
#         """Проверить последние отзывы на негативные тренды"""
#         try:
#             logger.info("Проверка последних отзывов...")
#
#             async for session in get_async_session():
#                 repo = ReviewAnalyticsRepository(session)
#
#                 # Получаем последние отзывы за 2 часа
#                 recent_reviews = await repo.get_recent_activity(minutes_back=120, limit=20)
#
#                 if len(recent_reviews) >= self.notification_threshold:
#                     negative_trend = self._detect_negative_trend(recent_reviews)
#
#                     if negative_trend and self._can_send_notification():
#                         await self._send_negative_trend_alert(negative_trend)
#                         self.last_notification_time = datetime.now()
#
#                 break  # Выходим из async generator
#
#         except Exception as e:
#             logger.error(f"Ошибка при проверке отзывов: {e}")
#
#     def _detect_negative_trend(self, reviews: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
#         """Определить негативный тренд"""
#         if len(reviews) < self.notification_threshold:
#             return None
#
#         # Анализируем последние отзывы
#         latest_reviews = reviews[:self.notification_threshold]
#         negative_count = sum(1 for review in latest_reviews if review.get('rating') == 'negative')
#
#         negative_percentage = (negative_count / len(latest_reviews)) * 100
#
#         if negative_percentage >= 70:  # 70% или больше негативных
#             return {
#                 'total_reviews': len(latest_reviews),
#                 'negative_count': negative_count,
#                 'negative_percentage': round(negative_percentage, 1),
#                 'reviews': latest_reviews[:3],  # Показываем только первые 3
#                 'alert_type': 'negative_trend'
#             }
#
#         return None
#
#     def _can_send_notification(self) -> bool:
#         """Проверить cooldown"""
#         if self.last_notification_time is None:
#             return True
#
#         time_diff = datetime.now() - self.last_notification_time
#         return time_diff.total_seconds() > (self.notification_cooldown_minutes * 60)
#
#     async def _send_negative_trend_alert(self, trend_data: Dict[str, Any]):
#         """Отправить уведомление о негативном тренде"""
#         if not self.telegram_bot_url:
#             logger.warning("Telegram URL не настроен, уведомление не отправлено")
#             return
#
#         message = {
#             "alert_type": "negative_trend",
#             "message": f"🚨 ВНИМАНИЕ: Негативный тренд в отзывах!\n\n"
#                        f"📊 Последних отзывов: {trend_data['total_reviews']}\n"
#                        f"👎 Негативных: {trend_data['negative_count']} ({trend_data['negative_percentage']}%)\n"
#                        f"⏰ Время: {datetime.now().strftime('%H:%M %d.%m.%Y')}\n\n"
#                        f"Требуется внимание к качеству обслуживания!",
#             "priority": "high",
#             "data": trend_data
#         }
#
#         await self._send_telegram_notification(message)
#
#     async def send_daily_report(self):
#         """Ежедневный отчет"""
#         try:
#             logger.info("Отправка ежедневного отчета...")
#
#             async for session in get_async_session():
#                 repo = ReviewAnalyticsRepository(session)
#
#                 # Статистика за вчера
#                 yesterday = datetime.now() - timedelta(days=1)
#                 filters_yesterday = ReviewFilters(
#                     date_from=yesterday.replace(hour=0, minute=0, second=0),
#                     date_to=yesterday.replace(hour=23, minute=59, second=59)
#                 )
#
#                 sentiment_data = await repo.get_sentiment_distribution(filters_yesterday)
#                 total_reviews = sum(sentiment_data.values())
#
#                 if total_reviews > 0:
#                     positive_pct = round((sentiment_data.get('positive', 0) / total_reviews * 100), 1)
#                     negative_pct = round((sentiment_data.get('negative', 0) / total_reviews * 100), 1)
#
#                     message = {
#                         "alert_type": "daily_report",
#                         "message": f"📈 Ежедневный отчет за {yesterday.strftime('%d.%m.%Y')}\n\n"
#                                    f"📊 Всего отзывов: {total_reviews}\n"
#                                    f"✅ Положительных: {positive_pct}%\n"
#                                    f"❌ Отрицательных: {negative_pct}%",
#                         "priority": "low",
#                         "data": {"total_reviews": total_reviews, "sentiment_distribution": sentiment_data}
#                     }
#
#                     await self._send_telegram_notification(message)
#
#                 break
#
#         except Exception as e:
#             logger.error(f"Ошибка при отправке отчета: {e}")
#
#     async def _send_telegram_notification(self, notification_data: Dict[str, Any]):
#         """Отправить уведомление в Telegram"""
#         try:
#             async with httpx.AsyncClient(timeout=30.0) as client:
#                 response = await client.post(
#                     self.telegram_bot_url,
#                     json=notification_data,
#                     headers={"Content-Type": "application/json"}
#                 )
#
#                 if response.status_code == 200:
#                     logger.info(f"Уведомление отправлено: {notification_data['alert_type']}")
#                 else:
#                     logger.error(f"Ошибка отправки: {response.status_code} - {response.text}")
#
#         except Exception as e:
#             logger.error(f"Ошибка при отправке в Telegram: {e}")
#
#     async def manual_check(self) -> Dict[str, Any]:
#         """Ручная проверка для тестирования"""
#         await self.check_recent_reviews()
#         return {
#             "status": "manual_check_completed",
#             "timestamp": datetime.now().isoformat(),
#             "last_notification": self.last_notification_time.isoformat() if self.last_notification_time else None,
#             "monitoring_enabled": self.is_monitoring_enabled
#         }
