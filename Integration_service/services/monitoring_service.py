
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from Integration_service.repository import ReviewAnalyticsRepository
from Integration_service.core.database import get_async_session
from Integration_service.schemas.processed_review import ReviewFilters
from Integration_service.core.config import configs

logger = logging.getLogger(__name__)


class ReviewMonitoringService:
    def __init__(self,
                 telegram_bot_url: str = None,
                 notification_threshold: int = 5,
                 check_interval_minutes: int = 2):

        self.telegram_bot_url = telegram_bot_url or getattr(configs, 'TELEGRAM_BOT_URL',
                                                            "http://localhost:8010/send-notification")
        self.notification_threshold = notification_threshold
        self.check_interval_minutes = check_interval_minutes
        self.scheduler = AsyncIOScheduler()

        self.last_notification_time = {}
        self.notification_cooldown_minutes = 60
        self.is_monitoring_enabled = bool(self.telegram_bot_url)

    async def start_monitoring(self):
        """Запустить планировщик мониторинга"""
        if not self.is_monitoring_enabled:
            logger.warning("Мониторинг отключен: не указан TELEGRAM_BOT_URL")
            return

        logger.info("🚀 Запуск службы мониторинга отзывов PulseAI")

        self.scheduler.add_job(
            func=self.check_negative_trends_by_products,
            trigger=IntervalTrigger(minutes=self.check_interval_minutes),
            id='negative_trends_monitoring',
            name='Мониторинг негативных трендов по продуктам',
            replace_existing=True,
            max_instances=1
        )

        self.scheduler.add_job(
            func=self.send_daily_report,
            trigger=CronTrigger(hour=9, minute=0),
            id='daily_report',
            name='Ежедневный отчет',
            replace_existing=True
        )

        self.scheduler.start()
        logger.info(f"📊 Мониторинг активирован. Проверка каждые {self.check_interval_minutes} минут")

    async def stop_monitoring(self):
        """Остановить планировщик"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("⏹️ Мониторинг остановлен")

    async def check_negative_trends_by_products(self):
        """Проверка негативных трендов по продуктам"""
        try:
            logger.info("🔍 Проверка негативных трендов по продуктам...")

            async for session in get_async_session():
                repo = ReviewAnalyticsRepository(session)

                recent_time = datetime.now() - timedelta(hours=2)
                filters = ReviewFilters(date_from=recent_time)

                products_stats = await repo.get_product_sentiment_analysis(
                    limit=10,
                    filters=filters
                )

                for product_stats in products_stats:
                    await self._analyze_product_trend(product_stats, repo)

                break

        except Exception as e:
            logger.error(f"❌ Ошибка при проверке трендов: {e}")

    async def _analyze_product_trend(self, product_stats: Dict[str, Any], repo: ReviewAnalyticsRepository):
        """Анализ тренда по конкретному продукту"""
        try:
            product_name = product_stats.get('product', 'Unknown')
            total_reviews = product_stats.get('total_reviews', 0)
            negative_count = product_stats.get('negative_count', 0)
            negative_rate = product_stats.get('negative_rate', 0)

            logger.info(
                f"📊 Анализ продукта '{product_name}': {negative_count} негативных из {total_reviews} ({negative_rate}%)")

            condition1 = total_reviews >= self.notification_threshold
            condition2 = negative_rate >= 60
            condition3 = self._can_send_notification(product_name)

            logger.info(
                f"✅ Условия для '{product_name}': reviews>={self.notification_threshold}? {condition1}, negative>=60%? {condition2}, cooldown? {condition3}")

            if condition1 and condition2 and condition3:
                logger.info(
                    f"🚨 ВСЕ УСЛОВИЯ ВЫПОЛНЕНЫ! Отправляем уведомление для '{product_name}' ({negative_rate}% негативных)")

                regions_list = await self._get_problem_regions(product_name, repo)

                await self._send_negative_trend_notification({
                    'product': product_name,
                    'negative_count': negative_count,
                    'total_count': total_reviews,
                    'negative_percentage': negative_rate,
                    'regions': regions_list,
                    'timestamp': datetime.now()
                })

                self.last_notification_time[product_name] = datetime.now()
            else:
                logger.info(f"❌ Условия НЕ выполнены для '{product_name}'")

        except Exception as e:
            logger.error(f"❌ Ошибка анализа продукта {product_stats}: {e}")

    async def _get_problem_regions(self, product_name: str, repo: ReviewAnalyticsRepository) -> List[str]:
        """Получить регионы с проблемами по продукту"""
        try:
            recent_time = datetime.now() - timedelta(hours=2)
            filters = ReviewFilters(
                product=product_name,
                date_from=recent_time,
                rating='negative'
            )

            regional_stats = await repo.get_regional_stats(filters, limit=5)

            regions = []
            for stats in regional_stats[:3]:
                region_name = stats.get('region_name', 'Unknown')
                if region_name != 'Unknown':
                    regions.append(region_name)

            return regions if regions else ["Различные регионы"]

        except Exception as e:
            logger.warning(f"⚠️ Не удалось получить регионы для {product_name}: {e}")
            return ["Различные регионы"]

    def _can_send_notification(self, product_name: str) -> bool:
        """Проверить cooldown для продукта"""
        last_time = self.last_notification_time.get(product_name)

        if last_time is None:
            return True

        time_diff = datetime.now() - last_time
        can_send = time_diff.total_seconds() > (self.notification_cooldown_minutes * 60)

        if not can_send:
            logger.info(
                f"⏳ Cooldown активен для '{product_name}' (осталось {self.notification_cooldown_minutes - int(time_diff.total_seconds() / 60)} мин)")

        return can_send

    async def _send_negative_trend_notification(self, trend_data: Dict[str, Any]):
        """Отправить уведомление о негативном тренде"""
        try:
            product = trend_data['product']
            negative_count = trend_data['negative_count']
            total_count = trend_data['total_count']
            negative_percentage = trend_data['negative_percentage']
            regions = trend_data['regions']
            timestamp = trend_data['timestamp']

            regions_text = ", ".join(regions) if regions else "Различные регионы"

            message_text = f"""🚨 <b>УВЕДОМЛЕНИЕ!</b>

📊 Обнаружен резкий рост негативных отзывов по продукту <b>"{product}"</b>

📈 <b>Статистика за последние 2 часа:</b>
• Всего отзывов: {total_count}
• Негативных: {negative_count} ({negative_percentage}%)
• Регионы: {regions_text}

🤖 PulseAI Monitoring System"""

            payload = {
                "text": message_text,
                "parse_mode": "HTML",
                "disable_notification": False
            }

            logger.info(f"📤 Отправляем уведомление в Telegram...")
            success = await self._send_to_telegram_bot(payload)

            if success:
                logger.info(f"✅ Уведомление отправлено для '{product}' ({negative_percentage}% негативных)")
            else:
                logger.error(f"❌ Не удалось отправить уведомление для '{product}'")

        except Exception as e:
            logger.error(f"❌ Ошибка отправки уведомления: {e}")

    async def _send_to_telegram_bot(self, payload: Dict[str, Any]) -> bool:
        """Отправить POST запрос в Telegram бота"""
        try:
            logger.info(f"🌐 Отправляю POST запрос на {self.telegram_bot_url}")

            timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.telegram_bot_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                logger.info(f"📡 Ответ от бота: {response.status_code}")

                if response.status_code == 200:
                    logger.info(f"✅ Уведомление успешно отправлено в бота")
                    return True
                else:
                    logger.error(f"❌ Ошибка отправки в бота: {response.status_code} - {response.text}")
                    return False

        except httpx.TimeoutException:
            logger.error("⏱️ Timeout при отправке в Telegram бота")
            return False
        except httpx.ConnectError as e:
            logger.error(f"🔌 Ошибка подключения к боту: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка при отправке в бота: {e}")
            return False

    async def send_daily_report(self):
        """Ежедневный отчет"""
        pass


monitoring_service = ReviewMonitoringService()
