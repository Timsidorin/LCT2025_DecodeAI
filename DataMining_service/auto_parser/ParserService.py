# parser_service.py
import asyncio

from DataMining_service.services.ReviewService import ReviewService
from ReviewParser import ReviewMonitor as RM
from DataMining_service.shemas.review import ReviewCreate
from DataMining_service.core.database import get_async_session
from DataMining_service.core.broker import KafkaBrokerManager
from DataMining_service.raw_repository import RawReviewRepository


parser = RM()


async def create_review_service():
    """Создает сервис для работы с сырыми отзывами"""
    async for session in get_async_session():
        kafka_broker = KafkaBrokerManager()
        await kafka_broker.connect()

        try:
            repo = RawReviewRepository(session)
            review_service = ReviewService(kafka_broker, repo)
            yield review_service
        finally:
            await kafka_broker.close()
            await session.close()
        break


async def get_and_publish_reviews():
    """
    Получает новые отзывы из парсера и обрабатывает их:
    1. Определяет пол по тексту
    2. Сохраняет в таблицу raw_reviews
    3. Отправляет в Kafka топик raw_reviews для ML обработки
    """
    result = parser.get_new_reviews()

    if result and result.get("reviews"):
        print(f"📥 Получено {len(result['reviews'])} новых отзывов из парсера")

        async for review_service in create_review_service():
            success_count = 0
            error_count = 0

            for review in result["reviews"]:
                try:
                    serialized_review = ReviewCreate(
                        source=review["source_id"],
                        text=review["text"],
                        datetime_review=review["datetime_review"],
                        city=review.get("city"),
                        region=review.get("region"),
                        region_code=review.get("region_code")
                    )

                    response = await review_service.create_review(serialized_review)

                except Exception as e:
                    print(f"Ошибка обработки отзыва: {e}")

            break
    else:
        print("📭 Новых отзывов нет")


async def main():
    try:
        await get_and_publish_reviews()
        print("Парсер завершил работу")
    except Exception as e:
        print(f"ошибка парсера: {e}")


if __name__ == "__main__":
    asyncio.run(main())
