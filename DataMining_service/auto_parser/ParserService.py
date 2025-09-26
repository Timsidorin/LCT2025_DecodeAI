import asyncio
from ReviewParser import ReviewMonitor as RM
from shemas.review import ReviewCreate
from core.database import get_async_session
from core.broker import KafkaBrokerManager
from repository import ReviewRepository
from ReviewService import ReviewService

parser = RM()


async def create_review_service():
    """Создает сервис обзоров без зависимости от FastAPI"""
    async for session in get_async_session():
        kafka_broker = KafkaBrokerManager()
        await kafka_broker.connect()

        try:
            repo = ReviewRepository(session)
            review_service = ReviewService(kafka_broker, repo)
            yield review_service
        finally:
            await kafka_broker.close()
            await session.close()
        break


async def get_and_publish_reviews():
    """Получает и публикует новые отзывы"""
    result = parser.get_new_reviews()

    if result:
        async for review_service in create_review_service():
            for review in result["reviews"]:
                serialized_review = ReviewCreate(
                    source=review["source_id"],
                    text=review["text"],
                    datetime_review=review["datetime_review"],
                    rating=None,
                    product=None,
                    city=review.get("city"),
                    region_code=review.get("region_code"),
                )
                print(serialized_review)
                await review_service.create_review(serialized_review)
            break


if __name__ == "__main__":
    asyncio.run(get_and_publish_reviews())
