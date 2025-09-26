from typing import Optional
from uuid import uuid4
from datetime import datetime

from core.broker import KafkaBrokerManager
from repository import ReviewRepository
from shemas.review import ReviewCreate, ReviewResponse


class ReviewService:
    """Бизнес-логика для работы с отзывами."""

    def __init__(self, broker: KafkaBrokerManager, repo: ReviewRepository):
        self.broker = broker
        self.repo = repo

    async def create_review(self, review: ReviewCreate) -> ReviewResponse:
        """Создать новый отзыв с публикацией в Kafka и сохранением в БД."""
        review_id = str(uuid4())
        datetime_review = review.datetime_review or datetime.now()

        # Публикуем в Kafka
        await self._publish_to_kafka(review_id, review, datetime_review)

        # Подготавливаем данные для БД
        review_dict = review.model_dump(exclude={"uuid"})
        review_dict["datetime_review"] = datetime_review

        # Создаем отзыв в БД
        db_review = await self.repo.add_review(**review_dict)

        # Формируем ответ
        return self._build_response(db_review)

    async def _publish_to_kafka(
        self, review_id: str, review: ReviewCreate, datetime_review: datetime
    ):
        """Публикация отзыва в Kafka."""
        message = {
            "id": review_id,
            "created_at": datetime_review.isoformat(),
            "source": review.source,
            "text": review.text,
            "rating": review.rating.value if review.rating else None,
            "product": review.product,
            "gender": review.gender.value if review.gender else None,
            "city": review.city,
            "region_code": review.region_code,
            "datetime_review": (
                review.datetime_review.isoformat() if review.datetime_review else None
            ),
        }

        await self.broker.publish(
            topic="raw_reviews", message=message, key=review_id.encode("utf-8")
        )

    def _build_response(self, db_review) -> ReviewResponse:
        """Построение ответа из объекта БД."""
        # Конвертируем rating из БД в enum
        rating = None
        if db_review.rating:
            try:
                rating = SentimentType(
                    db_review.rating.value
                    if hasattr(db_review.rating, "value")
                    else db_review.rating
                )
            except (ValueError, AttributeError):
                rating = None

        gender = None
        if db_review.gender:
            try:
                from shemas.review import Gender

                gender = Gender(
                    db_review.gender.value
                    if hasattr(db_review.gender, "value")
                    else db_review.gender
                )
            except (ValueError, AttributeError):
                gender = None

        return ReviewResponse(
            uuid=db_review.uuid,
            source=db_review.source,
            text=db_review.text,
            rating=rating,
            product=db_review.product,
            gender=gender,
            city=db_review.city,
            region_code=db_review.region_code,
            datetime_review=db_review.datetime_review,
            created_at=db_review.created_at,
        )

    async def get_review_by_id(self, review_id: str) -> Optional[ReviewResponse]:
        """Получить отзыв по ID."""
        db_review = await self.repo.get_review_by_id(review_id)
        if db_review:
            return self._build_response(db_review)
        return None

    async def get_reviews(
        self,
        limit: int = 10,
        offset: int = 0,
        source: Optional[str] = None,
        city: Optional[str] = None,
        region_code: Optional[str] = None,
    ) -> list[ReviewResponse]:
        """Получить список отзывов с фильтрацией."""
        db_reviews = await self.repo.get_reviews(
            limit=limit,
            offset=offset,
            source=source,
            city=city,
            region_code=region_code,
        )
        return [self._build_response(review) for review in db_reviews]

    # async def update_review_analysis(self,
    #                                review_id: str,
    #                                rating: Optional[SentimentType] = None,
    #                                gender: Optional[str] = None) -> Optional[ReviewResponse]:
    #     """Обновить результаты анализа отзыва."""
    #     db_review = await self.repo.update_review_analysis(
    #         review_id=review_id,
    #         rating=rating.value if rating else None,
    #         gender=gender
    #     )
    #     if db_review:
    #         return self._build_response(db_review)
    #     return None

    async def get_reviews_by_region(
        self, region_code: str, limit: int = 10
    ) -> list[ReviewResponse]:
        """Получить отзывы по коду региона."""
        db_reviews = await self.repo.get_reviews(limit=limit, region_code=region_code)
        return [self._build_response(review) for review in db_reviews]

    async def get_stats_by_region(self) -> dict:
        """Получить статистику отзывов по регионам."""
        return await self.repo.get_reviews_stats_by_region()
