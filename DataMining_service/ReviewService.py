from typing import Optional
from uuid import uuid4
from datetime import datetime

from core.broker import KafkaBrokerManager
from repository import ReviewRepository  # Используем ReviewRepository
from shemas.review import ReviewCreate, ReviewResponse, SentimentType


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

        db_review = await self.repo.add_review(
            text=review.text,
            datetime_review=datetime_review,
            source=review.source,
            product=review.product,
            rating=review.rating,
            gender=review.gender,
            city=review.city
        )

        # Формируем ответ
        return self._build_response(db_review)

    async def _publish_to_kafka(self, review_id: str, review: ReviewCreate, datetime_review: datetime):
        """Публикация отзыва в Kafka."""
        message = {
            "id": review_id,
            "created_at": datetime_review.isoformat(),
            **review.model_dump(exclude={"datetime_review"})
        }

        await self.broker.publish(
            topic="raw_reviews",
            message=message,
            key=review_id.encode('utf-8')
        )

    def _build_response(self, db_review) -> ReviewResponse:
        """Построение ответа из объекта БД."""
        rating = None
        if db_review.rating:
            try:
                rating = SentimentType(db_review.rating)
            except ValueError:
                rating = None

        gender = None
        if db_review.gender:
            try:
                from shemas.review import Gender
                gender = Gender(db_review.gender)
            except ValueError:
                gender = None

        return ReviewResponse(
            uuid=db_review.uuid,
            source=db_review.source,
            text=db_review.text,
            rating=rating,
            product=db_review.product,
            gender=gender,
            city=db_review.city,
            datetime_review=db_review.datetime_review,
            created_at=db_review.created_at
        )
