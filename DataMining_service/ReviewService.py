from typing import Optional
from uuid import uuid4
from datetime import datetime

from core.broker import KafkaBrokerManager
from repository import RawReviewRepository
from shemas.feedback import ReviewCreate, ReviewResponse


class ReviewService:
    """Бизнес-логика для работы с отзывами."""

    def __init__(self, broker: KafkaBrokerManager, repo: RawReviewRepository):
        self.broker = broker
        self.repo = repo

    async def create_review(self, review: ReviewCreate) -> ReviewResponse:
        """Создать новый отзыв с публикацией в Kafka и сохранением в БД."""
        review_id = str(uuid4())
        datetime_review = review.datetime_review or datetime.now()

        # Публикуем в Kafka
        await self._publish_to_kafka(review_id, review, datetime_review)

        # Сохраняем в БД
        db_review = await self.repo.add_raw_review(
            source_id=review.source_id,
            text=review.text,
            datetime_review=datetime_review,
            product=review.product,
            rating=[str(review.rating)] if review.rating is not None else None,
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
        if db_review.rating and db_review.rating[0]:
            try:
                rating = int(db_review.rating[0])
            except (ValueError, IndexError):
                rating = None

        return ReviewResponse(
            uuid=str(db_review.uuid),
            source_id=db_review.source_id,
            text=db_review.text,
            rating=rating,
            datetime_review=db_review.datetime_review,
            product=db_review.product,
        )
