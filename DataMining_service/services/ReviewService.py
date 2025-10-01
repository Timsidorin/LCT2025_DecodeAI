# ReviewService.py
from typing import Optional
from uuid import UUID
from datetime import datetime

from DataMining_service.core.broker import KafkaBrokerManager
from DataMining_service.raw_repository import RawReviewRepository
from DataMining_service.shemas.review import ReviewCreate, RawReviewResponse
from DataMining_service.services.GenderDetector import GenderDetector


class ReviewService:
    """
    Сервис для работы с сырыми отзывами:
    1. Определение пола по тексту
    2. Сохранение в таблицу raw_reviews
    3. Отправка в Kafka топик raw_reviews для ML обработки
    """

    def __init__(self, broker: KafkaBrokerManager, repo: RawReviewRepository):
        self.broker = broker
        self.repo = repo
        self.gender_detector = GenderDetector()

    async def create_review(self, review: ReviewCreate) -> RawReviewResponse:
        """
        Создать новый сырой отзыв с определением пола, публикацией в Kafka и сохранением в БД
        """
        datetime_review = review.datetime_review or datetime.utcnow()

        gender = review.gender
        if not gender:
            gender = self.gender_detector.detect_gender(review.text)
            print(f"🔍 Определен пол: {gender}")


        db_review = await self._save_to_database(review, datetime_review, gender)

        if not db_review:
            raise Exception("Ошибка сохранения отзыва в БД")
        await self._publish_to_kafka_for_ml(db_review)

        return self._build_response(db_review)

    async def _save_to_database(self, review: ReviewCreate, datetime_review: datetime, gender: Optional[str]):
        """Сохранение отзыва в таблицу raw_reviews"""
        try:
            db_review = await self.repo.add_raw_review(
                text=review.text,
                datetime_review=datetime_review,
                source=review.source,
                gender=gender,
                city=review.city,
                region=review.region,
                region_code=review.region_code
            )
            return db_review
        except Exception as e:
            print(f"❌ Ошибка сохранения в БД: {e}")
            return None

    async def _publish_to_kafka_for_ml(self, db_review):
        """
        Публикация отзыва в Kafka топик 'raw_reviews' для ML обработки

        Формат: {"data": [{"id": "uuid", "text": "текст", "gender": "М"}]}
        """
        message = {
            "data": [
                {
                    "id": str(db_review.uuid),
                    "text": db_review.text,
                    "gender": db_review.gender,
                    "source": db_review.source,
                    "city": db_review.city,
                    "region": db_review.region,
                    "region_code": db_review.region_code,
                    "datetime_review": db_review.datetime_review.isoformat()
                }
            ]
        }

        try:
            await self.broker.publish(
                topic="raw_reviews",
                message=message,
                key=str(db_review.uuid).encode("utf-8")
            )
        except Exception as e:
            raise

    def _build_response(self, db_review) -> RawReviewResponse:
        """Построение ответа из объекта БД"""
        return RawReviewResponse(
            uuid=db_review.uuid,
            source=db_review.source,
            text=db_review.text,
            gender=db_review.gender,
            city=db_review.city,
            region=db_review.region,
            region_code=db_review.region_code,
            datetime_review=db_review.datetime_review,
            created_at=db_review.created_at
        )
