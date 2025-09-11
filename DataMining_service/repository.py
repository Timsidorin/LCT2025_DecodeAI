# repository.py
from datetime import datetime
from typing import Optional
from uuid import uuid4, UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from models.review import Review
from shemas.review import SentimentType, Gender


class ReviewRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_review(
            self,
            text: str,
            datetime_review: datetime,
            source: str = "API",
            product: Optional[str] = None,
            rating: Optional[SentimentType] = None,
            gender: Optional[Gender] = None,
            city: Optional[str] = None
    ):
        """
        Добавление отзыва

        Args:
            text: Текст отзыва
            datetime_review: Дата и время создания отзыва
            source: Источник отзыва (по умолчанию "API")
            product: Название продукта (необязательно)
            rating: Тональность отзыва (необязательно)
            gender: Пол автора (необязательно)
            city: Город автора (необязательно)
        """
        try:
            review = Review(
                uuid=uuid4(),
                source=source,
                text=text,
                datetime_review=datetime_review,
                product=product,
                rating=rating.value if rating else None,
                gender=gender.value if gender else None,
                city=city
            )

            self.session.add(review)
            await self.session.commit()
            await self.session.refresh(review)
            return review

        except IntegrityError:
            await self.session.rollback()
            return None

    async def get_review_by_uuid(self, review_uuid: UUID) -> Optional[Review]:
        """Получение отзыва по UUID"""
        query = select(Review).where(Review.uuid == review_uuid)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_reviews_by_source(self, source: str) -> list[Review]:
        """Получение всех отзывов по источнику"""
        query = select(Review).where(Review.source == source)
        result = await self.session.execute(query)
        return result.scalars().all()
