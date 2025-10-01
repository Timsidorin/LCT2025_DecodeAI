# repositories/processed_repository.py
from datetime import datetime
from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from MLProcessing_service.models.ProcessedReviewModel import ProcessedReview


class ProcessedReviewRepository:
    """Репозиторий для работы с обработанными отзывами (processed_reviews)"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_processed_review(
            self,
            text: str,
            datetime_review: datetime,
            source: str = "API",
            rating: Optional[str] = None,
            product: Optional[str] = None,
            gender: Optional[str] = None,
            city: Optional[str] = None,
            region: Optional[str] = None,
            region_code: Optional[str] = None,
    ) -> Optional[ProcessedReview]:
        """
        Добавление обработанного отзыва в таблицу processed_reviews
        """
        try:
            processed_review = ProcessedReview(
                source=source,
                text=text,
                rating=rating,
                product=product,
                gender=gender,
                city=city,
                region=region,
                region_code=region_code,
                datetime_review=datetime_review
            )

            self.session.add(processed_review)
            await self.session.commit()
            await self.session.refresh(processed_review)

            return processed_review

        except IntegrityError as e:
            await self.session.rollback()
            print(f"Ошибка добавления обработанного отзыва: {e}")
            return None
        except Exception as e:
            await self.session.rollback()
            print(f"Неожиданная ошибка: {e}")
            return None

    async def add_processed_reviews_batch(
            self,
            reviews_data: list[dict]
    ) -> tuple[int, int]:
        """
        Массовое добавление обработанных отзывов

        Returns:
            tuple: (количество успешно добавленных, количество ошибок)
        """
        success_count = 0
        error_count = 0

        try:
            reviews = []
            for data in reviews_data:
                processed_review = ProcessedReview(
                    source=data.get("source", "API"),
                    text=data["text"],
                    rating=data.get("rating"),
                    product=data.get("product"),
                    gender=data.get("gender"),
                    city=data.get("city"),
                    region=data.get("region"),
                    region_code=data.get("region_code"),
                    datetime_review=data["datetime_review"]
                )
                reviews.append(processed_review)

            self.session.add_all(reviews)
            await self.session.commit()
            success_count = len(reviews)

        except IntegrityError as e:
            await self.session.rollback()
            error_count = len(reviews_data)
            print(f"Ошибка массового добавления: {e}")
        except Exception as e:
            await self.session.rollback()
            error_count = len(reviews_data)
            print(f"Неожиданная ошибка массового добавления: {e}")

        return success_count, error_count
