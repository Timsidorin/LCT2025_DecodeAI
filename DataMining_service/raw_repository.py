# raw_repository.py
from datetime import datetime
from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from DataMining_service.models.raw_reviews import RawReview


class RawReviewRepository:
    """Репозиторий для работы с сырыми отзывами (raw_reviews)"""
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_raw_review(
        self,
        text: str,
        datetime_review: datetime,
        source: str = "API",
        gender: Optional[str] = None,
        city: Optional[str] = None,
        region: Optional[str] = None,
        region_code: Optional[str] = None,
    ) -> Optional[RawReview]:
        """Добавление сырого отзыва в таблицу raw_reviews"""
        try:
            raw_review = RawReview(
                source=source,
                text=text,
                gender=gender,
                datetime_review=datetime_review,
                city=city,
                region=region,
                region_code=region_code
            )

            self.session.add(raw_review)
            await self.session.commit()
            await self.session.refresh(raw_review)

            return raw_review

        except IntegrityError as e:
            await self.session.rollback()
            return None
        except Exception as e:
            await self.session.rollback()
            return None
