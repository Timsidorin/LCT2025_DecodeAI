# repository.py
from datetime import datetime
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession
from models.feedback import RawReview
from uuid import uuid4


class RawReviewRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_raw_review(self, source_id: int, text: str, datetime_review: datetime, product: str,
                             rating: List[str] = None):
        raw_review = RawReview(
            uuid=uuid4(),
            source_id=source_id,
            text=text,
            datetime_review=datetime_review,
            product=product,
            rating=rating
        )

        self.session.add(raw_review)
        await self.session.commit()
        await self.session.refresh(raw_review)
        return raw_review