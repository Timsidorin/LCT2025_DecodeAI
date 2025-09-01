from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Optional
from datetime import datetime
from models.feedback import RawReview


class RawReviewRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_raw_review(
        self,
        source_id: int,
        text: str,
        created_at: list[datetime],
        product: Optional[str] = None,
        rating: Optional[list[str]] = None,
    ) -> RawReview:
        raw_review = RawReview(
            source_id=source_id,
            text=text,
            created_at=created_at,
            product=product,
            rating=rating,
        )
        self.session.add(raw_review)
        await self.session.commit()
        await self.session.refresh(raw_review)
        return raw_review
