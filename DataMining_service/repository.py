# repository.py
from datetime import datetime
from typing import Optional, List
from uuid import uuid4, UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from sqlalchemy.exc import IntegrityError
from models.review import Review


class ReviewRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_review(
            self,
            text: str,
            datetime_review: datetime,
            source: str = "API",
            product: Optional[str] = None,
            rating: Optional[str] = None,
            gender: Optional[str] = None,
            city: Optional[str] = None,
            region_code: Optional[str] = None  # Новое поле
    ) -> Optional[Review]:
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
            region_code: Код региона (необязательно)
        """
        try:
            review = Review(
                uuid=uuid4(),
                source=source,
                text=text,
                datetime_review=datetime_review,
                product=product,
                rating=rating,
                gender=gender,
                city=city,
                region_code=region_code
            )

            self.session.add(review)
            await self.session.commit()
            await self.session.refresh(review)
            return review

        except IntegrityError:
            await self.session.rollback()
            return None

    async def get_review_by_id(self, review_id: str) -> Optional[Review]:
        """Получение отзыва по UUID (строка)"""
        try:
            review_uuid = UUID(review_id)
            return await self.get_review_by_uuid(review_uuid)
        except ValueError:
            return None

    async def get_review_by_uuid(self, review_uuid: UUID) -> Optional[Review]:
        """Получение отзыва по UUID"""
        query = select(Review).where(Review.uuid == review_uuid)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    # async def get_reviews(
    #         self,
    #         limit: int = 10,
    #         offset: int = 0,
    #         source: Optional[str] = None,
    #         city: Optional[str] = None,
    #         region_code: Optional[str] = None,
    #         rating: Optional[SentimentType] = None
    # ) -> List[Review]:
    #     """Получение отзывов с фильтрацией и пагинацией"""
    #     query = select(Review)
    #     if source:
    #         query = query.where(Review.source == source)
    #     if city:
    #         query = query.where(Review.city == city)
    #     if region_code:
    #         query = query.where(Review.region_code == region_code)
    #     if rating:
    #         db_rating = SentimentTypeDB(rating.value)
    #         query = query.where(Review.rating == db_rating)
    #
    #     # Сортировка по дате создания (новые сначала)
    #     query = query.order_by(desc(Review.created_at))
    #
    #     # Пагинация
    #     query = query.offset(offset).limit(limit)
    #
    #     result = await self.session.execute(query)
    #     return result.scalars().all()

    async def get_reviews_by_source(self, source: str) -> List[Review]:
        """Получение всех отзывов по источнику"""
        query = select(Review).where(Review.source == source).order_by(desc(Review.created_at))
        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_reviews_by_region(self, region_code: str, limit: int = 10) -> List[Review]:
        """Получение отзывов по коду региона"""
        query = (
            select(Review)
            .where(Review.region_code == region_code)
            .order_by(desc(Review.created_at))
            .limit(limit)
        )
        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_reviews_by_city(self, city: str, limit: int = 10) -> List[Review]:
        """Получение отзывов по городу"""
        query = (
            select(Review)
            .where(Review.city == city)
            .order_by(desc(Review.created_at))
            .limit(limit)
        )
        result = await self.session.execute(query)
        return result.scalars().all()

    # async def update_review_analysis(
    #         self,
    #         review_id: str,
    #         rating: Optional[str] = None,
    #         gender: Optional[str] = None
    # ) -> Optional[Review]:
    #     """Обновление результатов анализа отзыва"""
    #     try:
    #         review_uuid = UUID(review_id)
    #         review = await self.get_review_by_uuid(review_uuid)
    #
    #         if not review:
    #             return None
    #
    #         if rating:
    #             try:
    #                 review.rating = SentimentTypeDB(rating)
    #             except ValueError:
    #                 pass
    #
    #         if gender:
    #             try:
    #                 review.gender = GenderDB(gender)
    #             except ValueError:
    #                 pass
    #
    #         await self.session.commit()
    #         await self.session.refresh(review)
    #         return review
    #
    #     except (ValueError, IntegrityError):
    #         await self.session.rollback()
    #         return None

    async def get_reviews_stats(self) -> dict:
        """Получение общей статистики по отзывам"""
        # Общее количество
        total_query = select(func.count(Review.uuid))
        total_result = await self.session.execute(total_query)
        total_count = total_result.scalar()

        source_query = (
            select(Review.source, func.count(Review.uuid))
            .group_by(Review.source)
        )
        source_result = await self.session.execute(source_query)
        sources_stats = {row[0]: row[1] for row in source_result.fetchall()}


        rating_query = (
            select(Review.rating, func.count(Review.uuid))
            .where(Review.rating.is_not(None))
            .group_by(Review.rating)
        )
        rating_result = await self.session.execute(rating_query)
        ratings_stats = {row[0].value if row[0] else None: row[1] for row in rating_result.fetchall()}

        return {
            "total_reviews": total_count,
            "by_source": sources_stats,
            "by_rating": ratings_stats
        }

    async def get_reviews_stats_by_region(self) -> dict:
        """Получение статистики отзывов по регионам"""
        query = (
            select(
                Review.region_code,
                Review.city,
                func.count(Review.uuid).label('count')
            )
            .where(Review.region_code.is_not(None))
            .group_by(Review.region_code, Review.city)
            .order_by(desc('count'))
        )

        result = await self.session.execute(query)

        stats = {}
        for row in result.fetchall():
            region_code, city, count = row

            if region_code not in stats:
                stats[region_code] = {
                    'total_reviews': 0,
                    'cities': {}
                }

            stats[region_code]['total_reviews'] += count
            stats[region_code]['cities'][city] = count

        return stats

    async def get_recent_reviews(self, hours: int = 24, limit: int = 50) -> List[Review]:
        """Получение недавних отзывов за указанное количество часов"""
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)

        query = (
            select(Review)
            .where(Review.created_at >= cutoff_time)
            .order_by(desc(Review.created_at))
            .limit(limit)
        )

        result = await self.session.execute(query)
        return result.scalars().all()

    async def count_reviews(
            self,
            source: Optional[str] = None,
            city: Optional[str] = None,
            region_code: Optional[str] = None
    ) -> int:
        """Подсчет количества отзывов с фильтрацией"""
        query = select(func.count(Review.uuid))

        if source:
            query = query.where(Review.source == source)
        if city:
            query = query.where(Review.city == city)
        if region_code:
            query = query.where(Review.region_code == region_code)

        result = await self.session.execute(query)
        return result.scalar()

    async def delete_review(self, review_id: str) -> bool:
        """Удаление отзыва по ID"""
        try:
            review_uuid = UUID(review_id)
            review = await self.get_review_by_uuid(review_uuid)

            if review:
                await self.session.delete(review)
                await self.session.commit()
                return True
            return False

        except (ValueError, IntegrityError):
            await self.session.rollback()
            return False
