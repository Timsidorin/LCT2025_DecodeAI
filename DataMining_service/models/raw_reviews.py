# models/raw_review.py
from datetime import datetime
from typing import Optional
from uuid import uuid4
from sqlalchemy import String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from DataMining_service.core.database import Base


class RawReview(Base):
    """Модель для таблицы сырых отзывов (raw_reviews)"""

    __tablename__ = "raw_reviews"

    uuid: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
        comment="Уникальный идентификатор отзыва"
    )

    source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="API",
        server_default="API",
        comment="Источник отзыва"
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Текст отзыва"
    )

    gender: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        comment="Пол автора: М (мужчина) или Ж (женщина)"
    )

    city: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Город автора отзыва"
    )

    region: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Регион"
    )

    region_code: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        comment="Код региона"
    )

    datetime_review: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Дата и время написания отзыва"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
        comment="Время добавления в систему"
    )

    def __repr__(self):
        return f"<RawReview(uuid={self.uuid}, source={self.source}, gender={self.gender})>"
