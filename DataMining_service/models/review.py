# models.py
from datetime import datetime
from typing import Optional
from uuid import uuid4
from sqlalchemy import String, Text, DateTime, Enum as SQLEnum
from enum import Enum as PyEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from core.database import Base


class SentimentTypeDB(PyEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class GenderDB(PyEnum):
    MALE = "male"
    FEMALE = "female"


class Review(Base):
    """
    Модель для таблицы отзывов
    """
    __tablename__ = "reviews"

    uuid: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="API",
        comment="Источник отзыва"
    )
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Текст отзыва"
    )

    rating: Mapped[Optional[SentimentTypeDB]] = mapped_column(
        SQLEnum(SentimentTypeDB),
        nullable=True
    )
    product: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Название продукта"
    )
    gender: Mapped[Optional[GenderDB]] = mapped_column(
        SQLEnum(GenderDB),
        nullable=True,
        comment="Пол автора отзыва"
    )
    city: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Город автора отзыва"
    )

    region_code: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="Регион (автоопределение)"
    )

    datetime_review: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Дата и время написания отзыва"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.now,
        comment="Время добавления в систему"
    )
