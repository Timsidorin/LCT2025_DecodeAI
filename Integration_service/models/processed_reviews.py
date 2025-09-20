# models.py
from datetime import datetime
from typing import Optional
from uuid import uuid4
from sqlalchemy import String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from core.database import Base


class Review(Base):
    """
    Модель для таблицы обработанных отзывов
    """
    __tablename__ = "processed_reviews"

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

    # Убираем enum, используем простые строки
    rating: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Тональность отзыва (positive/negative/neutral)"
    )
    product: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Название продукта"
    )
    gender: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="Пол автора отзыва (male/female/unknown)"
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
