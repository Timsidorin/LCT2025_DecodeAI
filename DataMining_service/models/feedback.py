import uuid
from datetime import datetime
from typing import Optional, List, Dict

from pydantic import UUID4
from sqlalchemy import ForeignKey, JSON, Enum, text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from core.database import Base
import sqlalchemy as sa





class raw_reviews(Base):
    """
    Таблица 'сырых отзывов'
    """
    __tablename__ = "reviews.raw_reviews"
    uuid: Mapped[UUID4] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    review_body: Mapped[str] = mapped_column(sa.Text, nullable=False)
    product_name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    time_to_publish: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
