# models.py
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import Column, Integer, Text, DateTime, String
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class RawReview(Base):
    """
    Модель для таблицы 'raw_reviews'
    """
    __tablename__ = "raw_reviews"

    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    datetime_review = Column(DateTime(timezone=True), nullable=False)
    product = Column(String)
    rating = Column(ARRAY(String(1)))