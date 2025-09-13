from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncAttrs,
    AsyncSession,
)
from sqlalchemy.orm import DeclarativeBase, declared_attr
from core.config import get_db_url
from sqlalchemy import MetaData
import asyncio
from typing import AsyncGenerator
from contextlib import asynccontextmanager


DATABASE_URL = get_db_url()
engine = create_async_engine(DATABASE_URL)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(AsyncAttrs, DeclarativeBase):
    __abstract__ = True
    __table_args__ = {"extend_existing": True}

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return f"{cls.__name__.lower()}s"


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency для FastAPI - возвращает генератор сессий"""
    async with async_session() as session:
        yield session


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Контекст-менеджер для использования вне FastAPI"""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_session() -> AsyncSession:
    """Создает новую сессию для ручного управления"""
    return async_session()
