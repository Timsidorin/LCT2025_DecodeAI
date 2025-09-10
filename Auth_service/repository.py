from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from models import User
from users_shema import UserRegister
from security import *


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_user(self, user_data: UserRegister) -> Optional[User]:
        """Создание нового пользователя"""
        try:
            # Хешируем пароль
            hashed_password = get_password_hash(user_data.password)

            # Создаем пользователя
            new_user = User(
                phone_number=user_data.phone_number,
                password=hashed_password,
                first_name=user_data.first_name,
                last_name=user_data.last_name
            )

            self.session.add(new_user)
            await self.session.commit()
            await self.session.refresh(new_user)

            return new_user
        except IntegrityError:
            await self.session.rollback()
            return None

    async def get_user_by_phone(self, phone_number: str) -> Optional[User]:
        """Получение пользователя по номеру телефона"""
        query = select(User).where(User.phone_number == phone_number)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_user_by_uuid(self, user_uuid: UUID) -> Optional[User]:
        """Получение пользователя по UUID"""
        query = select(User).where(User.uuid == user_uuid)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def authenticate_user(self, phone_number: str, password: str) -> Optional[User]:
        """Аутентификация пользователя"""
        user = await self.get_user_by_phone(phone_number)
        if not user or not verify_password(password, user.password):
            return None
        return user
