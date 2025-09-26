import httpx
import logging
from fastapi import HTTPException, status
from MLProcessing_service.core.config import configs

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self):
        # URL вашего auth-сервиса
        self.auth_service_url = configs.AUTH_SERVICE_URL

    async def verify_token(self, token: str) -> dict:
        """Проверка токена через внешний auth-сервис"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.auth_service_url}/api/auth/me",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10.0,
                )
                if response.status_code == 200:
                    user_data = response.json()
                    logger.info(
                        f"Токен успешно проверен для пользователя: {user_data.get('username', 'unknown')}"
                    )
                    return user_data
                elif response.status_code == 401:
                    logger.warning("Получен недействительный токен")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Недействительный токен",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                else:
                    logger.error(f"Auth-сервис вернул ошибку: {response.status_code}")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Сервис авторизации недоступен",
                    )

        except httpx.TimeoutException:
            logger.error("Таймаут при обращении к auth-сервису")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Сервис авторизации недоступен",
            )
        except httpx.RequestError as e:
            logger.error(f"Ошибка при обращении к auth-сервису: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ошибка соединения с сервисом авторизации",
            )
