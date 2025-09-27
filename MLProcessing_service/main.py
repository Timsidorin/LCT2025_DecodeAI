# main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from models.ReviewInferenceModel import ReviewInput, PredictResponse, PredictRequest
from core.config import configs
from core.database import get_async_session
from services.auth_service import AuthService
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# FastStream импорты
from faststream.kafka.fastapi import KafkaRouter
from pydantic import BaseModel, Field
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scheduler = None
auth_service = AuthService()
security = HTTPBearer()


# Модель для сырых отзывов из Kafka
class RawReview(BaseModel):
    text: str = Field(..., description="Текст отзыва")

kafka_router = KafkaRouter(
    configs.BOOTSTRAP_SERVICE,
    schema_url="/asyncapi",
    include_in_schema=True
)

@kafka_router.subscriber("raw_reviews", description="Подписчик на сырые отзывы из Kafka")
async def process_raw_review(msg: RawReview):
    """
    Обработчик сырых отзывов из Kafka topic "raw_reviews"

    Получает отзыв, обрабатывает его через ML модель
    и сохраняет результаты в базу данных
    """
    try:
        review_data = {
            "text": msg.text,
        }

        # analysis_result = await analyze_sentiment_and_topics(msg.text)

        logger.info(f"Отзыв {msg.review_id} успешно обработан")

    except Exception as e:
        logger.error(f"Ошибка при обработке отзыва {msg.review_id}: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск приложения...")

    # Запуск Kafka router
    await kafka_router.broker.start()
    logger.info("Kafka брокер подключен и подписчик активен")

    try:
        yield
    finally:
        # Закрытие Kafka router
        await kafka_router.broker.close()
        logger.info("Kafka брокер отключен")


app = FastAPI(title=configs.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(kafka_router)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Проверка JWT токена через внешний auth-сервис"""
    return await auth_service.verify_token(credentials.credentials)


# Защищенный endpoint для анализа отзывов
@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment_and_topics(
        request: PredictRequest, current_user: dict = Depends(verify_token)
):
    """
    Защищенный endpoint для анализа тональности и выделения тем из отзывов

    Требует Bearer токен в заголовке Authorization.
    Токен проверяется через внешний auth-сервис.

    Принимает список отзывов и возвращает для каждого:
    - Выявленные темы
    - Тональность для каждой темы
    """
    pass


# Тестовый endpoint для проверки аутентификации
@app.get("/test-auth")
async def test_auth(current_user: dict = Depends(verify_token)):
    """Тестовый endpoint для проверки аутентификации"""
    return {
        "message": "Аутентификация успешна",
        "user": current_user.get("username", "unknown"),
        "service": "ML Processing Service",
    }


@app.post("/test-kafka")
async def test_kafka_publish():
    """Тестовый endpoint для публикации сообщения в Kafka"""
    test_review = RawReview(
        text="Отличный банк, очень доволен обслуживанием!"
    )

    await kafka_router.broker.publish(test_review, "raw_reviews")
    return {"message": "Тестовое сообщение отправлено в Kafka"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=configs.HOST, port=configs.PORT, reload=True)
