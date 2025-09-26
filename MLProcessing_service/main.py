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
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# kafka_broker = KafkaBrokerManager()
scheduler = None

auth_service = AuthService()
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler
    # await kafka_broker.connect()
    logger.info("Kafka брокер подключен")

    try:
        yield
    finally:
        # await kafka_broker.close()
        logger.info("Kafka брокер отключен")


app = FastAPI(title=configs.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=configs.HOST, port=configs.PORT, reload=True)
