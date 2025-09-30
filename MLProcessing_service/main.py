# main.py
from contextlib import asynccontextmanager
import asyncio
import json
import tempfile
import os

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from models.ReviewInferenceModel import ReviewInput, PredictResponse, PredictRequest, PredictionOutput
from core.config import configs
from core.database import get_async_session
from services.auth_service import AuthService
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# FastStream импорты
from faststream.kafka.fastapi import KafkaRouter
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scheduler = None
auth_service = AuthService()
security = HTTPBearer()


class RawReview(BaseModel):
    text: str = Field(..., description="Текст отзыва")
    review_id: int = Field(None, description="ID отзыва")


kafka_router = KafkaRouter(
    configs.BOOTSTRAP_SERVICE,
    schema_url="/asyncapi",
    include_in_schema=True
)


# ========================= ML ФУНКЦИИ =========================

async def analyze_sentiment_and_topics_batch(reviews_data: list) -> list:
    """
    Асинхронная функция для пакетного анализа отзывов через ML модель
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as input_file:
            json.dump(reviews_data, input_file, ensure_ascii=False, indent=2)
            input_file_path = input_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
            output_file_path = output_file.name

        try:
            batch_size = min(len(reviews_data), 32)
            command = [
                "python",
                "analyze_single_review.py",
                "--input", input_file_path,
                "--output", output_file_path,
                "--workers", "8",
                "--batch-size", str(batch_size)
            ]

            logger.info(f"Запуск ML модели: {' '.join(command)}")

            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=300.0  # 5 минут таймаут
                )
            except asyncio.TimeoutError:
                process.kill()
                raise HTTPException(status_code=504, detail="ML model timeout")

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                logger.error(f"ML model error: {error_msg}")
                raise HTTPException(status_code=500, detail=f"ML model failed: {error_msg}")

            with open(output_file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return result if isinstance(result, list) else [result]

        finally:
            try:
                os.unlink(input_file_path)
                os.unlink(output_file_path)
            except OSError as e:
                logger.warning(f"Не удалось удалить временные файлы: {e}")

    except Exception as e:
        logger.error(f"Error in ML batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


async def analyze_single_review(review_text: str, review_id: int) -> dict:
    """
    Анализ одного отзыва (для Kafka обработки)
    """
    reviews_data = [{"id": review_id, "text": review_text}]
    result = await analyze_sentiment_and_topics_batch(reviews_data)

    if isinstance(result, list) and len(result) > 0:
        return result[0]
    return {"id": review_id, "topics": [], "sentiments": []}



@kafka_router.subscriber("raw_reviews", description="Подписчик на сырые отзывы из Kafka")
async def process_raw_review(msg: RawReview):
    """
    Обработчик сырых отзывов из Kafka topic "raw_reviews"

    Получает отзыв, обрабатывает его через ML модель
    и сохраняет результаты в базу данных
    """
    try:
        logger.info(f"Получен отзыв для обработки: ID={msg.review_id}, текст: {msg.text[:50]}...")

        analysis_result = await analyze_single_review(msg.text, msg.review_id or 0)

        logger.info(f"Отзыв {msg.review_id} успешно обработан: {analysis_result}")

        #добавить сохранение в БД
        # async with get_async_session() as session:
        #     await save_analysis_to_db(session, analysis_result)

    except Exception as e:
        logger.error(f"Ошибка при обработке отзыва {msg.review_id}: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск приложения...")

    # Запуск Kafka router
    try:
        await kafka_router.broker.start()
        logger.info("Kafka брокер подключен и подписчик активен")
    except Exception as e:
        logger.error(f"Ошибка подключения к Kafka: {e}")

    try:
        yield
    finally:
        try:
            await kafka_router.broker.close()
            logger.info("Kafka брокер отключен")
        except Exception as e:
            logger.error(f"Ошибка отключения Kafka: {e}")


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
    try:
        logger.info(
            f"Получен запрос на анализ {len(request.data)} отзывов от {current_user.get('username', 'unknown')}")

        if len(request.data) == 0:
            raise HTTPException(status_code=400, detail="Список отзывов не может быть пустым")

        if len(request.data) > 100:
            raise HTTPException(status_code=400, detail="Максимальное количество отзывов за один запрос: 100")

        reviews_data = []
        for review in request.data:
            if not review.text.strip():
                continue
            reviews_data.append({
                "id": review.id,
                "text": review.text.strip()
            })

        if not reviews_data:
            raise HTTPException(status_code=400, detail="Не найдено отзывов с непустым текстом")

        ml_results = await analyze_sentiment_and_topics_batch(reviews_data)

        predictions = []
        results_by_id = {}
        for result in ml_results:
            if isinstance(result, dict) and "id" in result:
                results_by_id[result["id"]] = result
        for review in request.data:
            ml_result = results_by_id.get(review.id, {})

            prediction = PredictionOutput(
                id=review.id,
                topics=ml_result.get("topics", []),
                sentiments=ml_result.get("sentiments", [])
            )
            predictions.append(prediction)

        logger.info(f"Успешно обработано {len(predictions)} отзывов")
        return PredictResponse(predictions=predictions)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка в endpoint predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


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
        text="Отличный банк, очень доволен обслуживанием!",
        review_id=999
    )

    try:
        await kafka_router.broker.publish(test_review, "raw_reviews")
        return {
            "message": "Тестовое сообщение отправлено в Kafka",
            "review_id": test_review.review_id,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Ошибка отправки в Kafka: {str(e)}")
        return {"error": str(e), "status": "failed"}


@app.post("/test-predict")
async def test_predict_no_auth():
    """Тестовый endpoint для проверки ML модели без аутентификации"""
    test_data = [
        {"id": 1, "text": "Очень понравилось обслуживание в отделении, но мобильное приложение часто зависает."},
        {"id": 2, "text": "Кредитную карту одобрили быстро, но лимит слишком маленький."}
    ]

    try:
        result = await analyze_sentiment_and_topics_batch(test_data)
        return {
            "input": test_data,
            "ml_results": result,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Ошибка в тестовом endpoint: {str(e)}")
        return {
            "input": test_data,
            "error": str(e),
            "status": "failed"
        }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=configs.HOST, port=configs.PORT, reload=True)
