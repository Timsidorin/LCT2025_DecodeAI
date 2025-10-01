# MLProcessing_service/main.py
from contextlib import asynccontextmanager
import json
import os
import subprocess
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from MLProcessing_service.processed_repository import ProcessedReviewRepository
from models.ReviewInferenceModel import PredictResponse, PredictRequest, PredictionOutput
from core.config import configs
from core.database import get_async_session
from faststream.kafka.fastapi import KafkaRouter
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RawReviewMessage(BaseModel):
    """Модель сообщения из Kafka топика raw_reviews"""
    data: list[dict] = Field(..., description="Список отзывов")


kafka_router = KafkaRouter(
    configs.BOOTSTRAP_SERVICE,
    schema_url="/asyncapi",
    include_in_schema=True
)


def sync_ml_analysis(data):
    """ вызов ML модели"""
    input_file_path = None
    output_file_path = None
    try:
        root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        input_file_path = os.path.join(root_dir, "api_input.json")
        output_file_path = os.path.join(root_dir, "api_output.json")

        input_data = {"data": data}
        with open(input_file_path, 'w', encoding='utf-8') as f:
            json.dump(input_data, f, ensure_ascii=False, indent=2)

        command = [
            "python",
            "analyze_single_review.py",
            "--input", "api_input.json",
            "--output", "api_output.json",
            "--workers", "8",
            "--batch-size", "8"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            encoding='utf-8',
            cwd=root_dir
        )

        if result.returncode != 0:
            error_msg = f"ML модель завершилась с кодом {result.returncode}"
            if result.stderr:
                error_msg += f". STDERR: {result.stderr}"
            raise Exception(error_msg)

        if not os.path.exists(output_file_path):
            raise FileNotFoundError("ML модель не создала выходной файл")

        if os.path.getsize(output_file_path) == 0:
            raise ValueError("Выходной файл пуст")

        with open(output_file_path, 'r', encoding='utf-8') as f:
            ml_result = json.load(f)

        if isinstance(ml_result, dict) and "predictions" in ml_result:
            return ml_result["predictions"]
        elif isinstance(ml_result, list):
            return ml_result
        else:
            return [ml_result] if ml_result else []

    finally:
        for file_path in [input_file_path, output_file_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except OSError:
                    pass


async def analyze_sentiment_and_topics_batch(reviews_data: list) -> list:
    """Пакетный анализ отзывов через ML модель"""
    try:
        root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        script_path = os.path.join(root_dir, "analyze_single_review.py")

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"ML скрипт не найден: {script_path}")
        result = sync_ml_analysis(reviews_data)
        return result

    except Exception as e:
        logger.error(f"Ошибка в ML анализе: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def process_ml_result(review_data: dict, ml_prediction: dict) -> dict:
    """
    Обработка результата ML и подготовка данных для БД

    ML модель возвращает:
    {"id": 1, "topics": ["Обслуживание", "Мобильное приложение"],
     "sentiments": ["положительно", "отрицательно"]}

    В БД сохраняется:
    rating: "positive"/"negative"/"neutral"
    product: "Обслуживание"
    """
    topics = ml_prediction.get("topics", [])
    sentiments = ml_prediction.get("sentiments", [])
    product = topics[0] if topics else None

    sentiment_to_english = {
        "отрицательно": "negative",
        "положительно": "positive",
        "нейтрально": "neutral"
    }

    rating = "neutral"
    if "отрицательно" in sentiments:
        rating = "negative"
    elif all(s == "положительно" for s in sentiments) and len(sentiments) > 0:
        rating = "positive"
    elif "нейтрально" in sentiments:
        rating = "neutral"

    return {
        "text": review_data["text"],
        "source": review_data.get("source", "API"),
        "gender": review_data.get("gender"),
        "city": review_data.get("city"),
        "region": review_data.get("region"),
        "region_code": review_data.get("region_code"),
        "datetime_review": datetime.fromisoformat(review_data["datetime_review"]),
        "rating": rating,
        "product": product
    }


@kafka_router.subscriber("raw_reviews")
async def process_raw_review(msg: RawReviewMessage):
    """
    Обработчик сообщений из Kafka топика raw_reviews

    Формат входящего сообщения:
    {
        "data": [
            {
                "id": "uuid",
                "text": "текст отзыва",
                "gender": "М",
                "source": "banki.ru",
                "city": "Москва",
                "region": "г Москва",
                "region_code": "RU-MOW",
                "datetime_review": "2025-10-01T20:00:00"
            }
        ]
    }
    """
    try:
        logger.info(f"📨 Получено сообщение из Kafka: {len(msg.data)} отзывов")
        reviews_for_ml = []
        for idx, review in enumerate(msg.data):
            reviews_for_ml.append({
                "id": idx,
                "text": review["text"]
            })

        logger.info(f"🤖 Отправка в ML модель...")
        ml_results = await analyze_sentiment_and_topics_batch(reviews_for_ml)
        logger.info(f"ML обработка завершена: {len(ml_results)} результатов")

        async for session in get_async_session():
            repo = ProcessedReviewRepository(session)

            for idx, review_data in enumerate(msg.data):
                try:
                    ml_prediction = ml_results[idx] if idx < len(ml_results) else {}
                    processed_data = process_ml_result(review_data, ml_prediction)
                    result = await repo.add_processed_review(**processed_data)

                    if result:
                        logger.info(
                            f"Отзыв сохранен в processed_reviews: {result.uuid} "
                            f"(тональность: {result.rating}, продукт: {result.product})"
                        )
                    else:
                        logger.error(f"❌ Ошибка сохранения отзыва в БД")

                except Exception as e:
                    logger.error(f"❌ Ошибка обработки отзыва {idx}: {str(e)}")
                    continue

            break

        logger.info(f"Обработка пакета завершена успешно")

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке сообщения: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    try:
        await kafka_router.broker.start()
        logger.info("✅ Kafka подключен, слушаем топик raw_reviews")
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к Kafka: {e}")

    try:
        yield
    finally:
        try:
            await kafka_router.broker.close()
            logger.info("✅ Kafka отключен")
        except Exception as e:
            logger.error(f"❌ Ошибка отключения Kafka: {e}")


app = FastAPI(
    title=configs.PROJECT_NAME,
    description="ML Processing Service - обработка отзывов через ML модель",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kafka_router)


@app.post("/predict", response_model=PredictResponse, description="Ручной прогон через ML модель")
async def predict_sentiment_and_topics(request: PredictRequest):
    """
    Endpoint для ручного тестирования ML модели
    Не используется в основном потоке обработки
    """
    try:
        if len(request.data) == 0:
            raise HTTPException(status_code=400, detail="Список отзывов не может быть пустым")

        reviews_data = []
        for review in request.data:
            if review.text.strip():
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

        return PredictResponse(predictions=predictions)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка в endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=configs.HOST, port=configs.PORT, reload=True)
