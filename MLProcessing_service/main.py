# MLProcessing_service/main.py
import json
import os
import subprocess
import uuid
import threading
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faststream.kafka.fastapi import KafkaRouter, KafkaMessage
from pydantic import BaseModel, Field
import logging

from core.database import get_async_session
from core.config import configs
from processed_repository import ProcessedReviewRepository
from models.ReviewInferenceModel import PredictResponse, PredictRequest, PredictionOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ML_MODEL_LOCK = threading.Lock()


class RawReviewMessage(BaseModel):
    data: list[dict] = Field(..., description="Список отзывов")


class KafkaMessageWrapper(BaseModel):
    msg: str


kafka_router = KafkaRouter(
    configs.BOOTSTRAP_SERVICE,
    schema_url="/asyncapi",
    include_in_schema=True
)


def sync_ml_analysis_optimized(data):
    input_file_path = None
    output_file_path = None

    try:
        temp_id = str(uuid.uuid4())[:8]
        root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

        input_file_path = os.path.join(root_dir, f"api_input_{temp_id}.json")
        output_file_path = os.path.join(root_dir, f"api_output_{temp_id}.json")

        with open(input_file_path, 'w', encoding='utf-8') as f:
            json.dump({"data": data}, f, ensure_ascii=False, indent=2)

        command = [
            "python", "-u", "analyze_single_review.py",
            "--input", f"api_input_{temp_id}.json",
            "--output", f"api_output_{temp_id}.json",
            "--workers", "4", "--batch-size", "4"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
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

        with open(output_file_path, 'r', encoding='utf-8') as f:
            ml_result = json.load(f)

        if isinstance(ml_result, dict) and "predictions" in ml_result:
            return ml_result["predictions"]
        elif isinstance(ml_result, list):
            return ml_result
        else:
            return [ml_result] if ml_result else []

    except subprocess.TimeoutExpired:
        raise Exception("ML модель превысила время выполнения")
    except Exception as e:
        logger.error(f"Ошибка в ML анализе: {str(e)}")
        raise
    finally:
        for file_path in [input_file_path, output_file_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except OSError as e:
                    logger.warning(f"Не удалось удалить файл {file_path}: {e}")


async def analyze_sentiment_and_topics_batch(reviews_data: list) -> list:
    try:
        logger.info(f"🤖 Начало ML анализа для {len(reviews_data)} отзывов...")

        with ML_MODEL_LOCK:
            result = sync_ml_analysis_optimized(reviews_data)

        logger.info("ML анализ полностью завершен")
        return result

    except Exception as e:
        logger.error(f"Критическая ошибка в ML анализе: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def process_ml_result(review_data: dict, ml_prediction: dict) -> dict:
    topics = ml_prediction.get("topics", [])
    sentiments = ml_prediction.get("sentiments", [])

    sentiment_to_english = {
        "отрицательно": "negative",
        "положительно": "positive",
        "нейтрально": "neutral"
    }

    first_sentiment = sentiments[0] if sentiments else "нейтрально"

    return {
        "text": review_data["text"],
        "source": review_data.get("source", "API"),
        "gender": review_data.get("gender"),
        "city": review_data.get("city"),
        "region": review_data.get("region"),
        "region_code": review_data.get("region_code"),
        "datetime_review": datetime.fromisoformat(review_data["datetime_review"]),
        "rating": sentiment_to_english.get(first_sentiment, "neutral"),
        "product": topics[0] if topics else None
    }


@kafka_router.subscriber("raw_reviews")
async def process_raw_review(message: KafkaMessage):
    try:
        raw_body = message.body
        message_text = raw_body.decode('utf-8') if isinstance(raw_body, bytes) else str(raw_body)
        logger.info(f"Получено сообщение из Kafka, размер: {len(message_text)} символов")

        message_data = json.loads(message_text)
        reviews_data = None

        if "data" in message_data:
            reviews_data = message_data["data"]
        elif "msg" in message_data:
            inner_data = json.loads(message_data["msg"])
            reviews_data = inner_data.get("data") if "data" in inner_data else None

        if not reviews_data or not isinstance(reviews_data, list):
            logger.error("Данные отзывов отсутствуют или не являются списком")
            return

        reviews_for_ml = []
        for idx, review in enumerate(reviews_data):
            if isinstance(review, dict) and "text" in review:
                reviews_for_ml.append({"id": idx, "text": review["text"]})

        if not reviews_for_ml:
            logger.error("Не найдено отзывов с валидным текстом")
            return

        ml_results = await analyze_sentiment_and_topics_batch(reviews_for_ml)
        logger.info(f"✅ ML обработка завершена: {len(ml_results)} результатов")

        ml_results_by_id = {result["id"]: result for result in ml_results if isinstance(result, dict) and "id" in result}

        async for session in get_async_session():
            repo = ProcessedReviewRepository(session)
            saved_count = 0

            for idx, review_data in enumerate(reviews_data):
                try:
                    if not isinstance(review_data, dict) or "text" not in review_data:
                        continue

                    ml_prediction = ml_results_by_id.get(idx, {})
                    processed_data = process_ml_result(review_data, ml_prediction)

                    if await repo.add_processed_review(**processed_data):
                        saved_count += 1
                    else:
                        logger.error(f"Ошибка сохранения отзыва {idx + 1} в БД")

                except Exception as e:
                    logger.error(f"Ошибка обработки отзыва {idx + 1}: {str(e)}")
                    continue

            break

        logger.info(f"Обработка завершена: сохранено {saved_count} отзывов")

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке сообщения: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await kafka_router.broker.start()
        logger.info("Kafka подключен, слушаем топик raw_reviews")
        logger.info("ML Processing Service готов к работе")
    except Exception as e:
        logger.error(f"Ошибка подключения к Kafka: {e}")

    try:
        yield
    finally:
        try:
            await kafka_router.broker.close()
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


@app.post("/predict", response_model=PredictResponse, summary="Ручной прогон через модель")
async def predict_sentiment_and_topics(request: PredictRequest):
    try:
        if len(request.data) == 0:
            raise HTTPException(status_code=400, detail="Список отзывов не может быть пустым")

        reviews_data = [
            {"id": review.id, "text": review.text.strip()}
            for review in request.data if review.text.strip()
        ]

        if not reviews_data:
            raise HTTPException(status_code=400, detail="Не найдено отзывов с непустым текстом")

        ml_results = await analyze_sentiment_and_topics_batch(reviews_data)
        results_by_id = {result["id"]: result for result in ml_results if isinstance(result, dict) and "id" in result}

        predictions = [
            PredictionOutput(
                id=review.id,
                topics=results_by_id.get(review.id, {}).get("topics", []),
                sentiments=results_by_id.get(review.id, {}).get("sentiments", [])
            )
            for review in request.data
        ]

        return PredictResponse(predictions=predictions)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка в API endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=configs.HOST, port=configs.PORT)