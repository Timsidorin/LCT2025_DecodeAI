from contextlib import asynccontextmanager
import json
import os
import subprocess

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.ReviewInferenceModel import PredictResponse, PredictRequest, PredictionOutput
from core.config import configs
from faststream.kafka.fastapi import KafkaRouter
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RawReview(BaseModel):
    text: str = Field(..., description="Текст отзыва")
    review_id: int = Field(None, description="ID отзыва")

kafka_router = KafkaRouter(
    configs.BOOTSTRAP_SERVICE,
    schema_url="/asyncapi",
    include_in_schema=True
)

def sync_ml_analysis(data):
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

async def analyze_single_review(review_text: str, review_id: int) -> dict:
    reviews_data = [{"id": review_id, "text": review_text}]
    result = await analyze_sentiment_and_topics_batch(reviews_data)

    if isinstance(result, list) and len(result) > 0:
        return result[0]
    return {"id": review_id, "topics": [], "sentiments": []}

@kafka_router.subscriber("raw_reviews")
async def process_raw_review(msg: RawReview):
    try:
        analysis_result = await analyze_single_review(msg.text, msg.review_id or 0)
        logger.info(f"Отзыв {msg.review_id} обработан")
    except Exception as e:
        logger.error(f"Ошибка при обработке отзыва {msg.review_id}: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await kafka_router.broker.start()
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

@app.post("/predict", response_model=PredictResponse, description="Проверка ML модели")
async def predict_sentiment_and_topics(request: PredictRequest):
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
