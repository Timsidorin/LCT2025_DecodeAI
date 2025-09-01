# app/main.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uuid
from datetime import datetime
import logging

from app.models import ReviewCreate, ReviewResponse
from app.services.kafka_broker import KafkaBrokerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

kafka_broker = KafkaBrokerManager() # Создаем экземпляр брокера


async def get_kafka_broker():
    return kafka_broker


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запуск приложения
    try:
        await kafka_broker.connect()
        yield
    except Exception as e:
        raise
    finally:
        await kafka_broker.close()


app = FastAPI(
    title="Pulse Review API",
    description="API для приема и обработки клиентских отзывов",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/reviews",
          response_model=ReviewResponse,
          status_code=status.HTTP_202_ACCEPTED,
          summary="Создать новый отзыв",
          description="Принимает отзыв и публикует его в Kafka для дальнейшей обработки")
async def create_review(
        review: ReviewCreate,
        broker: KafkaBrokerManager = Depends(get_kafka_broker)
):
    try:
        review_id = str(uuid.uuid4())
        created_at = datetime.now()

        kafka_message = {
            "id": review_id,
            "created_at": created_at.isoformat(),
            **review.dict()
        }

        # Публикуем сообщение в Kafka
        await broker.publish(
            topic="raw_reviews",
            message=kafka_message,
            key=review_id
        )

        response_data = {
            "id": review_id,
            "created_at": created_at,
            "status": "accepted",
            **review.dict()
        }

        return response_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.post("/reviews/batch",
          status_code=status.HTTP_202_ACCEPTED,
          summary="Создать несколько отзывов",
          description="Принимает несколько отзывов и публикует их в Kafka для обработки")
async def create_reviews_batch(
        reviews: list[ReviewCreate],
        broker: KafkaBrokerManager = Depends(get_kafka_broker)
):
    try:
        results = []

        for review in reviews:
            review_id = str(uuid.uuid4())
            created_at = datetime.now()
            kafka_message = {
                "id": review_id,
                "created_at": created_at.isoformat(),
                **review.dict()
            }

            await broker.publish(
                topic="raw_reviews",
                message=kafka_message,
                key=review_id
            )

            results.append({
                "id": review_id,
                "status": "accepted"
            })


        return {"results": results}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.get("/health")
async def health_check(broker: KafkaBrokerManager = Depends(get_kafka_broker)):
    return {
        "status": "healthy",
        "kafka_connected": broker.is_connected,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    return {
        "message": "Pulse Review API",
        "version": "1.0.0",
        "endpoints": {
            "Опубликовать отзыв": "/reviews",
            "Опубликовать пакет отзывов": "/reviews/batch",
            "Проверить состояние сервиса": "/health"
        }
    }