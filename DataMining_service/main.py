

from fastapi import FastAPI, APIRouter
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import configs
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uuid
from datetime import datetime
from repository import RawReviewRepository
from shemas.feedback import ReviewCreate,ReviewResponse
from core.broker import KafkaBrokerManager
from  core.database import get_async_session

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
          status_code=status.HTTP_202_ACCEPTED,
          summary="Создать новый отзыв",
          description="Принимает отзыв, публикует в Kafka и сохраняет в БД")
async def create_review(
        review: ReviewCreate,
        broker: KafkaBrokerManager = Depends(get_kafka_broker),
        session: AsyncSession = Depends(get_async_session),
):
    try:
        review_id = str(uuid.uuid4())
        created_at = datetime.now()

        kafka_message = {
            "id": review_id,
            "created_at": created_at.isoformat(),
            **review.model_dump()
        }

        # Публикуем сообщение в Kafka
        await broker.publish(
            topic="raw_reviews",
            message=kafka_message,
            key=review_id.encode('utf-8')
        )

        repo = RawReviewRepository(session)
        await repo.add_raw_review(
            source_id=review.source_id,
            text=review.text,
            created_at=[created_at],
            product=review.product,
            rating=[str(review.rating)] if review.rating is not None else None,
        )

        response_data = {
            "uuid": review_id,
            "created_at": created_at,
            "status": "accepted",
            **review.model_dump()
        }
        return response_data


    except Exception as e:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/")
async def root():
    return {
        "message": "Pulse Review API",
        "endpoints": {
            "Опубликовать отзыв": "/reviews",
            "Опубликовать пакет отзывов": "/reviews/batch",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=configs.HOST, port=configs.PORT, reload=True)