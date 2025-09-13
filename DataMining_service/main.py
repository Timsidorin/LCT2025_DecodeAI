from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import configs
from core.broker import KafkaBrokerManager
from core.database import get_async_session
from repository import ReviewRepository
from shemas.review import ReviewCreate, ReviewResponse
from ReviewService import ReviewService
from apscheduler.schedulers.asyncio import AsyncIOScheduler


kafka_broker = KafkaBrokerManager()
scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler
    await kafka_broker.connect()
    if scheduler is None:
        scheduler = AsyncIOScheduler()
        from auto_parser.ParserService import get_and_publish_reviews
        scheduler.add_job(
            get_and_publish_reviews,
            "interval",
            minutes=3,
            id="review_parser",
            replace_existing=True,
            max_instances=1
        )

        try:
            scheduler.start()
            print(f"Планировщик запущен с {len(scheduler.get_jobs())} задачами")
        except Exception as e:
            print(f"Ошибка запуска планировщика: {e}")

    try:
        yield
    finally:
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=True)
        await kafka_broker.close()


app = FastAPI(
    title="Pulse Review API",
    description="API для приема и обработки клиентских отзывов",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_review_service(
    session: AsyncSession = Depends(get_async_session),
    broker: KafkaBrokerManager = Depends(lambda: kafka_broker)
) -> ReviewService:
    repo = ReviewRepository(session)
    return ReviewService(broker, repo)


# Маршруты
@app.post(
    "/api/reviews",
    response_model=ReviewResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Создать новый отзыв",
    description="Принимает отзыв, публикует в Kafka и сохраняет в БД"
)
async def create_review(
    review: ReviewCreate,
    service: ReviewService = Depends(get_review_service)
) -> ReviewResponse:
    try:
        return await service.create_review(review)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при создании отзыва: {str(e)}"
        )


@app.get("/")
async def root():
    return {
        "name": "Pulse Review API",
        "version": "1.0.0",
        "endpoints": {"create_review": "/reviews"}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=configs.HOST, port=configs.PORT, reload=True)
