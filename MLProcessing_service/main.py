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


    try:
        yield
    finally:
        await kafka_broker.close()


app = FastAPI(
    title="Pulse Review API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Маршруты


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=configs.HOST, port=configs.PORT, reload=True)
