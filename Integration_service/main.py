
# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from routers.diagrams_router import router
from core.config import configs
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#kafka_broker = KafkaBrokerManager()
scheduler = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Сервис интеграции включен!")
    try:
        yield
    finally:
        logger.info("Сервис интеграции выключен!")


app = FastAPI(
    title=configs.PROJECT_NAME,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=configs.HOST, port=configs.PORT, reload=True)
