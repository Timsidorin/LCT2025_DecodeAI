# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from Integration_service.routers import dashboard_main, sentiment_analytics, regional_analytics, products_analytics, \
    cities_analytics, sources_analytics, demographics_analytics, trends_analytics, matrix_chart_router,statistics_router
from core.config import configs

from services.monitoring_service import monitoring_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager для запуска и остановки сервисов"""
    logger.info("🚀 Запуск сервиса интеграции PulseAI...")

    try:
        #await monitoring_service.start_monitoring()
        logger.info("✅ Система мониторинга запущена")

        yield

    except Exception as e:
        logger.error(f"❌ Ошибка при запуске сервисов: {e}")
        raise
    finally:
        logger.info("⏹️ Остановка сервисов...")

        try:
            await monitoring_service.stop_monitoring()
            logger.info("✅ Система мониторинга остановлена")
        except Exception as e:
            logger.error(f"❌ Ошибка при остановке мониторинга: {e}")

        logger.info("👋 Сервис интеграции выключен!")


app = FastAPI(
    title=configs.PROJECT_NAME,
    lifespan=lifespan,
    description="Сервис интеграции (BI)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(dashboard_main.router)
app.include_router(sentiment_analytics.router)
app.include_router(regional_analytics.router)
app.include_router(cities_analytics.router)
app.include_router(products_analytics.router)
app.include_router(sources_analytics.router)
app.include_router(demographics_analytics.router)
app.include_router(trends_analytics.router)
app.include_router(matrix_chart_router.router)
app.include_router(statistics_router.router)


@app.post("/monitoring/test-notification", tags=["Мониторинг"])
async def test_notification():
    """Отправить тестовое уведомление в бота"""
    try:
        result = await monitoring_service.send_test_notification()
        return result
    except Exception as e:
        logger.error(f"❌ Ошибка тестового уведомления: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка отправки: {str(e)}")




if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=configs.HOST, port=configs.PORT, reload=True)
