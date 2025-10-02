# main.py для бота-оповещателя PulseAI
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Union, Dict, Any

from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from config import configs

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация бота и диспетчера
BOT_TOKEN = configs.TOKEN_INFO_BOT
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
router = Router()

# FastAPI приложение
app = FastAPI(
    title=f"{configs.PROJECT_NAME} - Notification Bot",
    description="Telegram бот для отправки уведомлений о репутационных рисках"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Файл для хранения подписчиков
SUBSCRIBERS_FILE = "subscribers.json"


# ========== УПРАВЛЕНИЕ ПОДПИСЧИКАМИ ==========

class SubscriberStorage:
    def __init__(self, filename: str):
        self.filename = filename
        self.subscribers = self.load_subscribers()

    def load_subscribers(self) -> Dict[str, Dict]:
        """Загрузить подписчиков из файла"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Ошибка загрузки подписчиков: {e}")
            return {}

    def save_subscribers(self):
        """Сохранить подписчиков в файл"""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.subscribers, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения подписчиков: {e}")

    def add_subscriber(self, user_id: int, user_data: Dict):
        """Добавить подписчика"""
        user_key = str(user_id)
        self.subscribers[user_key] = {
            "user_id": user_id,
            "username": user_data.get("username"),
            "first_name": user_data.get("first_name"),
            "last_name": user_data.get("last_name"),
            "subscribed_at": datetime.now().isoformat(),
            "is_active": True,
            "last_activity": datetime.now().isoformat()
        }
        self.save_subscribers()
        logger.info(f"Добавлен подписчик: {user_id} (@{user_data.get('username', 'unknown')})")

    def remove_subscriber(self, user_id: int):
        """Удалить подписчика"""
        user_key = str(user_id)
        if user_key in self.subscribers:
            del self.subscribers[user_key]
            self.save_subscribers()
            logger.info(f"Удален подписчик: {user_id}")

    def deactivate_subscriber(self, user_id: int):
        """Деактивировать подписчика (бот заблокирован)"""
        user_key = str(user_id)
        if user_key in self.subscribers:
            self.subscribers[user_key]["is_active"] = False
            self.save_subscribers()
            logger.info(f"Деактивирован подписчик: {user_id}")

    def get_active_subscribers(self) -> List[int]:
        """Получить список активных подписчиков"""
        return [
            int(user_data["user_id"])
            for user_data in self.subscribers.values()
            if user_data.get("is_active", True)
        ]

    def get_subscribers_count(self) -> Dict[str, int]:
        """Получить статистику подписчиков"""
        total = len(self.subscribers)
        active = len(self.get_active_subscribers())
        return {
            "total": total,
            "active": active,
            "inactive": total - active
        }


# Инициализация хранилища подписчиков
subscriber_storage = SubscriberStorage(SUBSCRIBERS_FILE)


# ========== ОБРАБОТЧИКИ КОМАНД TELEGRAM ==========

@router.message(CommandStart())
async def cmd_start(message: Message):
    """Обработчик команды /start"""
    user = message.from_user

    # Добавляем пользователя в подписчики
    subscriber_storage.add_subscriber(user.id, {
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name
    })

    welcome_text = f"""
🤖 <b>Добро пожаловать в PulseAI Bot!</b>

Привет, {user.first_name}! 👋

Теперь вы будете получать уведомления о:
📊 Резких изменениях в отзывах
🚨 Критических негативных отзывах  
📈 Важных трендах по продуктам

<i>Вы можете отписаться в любое время командой /stop</i>


"""

    await message.answer(welcome_text)


@router.message(Command("stop"))
async def cmd_stop(message: Message):
    """Обработчик команды /stop"""
    user = message.from_user
    subscriber_storage.remove_subscriber(user.id)

    await message.answer(
        f"😔 <b>Вы отписались от уведомлений</b>\n\n"
        f"Если захотите снова получать уведомления о репутационных рисках, "
        f"просто отправьте /start"
    )


# Подключаем роутер к диспетчеру
dp.include_router(router)


# ========== PYDANTIC МОДЕЛИ ==========

class NotificationPayload(BaseModel):
    text: str
    parse_mode: str = "HTML"
    disable_notification: bool = False

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Текст уведомления не может быть пустым')
        if len(v) > 4096:
            raise ValueError('Текст уведомления превышает лимит Telegram (4096 символов)')
        return v.strip()


class ReviewAlertPayload(BaseModel):
    review_text: str
    source: str
    rating: str
    region: str = None
    product: str = None
    review_url: str = None


class TrendAlertPayload(BaseModel):
    trend_type: str  # "positive_spike", "negative_spike", "volume_drop"
    product: str
    region: str
    change_percentage: float
    period: str = "за последний час"


# ========== ФУНКЦИИ ОТПРАВКИ УВЕДОМЛЕНИЙ ==========

async def send_to_all_subscribers(text: str, parse_mode: str = "HTML", disable_notification: bool = False) -> Dict[
    str, Any]:
    """Отправить уведомление всем активным подписчикам"""
    active_subscribers = subscriber_storage.get_active_subscribers()

    if not active_subscribers:
        return {
            "ok": False,
            "message": "Нет активных подписчиков",
            "sent_count": 0,
            "failed_count": 0
        }

    sent_count = 0
    failed_count = 0
    failed_users = []

    for user_id in active_subscribers:
        try:
            await bot.send_message(
                chat_id=user_id,
                text=text,
                parse_mode=parse_mode if parse_mode != "None" else None,
                disable_notification=disable_notification,
                disable_web_page_preview=True
            )
            sent_count += 1

        except TelegramForbiddenError:
            subscriber_storage.deactivate_subscriber(user_id)
            failed_count += 1
            failed_users.append({"user_id": user_id, "error": "bot_blocked"})

        except TelegramBadRequest as e:
            failed_count += 1
            failed_users.append({"user_id": user_id, "error": str(e)})

        except Exception as e:
            failed_count += 1
            failed_users.append({"user_id": user_id, "error": f"unexpected: {str(e)}"})

        await asyncio.sleep(0.05)

    logger.info(f"Уведомление отправлено: {sent_count} успешно, {failed_count} неудачно")

    return {
        "ok": True,
        "message": f"Уведомление отправлено {sent_count} подписчикам",
        "sent_count": sent_count,
        "failed_count": failed_count,
        "total_subscribers": len(active_subscribers),
        "failed_users": failed_users[:10]
    }


# ========== FASTAPI ЭНДПОИНТЫ ==========

@app.post("/send-notification")
async def send_notification_endpoint(payload: NotificationPayload, background_tasks: BackgroundTasks):
    """Отправить уведомление всем подписчикам"""
    try:
        result = await send_to_all_subscribers(
            text=payload.text,
            parse_mode=payload.parse_mode,
            disable_notification=payload.disable_notification
        )
        return result

    except Exception as e:
        logger.error(f"Ошибка отправки уведомления: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка отправки уведомления: {str(e)}")


# ========== ЗАПУСК ПРИЛОЖЕНИЯ ==========

async def start_polling():
    """Запуск polling для Telegram бота"""
    try:
        logger.info("Запуск Telegram бота...")
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при запуске polling: {e}")


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    try:
        bot_info = await bot.get_me()
        logger.info(f"Бот запущен: @{bot_info.username} (ID: {bot_info.id})")

        stats = subscriber_storage.get_subscribers_count()
        logger.info(f"Загружено подписчиков: {stats['active']} активных из {stats['total']}")
        asyncio.create_task(start_polling())

    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении"""
    try:
        await bot.session.close()
        logger.info("Сессия бота закрыта")
    except Exception as e:
        logger.error(f"Ошибка при закрытии сессии: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=configs.HOST, port=configs.PORT)
