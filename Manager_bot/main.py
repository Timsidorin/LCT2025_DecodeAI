# main.py для бота-оповещателя
from datetime import datetime, timezone, timedelta
import logging
from typing import Union
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from config import configs
from fastapi.middleware.cors import CORSMiddleware

BOT_TOKEN = configs.TOKEN_INFO_BOT
logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
app = FastAPI(title=configs.PROJECT_NAME)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NotificationPayload(BaseModel):
    chat_identifier: Union[int, str]
    text: str


@app.post("/send-notification")
async def send_notification_endpoint(payload: NotificationPayload):

        return {"ok": True, "message": "Уведомление отправлено."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=configs.HOST, port=configs.PORT)