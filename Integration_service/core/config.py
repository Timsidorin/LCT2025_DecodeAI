from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
import os


class Configs(BaseSettings):
    # ------------ Веб-сервер ------------
    HOST: str = "localhost"
    PORT: int = 8009

    PROJECT_NAME:str = "Модуль интеграции с фронтендом"

    # ------------ БД ------------
    DB_HOST: Optional[str] = Field(default="127.127.126.5", env="DB_HOST")
    DB_PORT: Optional[int] = Field(default=5432, env="DB_PORT")
    DB_USER: Optional[str] = Field(default="postgres", env="DATABASE_USERNAME")
    DB_NAME: Optional[str] = Field(default="DecodeAI", env="DATABASE_NAME")
    DB_PASS: Optional[str] = Field(default="admin", env="DATABASE_PASSWORD")


    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    )


configs = Configs()


def get_db_url():
    return (
        f"postgresql+asyncpg://{configs.DB_USER}:{configs.DB_PASS}@"
        f"{configs.DB_HOST}:{configs.DB_PORT}/{configs.DB_NAME}"
    )
