from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
import os




class Configs(BaseSettings):

    HOST: str = "localhost"
    PORT: int = 8008

    PROJECT_NAME:str = "Модуль уведомлений"
    TOKEN_INFO_BOT: Optional[str] = Field(default="7882621367:AAFfSLtNXaMNjfaqz_OBzpB3NXKe5qcQR2M", env="TOKEN_INFO_BOT")


    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", ".env"),
        extra="ignore"
    )

configs = Configs()