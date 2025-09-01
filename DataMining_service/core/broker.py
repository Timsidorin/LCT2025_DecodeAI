# app/services/kafka_broker.py
from faststream.kafka import KafkaBroker
from typing import Optional, Any
import json
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class KafkaBrokerManager:
    """Управление подключением к Kafka """

    _instance: Optional['KafkaBrokerManager'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        if not hasattr(self, '_initialized'):
            self.bootstrap_servers = bootstrap_servers
            self._broker: Optional[KafkaBroker] = None
            self._initialized = True

    async def connect(self):
        """Установка соединения с Kafka"""
        try:
            self._broker = KafkaBroker(self.bootstrap_servers)
        except Exception as e:
            raise

    async def publish(self, topic: str, message: Any, key: Optional[str] = None):
        """
        Публикация сообщения в Kafka топик
        """
        if self._broker is None:
            await self.connect()

        try:
            if not isinstance(message, (str, bytes)):
                message = json.dumps(message)

            await self._broker.publish(
                message=message,
                topic=topic,
                key=key
            )
            logger.debug(f"Message published to {topic}")
        except Exception as e:
            logger.error(f"Failed to publish message to {topic}: {e}")
            raise

    async def start(self):
        """Запуск брокера"""
        if self._broker is None:
            await self.connect()

        await self._broker.start()
        logger.info("Kafka broker started")

    async def close(self):
        """Закрытие соединения с Kafka"""
        if self._broker:
            await self._broker.close()
            logger.info("Kafka connection closed")


    @property
    def is_connected(self) -> bool:
        """Проверка подключения к Kafka"""
        return self._broker is not None