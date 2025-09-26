from faststream.kafka import KafkaBroker
from typing import Optional, Any
from core.config import configs
import json
import asyncio


class KafkaBrokerManager:
    _instance: Optional["KafkaBrokerManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, bootstrap_server: str = configs.BOOTSTRAP_SERVICE):
        if not hasattr(self, "_initialized"):
            self.bootstrap_servers = bootstrap_server
            self._broker: Optional[KafkaBroker] = None
            self._is_started = False
            self._is_closed = False
            self._initialized = True

    async def connect(self):
        """Подключение к Kafka брокеру"""
        if self._broker is None or self._is_closed:
            self._broker = KafkaBroker(self.bootstrap_servers)

        if not self._is_started:
            try:
                await self._broker.start()
                self._is_started = True
                self._is_closed = False
                print(f"Kafka подключен к {self.bootstrap_servers}")
            except Exception as e:
                print(f"Ошибка подключения к Kafka: {e}")
                self._broker = None
                raise

    async def close(self):
        """Корректное закрытие соединения с Kafka"""
        if self._broker and self._is_started and not self._is_closed:
            try:
                await self._broker.close()
                print("Kafka соединение закрыто")
            except Exception as e:
                print(f"Ошибка при закрытии Kafka: {e}")
            finally:
                self._is_started = False
                self._is_closed = True

    async def publish(self, topic: str, message: Any, key: Optional[str] = None):
        """Публикация сообщения в Kafka топик"""
        if not self._is_started:
            await self.connect()

        try:
            # Сериализация сообщения
            if not isinstance(message, (str, bytes)):
                message = json.dumps(message, ensure_ascii=False, default=str)

            # Подготовка ключа
            if key is not None and isinstance(key, str):
                key = key.encode("utf-8")

            # Публикация
            await self._broker.publish(message=message, topic=topic, key=key)
            print(f"Сообщение опубликовано в топик {topic}")

        except Exception as e:
            print(f"Ошибка публикации в Kafka: {e}")
            # Попытка переподключения при ошибке
            self._is_started = False
            raise

    @property
    def is_connected(self) -> bool:
        """Проверка статуса соединения"""
        return self._broker is not None and self._is_started and not self._is_closed

    async def __aenter__(self):
        """Асинхронный контекст-менеджер вход"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекст-менеджер выход"""
        await self.close()

    def __del__(self):
        """Деструктор для принудительной очистки"""
        if hasattr(self, "_broker") and self._broker and self._is_started:
            try:
                # Создаем новый event loop если его нет
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Если loop уже запущен, создаем задачу
                    asyncio.create_task(self.close())
                else:
                    # Если loop не запущен, запускаем синхронно
                    loop.run_until_complete(self.close())
            except Exception:
                pass  # Игнорируем ошибки в деструкторе
