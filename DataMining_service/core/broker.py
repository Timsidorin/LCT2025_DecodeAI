from faststream.kafka import KafkaBroker
from typing import Optional, Any
from  core.config import configs
import json

class KafkaBrokerManager:
    _instance: Optional['KafkaBrokerManager'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, bootstrap_server: str = configs.BOOTSTRAP_SERVICE):
        if not hasattr(self, '_initialized'):
            self.bootstrap_servers = bootstrap_server
            self._broker: Optional[KafkaBroker] = None
            self._initialized = True

    async def connect(self):
        self._broker = KafkaBroker(self.bootstrap_servers)
        await self._broker.start()


    async def close(self):
        await self._broker.stop()

    async def publish(self, topic: str, message: Any, key: Optional[str] = None):
        if self._broker is None:
            await self.connect()

        if not isinstance(message, (str, bytes)):
            message = json.dumps(message)

        if key is not None and isinstance(key, str):
            key = key.encode('utf-8')

        await self._broker.publish(
            message=message,
            topic=topic,
            key=key
        )

    @property
    def is_connected(self) -> bool:
        return self._broker is not None
