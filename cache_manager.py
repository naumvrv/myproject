# cache_manager.py
import asyncio
import pickle
from datetime import datetime
from typing import Any, Optional
import redis.asyncio as redis
from config_loader import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD
import logging

# Попытка импорта logging_setup, с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("cache_manager_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("cache_manager")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

class LRUCache:
    def __init__(self, max_size: int):
        """Инициализация LRU-кэша.

        Args:
            max_size (int): Максимальный размер кэша.
        """
        self.max_size = max_size
        self.cache = {}
        self.order = []

    def get(self, key: str) -> tuple[Optional[Any], Optional[int]]:
        """Получение данных из кэша.

        Args:
            key (str): Ключ для поиска в кэше.

        Returns:
            tuple[Optional[Any], Optional[int]]: Данные и время истечения, или (None, None).
        """
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            expires = self.cache[key].get("expires")
            if expires and datetime.now().timestamp() > expires:
                del self.cache[key]
                self.order.remove(key)
                return None, None
            return self.cache[key]["data"], expires
        return None, None

    def set(self, key: str, value: Any, ttl: int) -> None:
        """Сохранение данных в кэш.

        Args:
            key (str): Ключ для данных.
            value (Any): Данные для сохранения.
            ttl (int): Время жизни в секундах.
        """
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.max_size:
            oldest_key = self.order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = {"data": value, "expires": int(datetime.now().timestamp()) + ttl}
        self.order.append(key)

class CacheManager:
    def __init__(self, redis_host: str = REDIS_HOST, redis_port: int = REDIS_PORT, 
                 redis_password: Optional[str] = REDIS_PASSWORD, max_size: int = 1000000):
        """Инициализация менеджера кэша с LRU и Redis.

        Args:
            redis_host (str): Хост Redis.
            redis_port (int): Порт Redis.
            redis_password (Optional[str]): Пароль Redis.
            max_size (int): Максимальный размер локального кэша.
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.redis_client: Optional[redis.Redis] = None
        self.redis_available = False
        self.local_cache = LRUCache(max_size=max_size)
        self.lock = asyncio.Lock()
        self._redis_connect_attempts = 0
        self._max_retries = 5
        logger.info(f"Инициализирован LRUCache с max_size={max_size}")
        asyncio.create_task(self._init_redis())

    async def _init_redis(self) -> None:
        """Инициализация подключения к Redis с повторными попытками."""
        while self._redis_connect_attempts < self._max_retries:
            try:
                self.redis_client = redis.Redis(
                    host=self.redis_host, port=self.redis_port, password=self.redis_password,
                    decode_responses=False
                )
                await self.redis_client.ping()
                self.redis_available = True
                logger.info("Успешное подключение к Redis")
                break
            except Exception as e:
                self._redis_connect_attempts += 1
                logger.error(f"Ошибка подключения к Redis (попытка {self._redis_connect_attempts}/{self._max_retries}): {e}")
                if self._redis_connect_attempts == self._max_retries:
                    logger.error("Не удалось подключиться к Redis после всех попыток")
                    self.redis_available = False
                    self.redis_client = None
                await asyncio.sleep(2 ** self._redis_connect_attempts)  # Экспоненциальная задержка

    async def check_redis_connection(self) -> bool:
        """Проверка состояния подключения к Redis."""
        if not self.redis_client or not self.redis_available:
            await self._init_redis()
        return self.redis_available

    async def get_data(self, key: str) -> Optional[Any]:
        """Получение данных из кэша.

        Args:
            key (str): Ключ для поиска.

        Returns:
            Optional[Any]: Данные из кэша или None.
        """
        if not isinstance(key, str):
            logger.error(f"Ожидается строка для ключа, получен {type(key)}")
            return None
        async with self.lock:
            try:
                data, expires = self.local_cache.get(key)
                if data is not None:
                    logger.debug(f"Данные получены из локального кэша для {key}")
                    return data

                if await self.check_redis_connection():
                    cached = await self.redis_client.get(key)
                    if cached:
                        cache_entry = pickle.loads(cached)
                        if float(cache_entry["expires"]) > datetime.now().timestamp():
                            self.local_cache.set(key, cache_entry["data"], 
                                               int(float(cache_entry["expires"]) - datetime.now().timestamp()))
                            logger.debug(f"Данные получены из Redis для {key}")
                            return cache_entry["data"]
                        else:
                            await self.redis_client.delete(key)
                            logger.debug(f"Данные для {key} в Redis устарели и удалены")
                return None
            except pickle.UnpicklingError as e:
                logger.error(f"Ошибка десериализации данных для {key}: {e}")
                return None
            except Exception as e:
                logger.error(f"Ошибка получения данных для {key}: {e}")
                return None

    async def save_data(self, key: str, data: Any, ttl: int = 3600) -> None:
        """Сохранение данных в кэш.

        Args:
            key (str): Ключ для данных.
            data (Any): Данные для сохранения.
            ttl (int): Время жизни в секундах.
        """
        if not isinstance(key, str):
            logger.error(f"Ожидается строка для ключа, получен {type(key)}")
            return
        try:
            # Проверка сериализуемости данных
            pickle.dumps(data)  # Предварительная проверка
            cache_entry = {
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "expires": int(datetime.now().timestamp()) + ttl
            }
            serialized = pickle.dumps(cache_entry)
            
            if self.redis_available and await self.check_redis_connection():
                await self.redis_client.setex(key, ttl, serialized)
                logger.debug(f"Данные сохранены в Redis для {key}, TTL={ttl}")
            
            self.local_cache.set(key, cache_entry["data"], ttl)
            logger.debug(f"Данные сохранены в локальный кэш для {key}")
        except pickle.PicklingError as e:
            logger.error(f"Данные для {key} не могут быть сериализованы: {e}")
        except Exception as e:
            logger.error(f"Ошибка сохранения данных для {key}: {e}")

    async def monitor_redis(self) -> None:
        """Мониторинг состояния Redis."""
        max_failures = 10
        failure_count = 0
        while failure_count < max_failures:
            try:
                if await self.check_redis_connection():
                    info = await self.redis_client.info()
                    keys = await self.redis_client.dbsize()
                    memory = info.get("used_memory_human", "N/A")
                    logger.info(f"Состояние Redis: {keys} ключей, память: {memory}")
                    failure_count = 0  # Сброс счетчика при успехе
                else:
                    logger.warning("Redis недоступен")
                    failure_count += 1
            except Exception as e:
                logger.error(f"Ошибка мониторинга Redis: {e}")
                failure_count += 1
            await asyncio.sleep(300)
        logger.error(f"Превышено максимальное количество неудачных попыток мониторинга Redis ({max_failures})")

    async def close(self) -> None:
        """Закрытие соединения с Redis."""
        if self.redis_client and self.redis_available:
            try:
                await self.redis_client.aclose()
                logger.info("Соединение с Redis закрыто")
            except Exception as e:
                logger.error(f"Ошибка закрытия Redis: {e}")

cache_manager = CacheManager()

async def init_cache() -> None:
    """Инициализация кэша и запуск мониторинга."""
    logger.info("Кэш инициализирован")
    asyncio.create_task(cache_manager.monitor_redis())

async def get_cached_data(key: str) -> Optional[Any]:
    """Глобальная функция для получения данных из кэша."""
    return await cache_manager.get_data(key)

async def save_cached_data(key: str, data: Any, ttl: int = 3600) -> None:
    """Глобальная функция для сохранения данных в кэш."""
    await cache_manager.save_data(key, data, ttl)

async def get_cached_sentiment(key: str) -> Optional[float]:
    """Глобальная функция для получения сентимента из кэша."""
    data = await get_cached_data(key)
    try:
        return float(data) if data is not None else None
    except (ValueError, TypeError):
        logger.error(f"Некорректный тип данных сентимента для {key}: {data}")
        return None

async def save_cached_sentiment(key: str, sentiment: float, ttl: int = 3600) -> None:
    """Глобальная функция для сохранения сентимента в кэш."""
    if not isinstance(sentiment, (int, float)):
        logger.error(f"Ожидается числовое значение сентимента, получен {type(sentiment)}")
        return
    await save_cached_data(key, float(sentiment), ttl)

if __name__ == "__main__":
    async def test():
        await init_cache()
        await save_cached_data("test_key", {"value": 42}, ttl=3600)
        data = await get_cached_data("test_key")
        logger.info(f"Полученные данные: {data}")
    asyncio.run(test())