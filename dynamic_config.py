# dynamic_config.py
import asyncio
import redis.asyncio as redis
import json
from typing import Dict, Any, Optional
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("dynamic_config_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dynamic_config")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from telegram_utils import send_async_message
    from config_loader import config_loader, SYMBOLS, STRATEGIES, LEVERAGE_RANGE, LOOKBACK_RANGE
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    async def send_async_message(msg: str) -> None:
        logger.warning(f"Telegram уведомления отключены: {msg}")
    raise SystemExit(1)

class DynamicConfig:
    def __init__(self):
        """Инициализация динамической конфигурации."""
        try:
            self.config = config_loader.get_config()
        except Exception as e:
            logger.error(f"Не удалось загрузить базовую конфигурацию: {e}")
            raise
        self.redis_client: Optional[redis.Redis] = None
        self._initialize_dynamic_params()
        asyncio.create_task(self._connect_redis())  # Запускаем подключение в фоновом режиме

    def _initialize_dynamic_params(self) -> None:
        """Инициализация динамических параметров начальными значениями."""
        self.config["LEVERAGE"] = LEVERAGE_RANGE[0]
        self.config["LOOKBACK"] = LOOKBACK_RANGE[0]

    async def _connect_redis(self, max_retries: int = 5, delay: int = 5) -> None:
        """
        Подключение к Redis для хранения обновлений.

        Args:
            max_retries (int): Максимальное количество попыток подключения (по умолчанию 5).
            delay (int): Задержка между попытками в секундах (по умолчанию 5).
        """
        if not isinstance(max_retries, int) or max_retries < 0:
            logger.error(f"max_retries должен быть неотрицательным числом, получен {max_retries}")
            max_retries = 5
        if not isinstance(delay, int) or delay < 0:
            logger.error(f"delay должен быть неотрицательным числом, получен {delay}")
            delay = 5

        attempt = 0
        while attempt < max_retries:
            try:
                self.redis_client = await redis.Redis(
                    host=self.config["REDIS_HOST"],
                    port=self.config["REDIS_PORT"],
                    password=self.config["REDIS_PASSWORD"] or None,
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Успешное подключение к Redis")
                return
            except Exception as e:
                attempt += 1
                error_msg = f"Не удалось подключиться к Redis (попытка {attempt}/{max_retries}): {e}"
                logger.warning(error_msg)
                if attempt == max_retries:
                    await send_async_message(f"⚠️ {error_msg}")
                    logger.error("Не удалось подключиться к Redis после всех попыток")
                await asyncio.sleep(delay)

    async def update_param(self, param_name: str, value: Any) -> None:
        """
        Обновление одного параметра конфигурации.

        Args:
            param_name (str): Имя параметра для обновления.
            value (Any): Новое значение параметра.
        """
        if not isinstance(param_name, str) or not param_name:
            logger.error(f"param_name должен быть непустой строкой, получен {param_name}")
            return
        if param_name in self.config:
            # Валидация типов для известных параметров
            if param_name in ["LEVERAGE", "LOOKBACK"] and not isinstance(value, int):
                logger.error(f"Значение для {param_name} должно быть int, получен {type(value)}")
                return
            if param_name == "AMOUNT_DEFAULT" and not isinstance(value, (int, float)):
                logger.error(f"Значение для {param_name} должно быть числом, получен {type(value)}")
                return
            self.config[param_name] = value
            logger.info(f"Параметр {param_name} обновлён: {value}")
            self._validate_dynamic_config()
            await send_async_message(f"✅ Параметр {param_name} обновлён: {value}")
            if self.redis_client and await self._check_redis_connection():
                try:
                    await self.redis_client.set(f"config:{param_name}", json.dumps(value))
                except Exception as e:
                    logger.error(f"Ошибка записи в Redis для {param_name}: {e}")
        else:
            error_msg = f"Попытка обновить несуществующий параметр: {param_name}"
            logger.error(error_msg)
            await send_async_message(f"⚠️ {error_msg}")

    async def update_dynamic_params(self, volatility: float, balance: float) -> None:
        """
        Динамическое обновление параметров на основе рыночных условий.

        Args:
            volatility (float): Текущая волатильность рынка.
            balance (float): Текущий баланс в USDT.
        """
        if not isinstance(volatility, (int, float)) or volatility < 0:
            logger.error(f"volatility должен быть неотрицательным числом, получен {volatility}")
            volatility = 0.0
        if not isinstance(balance, (int, float)) or balance < 0:
            logger.error(f"balance должен быть неотрицательным числом, получен {balance}")
            balance = 0.0

        try:
            leverage = min(LEVERAGE_RANGE[1], max(LEVERAGE_RANGE[0], int(20 / (1 + volatility * 10))))
            amount_default = min(0.1, max(0.001, balance * 0.01))
            lookback = min(LOOKBACK_RANGE[1], max(LOOKBACK_RANGE[0], int(100 * (1 + volatility * 5))))

            updates = {
                "LEVERAGE": leverage,
                "AMOUNT_DEFAULT": amount_default,
                "LOOKBACK": lookback
            }
            for param, value in updates.items():
                self.config[param] = value

            self._validate_dynamic_config()
            if self.redis_client and await self._check_redis_connection():
                for param, value in updates.items():
                    await self.redis_client.set(f"config:{param}", json.dumps(value))
            logger.info(f"Динамически обновлены параметры: {updates}")
        except Exception as e:
            logger.error(f"Ошибка динамического обновления параметров: {e}")
            await send_async_message(f"⚠️ Ошибка динамического обновления: {e}")

    def _validate_dynamic_config(self) -> None:
        """Валидация динамических параметров."""
        if not (LEVERAGE_RANGE[0] <= self.config["LEVERAGE"] <= LEVERAGE_RANGE[1]):
            logger.warning(f"LEVERAGE {self.config['LEVERAGE']} вне диапазона {LEVERAGE_RANGE}. Установлено: {LEVERAGE_RANGE[0]}")
            self.config["LEVERAGE"] = LEVERAGE_RANGE[0]
        if not (LOOKBACK_RANGE[0] <= self.config["LOOKBACK"] <= LOOKBACK_RANGE[1]):
            logger.warning(f"LOOKBACK {self.config['LOOKBACK']} вне диапазона {LOOKBACK_RANGE}. Установлено: {LOOKBACK_RANGE[0]}")
            self.config["LOOKBACK"] = LOOKBACK_RANGE[0]

    async def _check_redis_connection(self) -> bool:
        """
        Проверка подключения к Redis.

        Returns:
            bool: True если соединение активно, False в противном случае.
        """
        try:
            if not self.redis_client:
                await self._connect_redis()
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.warning(f"Ошибка проверки Redis: {e}")
            return False

    def get_config(self) -> Dict[str, Any]:
        """
        Получение текущей конфигурации.

        Returns:
            Dict[str, Any]: Словарь с динамическими настройками.
        """
        return self.config

    async def close(self) -> None:
        """Закрытие соединения с Redis."""
        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("Redis соединение закрыто")
            except Exception as e:
                logger.error(f"Ошибка закрытия Redis: {e}")

# Инициализация и экспорт переменных
dynamic_config = DynamicConfig()
config = dynamic_config.get_config()

LEVERAGE = config["LEVERAGE"]
LOOKBACK = config["LOOKBACK"]
AMOUNT_DEFAULT = config["AMOUNT_DEFAULT"]

async def initialize_config() -> None:
    """Инициализация конфигурации с проверкой Redis."""
    try:
        await dynamic_config._connect_redis()
        if not await dynamic_config._check_redis_connection():
            raise ValueError("Не удалось подключиться к Redis после всех попыток")
        logger.info("Конфигурация успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка инициализации конфигурации: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(initialize_config())
    logger.info(f"Динамическая конфигурация: {config}")