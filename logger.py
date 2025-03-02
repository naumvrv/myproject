# logger.py
import asyncio
import json
import uuid
import numpy as np
from typing import Dict, Optional, Union, Any, List
from datetime import datetime
import aiofiles
import os
import psutil
import redis.asyncio as redis
import logging

# Попытка импорта зависимостей с базовым логгером
try:
    from logging_setup import setup_logging
    logger = setup_logging("logger_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logger")
    logger.warning(f"Не удалось импортировать logging_setup: {e}")

try:
    from config_loader import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD
    from telegram_utils import send_async_message
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    async def send_async_message(msg: str) -> None:
        logger.warning(f"Telegram уведомления отключены: {msg}")
    raise SystemExit(1)

class TradeLogger:
    def __init__(self, log_file: str = "trade_log.jsonl", max_size: int = 10 * 1024 * 1024, 
                 backup_count: int = 5):
        """
        Инициализация логгера сделок.

        Args:
            log_file (str): Путь к файлу логов (по умолчанию "trade_log.jsonl").
            max_size (int): Максимальный размер файла логов в байтах (по умолчанию 10 MB).
            backup_count (int): Количество резервных копий (по умолчанию 5).

        Raises:
            ValueError: Если аргументы имеют некорректные типы или значения.
        """
        if not isinstance(log_file, str) or not log_file:
            raise ValueError(f"log_file должен быть непустой строкой, получено {log_file}")
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError(f"max_size должен быть положительным числом, получено {max_size}")
        if not isinstance(backup_count, int) or backup_count < 0:
            raise ValueError(f"backup_count должен быть неотрицательным числом, получено {backup_count}")

        self.log_file = log_file
        self.max_size = max_size
        self.backup_count = backup_count
        self.buffer: List[str] = []
        self.max_buffer_size = max(512 * 1024, int(psutil.virtual_memory().available * 0.01))  # 1% памяти
        self.buffer_size = 0
        self.lock = asyncio.Lock()
        self.redis_client: Optional[redis.Redis] = None
        asyncio.create_task(self._initialize_redis())

    async def _initialize_redis(self) -> None:
        """Асинхронная инициализация Redis клиента."""
        if not isinstance(REDIS_HOST, str) or not isinstance(REDIS_PORT, int) or (REDIS_PASSWORD is not None and not isinstance(REDIS_PASSWORD, str)):
            logger.error(f"Некорректные параметры Redis: host={REDIS_HOST}, port={REDIS_PORT}, password={REDIS_PASSWORD}")
            return

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD or None, decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Redis клиент успешно инициализирован")
                break
            except Exception as e:
                logger.error(f"Ошибка инициализации Redis клиента (попытка {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.redis_client = None
                    await send_async_message(f"⚠️ Не удалось подключиться к Redis после {max_retries} попыток")
                await asyncio.sleep(5 * (attempt + 1))

    async def log_trade(self, symbol: str, action: str, price: Optional[float], profit: Optional[float], 
                        grok_analysis: Optional[Dict[str, Any]] = None, predicted_price: Optional[float] = None, 
                        actual_change: Optional[float] = None) -> None:
        """
        Асинхронная запись сделки в лог.

        Args:
            symbol (str): Символ (например, "BTC/USDT:USDT").
            action (str): Действие (buy_long, sell_short и т.д.).
            price (Optional[float]): Цена сделки.
            profit (Optional[float]): Прибыль/убыток.
            grok_analysis (Optional[Dict[str, Any]]): Анализ Grok (опционально).
            predicted_price (Optional[float]): Предсказанная цена (опционально).
            actual_change (Optional[float]): Фактическое изменение (опционально).
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получено {symbol}")
            return
        if not isinstance(action, str) or not action:
            logger.error(f"action должен быть непустой строкой, получено {action}")
            return
        if price is not None and (not isinstance(price, (int, float, np.number)) or price <= 0):
            logger.error(f"price должен быть положительным числом или None, получено {price}")
            return
        if profit is not None and not isinstance(profit, (int, float, np.number)):
            logger.error(f"profit должен быть числом или None, получено {profit}")
            return
        if grok_analysis is not None and not isinstance(grok_analysis, dict):
            logger.error(f"grok_analysis должен быть словарем или None, получено {type(grok_analysis)}")
            return
        if predicted_price is not None and (not isinstance(predicted_price, (int, float, np.number)) or predicted_price <= 0):
            logger.error(f"predicted_price должен быть положительным числом или None, получено {predicted_price}")
            return
        if actual_change is not None and not isinstance(actual_change, (int, float, np.number)):
            logger.error(f"actual_change должен быть числом или None, получено {actual_change}")
            return

        try:
            log_entry = {
                "trade_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": action,
                "price": float(price) if price is not None else None,
                "profit": float(profit) if profit is not None else None,
                "grok_analysis": grok_analysis,
                "predicted_price": float(predicted_price) if predicted_price is not None else None,
                "actual_change": float(actual_change) if actual_change is not None else None
            }

            entry_str = json.dumps(log_entry) + "\n"
            entry_size = len(entry_str.encode("utf-8"))

            async with self.lock:
                self.buffer.append(entry_str)
                self.buffer_size += entry_size

                if os.path.exists(self.log_file) and os.path.getsize(self.log_file) + self.buffer_size > self.max_size:
                    await self.rotate_log()

                if self.buffer_size >= self.max_buffer_size:
                    await self.flush_buffer()

                if self.redis_client:
                    try:
                        await self.redis_client.rpush(f"trades:{symbol}", json.dumps(log_entry))
                    except Exception as e:
                        logger.error(f"Ошибка записи в Redis для {symbol}: {e}", exc_info=True)
                        await self._fallback_to_disk(log_entry, symbol)

            logger.info(f"Лог сделки добавлен в буфер: {symbol}, {action}, price={price}, profit={profit}")

        except Exception as e:
            logger.error(f"Ошибка записи лога для {symbol}: {e}", exc_info=True)

    async def _fallback_to_disk(self, log_entry: Dict[str, Union[str, float]], symbol: str) -> None:
        """Запись лога на диск в случае ошибки Redis.

        Args:
            log_entry (Dict[str, Union[str, float]]): Запись лога.
            symbol (str): Символ.
        """
        if not isinstance(log_entry, dict):
            logger.error(f"log_entry должен быть словарем, получено {type(log_entry)}")
            return
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получено {symbol}")
            return

        try:
            disk_path = f"logs/{symbol}_{datetime.now().strftime('%Y%m%d')}.jsonl"
            os.makedirs(os.path.dirname(disk_path), exist_ok=True)
            async with aiofiles.open(disk_path, "a", encoding="utf-8") as f:
                await f.write(json.dumps(log_entry) + "\n")
            logger.info(f"Лог сохранён на диск для {symbol}")
        except Exception as e:
            logger.error(f"Ошибка записи лога на диск для {symbol}: {e}", exc_info=True)

    async def rotate_log(self) -> None:
        """Ротация лог-файла при превышении максимального размера."""
        try:
            async with self.lock:
                if os.path.exists(self.log_file):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_name = f"trade_log_{timestamp}.jsonl"
                    os.rename(self.log_file, new_name)
                    logger.info(f"Лог-файл переименован: {new_name}")

                    log_dir = os.path.dirname(self.log_file) or "."
                    log_files = sorted(
                        [f for f in os.listdir(log_dir) if f.startswith("trade_log_") and f.endswith(".jsonl")],
                        reverse=True
                    )
                    for old_file in log_files[self.backup_count:]:
                        os.remove(os.path.join(log_dir, old_file))
                        logger.info(f"Удалён старый лог-файл: {old_file}")
        except Exception as e:
            logger.error(f"Ошибка ротации лога: {e}", exc_info=True)

    async def flush_buffer(self) -> None:
        """Запись буфера в файл."""
        try:
            async with self.lock:
                if self.buffer:
                    async with aiofiles.open(self.log_file, "a", encoding="utf-8") as f:
                        await f.writelines(self.buffer)
                    logger.info(f"Буфер записан в {self.log_file}, записей: {len(self.buffer)}")
                    self.buffer.clear()
                    self.buffer_size = 0
        except Exception as e:
            logger.error(f"Ошибка записи буфера в файл: {e}", exc_info=True)

    async def close(self) -> None:
        """Закрытие логгера и освобождение ресурсов."""
        try:
            async with self.lock:
                await self.flush_buffer()
                if self.redis_client:
                    try:
                        await self.redis_client.aclose()
                        logger.info("Redis клиент закрыт")
                    except Exception as e:
                        logger.error(f"Ошибка закрытия Redis клиента: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Ошибка закрытия логгера: {e}", exc_info=True)

# Глобальный экземпляр
trade_logger = TradeLogger()

async def log_trade(symbol: str, action: str, price: Optional[float], profit: Optional[float], 
                    grok_analysis: Optional[Dict[str, Any]] = None, predicted_price: Optional[float] = None, 
                    actual_change: Optional[float] = None) -> None:
    """Глобальная функция для записи лога сделки.

    Args:
        symbol (str): Символ.
        action (str): Действие.
        price (Optional[float]): Цена сделки.
        profit (Optional[float]): Прибыль/убыток.
        grok_analysis (Optional[Dict]): Анализ Grok.
        predicted_price (Optional[float]): Предсказанная цена.
        actual_change (Optional[float]): Фактическое изменение.
    """
    await trade_logger.log_trade(symbol, action, price, profit, grok_analysis, predicted_price, actual_change)

async def initialize_logger() -> None:
    """Инициализация логгера с подключением к Redis."""
    try:
        await trade_logger._initialize_redis()
        logger.info("Логгер инициализирован")
    except Exception as e:
        logger.error(f"Ошибка инициализации логгера: {e}", exc_info=True)

if __name__ == "__main__":
    async def test():
        await initialize_logger()
        await log_trade("BTC/USDT:USDT", "buy_long", 30000, 100, {"sentiment": "bullish"}, 30500, 500)
        await trade_logger.close()
    asyncio.run(test())