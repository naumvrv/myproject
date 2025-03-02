# fetch_data.py
import ccxt.async_support as ccxt_async
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Union
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("fetch_data_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fetch_data")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from config_loader import API_KEY, API_SECRET, API_PASSPHRASE, SYMBOLS, TESTNET
    from cache_manager import get_cached_data, save_cached_data
    from data_utils import validate_ohlcv, recover_ohlcv
    from telegram_utils import send_async_message
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    raise SystemExit(1)

class DataFetcher:
    def __init__(self, exchange_name: str = "okx"):
        """
        Инициализация загрузчика данных с биржи.

        Args:
            exchange_name (str): Название биржи (по умолчанию "okx").

        Raises:
            ValueError: Если биржа не поддерживается.
        """
        if not isinstance(exchange_name, str) or not exchange_name:
            logger.error(f"exchange_name должен быть непустой строкой, получен {exchange_name}")
            exchange_name = "okx"
        self.exchange_name = exchange_name.lower()
        self.exchanges = {
            "okx": ccxt_async.okx({
                "apiKey": API_KEY,
                "secret": API_SECRET,
                "password": API_PASSPHRASE,
                "enableRateLimit": True,
                "options": {"defaultType": "swap", "testnet": TESTNET},
            }),
            "binance": ccxt_async.binance({
                "apiKey": API_KEY,
                "secret": API_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }),
            "bybit": ccxt_async.bybit({
                "apiKey": API_KEY,
                "secret": API_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            })
        }
        if exchange_name not in self.exchanges:
            raise ValueError(f"Биржа {exchange_name} не поддерживается")
        self.exchange = self.exchanges[exchange_name]
        self.rate_limits = {
            "okx": 10,  # 10 запросов в секунду
            "binance": 20,
            "bybit": 5
        }
        self.rate_limit = self.rate_limits.get(exchange_name, 10)
        self.request_count = 0
        self.last_request_time = asyncio.get_event_loop().time()

    async def _check_rate_limit(self) -> None:
        """Проверка и соблюдение лимита запросов API."""
        try:
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - self.last_request_time
            if elapsed < 1.0 and self.request_count >= self.rate_limit:
                wait_time = 1.0 - elapsed + (self.request_count // self.rate_limit)
                logger.info(f"Достигнут лимит запросов для {self.exchange_name}, ждём {wait_time:.2f} сек")
                await asyncio.sleep(wait_time)
                self.request_count = 0
            self.request_count += 1
            self.last_request_time = current_time
        except Exception as e:
            logger.error(f"Ошибка проверки лимита запросов: {e}")
            await asyncio.sleep(1)  # Задержка на случай сбоя

    async def fetch_ohlcv(self, symbol: str, interval: str = "15m", limit: int = 1000, 
                         since: Optional[int] = None, retries: int = 3, 
                         delay: int = 5) -> Optional[pd.DataFrame]:
        """
        Асинхронная загрузка OHLCV данных.

        Args:
            symbol (str): Символ (например, "BTC/USDT:USDT").
            interval (str): Интервал свечей (по умолчанию "15m").
            limit (int): Количество свечей (по умолчанию 1000).
            since (Optional[int]): Начальная временная метка в миллисекундах.
            retries (int): Количество попыток при ошибке (по умолчанию 3).
            delay (int): Задержка между попытками в секундах (по умолчанию 5).

        Returns:
            Optional[pd.DataFrame]: DataFrame с OHLCV данными или None при ошибке.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return None
        if not isinstance(interval, str) or not interval:
            logger.error(f"interval должен быть непустой строкой, получен {interval}")
            return None
        if not isinstance(limit, int) or limit <= 0:
            logger.error(f"limit должен быть положительным числом, получен {limit}")
            return None
        if since is not None and not isinstance(since, int):
            logger.error(f"since должен быть int или None, получен {since}")
            return None
        if not isinstance(retries, int) or retries < 0:
            logger.error(f"retries должен быть неотрицательным числом, получен {retries}")
            return None
        if not isinstance(delay, int) or delay < 0:
            logger.error(f"delay должен быть неотрицательным числом, получен {delay}")
            return None

        key = f"ohlcv:{self.exchange_name}:{symbol}:{interval}"
        for attempt in range(retries):
            try:
                cached_data = await get_cached_data(key)
                if cached_data:
                    df = pd.DataFrame(cached_data)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    is_valid, validated_df = validate_ohlcv(df)
                    if is_valid and (not since or df["timestamp"].min() <= pd.Timestamp(since, unit="ms")):
                        logger.info(f"Использованы кэшированные OHLCV данные для {key}: {len(df)} свечей")
                        if df["timestamp"].max() < pd.Timestamp.now() - pd.Timedelta(minutes=15):
                            logger.info(f"Кэш устарел для {key}, обновляем данные")
                        else:
                            return validated_df.tail(limit)

                await self._check_rate_limit()
                ohlcv = await self.exchange.fetch_ohlcv(symbol, interval, since=since, limit=limit)
                if not isinstance(ohlcv, list) or not ohlcv:
                    raise ValueError(f"Некорректный ответ от API {self.exchange_name}: {ohlcv}")

                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                is_valid, validated_df = validate_ohlcv(df)
                if not is_valid:
                    raise ValueError("Некорректные OHLCV данные")

                validated_df = recover_ohlcv(validated_df)
                await save_cached_data(key, prepare_ohlcv_for_cache(validated_df), ttl=300)
                logger.info(f"OHLCV для {key} получены и сохранены в кэш")
                return validated_df
            except ccxt.RateLimitExceeded as e:
                logger.error(f"Превышен лимит запросов для {key}: {e}")
                await asyncio.sleep(delay * (attempt + 1) * 2)
            except Exception as e:
                logger.error(f"Ошибка получения OHLCV для {key} (попытка {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    await send_async_message(f"⚠️ Не удалось получить OHLCV для {symbol} ({interval}) на {self.exchange_name} после {retries} попыток")
                    return None
                await asyncio.sleep(delay)

    async def fetch_order_book(self, symbol: str, limit: int = 10, retries: int = 3, 
                              delay: int = 5) -> Dict[str, float]:
        """
        Асинхронная загрузка книги ордеров.

        Args:
            symbol (str): Символ.
            limit (int): Количество уровней книги ордеров (по умолчанию 10).
            retries (int): Количество попыток при ошибке (по умолчанию 3).
            delay (int): Задержка между попытками в секундах (по умолчанию 5).

        Returns:
            Dict[str, float]: Словарь с данными книги ордеров (spread, liquidity).
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return {"spread": 0, "liquidity": 0}
        if not isinstance(limit, int) or limit <= 0:
            logger.error(f"limit должен быть положительным числом, получен {limit}")
            return {"spread": 0, "liquidity": 0}
        if not isinstance(retries, int) or retries < 0:
            logger.error(f"retries должен быть неотрицательным числом, получен {retries}")
            return {"spread": 0, "liquidity": 0}
        if not isinstance(delay, int) or delay < 0:
            logger.error(f"delay должен быть неотрицательным числом, получен {delay}")
            return {"spread": 0, "liquidity": 0}

        key = f"order_book:{self.exchange_name}:{symbol}"
        for attempt in range(retries):
            try:
                cached_data = await get_cached_data(key)
                if cached_data and "spread" in cached_data and "liquidity" in cached_data:
                    logger.info(f"Использованы кэшированные данные книги ордеров для {symbol}")
                    return cached_data

                await self._check_rate_limit()
                order_book = await self.exchange.fetch_order_book(symbol, limit=limit)
                if not isinstance(order_book, dict) or not order_book.get("asks") or not order_book.get("bids"):
                    raise ValueError(f"Некорректная книга ордеров: {order_book}")

                spread = order_book["asks"][0][0] - order_book["bids"][0][0] if order_book["asks"] and order_book["bids"] else 0
                liquidity = sum(bid[1] for bid in order_book["bids"][:5]) + sum(ask[1] for ask in order_book["asks"][:5])
                result = {"spread": spread, "liquidity": liquidity}
                
                await save_cached_data(key, result, ttl=300)
                logger.info(f"Книга ордеров для {symbol} получена")
                return result
            except Exception as e:
                logger.error(f"Ошибка получения книги ордеров для {symbol} (попытка {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    await send_async_message(f"⚠️ Не удалось получить книгу ордеров для {symbol} на {self.exchange_name}")
                    return {"spread": 0, "liquidity": 0}
                await asyncio.sleep(delay)

    async def fetch_funding_rate(self, symbol: str, retries: int = 3, delay: int = 5) -> float:
        """
        Асинхронная загрузка ставки фондирования.

        Args:
            symbol (str): Символ.
            retries (int): Количество попыток при ошибке (по умолчанию 3).
            delay (int): Задержка между попытками в секундах (по умолчанию 5).

        Returns:
            float: Ставка фондирования или 0 при ошибке.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return 0
        if not isinstance(retries, int) or retries < 0:
            logger.error(f"retries должен быть неотрицательным числом, получен {retries}")
            return 0
        if not isinstance(delay, int) or delay < 0:
            logger.error(f"delay должен быть неотрицательным числом, получен {delay}")
            return 0

        key = f"funding_rate:{self.exchange_name}:{symbol}"
        for attempt in range(retries):
            try:
                cached_data = await get_cached_data(key)
                if cached_data is not None and isinstance(cached_data, (int, float)):
                    logger.info(f"Использован кэшированный funding rate для {symbol}")
                    return float(cached_data)

                await self._check_rate_limit()
                funding_rate = await self.exchange.fetch_funding_rate(symbol)
                if not isinstance(funding_rate, dict):
                    raise ValueError(f"Некорректный ответ API для funding rate: {funding_rate}")
                rate = funding_rate.get("fundingRate")
                if rate is None or not isinstance(rate, (int, float)):
                    raise ValueError(f"Некорректный funding rate: {rate}")

                await save_cached_data(key, float(rate), ttl=3600)
                logger.info(f"Funding rate для {symbol}: {rate}")
                return float(rate)
            except Exception as e:
                logger.error(f"Ошибка получения funding rate для {symbol} (попытка {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    await send_async_message(f"⚠️ Не удалось получить funding rate для {symbol} на {self.exchange_name}")
                    return 0
                await asyncio.sleep(delay)

    async def fetch_open_interest(self, symbol: str, retries: int = 3, delay: int = 5) -> float:
        """
        Асинхронная загрузка открытого интереса.

        Args:
            symbol (str): Символ.
            retries (int): Количество попыток при ошибке (по умолчанию 3).
            delay (int): Задержка между попытками в секундах (по умолчанию 5).

        Returns:
            float: Открытый интерес или 0 при ошибке.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return 0
        if not isinstance(retries, int) or retries < 0:
            logger.error(f"retries должен быть неотрицательным числом, получен {retries}")
            return 0
        if not isinstance(delay, int) or delay < 0:
            logger.error(f"delay должен быть неотрицательным числом, получен {delay}")
            return 0

        key = f"open_interest:{self.exchange_name}:{symbol}"
        for attempt in range(retries):
            try:
                cached_data = await get_cached_data(key)
                if cached_data is not None and isinstance(cached_data, (int, float)):
                    logger.info(f"Использован кэшированный open interest для {symbol}")
                    return float(cached_data)

                await self._check_rate_limit()
                open_interest = await self.exchange.fetch_open_interest(symbol)
                if not isinstance(open_interest, dict):
                    raise ValueError(f"Некорректный ответ API для open interest: {open_interest}")
                oi_value = open_interest.get("openInterest") or open_interest.get("info", {}).get("oi")
                if oi_value is None or not isinstance(oi_value, (int, float)):
                    raise ValueError(f"Некорректный open interest: {oi_value}")

                await save_cached_data(key, float(oi_value), ttl=3600)
                logger.info(f"Open interest для {symbol}: {oi_value}")
                return float(oi_value)
            except Exception as e:
                logger.error(f"Ошибка получения open interest для {symbol} (попытка {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    await send_async_message(f"⚠️ Не удалось получить open interest для {symbol} на {self.exchange_name}")
                    return 0
                await asyncio.sleep(delay)

    async def fetch_all_data(self, symbol: str, interval: str = "15m", ohlcv_limit: int = 1000, 
                            order_book_limit: int = 10) -> Dict[str, Union[pd.DataFrame, Dict, float]]:
        """
        Асинхронная загрузка всех данных (OHLCV, книга ордеров, funding rate, open interest).

        Args:
            symbol (str): Символ.
            interval (str): Интервал для OHLCV (по умолчанию "15m").
            ohlcv_limit (int): Лимит для OHLCV (по умолчанию 1000).
            order_book_limit (int): Лимит для книги ордеров (по умолчанию 10).

        Returns:
            Dict[str, Union[pd.DataFrame, Dict, float]]: Словарь с данными или значениями по умолчанию при ошибке.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return {"ohlcv": None, "order_book": {"spread": 0, "liquidity": 0}, "funding_rate": 0, "open_interest": 0}
        if not isinstance(interval, str) or not interval:
            logger.error(f"interval должен быть непустой строкой, получен {interval}")
            return {"ohlcv": None, "order_book": {"spread": 0, "liquidity": 0}, "funding_rate": 0, "open_interest": 0}
        if not isinstance(ohlcv_limit, int) or ohlcv_limit <= 0:
            logger.error(f"ohlcv_limit должен быть положительным числом, получен {ohlcv_limit}")
            return {"ohlcv": None, "order_book": {"spread": 0, "liquidity": 0}, "funding_rate": 0, "open_interest": 0}
        if not isinstance(order_book_limit, int) or order_book_limit <= 0:
            logger.error(f"order_book_limit должен быть положительным числом, получен {order_book_limit}")
            return {"ohlcv": None, "order_book": {"spread": 0, "liquidity": 0}, "funding_rate": 0, "open_interest": 0}

        tasks = [
            self.fetch_ohlcv(symbol, interval=interval, limit=ohlcv_limit),
            self.fetch_order_book(symbol, limit=order_book_limit),
            self.fetch_funding_rate(symbol),
            self.fetch_open_interest(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ohlcv, order_book, funding_rate, open_interest = results
        
        result = {
            "ohlcv": ohlcv if isinstance(ohlcv, pd.DataFrame) else None,
            "order_book": order_book if isinstance(order_book, dict) else {"spread": 0, "liquidity": 0},
            "funding_rate": float(funding_rate) if isinstance(funding_rate, (int, float)) else 0,
            "open_interest": float(open_interest) if isinstance(open_interest, (int, float)) else 0
        }
        for i, task_result in enumerate(results):
            if isinstance(task_result, Exception):
                logger.error(f"Ошибка в задаче {i} fetch_all_data для {symbol}: {task_result}")
        logger.info(f"Получены все данные для {symbol} с {self.exchange_name}")
        return result

    async def close(self) -> None:
        """Закрытие соединения с биржей."""
        try:
            await self.exchange.close()
            logger.info(f"Соединение с биржей {self.exchange_name} закрыто")
        except Exception as e:
            logger.error(f"Ошибка закрытия соединения с биржей {self.exchange_name}: {e}")

fetcher = DataFetcher()  # По умолчанию OKX

fetch_ohlcv = fetcher.fetch_ohlcv
fetch_order_book = fetcher.fetch_order_book
fetch_funding_rate = fetcher.fetch_funding_rate
fetch_open_interest = fetcher.fetch_open_interest
fetch_all_data = fetcher.fetch_all_data