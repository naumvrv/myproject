# data_store.py
import asyncio
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("data_store_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("data_store")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from data_utils import validate_ohlcv, validate_additional_data, prepare_ohlcv_for_cache
    from grok3_analyze import grok3_analyze
    from cache_manager import get_cached_data, save_cached_data
    from config_loader import SYMBOLS
    from telegram_utils import send_async_message
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    raise SystemExit(1)

class DataStore:
    def __init__(self, exchange: ccxt_async.Exchange, symbols: List[str] = list(SYMBOLS.keys()), 
                 interval: str = "15m", lookback: int = 200):
        """Инициализация хранилища данных.

        Args:
            exchange (ccxt_async.Exchange): Объект биржи.
            symbols (List[str]): Список символов для торговли.
            interval (str): Интервал свечей по умолчанию.
            lookback (int): Количество свечей для анализа.
        """
        if not isinstance(exchange, ccxt_async.Exchange):
            raise ValueError(f"exchange должен быть объектом ccxt_async.Exchange, получен {type(exchange)}")
        if not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols):
            raise ValueError(f"symbols должен быть списком строк, получен {symbols}")
        if not isinstance(interval, str) or not interval:
            raise ValueError(f"interval должен быть непустой строкой, получен {interval}")
        if not isinstance(lookback, int) or lookback <= 0:
            raise ValueError(f"lookback должен быть положительным числом, получен {lookback}")

        self.exchange = exchange
        self.symbols = symbols
        self.default_interval = interval
        self.lookback = lookback
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {symbol: {} for symbol in symbols}
        self.last_timestamp: Dict[str, Dict[str, Optional[int]]] = {symbol: {} for symbol in symbols}
        self.additional_data: Dict[str, Dict[str, Dict[str, float]]] = {symbol: {} for symbol in symbols}
        self.volatility_threshold = 0.02
        self.current_interval: Dict[str, str] = {symbol: interval for symbol in symbols}
        self.max_data_size = lookback * 2
        self.cleanup_interval = 3600  # Очистка каждые 1 час
        asyncio.create_task(self.auto_cleanup())

    async def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 3000, since: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Загрузка OHLCV данных с биржи.

        Args:
            symbol (str): Символ (например, "BTC/USDT:USDT").
            interval (str): Интервал свечей.
            limit (int): Максимальное количество свечей.
            since (Optional[int]): Начальная временная метка в миллисекундах.

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

        key = f"ohlcv:{symbol}:{interval}"
        for attempt in range(3):
            try:
                cached_data = await get_cached_data(key)
                if cached_data and len(cached_data) >= self.lookback + 26:
                    df = pd.DataFrame(cached_data)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    is_valid, validated_df = validate_ohlcv(df)
                    if is_valid:
                        logger.info(f"Использованы кэшированные данные для {key}: {len(df)} свечей")
                        return validated_df.tail(limit)

                logger.info(f"Запрос OHLCV для {symbol}, interval={interval}, limit={limit}" + (f", since={since}" if since else ""))
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit, since=since)
                if not isinstance(ohlcv, list) or not ohlcv:
                    raise ValueError(f"Некорректный ответ от API: {ohlcv}")

                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                is_valid, validated_df = validate_ohlcv(df)
                if not is_valid:
                    raise ValueError("Некорректные OHLCV данные")

                logger.info(f"Получено {len(validated_df)} свечей для {symbol} ({interval})")
                await save_cached_data(key, prepare_ohlcv_for_cache(validated_df))
                return validated_df
            except Exception as e:
                logger.error(f"Ошибка получения OHLCV для {symbol} ({interval}) (попытка {attempt+1}/3): {e}")
                if attempt == 2:
                    await send_async_message(f"⚠️ Не удалось получить OHLCV для {symbol} ({interval})")
                    return None
                await asyncio.sleep(5)

    async def fetch_additional_data(self, symbol: str) -> Dict[str, float]:
        """Загрузка дополнительных данных для символа.

        Args:
            symbol (str): Символ (например, "BTC/USDT:USDT").

        Returns:
            Dict[str, float]: Словарь с дополнительными данными.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return {"funding_rate": 0, "open_interest": 0, "spread": 0, "liquidity": 0, "volume": 0, "sentiment": 0}
        try:
            funding_rate = await self.exchange.fetch_funding_rate(symbol)
            order_book = await self.exchange.fetch_order_book(symbol, limit=10)
            bid = order_book["bids"][0][0] if order_book["bids"] else 0
            ask = order_book["asks"][0][0] if order_book["asks"] else 0
            spread = ask - bid if bid and ask else 0
            liquidity = sum([bid[1] for bid in order_book["bids"]]) + sum([ask[1] for ask in order_book["asks"]])
            ticker = await self.exchange.fetch_ticker(symbol)
            volume = float(ticker.get("baseVolume", 0)) if isinstance(ticker, dict) else 0
            
            df = self.data[symbol].get(self.current_interval[symbol], pd.DataFrame())
            sentiment = 0.0
            if not df.empty and len(df) > 1:
                latest_price = df["close"].iloc[-1]
                grok_analysis = await grok3_analyze(latest_price, {
                    "volatility": df["close"].pct_change().std()
                })
                sentiment = float(grok_analysis.get("avg_sentiment", 0.0))

            return {
                "funding_rate": float(funding_rate.get("fundingRate", 0)),
                "open_interest": float(funding_rate.get("openInterest", 0)),
                "spread": spread,
                "liquidity": liquidity,
                "volume": volume,
                "sentiment": sentiment
            }
        except Exception as e:
            logger.error(f"Ошибка получения дополнительных данных для {symbol}: {e}")
            return {"funding_rate": 0, "open_interest": 0, "spread": 0, "liquidity": 0, "volume": 0, "sentiment": 0}

    async def adjust_interval(self, symbol: str) -> str:
        """Динамическая настройка интервала свечей.

        Args:
            symbol (str): Символ.

        Returns:
            str: Текущий интервал для символа.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return self.current_interval.get(symbol, self.default_interval)
        try:
            df, additional = self.get_data_for_feature_engineering(symbol, self.current_interval[symbol])
            if df.empty or len(df) < 2:
                logger.warning(f"Недостаточно данных для {symbol} для настройки интервала")
            else:
                volatility = df["close"].pct_change().std()
                funding_rate = abs(additional.get("funding_rate", 0))
                volume = df["volume"].mean()
                liquidity = additional.get("liquidity", 0)
                self.volatility_threshold = max(0.01, min(0.05, volatility * 0.5))

                intervals = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
                vol_score = min(1.0, volatility / self.volatility_threshold)
                vol_interval = int(np.interp(vol_score, [0, 1], [60, 5]))
                liq_score = min(1.0, liquidity / 1000)
                liq_interval = int(np.interp(liq_score, [0, 1], [15, 60]))
                final_interval = min(vol_interval, liq_interval)
                closest_interval = min(intervals, key=lambda x: abs(intervals[x] - final_interval))
                self.current_interval[symbol] = closest_interval
                logger.info(f"Интервал для {symbol} установлен: {self.current_interval[symbol]} (vol={volatility:.4f}, liq={liquidity})")
        
            await self.fetch_and_update_data(symbol, self.current_interval[symbol])
            return self.current_interval[symbol]
        except Exception as e:
            logger.error(f"Ошибка настройки интервала для {symbol}: {e}")
            return self.current_interval.get(symbol, self.default_interval)

    async def initial_data_fetch(self, symbol: str, interval: Optional[str] = None) -> None:
        """Инициализация данных для символа.

        Args:
            symbol (str): Символ.
            interval (Optional[str]): Интервал свечей.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return
        interval = interval or self.current_interval[symbol]
        if not isinstance(interval, str) or not interval:
            logger.error(f"interval должен быть непустой строкой, получен {interval}")
            return

        limit = 500
        required_candles = self.lookback + 26
        key = f"ohlcv:{symbol}:{interval}"
        max_attempts = 20
        
        try:
            cached_data = await get_cached_data(key)
            if cached_data:
                df = pd.DataFrame(cached_data)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                is_valid, validated_df = validate_ohlcv(df)
                if is_valid and len(validated_df) >= required_candles:
                    self.data[symbol][interval] = validated_df
                    self.last_timestamp[symbol][interval] = int(validated_df["timestamp"].max().timestamp() * 1000)
                    logger.info(f"Использованы кэшированные данные для {key}: {len(validated_df)} свечей")
                    return

            ohlcv_list = []
            for attempt in range(max_attempts):
                new_data = await self.fetch_ohlcv(symbol, interval, limit=limit)
                if new_data is None or new_data.empty:
                    logger.warning(f"Прерывание загрузки для {symbol} ({interval}): данные недоступны после попытки {attempt+1}")
                    break
                ohlcv_list.extend(new_data.to_dict("records"))
                if len(new_data) < limit:
                    logger.info(f"Получено меньше данных, чем limit ({len(new_data)}/{limit}), конец истории")
                    break
                since = int(new_data["timestamp"].max().timestamp() * 1000) + 1
                await asyncio.sleep(1)

            if ohlcv_list:
                df = pd.DataFrame(ohlcv_list)
                is_valid, validated_df = validate_ohlcv(df)
                if is_valid:
                    self.data[symbol][interval] = validated_df.tail(self.max_data_size)
                    self.last_timestamp[symbol][interval] = int(validated_df["timestamp"].max().timestamp() * 1000)
                    await save_cached_data(key, prepare_ohlcv_for_cache(self.data[symbol][interval]))
                    logger.info(f"Изначально загружено {len(self.data[symbol][interval])} свечей для {symbol} ({interval})")
                else:
                    logger.error(f"Некорректные данные при инициализации для {symbol} ({interval})")
                    await send_async_message(f"⚠️ Некорректные данные при инициализации для {symbol} ({interval})")

            self.additional_data[symbol][interval] = await self.fetch_additional_data(symbol)
        except Exception as e:
            logger.error(f"Ошибка инициализации данных для {symbol} ({interval}): {e}")

    async def fetch_and_update_data(self, symbol: str, interval: Optional[str] = None, 
                                   retries: int = 3, delay: int = 5, force_update: bool = False) -> None:
        """Обновление данных для символа.

        Args:
            symbol (str): Символ.
            interval (Optional[str]): Интервал свечей.
            retries (int): Количество попыток.
            delay (int): Задержка между попытками в секундах.
            force_update (bool): Принудительное обновление.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return
        if interval is not None and (not isinstance(interval, str) or not interval):
            logger.error(f"interval должен быть непустой строкой или None, получен {interval}")
            return
        if not isinstance(retries, int) or retries < 0:
            logger.error(f"retries должен быть неотрицательным числом, получен {retries}")
            return
        if not isinstance(delay, int) or delay < 0:
            logger.error(f"delay должен быть неотрицательным числом, получен {delay}")
            return
        if not isinstance(force_update, bool):
            logger.error(f"force_update должен быть булевым значением, получен {force_update}")
            return

        interval = interval or self.current_interval[symbol]
        if interval not in self.data[symbol]:
            self.data[symbol][interval] = pd.DataFrame()
            self.last_timestamp[symbol][interval] = None

        current_time = int(pd.Timestamp.now().timestamp() * 1000)  # Исправлено на миллисекунды
        last_fetch_time = self.last_timestamp[symbol].get(interval)
        if force_update or last_fetch_time is None or (current_time - last_fetch_time > 15 * 60 * 1000):  # Исправлено сравнение
            since = self.last_timestamp[symbol].get(interval) if last_fetch_time is not None else None
            if since is None or force_update:
                since = None

            for attempt in range(retries):
                try:
                    new_data = await self.fetch_ohlcv(symbol, interval, since=since, limit=500)
                    if new_data is not None and not new_data.empty:
                        if not self.data[symbol][interval].empty:
                            new_data = new_data[new_data["timestamp"] > self.data[symbol][interval]["timestamp"].max()]
                        if not new_data.empty:
                            self.data[symbol][interval] = pd.concat([self.data[symbol][interval], new_data]).tail(self.max_data_size)
                            self.last_timestamp[symbol][interval] = int(self.data[symbol][interval]["timestamp"].max().timestamp() * 1000)
                            await save_cached_data(f"ohlcv:{symbol}:{interval}", prepare_ohlcv_for_cache(self.data[symbol][interval]))
                            logger.info(f"Обновлены данные для {symbol} ({interval}): {len(new_data)} новых свечей")
                        else:
                            logger.info(f"Нет новых данных для {symbol} ({interval})")
                        self.additional_data[symbol][interval] = await self.fetch_additional_data(symbol)
                        break
                except Exception as e:
                    logger.error(f"Ошибка обновления данных для {symbol} ({interval}) (попытка {attempt+1}/{retries}): {e}")
                    if attempt == retries - 1:
                        await send_async_message(f"⚠️ Не удалось обновить данные для {symbol} ({interval}) после {retries} попыток")
                    await asyncio.sleep(delay)

    def get_data_for_feature_engineering(self, symbol: str, interval: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Получение данных для инженерии признаков.

        Args:
            symbol (str): Символ.
            interval (Optional[str]): Интервал свечей.

        Returns:
            Tuple[pd.DataFrame, Dict[str, float]]: OHLCV данные и дополнительные данные.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return pd.DataFrame(), {}
        interval = interval or self.current_interval[symbol]
        if interval not in self.data[symbol] or self.data[symbol][interval].empty:
            logger.warning(f"Нет данных для {symbol} ({interval})")
            return pd.DataFrame(), {}
        df = self.data[symbol][interval].tail(self.lookback + 26)
        additional = self.additional_data[symbol].get(interval, {})
        return df, additional

    def _interval_to_minutes(self, interval: str) -> int:
        """Преобразование интервала в минуты.

        Args:
            interval (str): Интервал свечей.

        Returns:
            int: Количество минут.
        """
        if not isinstance(interval, str) or not interval:
            logger.error(f"interval должен быть непустой строкой, получен {interval}")
            return 15
        intervals = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
        return intervals.get(interval, 15)

    async def cleanup_old_intervals(self, symbol: str) -> None:
        """Очистка устаревших интервалов.

        Args:
            symbol (str): Символ.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return
        current = self.current_interval[symbol]
        current_time = int(pd.Timestamp.now().timestamp() * 1000)
        for interval in list(self.data[symbol].keys()):
            if interval != current and (current_time - self.last_timestamp[symbol].get(interval, 0)) > self.cleanup_interval * 1000:
                del self.data[symbol][interval]
                del self.last_timestamp[symbol][interval]
                del self.additional_data[symbol][interval]
                logger.info(f"Удалены устаревшие данные для {symbol} ({interval})")

    async def auto_cleanup(self) -> None:
        """Автоматическая очистка устаревших данных."""
        while True:
            for symbol in self.symbols:
                await self.cleanup_old_intervals(symbol)
            await asyncio.sleep(self.cleanup_interval)

    async def close(self) -> None:
        """Закрытие соединения с биржей."""
        try:
            await self.exchange.close()
            logger.info("Соединение с биржей закрыто")
        except Exception as e:
            logger.error(f"Ошибка закрытия соединения с биржей: {e}")

if __name__ == "__main__":
    async def test():
        exchange = ccxt_async.okx({"enableRateLimit": True})
        store = DataStore(exchange, list(SYMBOLS.keys()))
        await store.initial_data_fetch("BTC/USDT:USDT")
        ohlcv, additional = store.get_data_for_feature_engineering("BTC/USDT:USDT")
        logger.info(f"OHLCV: {ohlcv.shape}, Additional: {additional}")
        await store.close()
    asyncio.run(test())