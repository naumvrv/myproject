# data_utils.py
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("data_utils_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("data_utils")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from telegram_utils import send_async_message
except ImportError as e:
    logger.error(f"Не удалось импортировать telegram_utils: {e}")
    async def send_async_message(msg: str) -> None:
        logger.warning(f"Telegram уведомления отключены: {msg}")

class DataUtils:
    @staticmethod
    def validate_ohlcv(df: Union[pd.DataFrame, List[Dict]]) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Проверка структуры и корректности OHLCV данных.

        Args:
            df: Данные OHLCV в формате DataFrame или списка словарей.

        Returns:
            Tuple[bool, Optional[pd.DataFrame]]: (валидность, обработанный DataFrame или None).
        """
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        try:
            if not isinstance(df, (pd.DataFrame, list)):
                logger.error(f"Ожидается pd.DataFrame или list, получен {type(df)}")
                return False, None

            if isinstance(df, list):
                df = pd.DataFrame(df)
            if not isinstance(df, pd.DataFrame):
                logger.error(f"После преобразования df не является DataFrame: {type(df)}")
                return False, None

            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Отсутствуют обязательные столбцы в OHLCV: {list(df.columns)}")
                return False, None

            numeric_cols = ["open", "high", "low", "close", "volume"]
            if df[numeric_cols].isnull().any().any():
                logger.warning("Обнаружены пропуски в OHLCV, данные будут интерполированы")
                df[numeric_cols] = df[numeric_cols].interpolate(method="linear").ffill().bfill()

            if df[numeric_cols].lt(0).any().any():
                logger.warning("Обнаружены отрицательные значения в OHLCV")
                return False, None

            if df["timestamp"].duplicated().any():
                logger.warning("Обнаружены дубликаты timestamp в OHLCV, удаляем")
                df = df.drop_duplicates(subset="timestamp")

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")  # Исправлено
            if df["timestamp"].isnull().any():
                logger.error("Некорректные временные метки в OHLCV")
                return False, None

            return True, df.sort_values("timestamp")
        except Exception as e:
            logger.error(f"Ошибка валидации OHLCV: {e}")
            return False, None

    @staticmethod
    def recover_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Восстановление OHLCV данных (интерполяция, сглаживание).

        Args:
            df: DataFrame с OHLCV данными.

        Returns:
            pd.DataFrame: Обработанный DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Ожидается pd.DataFrame, получен {type(df)}")
            return pd.DataFrame()
        required_cols = ["open", "high", "low", "close", "volume", "timestamp"]
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Отсутствуют необходимые столбцы в df: {df.columns}")
            return df
        try:
            df = df.drop_duplicates(subset="timestamp")
            numeric_cols = ["open", "high", "low", "close", "volume"]
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear").ffill().bfill()

            if "close" in df.columns and df["close"].pct_change().abs().max() > 0.5:
                logger.warning("Обнаружены аномальные скачки цен, применяем сглаживание")
                df["close"] = df["close"].rolling(window=3, min_periods=1).mean()

            return df
        except Exception as e:
            logger.error(f"Ошибка восстановления OHLCV: {e}")
            return df

    @staticmethod
    def validate_additional_data(data: Dict[str, Any]) -> Dict[str, float]:
        """
        Проверка и валидация дополнительных данных (funding rate, open interest и т.д.).

        Args:
            data: Словарь с дополнительными данными.

        Returns:
            Dict[str, float]: Валидированный словарь с численными значениями.
        """
        if not isinstance(data, dict):
            logger.error(f"Ожидается dict, получен {type(data)}")
            return {
                "funding_rate": 0.0,
                "open_interest": 0.0,
                "spread": 0.0,
                "liquidity": 0.0,
                "volume": 0.0,
                "sentiment": 0.0
            }
        validated = {
            "funding_rate": 0.0,
            "open_interest": 0.0,
            "spread": 0.0,
            "liquidity": 0.0,
            "volume": 0.0,
            "sentiment": 0.0
        }
        try:
            for key, default in validated.items():
                value = data.get(key, default)
                if not isinstance(value, (int, float)):
                    logger.warning(f"Некорректное значение {key}: {value}. Установлено: {default}")
                    validated[key] = default
                else:
                    validated[key] = float(value)
            return validated
        except Exception as e:
            logger.error(f"Ошибка валидации дополнительных данных: {e}")
            return validated

    @staticmethod
    def prepare_ohlcv_for_cache(df: Union[pd.DataFrame, List[Dict]]) -> List[Dict]:
        """
        Подготовка OHLCV данных для кэширования.

        Args:
            df: DataFrame или список словарей с OHLCV данными.

        Returns:
            List[Dict]: Список словарей с ISO-форматированными временными метками.
        """
        required_keys = ["timestamp", "open", "high", "low", "close", "volume"]
        try:
            if not isinstance(df, (pd.DataFrame, list)):
                logger.error(f"Ожидается pd.DataFrame или list, получен {type(df)}")
                return []

            if isinstance(df, pd.DataFrame):
                data = df.to_dict("records")
            else:
                data = df

            if not isinstance(data, list):
                logger.error(f"После преобразования data не является списком: {type(data)}")
                return []

            prepared_data = []
            for entry in data:
                if not isinstance(entry, dict) or not all(k in entry for k in required_keys):
                    logger.warning(f"Некорректная структура OHLCV: {entry}")
                    continue
                if isinstance(entry["timestamp"], pd.Timestamp):
                    entry["timestamp"] = entry["timestamp"].isoformat()
                prepared_data.append(entry)
            return prepared_data
        except Exception as e:
            logger.error(f"Ошибка подготовки OHLCV для кэширования: {e}")
            return []

    @staticmethod
    def normalize_data(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
        """
        Нормализация числовых данных в DataFrame.

        Args:
            df: DataFrame с данными.
            columns: Список столбцов для нормализации.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]: Нормализованный DataFrame и параметры нормализации.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Ожидается pd.DataFrame, получен {type(df)}")
            return df, {}
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            logger.error(f"columns должен быть списком строк, получен {columns}")
            return df, {}
        try:
            normalized_df = df.copy()
            scalers = {}
            for col in columns:
                if col in df.columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if pd.isna(min_val) or pd.isna(max_val):
                        logger.warning(f"Столбец {col} содержит NaN, нормализация невозможна")
                        scalers[col] = (0.0, 0.0)
                        continue
                    if max_val > min_val:
                        normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
                        scalers[col] = (min_val, max_val)
                    else:
                        normalized_df[col] = df[col]
                        scalers[col] = (min_val, min_val)
                else:
                    logger.warning(f"Столбец {col} отсутствует в DataFrame")
            return normalized_df, scalers
        except Exception as e:
            logger.error(f"Ошибка нормализации данных: {e}")
            return df, {}

    @staticmethod
    async def validate_and_notify(df: Union[pd.DataFrame, List[Dict]], symbol: str, 
                                 min_size: int = 60) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Проверка OHLCV данных с уведомлением через Telegram при ошибке.

        Args:
            df: Данные OHLCV.
            symbol: Символ (например, "BTC/USDT:USDT").
            min_size: Минимальный размер данных (по умолчанию 60).

        Returns:
            Tuple[bool, Optional[pd.DataFrame]]: (валидность, обработанный DataFrame или None).
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return False, None
        if not isinstance(min_size, int) or min_size <= 0:
            logger.error(f"min_size должен быть положительным числом, получен {min_size}")
            return False, None
        is_valid, processed_df = DataUtils.validate_ohlcv(df)
        if not is_valid:
            await send_async_message(f"⚠️ Некорректные OHLCV данные для {symbol}")
            return False, None
        if len(processed_df) < min_size:
            logger.warning(f"Недостаточно данных для {symbol}: {len(processed_df)} < {min_size}")
            await send_async_message(f"⚠️ Недостаточно данных для {symbol}: {len(processed_df)} < {min_size}")
            return False, None
        return True, processed_df

# Экспорт функций для удобства
validate_ohlcv = DataUtils.validate_ohlcv
recover_ohlcv = DataUtils.recover_ohlcv
validate_additional_data = DataUtils.validate_additional_data
prepare_ohlcv_for_cache = DataUtils.prepare_ohlcv_for_cache
normalize_data = DataUtils.normalize_data
validate_and_notify = DataUtils.validate_and_notify

if __name__ == "__main__":
    # Тестовый пример
    test_data = [
        {"timestamp": "2023-01-01T00:00:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
        {"timestamp": "2023-01-01T00:15:00", "open": 100, "high": 102, "low": 98, "close": 101, "volume": 1100}
    ]
    is_valid, df = validate_ohlcv(test_data)
    logger.info(f"Валидация успешна: {is_valid}, DataFrame: {df.shape if df is not None else 'None'}")