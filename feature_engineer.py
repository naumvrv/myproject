# feature_engineer.py
import pandas as pd
import numpy as np
import talib
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import MinMaxScaler
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("feature_engineer_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("feature_engineer")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

class FeatureEngineer:
    def __init__(self, cache_dir: str = "features_cache"):
        """
        Инициализация инженера признаков.

        Args:
            cache_dir (str): Директория для кэширования признаков.
        """
        if not isinstance(cache_dir, str) or not cache_dir:
            logger.error(f"cache_dir должен быть непустой строкой, получен {cache_dir}")
            cache_dir = "features_cache"
        self.cache_dir = cache_dir
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Не удалось создать директорию {cache_dir}: {e}")
            raise

        self.all_indicators = {
            "rsi": {"func": lambda df: talib.RSI(df["close"], timeperiod=14), "timeperiod": 14},
            "macd": {"func": lambda df: talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)[0], "timeperiod": 26},
            "macd_signal": {"func": lambda df: talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)[1], "timeperiod": 26},
            "atr": {"func": lambda df: talib.ATR(df["high"], df["low"], df["close"], timeperiod=14), "timeperiod": 14},
            "adx": {"func": lambda df: talib.ADX(df["high"], df["low"], df["close"], timeperiod=14), "timeperiod": 14},
            "ema_short": {"func": lambda df: talib.EMA(df["close"], timeperiod=12), "timeperiod": 12},
            "ema_long": {"func": lambda df: talib.EMA(df["close"], timeperiod=26), "timeperiod": 26},
            "volatility": {"func": lambda df: df["close"].pct_change().rolling(window=20).std() * np.sqrt(365), "timeperiod": 20},
            "bb_upper": {"func": lambda df: talib.BBANDS(df["close"], timeperiod=20)[0], "timeperiod": 20},
            "bb_lower": {"func": lambda df: talib.BBANDS(df["close"], timeperiod=20)[2], "timeperiod": 20},
            "ichimoku_tenkan": {"func": lambda df: (df["high"].rolling(window=9).max() + df["low"].rolling(window=9).min()) / 2, "timeperiod": 9},
            "ichimoku_kijun": {"func": lambda df: (df["high"].rolling(window=26).max() + df["low"].rolling(window=26).min()) / 2, "timeperiod": 26},
            "vwap": {"func": lambda df: ((df["high"] + df["low"] + df["close"]) / 3 * df["volume"]).cumsum() / df["volume"].cumsum(), "timeperiod": 1}
        }
        self.selected_indicators: List[str] = list(self.all_indicators.keys())

    def select_indicators(self, strategy: str, volatility: float, rl_model: Optional[Any] = None, 
                         state: Optional[np.ndarray] = None) -> None:
        """
        Динамический выбор индикаторов на основе стратегии, волатильности и RL модели.

        Args:
            strategy (str): Название стратегии.
            volatility (float): Уровень волатильности.
            rl_model (Optional[Any]): Модель RL (опционально).
            state (Optional[np.ndarray]): Состояние для RL модели (опционально).
        """
        if not isinstance(strategy, str) or not strategy:
            logger.error(f"strategy должен быть непустой строкой, получен {strategy}")
            strategy = "trend"
        if not isinstance(volatility, (int, float)) or volatility < 0:
            logger.error(f"volatility должен быть неотрицательным числом, получен {volatility}")
            volatility = 0.0
        if state is not None and not isinstance(state, np.ndarray):
            logger.error(f"state должен быть np.ndarray или None, получен {type(state)}")
            state = None

        try:
            if rl_model and state is not None:
                action = rl_model.get_action(state)
                if not isinstance(action, int):
                    logger.error(f"RL модель вернула некорректное действие: {action}")
                    raise ValueError("Некорректное действие от RL модели")
                self.selected_indicators = [list(self.all_indicators.keys())[i] 
                                          for i in range(len(self.all_indicators)) if action & (1 << i)]
                if not self.selected_indicators:
                    logger.warning("RL модель не выбрала индикаторы, используются все по умолчанию")
                    self.selected_indicators = list(self.all_indicators.keys())
            else:
                if strategy == "trend":
                    self.selected_indicators = ["ema_short", "ema_long", "macd", "macd_signal", "adx"]
                elif strategy == "scalping":
                    self.selected_indicators = ["rsi", "atr", "bb_upper", "bb_lower"]
                elif strategy == "mean_reversion":
                    self.selected_indicators = ["rsi", "volatility", "ichimoku_tenkan", "ichimoku_kijun"]
                else:
                    self.selected_indicators = ["rsi", "macd", "atr", "volatility"]
                if volatility > 0.02:
                    self.selected_indicators.extend(["atr", "volatility"])
                self.selected_indicators = list(set(self.selected_indicators))  # Удаление дубликатов
            logger.info(f"Выбраны индикаторы: {self.selected_indicators}")
        except Exception as e:
            logger.error(f"Ошибка выбора индикаторов: {e}")
            self.selected_indicators = list(self.all_indicators.keys())  # Fallback

    def calculate_technical_indicators(self, df: pd.DataFrame, volatility: float, strategy: str, 
                                      rl_model: Optional[Any] = None, state: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Расчёт технических индикаторов для DataFrame.

        Args:
            df (pd.DataFrame): DataFrame с OHLCV данными.
            volatility (float): Уровень волатильности.
            strategy (str): Название стратегии.
            rl_model (Optional[Any]): Модель RL (опционально).
            state (Optional[np.ndarray]): Состояние для RL модели (опционально).

        Returns:
            pd.DataFrame: DataFrame с добавленными индикаторами.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error(f"df должен быть pd.DataFrame, получен {type(df)}")
            return pd.DataFrame()
        required_cols = ["close", "high", "low", "volume"]
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Отсутствуют необходимые столбцы в df: {df.columns}")
            return df
        if not isinstance(volatility, (int, float)) or volatility < 0:
            logger.error(f"volatility должен быть неотрицательным числом, получен {volatility}")
            volatility = 0.0
        if not isinstance(strategy, str) or not strategy:
            logger.error(f"strategy должен быть непустой строкой, получен {strategy}")
            strategy = "trend"
        if state is not None and not isinstance(state, np.ndarray):
            logger.error(f"state должен быть np.ndarray или None, получен {type(state)}")
            state = None

        try:
            self.select_indicators(strategy, volatility, rl_model, state)
            df_processed = df.copy()
            for indicator in self.selected_indicators:
                df_processed[indicator] = self.all_indicators[indicator]["func"](df_processed)
                if df_processed[indicator].isnull().all():
                    logger.warning(f"Индикатор {indicator} содержит только NaN")
            df_processed = df_processed.interpolate(method="linear").ffill().bfill()
            return df_processed
        except Exception as e:
            logger.error(f"Ошибка вычисления технических индикаторов: {e}")
            return df

    def prepare_features(self, ohlcv_dict: Dict[str, pd.DataFrame], symbols: List[str], 
                        additional_data_dict: Optional[Dict[str, Dict[str, float]]] = None, 
                        interval_minutes: int = 15, strategy: str = "trend", 
                        rl_model: Optional[Any] = None, state: Optional[np.ndarray] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame, MinMaxScaler]]:
        """
        Подготовка признаков для моделей.

        Args:
            ohlcv_dict (Dict[str, pd.DataFrame]): Словарь с OHLCV данными для символов.
            symbols (List[str]): Список символов.
            additional_data_dict (Optional[Dict[str, Dict[str, float]]]): Словарь с дополнительными данными.
            interval_minutes (int): Интервал свечей в минутах.
            strategy (str): Название стратегии.
            rl_model (Optional[Any]): Модель RL (опционально).
            state (Optional[np.ndarray]): Состояние для RL модели (опционально).

        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame, MinMaxScaler]]: Словарь с признаками для каждого символа.
        """
        if not isinstance(ohlcv_dict, dict):
            logger.error(f"ohlcv_dict должен быть словарем, получен {type(ohlcv_dict)}")
            return {}
        if not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols):
            logger.error(f"symbols должен быть списком строк, получен {symbols}")
            return {}
        if additional_data_dict is not None and not isinstance(additional_data_dict, dict):
            logger.error(f"additional_data_dict должен быть словарем или None, получен {type(additional_data_dict)}")
            return {}
        if not isinstance(interval_minutes, int) or interval_minutes <= 0:
            logger.error(f"interval_minutes должен быть положительным числом, получен {interval_minutes}")
            interval_minutes = 15
        if not isinstance(strategy, str) or not strategy:
            logger.error(f"strategy должен быть непустой строкой, получен {strategy}")
            strategy = "trend"
        if state is not None and not isinstance(state, np.ndarray):
            logger.error(f"state должен быть np.ndarray или None, получен {type(state)}")
            state = None

        features_dict = {}
        min_data_size = 60

        for symbol in symbols:
            try:
                df = ohlcv_dict.get(symbol, pd.DataFrame())
                if not isinstance(df, pd.DataFrame):
                    logger.warning(f"Для {symbol} ожидается pd.DataFrame, получен {type(df)}")
                    continue
                if df.empty or len(df) < min_data_size:
                    logger.warning(f"Недостаточно данных для {symbol}: {len(df)} свечей < {min_data_size}")
                    continue
                
                required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
                if not all(col in df.columns for col in required_columns):
                    logger.error(f"Отсутствуют необходимые столбцы в данных для {symbol}: {df.columns}")
                    continue

                volatility = df["close"].pct_change().std() if len(df) > 1 else 0
                lookback = max(min_data_size, min(500, int(100 * (1 + volatility * 10) / (interval_minutes / 15))))
                if strategy in ["scalping", "countertrend"]:
                    lookback = min(lookback, 200)
                logger.info(f"Динамический lookback для {symbol}: {lookback}")

                df = self.calculate_technical_indicators(df, volatility, strategy, rl_model, state)
                if df.empty:
                    logger.error(f"После расчета индикаторов для {symbol} получен пустой DataFrame")
                    continue

                additional_data = additional_data_dict.get(symbol, {}) if additional_data_dict else {}
                df["funding_rate"] = additional_data.get("funding_rate", 0)
                df["open_interest"] = additional_data.get("open_interest", 0)
                df["sentiment"] = additional_data.get("sentiment", 0)

                if len(df) < lookback:
                    logger.error(f"Недостаточно данных после обработки для {symbol}: {len(df)} < {lookback}")
                    continue

                feature_columns = [col for col in ["close"] + self.selected_indicators + ["funding_rate", "open_interest", "sentiment"] if col in df.columns]
                X = np.array([df[feature_columns].iloc[i:i + lookback].values 
                             for i in range(len(df) - lookback + 1)])
                y = df["close"].iloc[lookback - 1:].values  # Исправлено выравнивание
                if X.size == 0 or X.shape[1] != lookback or X.shape[2] != len(feature_columns) or len(X) != len(y):
                    logger.error(f"Ошибка создания признаков для {symbol}: X.shape={X.shape}, y.shape={y.shape}")
                    continue

                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X_reshaped = X.reshape(-1, X.shape[2])
                X_normalized = scaler_X.fit_transform(X_reshaped).reshape(X.shape)
                y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
                X = np.array(X_normalized, dtype=np.float32)
                y = np.array(y_normalized, dtype=np.float32)

                cache_path_df = os.path.join(self.cache_dir, f"{symbol.replace('/', '_')}_features.csv")
                cache_path_X = os.path.join(self.cache_dir, f"{symbol.replace('/', '_')}_X.npy")
                cache_path_y = os.path.join(self.cache_dir, f"{symbol.replace('/', '_')}_y.npy")
                try:
                    df.to_csv(cache_path_df, index=False)
                    np.save(cache_path_X, X)
                    np.save(cache_path_y, y)
                    logger.info(f"Признаки для {symbol}: {X.shape}, сохранены в {cache_path_X} и {cache_path_y}")
                except OSError as e:
                    logger.error(f"Ошибка сохранения признаков для {symbol}: {e}")

                features_dict[symbol] = (X, y, df, scaler_y)
            except Exception as e:
                logger.error(f"Ошибка создания признаков для {symbol}: {e}")

        return features_dict

    def load_cached_features(self, symbol: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Optional[MinMaxScaler]]:
        """
        Загрузка кэшированных признаков.

        Args:
            symbol (str): Символ.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Optional[MinMaxScaler]]: Кортеж с признаками или None при ошибке.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return None, None, None, None
        try:
            cache_path_X = os.path.join(self.cache_dir, f"{symbol.replace('/', '_')}_X.npy")
            cache_path_y = os.path.join(self.cache_dir, f"{symbol.replace('/', '_')}_y.npy")
            cache_path_df = os.path.join(self.cache_dir, f"{symbol.replace('/', '_')}_features.csv")
            if os.path.exists(cache_path_X) and os.path.exists(cache_path_y) and os.path.exists(cache_path_df):
                X = np.load(cache_path_X)
                y = np.load(cache_path_y)
                df = pd.read_csv(cache_path_df)
                scaler_y = MinMaxScaler()
                scaler_y.fit(y.reshape(-1, 1))
                logger.info(f"Загружены кэшированные признаки для {symbol}: {X.shape}")
                return X, y, df, scaler_y
            else:
                logger.warning(f"Кэшированные файлы для {symbol} не найдены")
            return None, None, None, None
        except Exception as e:
            logger.error(f"Ошибка загрузки кэшированных признаков для {symbol}: {e}")
            return None, None, None, None

if __name__ == "__main__":
    engineer = FeatureEngineer()
    test_df = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="15min"),
        "open": np.random.rand(100) * 100,
        "high": np.random.rand(100) * 100,
        "low": np.random.rand(100) * 100,
        "close": np.random.rand(100) * 100,
        "volume": np.random.rand(100) * 1000
    })
    features = engineer.prepare_features({"BTC/USDT:USDT": test_df}, ["BTC/USDT:USDT"])
    logger.info(f"Подготовлены признаки: {features.keys()}")