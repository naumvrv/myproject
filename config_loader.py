# config_loader.py
from dotenv import load_dotenv
import os
import json
from typing import Dict, Tuple, Optional, Any
from cryptography.fernet import Fernet, InvalidToken
import logging

# Попытка импорта logging_setup, с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("config_loader_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("config_loader")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

class ConfigLoader:
    def __init__(self):
        """Инициализация загрузчика конфигурации из файла .env."""
        load_dotenv()
        try:
            self.fernet = self._load_encryption_key()
        except ValueError as e:
            logger.error(f"Не удалось загрузить ключ шифрования: {e}")
            raise
        self.config = self._load_config()
        self._validate_config()

    def _load_encryption_key(self) -> Fernet:
        """Загрузка или генерация ключа шифрования для Fernet.

        Returns:
            Fernet: Объект Fernet для шифрования/дешифрования.
        Raises:
            ValueError: Если ключ некорректен или не может быть использован.
        """
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            logger.warning("ENCRYPTION_KEY не найден в .env. Генерируется новый ключ.")
            encryption_key = Fernet.generate_key().decode()
            logger.info(f"Сгенерирован новый ключ: {encryption_key}. Добавьте его в .env.")
            # Попытка автоматически добавить ключ в .env
            try:
                with open(".env", "a", encoding="utf-8") as env_file:
                    env_file.write(f"\nENCRYPTION_KEY={encryption_key}\n")
                logger.info("Новый ключ добавлен в .env")
            except IOError as e:
                logger.warning(f"Не удалось записать ключ в .env: {e}. Сохраните вручную.")
        try:
            return Fernet(encryption_key.encode())
        except (InvalidToken, ValueError) as e:
            logger.error(f"Некорректный ключ шифрования: {e}")
            raise ValueError(f"Некорректный ключ шифрования: {e}")

    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из .env.

        Returns:
            Dict[str, Any]: Словарь с настройками.
        Raises:
            ValueError: Если отсутствуют обязательные ключи или данные некорректны.
        """
        required_keys = [
            "OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE", "TESTNET",
            "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "NEWSAPI_KEY",
            "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT", "FRED_API_KEY"
        ]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            error_msg = f"Отсутствуют переменные окружения: {', '.join(missing_keys)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            config = {
                "API_KEY": os.getenv("OKX_API_KEY"),
                "API_SECRET": os.getenv("OKX_API_SECRET"),
                "API_PASSPHRASE": os.getenv("OKX_API_PASSPHRASE"),
                "TESTNET": os.getenv("TESTNET", "False").lower() == "true",
                "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN"),
                "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
                "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY"),
                "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID"),
                "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET"),
                "REDDIT_USER_AGENT": os.getenv("REDDIT_USER_AGENT"),
                "FRED_API_KEY": os.getenv("FRED_API_KEY"),
                "REDIS_HOST": os.getenv("REDIS_HOST", "localhost"),
                "REDIS_PORT": int(os.getenv("REDIS_PORT", "6379")),
                "REDIS_PASSWORD": os.getenv("REDIS_PASSWORD", ""),
                "LOG_FILE": os.getenv("LOG_FILE", "trade_log.jsonl"),
                "DASH_PORT": int(os.getenv("DASH_PORT", "8050")),
                "SYMBOLS": self._load_symbols(),
                "STRATEGIES": self._load_strategies(),
                "LEVERAGE_RANGE": tuple(map(int, os.getenv("LEVERAGE_RANGE", "5-20").split("-"))),
                "LOOKBACK_RANGE": tuple(map(int, os.getenv("LOOKBACK_RANGE", "100-500").split("-"))),
                "AMOUNT_DEFAULT": float(os.getenv("AMOUNT_DEFAULT", "0.01")),
                "INTERVAL_DEFAULT": os.getenv("INTERVAL_DEFAULT", "15m"),
                "MAX_DAILY_LOSS": float(os.getenv("MAX_DAILY_LOSS", "100")),
                "MAX_WEEKLY_LOSS": float(os.getenv("MAX_WEEKLY_LOSS", "500")),
                "RL_GAMMA": float(os.getenv("RL_GAMMA", "0.95")),
                "RL_LEARNING_RATE": float(os.getenv("RL_LEARNING_RATE", "0.001")),
                "RL_EPSILON": float(os.getenv("RL_EPSILON", "0.1")),
                "RL_EPSILON_MIN": float(os.getenv("RL_EPSILON_MIN", "0.01")),
                "RL_EPSILON_DECAY": float(os.getenv("RL_EPSILON_DECAY", "0.995"))
            }
            return config
        except (ValueError, TypeError) as e:
            logger.error(f"Ошибка преобразования параметров конфигурации: {e}")
            raise ValueError(f"Ошибка в формате параметров конфигурации: {e}")

    def _load_symbols(self) -> Dict[str, Dict[str, Any]]:
        """Загрузка символов из SYMBOLS_JSON.

        Returns:
            Dict[str, Dict[str, Any]]: Словарь символов с их настройками.
        """
        default_symbols = {"BTC/USDT:USDT": {"interval": "15m", "amount": 0.01}}
        symbols_json = os.getenv("SYMBOLS_JSON")
        try:
            if symbols_json:
                symbols = json.loads(symbols_json)
                for symbol, settings in symbols.items():
                    if not isinstance(settings, dict) or not all(k in settings for k in ["interval", "amount"]):
                        logger.error(f"Некорректная структура SYMBOLS_JSON для {symbol}: {settings}")
                        raise ValueError(f"Некорректная структура SYMBOLS для {symbol}")
                    if settings["interval"] not in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                        logger.error(f"Недопустимый интервал для {symbol}: {settings['interval']}")
                        raise ValueError(f"Недопустимый интервал для {symbol}: {settings['interval']}")
                    if not isinstance(settings["amount"], (int, float)) or settings["amount"] <= 0:
                        logger.error(f"Некорректный amount для {symbol}: {settings['amount']}")
                        raise ValueError(f"Некорректный amount для {symbol}: {settings['amount']}")
                return symbols
            return default_symbols
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Ошибка загрузки SYMBOLS_JSON: {e}. Используется значение по умолчанию")
            return default_symbols

    def _load_strategies(self) -> Dict[str, Dict[str, float]]:
        """Загрузка стратегий из STRATEGIES_JSON.

        Returns:
            Dict[str, Dict[str, float]]: Словарь стратегий с их параметрами.
        """
        default_strategies = {"trend": {"entry_threshold": 0.005, "stop_loss_factor": 1.0, "take_profit_factor": 1.5}}
        strategies_json = os.getenv("STRATEGIES_JSON")
        try:
            if strategies_json:
                strategies = json.loads(strategies_json)
                for strat, params in strategies.items():
                    if not isinstance(params, dict) or not all(k in params for k in ["entry_threshold", "stop_loss_factor", "take_profit_factor"]):
                        logger.error(f"Некорректная структура STRATEGIES_JSON для {strat}: {params}")
                        raise ValueError(f"Некорректная структура STRATEGIES для {strat}")
                    for key, value in params.items():
                        if not isinstance(value, (int, float)):
                            logger.error(f"Некорректное значение {key} в стратегии {strat}: {value}")
                            raise ValueError(f"Некорректное значение {key} в стратегии {strat}")
                return strategies
            return default_strategies
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Ошибка загрузки STRATEGIES_JSON: {e}. Используется значение по умолчанию")
            return default_strategies

    def _validate_config(self) -> None:
        """Валидация загруженной конфигурации."""
        valid_intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
        if self.config["INTERVAL_DEFAULT"] not in valid_intervals:
            logger.warning(f"INTERVAL_DEFAULT {self.config['INTERVAL_DEFAULT']} недопустим. Установлено: 15m")
            self.config["INTERVAL_DEFAULT"] = "15m"
        if not isinstance(self.config["AMOUNT_DEFAULT"], (int, float)) or self.config["AMOUNT_DEFAULT"] <= 0:
            logger.warning(f"AMOUNT_DEFAULT {self.config['AMOUNT_DEFAULT']} некорректен. Установлено: 0.01")
            self.config["AMOUNT_DEFAULT"] = 0.01
        if not isinstance(self.config["MAX_DAILY_LOSS"], (int, float)) or self.config["MAX_DAILY_LOSS"] < 0:
            logger.warning(f"MAX_DAILY_LOSS {self.config['MAX_DAILY_LOSS']} некорректен. Установлено: 100")
            self.config["MAX_DAILY_LOSS"] = 100
        if (not isinstance(self.config["MAX_WEEKLY_LOSS"], (int, float)) or 
            self.config["MAX_WEEKLY_LOSS"] < 0 or 
            self.config["MAX_WEEKLY_LOSS"] < self.config["MAX_DAILY_LOSS"]):
            logger.warning(f"MAX_WEEKLY_LOSS {self.config['MAX_WEEKLY_LOSS']} некорректен. Установлено: 500")
            self.config["MAX_WEEKLY_LOSS"] = 500
        if not 0 <= self.config["RL_GAMMA"] <= 1:
            logger.warning(f"RL_GAMMA {self.config['RL_GAMMA']} вне диапазона 0-1. Установлено: 0.95")
            self.config["RL_GAMMA"] = 0.95
        if not 0 < self.config["RL_LEARNING_RATE"] < 1:
            logger.warning(f"RL_LEARNING_RATE {self.config['RL_LEARNING_RATE']} вне диапазона 0-1. Установлено: 0.001")
            self.config["RL_LEARNING_RATE"] = 0.001
        if not 0 <= self.config["RL_EPSILON"] <= 1:
            logger.warning(f"RL_EPSILON {self.config['RL_EPSILON']} вне диапазона 0-1. Установлено: 0.1")
            self.config["RL_EPSILON"] = 0.1
        if not 0 <= self.config["RL_EPSILON_MIN"] < self.config["RL_EPSILON"]:
            logger.warning(f"RL_EPSILON_MIN {self.config['RL_EPSILON_MIN']} некорректен. Установлено: 0.01")
            self.config["RL_EPSILON_MIN"] = 0.01
        if not 0 < self.config["RL_EPSILON_DECAY"] <= 1:
            logger.warning(f"RL_EPSILON_DECAY {self.config['RL_EPSILON_DECAY']} вне диапазона 0-1. Установлено: 0.995")
            self.config["RL_EPSILON_DECAY"] = 0.995
        # Валидация диапазонов
        if (not isinstance(self.config["LEVERAGE_RANGE"], tuple) or len(self.config["LEVERAGE_RANGE"]) != 2 or 
            not all(isinstance(x, int) and x > 0 for x in self.config["LEVERAGE_RANGE"]) or 
            self.config["LEVERAGE_RANGE"][0] > self.config["LEVERAGE_RANGE"][1]):
            logger.warning(f"LEVERAGE_RANGE {self.config['LEVERAGE_RANGE']} некорректен. Установлено: (5, 20)")
            self.config["LEVERAGE_RANGE"] = (5, 20)
        if (not isinstance(self.config["LOOKBACK_RANGE"], tuple) or len(self.config["LOOKBACK_RANGE"]) != 2 or 
            not all(isinstance(x, int) and x > 0 for x in self.config["LOOKBACK_RANGE"]) or 
            self.config["LOOKBACK_RANGE"][0] > self.config["LOOKBACK_RANGE"][1]):
            logger.warning(f"LOOKBACK_RANGE {self.config['LOOKBACK_RANGE']} некорректен. Установлено: (100, 500)")
            self.config["LOOKBACK_RANGE"] = (100, 500)

    def get_config(self) -> Dict[str, Any]:
        """Получение загруженной конфигурации.

        Returns:
            Dict[str, Any]: Словарь с настройками.
        """
        return self.config

# Инициализация и экспорт переменных
try:
    config_loader = ConfigLoader()
    config = config_loader.get_config()
except Exception as e:
    logger.error(f"Не удалось инициализировать ConfigLoader: {e}")
    raise SystemExit(1)

# Экспортируемые переменные с описанием
API_KEY = config["API_KEY"]  # Ключ API для OKX
API_SECRET = config["API_SECRET"]  # Секретный ключ для OKX
API_PASSPHRASE = config["API_PASSPHRASE"]  # Пароль для OKX API
TESTNET = config["TESTNET"]  # Флаг использования тестовой сети OKX (True/False)
TELEGRAM_TOKEN = config["TELEGRAM_TOKEN"]  # Токен для Telegram бота
TELEGRAM_CHAT_ID = config["TELEGRAM_CHAT_ID"]  # ID чата для Telegram уведомлений
NEWSAPI_KEY = config["NEWSAPI_KEY"]  # Ключ API для NewsAPI
REDDIT_CLIENT_ID = config["REDDIT_CLIENT_ID"]  # ID клиента для Reddit API
REDDIT_CLIENT_SECRET = config["REDDIT_CLIENT_SECRET"]  # Секрет клиента для Reddit API
REDDIT_USER_AGENT = config["REDDIT_USER_AGENT"]  # User-Agent для Reddit API
FRED_API_KEY = config["FRED_API_KEY"]  # Ключ API для FRED (Federal Reserve Economic Data)
REDIS_HOST = config["REDIS_HOST"]  # Хост Redis сервера
REDIS_PORT = config["REDIS_PORT"]  # Порт Redis сервера
REDIS_PASSWORD = config["REDIS_PASSWORD"]  # Пароль для Redis (опционально)
LOG_FILE = config["LOG_FILE"]  # Путь к файлу логов торговли
DASH_PORT = config["DASH_PORT"]  # Порт для Dash дашборда
SYMBOLS = config["SYMBOLS"]  # Словарь символов для торговли (например, BTC/USDT:USDT)
STRATEGIES = config["STRATEGIES"]  # Словарь торговых стратегий с параметрами
LEVERAGE_RANGE = config["LEVERAGE_RANGE"]  # Диапазон рычага (min, max)
LOOKBACK_RANGE = config["LOOKBACK_RANGE"]  # Диапазон lookback периода (min, max)
AMOUNT_DEFAULT = config["AMOUNT_DEFAULT"]  # Значение объема по умолчанию для сделок
INTERVAL_DEFAULT = config["INTERVAL_DEFAULT"]  # Интервал свечей по умолчанию (например, "15m")
MAX_DAILY_LOSS = config["MAX_DAILY_LOSS"]  # Максимальный дневной убыток
MAX_WEEKLY_LOSS = config["MAX_WEEKLY_LOSS"]  # Максимальный недельный убыток
RL_GAMMA = config["RL_GAMMA"]  # Гамма для RL (дисконтирование награды)
RL_LEARNING_RATE = config["RL_LEARNING_RATE"]  # Скорость обучения для RL
RL_EPSILON = config["RL_EPSILON"]  # Начальное значение epsilon для RL
RL_EPSILON_MIN = config["RL_EPSILON_MIN"]  # Минимальное значение epsilon для RL
RL_EPSILON_DECAY = config["RL_EPSILON_DECAY"]  # Скорость затухания epsilon для RL

if __name__ == "__main__":
    logger.info(f"Конфигурация загружена: {config}")