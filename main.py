# main.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import asyncio
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
import pickle
import sys
import platform
import tensorflow as tf
import warnings
from typing import Dict, List, Optional, Tuple, Any
import aiohttp
import logging

# Попытка импорта зависимостей с базовым логгером
try:
    from logging_setup import setup_logging
    logger = setup_logging("main_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")
    logger.warning(f"Не удалось импортировать logging_setup: {e}")

try:
    from config_loader import API_KEY, API_SECRET, API_PASSPHRASE, TESTNET, SYMBOLS, LOG_FILE, FRED_API_KEY
    from dynamic_config import dynamic_config, LEVERAGE, LOOKBACK, initialize_config
    from cache_manager import init_cache
    from data_store import DataStore
    from grok3_analyze import grok3_analyze
    from trading_models import TradingModels
    from trade_executor import TradeExecutor
    from telegram_handler import TelegramHandler
    from feature_engineer import FeatureEngineer
    from logger import log_trade, trade_logger, initialize_logger
    from generate_report import auto_generate_reports
    from data_utils import validate_and_notify
    from sklearn.preprocessing import MinMaxScaler
    from telegram_utils import send_async_message
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    async def send_async_message(msg: str, **kwargs) -> bool:
        logger.warning(f"Telegram уведомления отключены: {msg}")
        return False
    raise SystemExit(1)

warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.get_logger().setLevel("ERROR")

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def fetch_fred_data(series_id: str = "FEDFUNDS", retries: int = 3, delay: int = 5) -> float:
    """Получение данных FRED (ставка федерального фонда).

    Args:
        series_id (str): Идентификатор серии данных (по умолчанию "FEDFUNDS").
        retries (int): Количество попыток (по умолчанию 3).
        delay (int): Задержка между попытками в секундах (по умолчанию 5).

    Returns:
        float: Последняя ставка FRED или 0.0 при ошибке.
    """
    if not isinstance(series_id, str) or not series_id:
        logger.error(f"series_id должен быть непустой строкой, получено {series_id}")
        return 0.0
    if not isinstance(retries, int) or retries < 0:
        logger.error(f"retries должен быть неотрицательным числом, получено {retries}")
        retries = 3
    if not isinstance(delay, int) or delay < 0:
        logger.error(f"delay должен быть неотрицательным числом, получено {delay}")
        delay = 5

    for attempt in range(retries):
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if not isinstance(data, dict) or "observations" not in data:
                        raise ValueError(f"Некорректный ответ API FRED: {data}")
                    latest_rate = float(data["observations"][-1]["value"])
                    logger.info(f"Получена ставка FRED FEDFUNDS: {latest_rate}")
                    return latest_rate
        except Exception as e:
            logger.error(f"Ошибка получения данных FRED (попытка {attempt+1}/{retries}): {e}", exc_info=True)
            if attempt == retries - 1:
                await send_async_message(f"⚠️ Не удалось получить данные FRED после {retries} попыток")
                return 0.0
            await asyncio.sleep(delay)

async def initialize_exchange() -> ccxt_async.Exchange:
    """Инициализация биржи OKX.

    Returns:
        ccxt_async.Exchange: Объект биржи.

    Raises:
        Exception: Если инициализация не удалась.
    """
    logger.info("Начало инициализации биржи OKX")
    exchange = ccxt_async.okx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "password": API_PASSPHRASE,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "testnet": TESTNET},
    })
    try:
        logger.info("Загрузка рынков OKX")
        await exchange.load_markets()
        logger.info("Биржа инициализирована успешно")
        return exchange
    except Exception as e:
        logger.error(f"Ошибка инициализации биржи: {e}", exc_info=True)
        raise

async def initialize_models(data_store: DataStore, feature_engineer: FeatureEngineer, 
                            exchange: ccxt_async.Exchange, models_dict_path: str) -> Dict[str, Tuple[TradingModels, pd.DataFrame, MinMaxScaler, int]]:
    """Инициализация моделей для каждого символа.

    Args:
        data_store (DataStore): Объект хранилища данных.
        feature_engineer (FeatureEngineer): Объект инженерии признаков.
        exchange (ccxt_async.Exchange): Объект биржи.
        models_dict_path (str): Путь для сохранения/загрузки состояния моделей.

    Returns:
        Dict[str, Tuple[TradingModels, pd.DataFrame, MinMaxScaler, int]]: Словарь моделей для символов.
    """
    if not isinstance(data_store, DataStore):
        logger.error(f"data_store должен быть экземпляром DataStore, получено {type(data_store)}")
        return {}
    if not isinstance(feature_engineer, FeatureEngineer):
        logger.error(f"feature_engineer должен быть экземпляром FeatureEngineer, получено {type(feature_engineer)}")
        return {}
    if not isinstance(exchange, ccxt_async.Exchange):
        logger.error(f"exchange должен быть экземпляром ccxt_async.Exchange, получено {type(exchange)}")
        return {}
    if not isinstance(models_dict_path, str) or not models_dict_path:
        logger.error(f"models_dict_path должен быть непустой строкой, получено {models_dict_path}")
        return {}

    logger.info("Начало инициализации моделей")
    models_dict = {}
    if os.path.exists(models_dict_path):
        try:
            with open(models_dict_path, "rb") as f:
                models_dict = pickle.load(f)
            logger.info("Модели восстановлены из сохраненного состояния")
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния моделей: {e}", exc_info=True)

    min_data_size = LOOKBACK + 26
    for symbol in SYMBOLS:
        if symbol not in models_dict:
            logger.info(f"Инициализация данных для {symbol}")
            await data_store.initial_data_fetch(symbol)
            ohlcv, additional_data = data_store.get_data_for_feature_engineering(symbol)
            is_valid, validated_df = await validate_and_notify(ohlcv, symbol, min_data_size)
            if not is_valid:
                continue

            features_dict = feature_engineer.prepare_features({symbol: validated_df}, [symbol], {symbol: additional_data})
            if symbol not in features_dict:
                logger.warning(f"Не удалось подготовить признаки для {symbol}, пропускаем")
                continue
            X, y, df, scaler_y = features_dict[symbol]
            if X.shape[0] < min_data_size:
                logger.warning(f"Недостаточно данных после обработки для {symbol}: {X.shape[0]} < {min_data_size}")
                continue

            balance_response = await exchange.fetch_balance()
            balance = float(balance_response.get("total", {}).get("USDT", 0))
            grok_analysis = await grok3_analyze(df["close"].iloc[-1], {
                "volatility": df["volatility"].iloc[-1] if "volatility" in df else 0,
                "rsi": df["rsi"].iloc[-1] if "rsi" in df else 50,
                "macd": df["macd"].iloc[-1] if "macd" in df else 0,
                "adx": df["adx"].iloc[-1] if "adx" in df else 25,
                "atr": df["atr"].iloc[-1] if "atr" in df else 0,
                "ichimoku_tenkan": df.get("ichimoku_tenkan", pd.Series([0])).iloc[-1],
                "ichimoku_kijun": df.get("ichimoku_kijun", pd.Series([0])).iloc[-1]
            })
            fred_rate = await fetch_fred_data()

            X_expanded = np.array(X, dtype=np.float32)
            if X_expanded.shape[0] == 0 or np.any(np.isnan(X_expanded)):
                logger.warning(f"Некорректные данные для {symbol}, пропуск")
                continue

            num_features = X.shape[2]
            input_features = num_features + 2
            X_expanded = np.pad(X_expanded, ((0, 0), (0, 0), (0, 2)), mode="constant")
            X_expanded[:, :, num_features] = balance
            X_expanded[:, :, num_features + 1] = fred_rate

            models = TradingModels(lookback=X.shape[1], input_features=input_features)
            await models.train_model(X_expanded, y, volatility=df["volatility"].iloc[-1] if "volatility" in df else 0)
            models_dict[symbol] = (models, df, scaler_y, X.shape[1])
            # Сохраняем состояние после добавления новой модели
            with open(models_dict_path, "wb") as f:
                pickle.dump(models_dict, f)

    logger.info("Модели инициализированы")
    return models_dict

async def trading_cycle(symbol: str, data_store: DataStore, executor: TradeExecutor, 
                        feature_engineer: FeatureEngineer, models_dict: Dict, exchange: ccxt_async.Exchange) -> None:
    """Основной торговый цикл для символа.

    Args:
        symbol (str): Символ (например, "BTC/USDT:USDT").
        data_store (DataStore): Объект хранилища данных.
        executor (TradeExecutor): Объект исполнителя сделок.
        feature_engineer (FeatureEngineer): Объект инженерии признаков.
        models_dict (Dict): Словарь моделей.
        exchange (ccxt_async.Exchange): Объект биржи.
    """
    if not isinstance(symbol, str) or symbol not in SYMBOLS:
        logger.error(f"symbol должен быть строкой из SYMBOLS, получено {symbol}")
        return
    if not isinstance(data_store, DataStore):
        logger.error(f"data_store должен быть экземпляром DataStore, получено {type(data_store)}")
        return
    if not isinstance(executor, TradeExecutor):
        logger.error(f"executor должен быть экземпляром TradeExecutor, получено {type(executor)}")
        return
    if not isinstance(feature_engineer, FeatureEngineer):
        logger.error(f"feature_engineer должен быть экземпляром FeatureEngineer, получено {type(feature_engineer)}")
        return
    if not isinstance(models_dict, dict):
        logger.error(f"models_dict должен быть словарем, получено {type(models_dict)}")
        return
    if not isinstance(exchange, ccxt_async.Exchange):
        logger.error(f"exchange должен быть экземпляром ccxt_async.Exchange, получено {type(exchange)}")
        return

    try:
        min_data_size = LOOKBACK + 26
        logger.info(f"Начало цикла для {symbol}")
        interval = await data_store.adjust_interval(symbol)
        await data_store.fetch_and_update_data(symbol, interval)
        ohlcv, additional_data = data_store.get_data_for_feature_engineering(symbol)
        is_valid, validated_df = await validate_and_notify(ohlcv, symbol, min_data_size)
        if not is_valid:
            return

        balance_response = await exchange.fetch_balance()
        balance = float(balance_response.get("total", {}).get("USDT", 0))
        await dynamic_config.update_dynamic_params(validated_df["close"].pct_change().std(), balance)

        if symbol not in models_dict:
            logger.warning(f"Модель для {symbol} не инициализирована, пропуск")
            return
        models, _, _, _ = models_dict[symbol]
        features_dict = feature_engineer.prepare_features(
            {symbol: validated_df}, [symbol], {symbol: additional_data}, 
            interval_minutes=15, strategy="trend", rl_model=models, state=validated_df.tail(LOOKBACK).values
        )
        if symbol not in features_dict:
            logger.warning(f"Не удалось подготовить признаки для {symbol} в цикле")
            return
        X, y, df, scaler_y = features_dict[symbol]
        if X.shape[0] < min_data_size:
            logger.warning(f"Недостаточно данных после обработки для {symbol}: {X.shape[0]} < {min_data_size}")
            return

        grok_analysis = await grok3_analyze(df["close"].iloc[-1], {
            "volatility": df["volatility"].iloc[-1] if "volatility" in df else 0,
            "rsi": df["rsi"].iloc[-1] if "rsi" in df else 50,
            "macd": df["macd"].iloc[-1] if "macd" in df else 0,
            "adx": df["adx"].iloc[-1] if "adx" in df else 25,
            "atr": df["atr"].iloc[-1] if "atr" in df else 0,
            "ichimoku_tenkan": df.get("ichimoku_tenkan", pd.Series([0])).iloc[-1],
            "ichimoku_kijun": df.get("ichimoku_kijun", pd.Series([0])).iloc[-1]
        })
        fred_rate = await fetch_fred_data()

        X_expanded = np.array(X, dtype=np.float32)
        if X_expanded.shape[0] == 0 or np.any(np.isnan(X_expanded)):
            logger.warning(f"Некорректные данные для {symbol} в цикле, пропуск")
            return

        num_features = X.shape[2]
        input_features = num_features + 2
        X_expanded = np.pad(X_expanded, ((0, 0), (0, 0), (0, 2)), mode="constant")
        X_expanded[:, :, num_features] = balance
        X_expanded[:, :, num_features + 1] = fred_rate

        models, _, scaler_y, model_lookback = models_dict[symbol]
        current_lookback = X.shape[1]
        if current_lookback != model_lookback:
            logger.info(f"Lookback для {symbol} изменился с {model_lookback} на {current_lookback}")
            models = TradingModels(lookback=current_lookback, input_features=input_features)
            await models.train_model(X_expanded, y, volatility=df["volatility"].iloc[-1] if "volatility" in df else 0)
            models_dict[symbol] = (models, df, scaler_y, current_lookback)

        state = X_expanded[-1:]
        current_price = (await exchange.fetch_ticker(symbol))["last"]
        await executor.check_open_positions(symbol, current_price)

        action_idx = models.get_action(state[0], volatility=df["volatility"].iloc[-1] if "volatility" in df else 0, sentiment=grok_analysis["sentiment"])
        action_map = {
            0: "hold", 1: "buy_long", 2: "sell_short", 3: "increase_amount",
            4: "decrease_amount", 5: "grid_buy", 6: "grid_sell", 7: "arbitrage"
        }
        action = action_map.get(action_idx, "hold")

        predicted_change_normalized = float(models.predict(state)[0][0])
        predicted_change_denormalized = scaler_y.inverse_transform([[predicted_change_normalized]])[0][0]
        adjusted_change = grok_analysis["predicted_change"]
        predicted_price = current_price + adjusted_change

        if action in ["buy_long", "sell_short", "grid_buy", "grid_sell"] and executor.positions[symbol] is None:
            await executor.execute_trade(
                action, current_price, grok_analysis, predicted_price, 
                log_trade, symbol=symbol, atr=df["atr"].iloc[-1], rsi=df["rsi"].iloc[-1], fred_rate=fred_rate
            )
            
            next_ohlcv, _ = data_store.get_data_for_feature_engineering(symbol)
            if not next_ohlcv.empty and len(next_ohlcv) >= min_data_size:
                next_features_dict = feature_engineer.prepare_features(
                    {symbol: next_ohlcv}, [symbol], {symbol: additional_data}, 
                    interval_minutes=15, strategy="trend", rl_model=models, state=next_ohlcv.tail(LOOKBACK).values
                )
                if symbol in next_features_dict:
                    next_X, _, next_df, _ = next_features_dict[symbol]
                    next_X_expanded = np.array(next_X, dtype=np.float32)
                    next_X_expanded = np.pad(next_X_expanded, ((0, 0), (0, 0), (0, 2)), mode="constant")
                    next_X_expanded[:, :, num_features] = balance
                    next_X_expanded[:, :, num_features + 1] = fred_rate
                    next_state = next_X_expanded[-1:]
                    current_price = (await exchange.fetch_ticker(symbol))["last"]
                    profit = ((next_df["close"].iloc[-1] - df["close"].iloc[-1]) * executor.amounts[symbol] * executor.leverage 
                             if action in ["buy_long", "grid_buy"] else 
                             (df["close"].iloc[-1] - next_df["close"].iloc[-1]) * executor.amounts[symbol] * executor.leverage)
                    actual_change = next_df["close"].iloc[-1] - df["close"].iloc[-1]
                    reward = profit / current_price if current_price != 0 else 0
                    done = executor.positions[symbol] is None
                    await log_trade(symbol, action, df["close"].iloc[-1], profit, grok_analysis, predicted_price, actual_change)
                    models.store_transition(state[0], action_idx, reward, next_state[0], done)
                    if len(models.memory) >= models.batch_size * 2:
                        models.replay()

        elif action in ["increase_amount", "decrease_amount"]:
            current_amount = executor.amounts[symbol]
            new_amount = current_amount * 1.1 if action == "increase_amount" else current_amount * 0.9
            executor.amounts[symbol] = new_amount
            await log_trade(symbol, action, current_price, None)

        if await models.should_retrain():
            await models.train_model(X_expanded, y, volatility=df["volatility"].iloc[-1] if "volatility" in df else 0)
            models_dict[symbol] = (models, df, scaler_y, current_lookback)
            logger.info(f"Модель для {symbol} переобучена")

    except Exception as e:
        logger.error(f"Ошибка в торговом цикле для {symbol}: {e}", exc_info=True)

async def main() -> None:
    """Главная функция бота."""
    logger.info("Начало main()")
    try:
        await initialize_config()
        logger.info("Конфигурация инициализирована")
    except Exception as e:
        logger.error(f"Ошибка в initialize_config: {e}", exc_info=True)
        raise
    
    start_time = asyncio.get_event_loop().time()
    max_attempts = 5
    models_dict_path = "models_dict_state.pkl"
    attempt = 0
    telegram_handler = None
    exchange = None

    while attempt < max_attempts:
        try:
            logger.info("Запуск бота...")
            await init_cache()
            logger.info("Кэш инициализирован")
            await initialize_logger()
            logger.info("Логгер инициализирован")
            exchange = await initialize_exchange()
            logger.info("Инициализация DataStore")
            data_store = DataStore(exchange, SYMBOLS.keys())
            logger.info("Инициализация TelegramHandler")
            telegram_handler = TelegramHandler(None, exchange)
            telegram_handler.set_start_time(start_time)
            logger.info("Инициализация TradeExecutor")
            executor = TradeExecutor(exchange, telegram_handler)
            telegram_handler.executor = executor
            logger.info("Инициализация FeatureEngineer")
            feature_engineer = FeatureEngineer()

            logger.info("Запуск инициализации моделей")
            models_dict = await initialize_models(data_store, feature_engineer, exchange, models_dict_path)
            logger.info("Отправка уведомления о запуске")
            await send_async_message("✅ Бот успешно запущен")
            logger.info("Бот успешно запущен")

            attempt = 0
            report_task = asyncio.create_task(auto_generate_reports())
            while True:
                if telegram_handler.paused:
                    logger.info("Бот на паузе")
                    await asyncio.sleep(60)
                    continue
                
                cycle_start = asyncio.get_event_loop().time()
                tasks = [trading_cycle(symbol, data_store, executor, feature_engineer, models_dict, exchange) 
                         for symbol in SYMBOLS]
                await asyncio.gather(*tasks)
                
                # Сохранение состояния моделей после успешного цикла
                with open(models_dict_path, "wb") as f:
                    pickle.dump(models_dict, f)
                logger.info("Состояние моделей сохранено после цикла")

                cycle_duration = asyncio.get_event_loop().time() - cycle_start
                if cycle_duration > 120:
                    await send_async_message(f"⚠️ Задержка цикла: {cycle_duration:.2f} секунд")
                await asyncio.sleep(max(15 - cycle_duration, 1))

        except Exception as e:
            attempt += 1
            logger.error(f"Критическая ошибка (попытка {attempt}/{max_attempts}): {e}", exc_info=True)
            escaped_error = str(e).replace("(", "\\(").replace(")", "\\)").replace("-", "\\-")
            if telegram_handler:
                await send_async_message(f"🚨 Критическая ошибка \\(попытка {attempt}/{max_attempts}\\): {escaped_error}")
            else:
                logger.warning("TelegramHandler не инициализирован, уведомление об ошибке не отправлено")
            if attempt >= max_attempts:
                logger.error("Превышено максимальное количество попыток. Завершение работы.")
                try:
                    with open(models_dict_path, "wb") as f:
                        pickle.dump(models_dict, f)
                    logger.info("Состояние моделей сохранено")
                    await trade_logger.close()
                    if telegram_handler:
                        telegram_handler.stop_polling()
                        await asyncio.sleep(1)
                    if exchange:
                        await exchange.close()
                        logger.info("Соединение с биржей закрыто")
                    if 'report_task' in locals():
                        report_task.cancel()
                        await asyncio.sleep(1)
                except Exception as shutdown_e:
                    logger.error(f"Ошибка при завершении: {shutdown_e}", exc_info=True)
                sys.exit(1)
            delay = min(300 * (2 ** (attempt - 1)), 3600)
            if telegram_handler:
                telegram_handler.stop_polling()
                await asyncio.sleep(1)
            if exchange:
                await exchange.close()
                logger.info("Соединение с биржей закрыто перед перезапуском")
            await asyncio.sleep(delay)
            logger.info("Перезапуск бота...")

if __name__ == "__main__":
    asyncio.run(main())