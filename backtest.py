# backtest.py
import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime, timedelta

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("backtest_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("backtest")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from config_loader import SYMBOLS, STRATEGIES
    from dynamic_config import dynamic_config
    from cache_manager import get_cached_data, save_cached_data
    from data_utils import validate_and_notify, DataUtils
    from plot_utils import generate_plots
    from feature_engineer import FeatureEngineer
    from trading_models import TradingModels
    from trade_executor import TradeExecutor
    from fetch_data import fetch_ohlcv, fetcher
    from telegram_utils import send_async_message
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    raise SystemExit(1)

import platform

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def fetch_historical_data(symbol: str, interval: str = "15m", lookback: int = 1000, 
                               since: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Асинхронная загрузка исторических данных OHLCV.

    Args:
        symbol (str): Символ (например, "BTC/USDT:USDT").
        interval (str): Интервал свечей (по умолчанию "15m").
        lookback (int): Количество свечей для загрузки.
        since (Optional[int]): Начальная временная метка в миллисекундах.

    Returns:
        Optional[pd.DataFrame]: DataFrame с OHLCV данными или None при ошибке.
    """
    if not isinstance(symbol, str) or not symbol:
        logger.error(f"symbol должен быть непустой строкой, получен {type(symbol)}")
        return None
    if not isinstance(interval, str) or not interval:
        logger.error(f"interval должен быть непустой строкой, получен {type(interval)}")
        return None
    if not isinstance(lookback, int) or lookback <= 0:
        logger.error(f"lookback должен быть положительным числом, получен {lookback}")
        return None

    try:
        limit = min(lookback * 2, 1000)  # Ограничение API OKX
        df_list = []
        current_since = since or int((pd.Timestamp.now().timestamp() - lookback * 60 * int(interval[:-1])) * 1000)
        
        while len(df_list) * limit < lookback:
            df = await fetch_ohlcv(symbol, interval=interval, limit=limit, since=current_since)
            is_valid, validated_df = await validate_and_notify(df, symbol)
            if not is_valid or validated_df.empty:
                logger.warning(f"Данные для {symbol} недоступны или некорректны")
                break
            
            df_list.append(validated_df)
            if len(validated_df) < limit:
                break
            
            current_since = int(validated_df["timestamp"].iloc[-1].timestamp() * 1000) + 1
            await asyncio.sleep(1)  # Избежание лимитов API
            
        if not df_list:
            return None
        
        combined_df = pd.concat(df_list).drop_duplicates(subset="timestamp").sort_values("timestamp")
        logger.info(f"Загружено {len(combined_df)} записей для {symbol}")
        return combined_df.tail(lookback)
    except Exception as e:
        logger.error(f"Ошибка загрузки данных для {symbol}: {e}")
        await send_async_message(f"⚠️ Ошибка загрузки данных для {symbol}: {e}")
        return None

async def run_backtest_for_symbol(symbol: str, executor: TradeExecutor, feature_engineer: FeatureEngineer,
                                 lookback: int, commission: float, train_split: float, slippage: float,
                                 total_profit: Dict, profits: Dict, equity_curves: Dict, wins: Dict) -> None:
    """Выполнение бэктеста для одного символа.

    Args:
        symbol (str): Символ для бэктеста.
        executor (TradeExecutor): Экземпляр TradeExecutor.
        feature_engineer (FeatureEngineer): Экземпляр FeatureEngineer.
        lookback (int): Количество свечей для анализа.
        commission (float): Комиссия за сделку.
        train_split (float): Доля данных для обучения.
        slippage (float): Проскальзывание цены.
        total_profit (Dict): Словарь общей прибыли.
        profits (Dict): Словарь списка прибылей.
        equity_curves (Dict): Словарь кривых эквити.
        wins (Dict): Словарь количества выигрышных сделок.
    """
    if not isinstance(symbol, str) or not symbol:
        logger.error(f"symbol должен быть непустой строкой, получен {type(symbol)}")
        return
    if not isinstance(executor, TradeExecutor):
        logger.error(f"executor должен быть TradeExecutor, получен {type(executor)}")
        return
    if not isinstance(feature_engineer, FeatureEngineer):
        logger.error(f"feature_engineer должен быть FeatureEngineer, получен {type(feature_engineer)}")
        return
    if not isinstance(lookback, int) or lookback <= 0:
        logger.error(f"lookback должен быть положительным числом, получен {lookback}")
        return
    if not isinstance(commission, float) or commission < 0:
        logger.error(f"commission должен быть неотрицательным числом, получен {commission}")
        return
    if not isinstance(train_split, float) or not 0 < train_split < 1:
        logger.error(f"train_split должен быть числом от 0 до 1, получен {train_split}")
        return
    if not isinstance(slippage, float) or slippage < 0:
        logger.error(f"slippage должен быть неотрицательным числом, получен {slippage}")
        return

    try:
        cached_result = await get_cached_data(f"backtest:{symbol}:{lookback}")
        if cached_result:
            logger.info(f"Использованы кэшированные результаты для {symbol}")
            for strategy in STRATEGIES:
                total_profit[symbol][strategy] = cached_result["total_profit"][strategy]
                profits[symbol][strategy] = cached_result["profits"][strategy]
                equity_curves[symbol][strategy] = cached_result["equity_curves"][strategy]
                wins[symbol][strategy] = cached_result["wins"][strategy]
            return

        df = await fetch_historical_data(symbol, interval=SYMBOLS[symbol]["interval"], lookback=lookback)
        min_data_size = lookback + 26
        if df is None or len(df) < min_data_size:
            logger.warning(f"Недостаточно данных для {symbol}: {len(df) if df is not None else 0}")
            return

        train_df, test_df = train_test_split(df, train_size=train_split, shuffle=False)
        if len(train_df) < min_data_size or len(test_df) < min_data_size:
            logger.warning(f"Недостаточно данных после разделения для {symbol}: train={len(train_df)}, test={len(test_df)}")
            return

        features_dict = feature_engineer.prepare_features({symbol: train_df}, [symbol])
        if symbol not in features_dict:
            logger.warning(f"Не удалось подготовить признаки для {symbol} (обучающая выборка)")
            return
        X_train, y_train_raw, train_df = features_dict[symbol]
        y_train = np.diff(train_df["close"])[lookback - 1:]  # Исправлено выравнивание
        if len(X_train) == 0 or len(y_train) != len(X_train):
            logger.warning(f"Несоответствие размеров X_train и y_train для {symbol}: {len(X_train)} vs {len(y_train)}")
            return

        model = TradingModels(lookback=lookback, input_features=X_train.shape[2])
        await model.train_model(X_train, y_train, volatility=train_df.get("volatility", pd.Series([0])).iloc[-1])

        features_dict = feature_engineer.prepare_features({symbol: test_df}, [symbol])
        if symbol not in features_dict:
            logger.warning(f"Не удалось подготовить признаки для {symbol} (тестовая выборка)")
            return
        X_test, y_test_raw, test_df = features_dict[symbol]
        if len(X_test) == 0:
            logger.warning(f"Пустой тестовый набор для {symbol}")
            return

        for strategy in STRATEGIES:
            for i in range(len(X_test)):
                if i + lookback - 1 >= len(test_df):
                    logger.warning(f"Индекс {i + lookback - 1} выходит за пределы test_df для {symbol}")
                    continue
                price = test_df["close"].iloc[i + lookback - 1]
                atr = test_df.get("atr", pd.Series([0])).iloc[i + lookback - 1]
                rsi = test_df.get("rsi", pd.Series([50])).iloc[i + lookback - 1]
                volatility = test_df.get("volatility", pd.Series([0])).iloc[i + lookback - 1]
                macd = test_df.get("macd", pd.Series([0])).iloc[i + lookback - 1]
                adx = test_df.get("adx", pd.Series([25])).iloc[i + lookback - 1]

                state = X_test[i].reshape(1, lookback, X_test.shape[2])
                predicted_change = model.predict(state, volatility=volatility)[0]
                predicted_price = price + predicted_change + (price * slippage)

                if abs(predicted_change / price) >= STRATEGIES[strategy]["entry_threshold"]:
                    profit = await executor.simulate_trade("buy_long", price, {"volatility": volatility}, 
                                                          predicted_price, symbol, atr, rsi, commission)
                    if profit is not None:
                        total_profit[symbol][strategy] += profit
                        profits[symbol][strategy].append(profit)
                        equity_curves[symbol][strategy].append(equity_curves[symbol][strategy][-1] + profit)
                        wins[symbol][strategy] += 1 if profit > 0 else 0

        await save_cached_data(f"backtest:{symbol}:{lookback}", {
            "total_profit": total_profit[symbol],
            "profits": profits[symbol],
            "equity_curves": equity_curves[symbol],
            "wins": wins[symbol]
        }, ttl=24 * 3600)

    except Exception as e:
        logger.error(f"Ошибка бэктеста для {symbol}: {e}")
        await send_async_message(f"⚠️ Ошибка бэктеста для {symbol}: {e}")

async def backtest(data_file: str = "historical_data.csv", lookback: int = 120, commission: float = 0.0001, 
                  train_split: float = 0.8, slippage: float = 0.001, generate_plots_flag: bool = True) -> None:
    """Основная функция для выполнения бэктеста.

    Args:
        data_file (str): Путь к файлу с историческими данными (не используется).
        lookback (int): Количество свечей для анализа.
        commission (float): Комиссия за сделку.
        train_split (float): Доля данных для обучения.
        slippage (float): Проскальзывание цены.
        generate_plots_flag (bool): Флаг генерации графиков.
    """
    if not isinstance(data_file, str):
        logger.error(f"data_file должен быть строкой, получен {type(data_file)}")
        return
    if not isinstance(lookback, int) or lookback <= 0:
        logger.error(f"lookback должен быть положительным числом, получен {lookback}")
        return
    if not isinstance(commission, float) or commission < 0:
        logger.error(f"commission должен быть неотрицательным числом, получен {commission}")
        return
    if not isinstance(train_split, float) or not 0 < train_split < 1:
        logger.error(f"train_split должен быть числом от 0 до 1, получен {train_split}")
        return
    if not isinstance(slippage, float) or slippage < 0:
        logger.error(f"slippage должен быть неотрицательным числом, получен {slippage}")
        return
    if not isinstance(generate_plots_flag, bool):
        logger.error(f"generate_plots_flag должен быть булевым значением, получен {type(generate_plots_flag)}")
        return

    try:
        if not hasattr(fetcher, "exchange") or fetcher.exchange is None:
            logger.error("fetcher.exchange не инициализирован")
            raise ValueError("fetcher.exchange не инициализирован")
        executor = TradeExecutor(exchange=fetcher.exchange, telegram_handler=None)
        feature_engineer = FeatureEngineer()
        total_profit = {symbol: {strategy: 0 for strategy in STRATEGIES} for symbol in SYMBOLS}
        profits = {symbol: {strategy: [] for strategy in STRATEGIES} for symbol in SYMBOLS}
        equity_curves = {symbol: {strategy: [0] for strategy in STRATEGIES} for symbol in SYMBOLS}
        wins = {symbol: {strategy: 0 for strategy in STRATEGIES} for symbol in SYMBOLS}

        tasks = [
            run_backtest_for_symbol(symbol, executor, feature_engineer, lookback, commission, train_split, slippage,
                                    total_profit, profits, equity_curves, wins)
            for symbol in SYMBOLS
        ]
        await asyncio.gather(*tasks)

        best_strategies = {}
        for symbol in SYMBOLS:
            if any(profits[symbol][strategy] for strategy in STRATEGIES):
                sharpe_ratios = {}
                for strategy in STRATEGIES:
                    profits_array = np.array(profits[symbol][strategy])
                    if len(profits_array) > 1 and np.std(profits_array, ddof=1) != 0:
                        sharpe = np.mean(profits_array) / np.std(profits_array, ddof=1) * np.sqrt(252)
                    else:
                        sharpe = 0.0
                    sharpe_ratios[strategy] = sharpe
                    logger.info(f"Бэктест для {symbol} ({strategy}): Профит: {total_profit[symbol][strategy]:.2f}, Sharpe: {sharpe:.2f}")
                    await send_async_message(f"📊 Бэктест для {symbol} ({strategy}): Профит: {total_profit[symbol][strategy]:.2f}, Sharpe: {sharpe:.2f}")
                
                best_strategy = max(sharpe_ratios, key=sharpe_ratios.get)
                best_strategies[symbol] = best_strategy
                
                if generate_plots_flag:
                    df = await fetch_historical_data(symbol, interval=SYMBOLS[symbol]["interval"], lookback=lookback)
                    if df is not None and not df.empty:
                        df["profit"] = [0] * len(df)  # Упрощение для примера
                        df["rsi"] = feature_engineer.calculate_technical_indicators(df, 0, best_strategy).get("rsi", pd.Series([50] * len(df)))
                        df["volatility"] = feature_engineer.calculate_technical_indicators(df, 0, best_strategy).get("volatility", pd.Series([0] * len(df)))
                        await generate_plots(df, symbol, "all")

        await save_cached_data("best_strategies", best_strategies, ttl=24 * 3600)
        logger.info(f"Лучшие стратегии: {best_strategies}")
    except Exception as e:
        logger.error(f"Ошибка выполнения бэктеста: {e}")
        await send_async_message(f"⚠️ Ошибка выполнения бэктеста: {e}")

async def auto_backtest() -> None:
    """Автоматический ежедневный запуск бэктеста."""
    while True:
        now = datetime.now()
        next_run = datetime(now.year, now.month, now.day, 0, 0) + timedelta(days=1)
        sleep_seconds = (next_run - now).total_seconds()
        await asyncio.sleep(sleep_seconds)
        await backtest()
        logger.info("Ежедневный бэктест выполнен")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(backtest())
        loop.create_task(auto_backtest())
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(fetcher.close())
        loop.close()
    except Exception as e:
        logger.error(f"Ошибка при запуске бэктеста: {e}")