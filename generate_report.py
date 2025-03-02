# generate_report.py
import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import aiofiles
import os
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("generate_report_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("generate_report")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from config_loader import SYMBOLS, STRATEGIES
    from telegram_utils import send_async_message
    from plot_utils import generate_plots
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    async def send_async_message(msg: str) -> None:
        logger.warning(f"Telegram уведомления отключены: {msg}")
    raise SystemExit(1)

async def read_logs(log_file: str = "trade_log.jsonl") -> List[Dict[str, Union[str, float]]]:
    """
    Асинхронное чтение логов из файла.

    Args:
        log_file (str): Путь к файлу логов (по умолчанию "trade_log.jsonl").

    Returns:
        List[Dict[str, Union[str, float]]]: Список словарей с данными логов.
    """
    if not isinstance(log_file, str) or not log_file:
        logger.error(f"log_file должен быть непустой строкой, получен {log_file}")
        return []
    try:
        if not os.path.exists(log_file):
            error_msg = f"Файл логов {log_file} не существует"
            logger.error(error_msg)
            await send_async_message(error_msg)
            return []

        logs = []
        async with aiofiles.open(log_file, "r", encoding="utf-8") as f:
            async for line in f:
                if line.strip():
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Ошибка разбора строки лога: {e}")
        return logs
    except Exception as e:
        logger.error(f"Ошибка при чтении логов ({log_file}): {e}")
        await send_async_message(f"⚠️ Ошибка при чтении логов: {e}")
        return []

async def filter_logs(logs: List[Dict[str, Union[str, float]]], period: str) -> List[Dict[str, Union[str, float]]]:
    """
    Фильтрация логов по периоду.

    Args:
        logs (List[Dict[str, Union[str, float]]]): Список логов.
        period (str): Период для фильтрации ("1h", "1d", "1w", "1m", "all").

    Returns:
        List[Dict[str, Union[str, float]]]: Отфильтрованный список логов.
    """
    if not isinstance(logs, list):
        logger.error(f"logs должен быть списком, получен {type(logs)}")
        return []
    if not isinstance(period, str) or period not in ["1h", "1d", "1w", "1m", "all"]:
        logger.error(f"period должен быть одним из ['1h', '1d', '1w', '1m', 'all'], получен {period}")
        return logs
    try:
        if period == "all":
            return logs
        
        time_delta = {"1h": timedelta(hours=1), "1d": timedelta(days=1), "1w": timedelta(weeks=1), 
                      "1m": timedelta(days=30)}[period]
        cutoff = datetime.now() - time_delta
        return [log for log in logs if datetime.strptime(log["timestamp"], "%Y-%m-%dT%H:%M:%S.%f") >= cutoff]
    except Exception as e:
        logger.error(f"Ошибка фильтрации логов по периоду {period}: {e}")
        return logs

def calculate_metrics(trades: List[Dict[str, Union[str, float]]]) -> Dict[str, float]:
    """
    Расчёт метрик торговли на основе отфильтрованных сделок.

    Args:
        trades (List[Dict[str, Union[str, float]]]): Список сделок.

    Returns:
        Dict[str, float]: Словарь с метриками.
    """
    if not isinstance(trades, list):
        logger.error(f"trades должен быть списком, получен {type(trades)}")
        return {
            "total_profit": 0.0, "win_rate": 0.0, "mae": 0.0, "max_drawdown": 0.0,
            "profit_factor": 0.0, "sortino_ratio": 0.0, "sharpe_ratio": 0.0
        }
    try:
        profits = np.array([trade["profit"] for trade in trades if isinstance(trade.get("profit"), (int, float))])
        if not profits.size:
            return {
                "total_profit": 0.0, "win_rate": 0.0, "mae": 0.0, "max_drawdown": 0.0,
                "profit_factor": 0.0, "sortino_ratio": 0.0, "sharpe_ratio": 0.0
            }

        total_profit = float(profits.sum())
        win_rate = len([p for p in profits if p > 0]) / len(profits) if len(profits) > 0 else 0.0
        mae = np.mean([abs(trade.get("predicted_price", 0) - trade.get("price", 0) - trade.get("actual_change", 0)) 
                      for trade in trades if isinstance(trade.get("predicted_price"), (int, float)) and isinstance(trade.get("actual_change"), (int, float))]) if trades else 0.0
        cumulative = np.cumsum(profits)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_drawdown = float(np.max(drawdown)) if drawdown.size > 0 else 0.0
        gross_profit = float(sum(p for p in profits if p > 0))
        gross_loss = float(-sum(p for p in profits if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")
        downside_deviation = float(np.std([p for p in profits if p < 0], ddof=1)) if any(p < 0 for p in profits) else 0.0
        sortino_ratio = float(np.mean(profits) / downside_deviation * np.sqrt(252)) if downside_deviation != 0 else float("inf")
        sharpe_ratio = float(np.mean(profits) / np.std(profits, ddof=1) * np.sqrt(252)) if len(profits) > 1 and np.std(profits, ddof=1) != 0 else 0.0

        return {
            "total_profit": total_profit, "win_rate": win_rate, "mae": mae, "max_drawdown": max_drawdown,
            "profit_factor": profit_factor, "sortino_ratio": sortino_ratio, "sharpe_ratio": sharpe_ratio
        }
    except Exception as e:
        logger.error(f"Ошибка расчёта метрик: {e}")
        return {
            "total_profit": 0.0, "win_rate": 0.0, "mae": 0.0, "max_drawdown": 0.0,
            "profit_factor": 0.0, "sortino_ratio": 0.0, "sharpe_ratio": 0.0
        }

def generate_symbol_report(trades: List[Dict[str, Union[str, float]]]) -> str:
    """
    Генерация отчёта по символам.

    Args:
        trades (List[Dict[str, Union[str, float]]]): Список сделок.

    Returns:
        str: Текстовый отчет по символам.
    """
    if not isinstance(trades, list):
        logger.error(f"trades должен быть списком, получен {type(trades)}")
        return ""
    symbol_report = ""
    for symbol in SYMBOLS:
        symbol_trades = [t for t in trades if isinstance(t, dict) and t.get("symbol") == symbol]
        if symbol_trades:
            symbol_profits = np.array([t["profit"] for t in symbol_trades if isinstance(t.get("profit"), (int, float))])
            symbol_win_rate = len([p for p in symbol_profits if p > 0]) / len(symbol_trades) if len(symbol_trades) > 0 else 0.0
            symbol_profit = float(symbol_profits.sum()) if symbol_profits.size > 0 else 0.0
            symbol_report += f"{symbol}: Профит: {symbol_profit:.2f} USDT, Сделок: {len(symbol_trades)}, Win Rate: {symbol_win_rate:.2%}\n"
    return symbol_report.strip()

def generate_action_report(trades: List[Dict[str, Union[str, float]]]) -> str:
    """
    Генерация отчёта по действиям.

    Args:
        trades (List[Dict[str, Union[str, float]]]): Список сделок.

    Returns:
        str: Текстовый отчет по действиям.
    """
    if not isinstance(trades, list):
        logger.error(f"trades должен быть списком, получен {type(trades)}")
        return ""
    action_report = ""
    actions = ["buy_long", "sell_short", "close_long", "close_short", "grid_buy", "grid_sell"]
    for action in actions:
        action_trades = [t for t in trades if isinstance(t, dict) and t.get("action") == action]
        if action_trades:
            action_profits = np.array([t["profit"] for t in action_trades if isinstance(t.get("profit"), (int, float))])
            action_win_rate = len([p for p in action_profits if p > 0]) / len(action_trades) if len(action_trades) > 0 else 0.0
            action_profit = float(action_profits.sum()) if action_profits.size > 0 else 0.0
            action_report += f"{action}: Профит: {action_profit:.2f} USDT, Сделок: {len(action_trades)}, Win Rate: {action_win_rate:.2%}\n"
    return action_report.strip()

def generate_strategy_report(trades: List[Dict[str, Union[str, float]]]) -> str:
    """
    Генерация отчёта по стратегиям.

    Args:
        trades (List[Dict[str, Union[str, float]]]): Список сделок.

    Returns:
        str: Текстовый отчет по стратегиям.
    """
    if not isinstance(trades, list):
        logger.error(f"trades должен быть списком, получен {type(trades)}")
        return ""
    strategy_report = ""
    for strategy in STRATEGIES:
        strategy_trades = [t for t in trades if isinstance(t, dict) and t.get("grok_analysis", {}).get("strategy", "trend") == strategy]
        if strategy_trades:
            strategy_profits = np.array([t["profit"] for t in strategy_trades if isinstance(t.get("profit"), (int, float))])
            strategy_win_rate = len([p for p in strategy_profits if p > 0]) / len(strategy_trades) if len(strategy_trades) > 0 else 0.0
            strategy_profit = float(strategy_profits.sum()) if strategy_profits.size > 0 else 0.0
            strategy_report += f"{strategy}: Профит: {strategy_profit:.2f} USDT, Сделок: {len(strategy_trades)}, Win Rate: {strategy_win_rate:.2%}\n"
    return strategy_report.strip()

def generate_indicator_report(trades: List[Dict[str, Union[str, float]]]) -> str:
    """
    Генерация отчёта по индикаторам.

    Args:
        trades (List[Dict[str, Union[str, float]]]): Список сделок.

    Returns:
        str: Текстовый отчет по индикаторам.
    """
    if not isinstance(trades, list):
        logger.error(f"trades должен быть списком, получен {type(trades)}")
        return ""
    indicator_report = ""
    indicators = ["rsi", "macd", "atr", "volatility"]
    for indicator in indicators:
        indicator_values = [t.get("grok_analysis", {}).get(indicator, 0) for t in trades if isinstance(t, dict) and isinstance(t.get("grok_analysis", {}).get(indicator), (int, float))]
        avg_value = float(np.mean(indicator_values)) if indicator_values else 0.0
        indicator_report += f"{indicator}: Среднее значение: {avg_value:.2f}\n"
    return indicator_report.strip()

async def generate_report(log_file: str = "trade_log.jsonl", period: str = "all") -> str:
    """
    Генерация отчёта по торговле.

    Args:
        log_file (str): Путь к файлу логов (по умолчанию "trade_log.jsonl").
        period (str): Период для отчёта ("1h", "1d", "1w", "1m", "all") (по умолчанию "all").

    Returns:
        str: Текстовый отчёт.
    """
    if not isinstance(log_file, str) or not log_file:
        logger.error(f"log_file должен быть непустой строкой, получен {log_file}")
        return "Ошибка: некорректный путь к файлу логов"
    if not isinstance(period, str) or period not in ["1h", "1d", "1w", "1m", "all"]:
        logger.error(f"period должен быть одним из ['1h', '1d', '1w', '1m', 'all'], получен {period}")
        period = "all"

    try:
        logs = await read_logs(log_file)
        if not logs:
            await send_async_message("Нет данных для генерации отчёта")
            logger.info("Нет данных для генерации отчёта")
            return "Нет данных"

        filtered_logs = await filter_logs(logs, period)
        trades = [log for log in filtered_logs if isinstance(log, dict) and log.get("profit") is not None and log.get("profit") != 0]
        if not trades:
            msg = f"Нет завершённых сделок за период {period}"
            await send_async_message(msg)
            logger.info(msg)
            return msg

        metrics = calculate_metrics(trades)
        symbol_report = generate_symbol_report(trades)
        action_report = generate_action_report(trades)
        strategy_report = generate_strategy_report(trades)
        indicator_report = generate_indicator_report(trades)

        report = (
            f"📊 Отчет о торговле ({period}):\n"
            f"Общий профит: {metrics['total_profit']:.2f} USDT\n"
            f"Количество сделок: {len(trades)}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"MAE предсказания: {metrics['mae']:.4f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2f} USDT\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"---\nПо символам:\n{symbol_report}\n"
            f"---\nПо действиям:\n{action_report}\n"
            f"---\nПо стратегиям:\n{strategy_report}\n"
            f"---\nПо индикаторам:\n{indicator_report}"
        )
        
        await send_async_message(report)
        logger.info(f"Сгенерирован отчёт: {report[:100]}...")

        try:
            # Используем trades вместо filtered_logs для генерации графиков сделок
            if trades:
                df = pd.DataFrame(trades)
                await generate_plots(df, "all_symbols", period, send_to_telegram=True)
                logger.info("Визуализация логов успешно выполнена")
            else:
                logger.warning("Нет данных сделок для визуализации")
        except Exception as e:
            error_msg = f"Ошибка визуализации логов: {e}"
            logger.error(error_msg)
            await send_async_message(error_msg)

        return report
    except Exception as e:
        logger.error(f"Ошибка генерации отчета: {e}")
        return f"Ошибка генерации отчета: {e}"

async def auto_generate_reports(period: str = "1h") -> None:
    """
    Автоматическая генерация ежечасных отчётов.

    Args:
        period (str): Период для отчетов ("1h", "1d", "1w", "1m", "all") (по умолчанию "1h").
    """
    if not isinstance(period, str) or period not in ["1h", "1d", "1w", "1m", "all"]:
        logger.error(f"period должен быть одним из ['1h', '1d', '1w', '1m', 'all'], получен {period}")
        period = "1h"
    while True:
        try:
            now = datetime.now()
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            sleep_seconds = (next_run - now).total_seconds()
            await asyncio.sleep(sleep_seconds)
            await generate_report(period=period)
            logger.info("Ежечасный отчёт выполнен")
        except Exception as e:
            logger.error(f"Ошибка в цикле автоматической генерации отчетов: {e}")
            await asyncio.sleep(3600)  # Задержка 1 час перед следующей попыткой

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(generate_report())
        loop.create_task(auto_generate_reports())
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("Программа завершена по сигналу прерывания")
        loop.close()
    except Exception as e:
        logger.error(f"Ошибка при запуске программы: {e}")
        loop.close()