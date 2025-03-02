# visualize_logs.py
import asyncio
import json
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import aiofiles
import os
import logging

# Попытка импорта зависимостей с базовым логгером
try:
    from logging_setup import setup_logging
    logger = setup_logging("visualize_logs_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("visualize_logs")
    logger.warning(f"Не удалось импортировать logging_setup: {e}")

try:
    from config_loader import SYMBOLS
    from cache_manager import get_cached_data, save_cached_data
    from telegram_utils import send_async_message
    from plot_utils import generate_plots
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    async def send_async_message(msg: str, **kwargs) -> bool:
        logger.warning(f"Telegram уведомления отключены: {msg}")
        return False
    raise SystemExit(1)

async def read_logs(log_file: str = "trade_log.jsonl") -> List[Dict[str, Union[str, float]]]:
    """
    Асинхронное чтение логов из файла.

    Args:
        log_file (str): Путь к файлу логов (по умолчанию "trade_log.jsonl").

    Returns:
        List[Dict[str, Union[str, float]]]: Список словарей с данными логов.

    Raises:
        ValueError: Если log_file не является строкой или пустой.
    """
    if not isinstance(log_file, str) or not log_file:
        logger.error(f"log_file должен быть непустой строкой, получено {log_file}")
        raise ValueError(f"log_file должен быть непустой строкой, получено {log_file}")

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
                        logger.error(f"Ошибка разбора строки лога: {e}", exc_info=True)
        if not logs:
            await send_async_message("Нет данных для визуализации")
            logger.info("Нет данных для визуализации")
        return logs
    except Exception as e:
        logger.error(f"Ошибка чтения логов ({log_file}): {e}", exc_info=True)
        await send_async_message(f"⚠️ Ошибка чтения логов: {e}")
        return []

async def visualize_logs(log_file: str = "trade_log.jsonl", period: str = "all", 
                         symbol: Optional[str] = None, send_to_telegram: bool = True) -> Optional[Dict[str, Any]]:
    """
    Генерация визуализации логов с использованием Plotly и кэшированием.

    Args:
        log_file (str): Путь к файлу логов (по умолчанию "trade_log.jsonl").
        period (str): Период для визуализации ("all", "1h", "1d", "1w", "1m"; по умолчанию "all").
        symbol (Optional[str]): Символ для фильтрации (опционально, по умолчанию None).
        send_to_telegram (bool): Отправлять ли визуализацию в Telegram (по умолчанию True).

    Returns:
        Optional[Dict[str, Any]]: Словарь с графиками или None при ошибке.
    """
    if not isinstance(log_file, str) or not log_file:
        logger.error(f"log_file должен быть непустой строкой, получено {log_file}")
        return None
    if not isinstance(period, str) or period not in ["all", "1h", "1d", "1w", "1m"]:
        logger.error(f"period должен быть одним из ['all', '1h', '1d', '1w', '1m'], получено {period}")
        return None
    if symbol is not None and (not isinstance(symbol, str) or symbol not in SYMBOLS):
        logger.error(f"symbol должен быть строкой из SYMBOLS или None, получено {symbol}")
        return None
    if not isinstance(send_to_telegram, bool):
        logger.error(f"send_to_telegram должен быть булевым значением, получено {send_to_telegram}")
        return None

    try:
        cache_key = f"visualization:{symbol or 'all'}:{period}"
        cached_data = await get_cached_data(cache_key)
        if cached_data and isinstance(cached_data, dict) and "figures" in cached_data and "output_file" in cached_data:
            logger.info(f"Использованы кэшированные графики для {cache_key}")
            if send_to_telegram and os.path.exists(cached_data["output_file"]):
                async with aiofiles.open(cached_data["output_file"], "rb") as photo:
                    await send_async_message(f"Графики за период {period} для {symbol or 'всех символов'} (из кэша)", 
                                            photo=await photo.read())
            return cached_data["figures"]

        logs = await read_logs(log_file)
        if not logs:
            return None

        # Фильтрация логов по периоду
        time_deltas = {"1h": timedelta(hours=1), "1d": timedelta(days=1), "1w": timedelta(weeks=1), "1m": timedelta(days=30)}
        cutoff = datetime.now() - time_deltas.get(period, timedelta(days=0)) if period != "all" else None
        if cutoff:
            logs = [log for log in logs if datetime.fromisoformat(log["timestamp"]) >= cutoff]

        # Фильтрация по символу или группировка по всем символам
        if symbol:
            filtered_logs = [log for log in logs if log.get("symbol") == symbol]
            target_symbol = symbol
        else:
            symbol_data = {sym: [log for log in logs if log.get("symbol") == sym] for sym in SYMBOLS}
            filtered_logs = [log for sym_logs in symbol_data.values() for log in sym_logs if sym_logs]
            target_symbol = "all_symbols"

        if not filtered_logs:
            msg = f"Нет данных для визуализации за период {period} и символ {symbol or 'всех символов'}"
            await send_async_message(msg)
            logger.info(msg)
            return None

        # Генерация графиков
        figures = await generate_plots(filtered_logs, target_symbol, period, send_to_telegram=False)
        if not figures or not isinstance(figures, dict):
            logger.error(f"Ошибка генерации графиков для {target_symbol}")
            return None

        # Сохранение и отправка в Telegram
        output_file = f"trade_visualization_{target_symbol}_{period}_{int(datetime.now().timestamp())}.png"
        profit_fig = figures.get("profit")
        if profit_fig:
            profit_fig.write_image(output_file)
            if send_to_telegram:
                async with aiofiles.open(output_file, "rb") as photo:
                    await send_async_message(f"Графики за период {period} для {symbol or 'всех символов'}", 
                                            photo=await photo.read())
            await save_cached_data(cache_key, {"figures": figures, "output_file": output_file}, ttl=3600)
            logger.info(f"Визуализация завершена для {log_file}, period={period}, symbol={symbol}")
            return figures
        else:
            logger.error(f"Не удалось получить график 'profit' для {target_symbol}")
            return None

    except Exception as e:
        logger.error(f"Ошибка визуализации логов: {e}", exc_info=True)
        await send_async_message(f"⚠️ Ошибка визуализации логов: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(visualize_logs())