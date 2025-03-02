# start_bot.py
import asyncio
import subprocess
import sys
from typing import Optional
from datetime import datetime
import aiofiles
import os
import logging

# Попытка импорта зависимостей с базовым логгером
try:
    from logging_setup import setup_logging
    logger = setup_logging("bot_start_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("start_bot")
    logger.warning(f"Не удалось импортировать logging_setup: {e}")

try:
    from telegram_utils import send_async_message
except ImportError as e:
    logger.error(f"Не удалось импортировать telegram_utils: {e}")
    async def send_async_message(msg: str) -> None:
        logger.warning(f"Telegram уведомления отключены: {msg}")

async def check_python() -> bool:
    """
    Проверка наличия и версии установленного Python.

    Returns:
        bool: True, если Python версии 3.8+ доступен, иначе False.

    Raises:
        Exception: Если возникла ошибка при проверке версии Python.
    """
    try:
        # Используем sys.executable для текущего интерпретатора Python
        process = await asyncio.create_subprocess_exec(
            sys.executable, "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        version_output = stdout.decode().strip() or stderr.decode().strip()
        if not version_output.startswith("Python"):
            raise ValueError(f"Некорректный вывод версии Python: {version_output}")
        
        version_str = version_output.split()[1]
        version = tuple(map(int, version_str.split('.')))
        min_version = (3, 8)
        if version < min_version:
            logger.error(f"Требуется Python 3.8+, найдена версия: {version_str}")
            await send_async_message(f"⚠️ Требуется Python 3.8+, найдена версия: {version_str}")
            return False
        
        logger.info(f"Python найден: {version_str}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
        logger.error(f"Ошибка проверки Python: {e}", exc_info=True)
        await send_async_message(f"⚠️ Ошибка проверки Python: {e}")
        return False

async def start_bot() -> None:
    """
    Асинхронный запуск и перезапуск бота.

    Raises:
        Exception: Если возникла критическая ошибка при запуске процесса.
    """
    if not await check_python():
        return

    attempt = 0
    max_attempts = 5
    min_delay = 5  # Минимальная задержка в секундах
    while attempt < max_attempts:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bot_log = f"bot_log_{timestamp}.txt"
        logger.info(f"Запуск бота, логи в {bot_log}")

        if not os.path.exists("main.py"):
            logger.error("Файл main.py не найден")
            await send_async_message("⚠️ Файл main.py не найден")
            break

        try:
            # Асинхронное открытие файла логов
            async with aiofiles.open(bot_log, "w", encoding="utf-8") as log_file:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "main.py",
                    stdout=log_file.file,
                    stderr=subprocess.STDOUT
                )
                returncode = await process.wait()

            if returncode != 0:
                attempt += 1
                logger.error(f"Бот завершился с ошибкой (попытка {attempt}/{max_attempts}). Проверьте {bot_log}.")
                await send_async_message(f"⚠️ Ошибка запуска бота (попытка {attempt}/{max_attempts}). Проверьте {bot_log}")
                if attempt >= max_attempts:
                    logger.error("Превышено максимальное количество попыток. Завершение.")
                    await send_async_message("🚨 Превышено максимальное количество попыток запуска.")
                    break
                delay = max(min_delay, min(300 * (2 ** (attempt - 1)), 3600))  # Экспоненциальный backoff с минимумом
                logger.info(f"Перезапуск через {delay} секунд...")
                await asyncio.sleep(delay)
            else:
                logger.info(f"Бот успешно завершил работу. Логи в {bot_log}")
                await send_async_message(f"✅ Бот завершил работу. Логи: {bot_log}")
                break

        except Exception as e:
            logger.error(f"Ошибка запуска процесса бота: {e}", exc_info=True)
            await send_async_message(f"⚠️ Ошибка запуска процесса бота: {e}")
            attempt += 1
            if attempt >= max_attempts:
                logger.error("Превышено максимальное количество попыток. Завершение.")
                await send_async_message("🚨 Превышено максимальное количество попыток запуска.")
                break
            delay = max(min_delay, min(300 * (2 ** (attempt - 1)), 3600))
            await asyncio.sleep(delay)

if __name__ == "__main__":
    asyncio.run(start_bot())