# logging_setup.py
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
import os

# Константы
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_BACKUP_COUNT = 3
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def setup_logging(filename: str, level: int = DEFAULT_LOG_LEVEL, 
                  max_bytes: int = DEFAULT_MAX_BYTES, 
                  backup_count: int = DEFAULT_BACKUP_COUNT) -> logging.Logger:
    """
    Настраивает логирование с ротацией файлов и выводом в консоль.

    Args:
        filename (str): Путь к файлу логов.
        level (int): Уровень логирования (по умолчанию logging.INFO).
        max_bytes (int): Максимальный размер файла логов в байтах (по умолчанию 5 MB).
        backup_count (int): Количество резервных копий (по умолчанию 3).

    Returns:
        logging.Logger: Настроенный объект логгера с именем, основанным на базовом имени файла.

    Raises:
        ValueError: Если аргументы имеют некорректные типы или значения.
        OSError: Если не удается создать или открыть файл логов.
    """
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"filename должен быть непустой строкой, получен {filename}")
    if not isinstance(level, int) or level not in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
        raise ValueError(f"level должен быть одним из уровней logging, получен {level}")
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError(f"max_bytes должен быть положительным числом, получен {max_bytes}")
    if not isinstance(backup_count, int) or backup_count < 0:
        raise ValueError(f"backup_count должен быть неотрицательным числом, получен {backup_count}")

    try:
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        handler = RotatingFileHandler(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
    except OSError as e:
        logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger(__name__)
        logger.error(f"Не удалось создать RotatingFileHandler для {filename}: {e}")
        raise

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    handler.setFormatter(formatter)
    
    # Используем базовое имя файла без расширения как имя логгера
    logger_name = os.path.splitext(os.path.basename(filename))[0]
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Проверяем и добавляем обработчики только при необходимости
    has_file_handler = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
    has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    
    if not has_file_handler:
        logger.addHandler(handler)
    if not has_stream_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    try:
        logger = setup_logging("test_log.txt")
        logger.info("Тестовое сообщение")
        logger.error("Тестовая ошибка")
    except Exception as e:
        print(f"Ошибка настройки логирования: {e}")