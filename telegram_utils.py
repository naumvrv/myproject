# telegram_utils.py
import asyncio
from aiogram import Bot
from typing import Optional, Dict, Union, Any
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("telegram_utils_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("telegram_utils")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from config_loader import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
except ImportError as e:
    logger.error(f"Не удалось импортировать config_loader: {e}")
    TELEGRAM_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    raise SystemExit(1)

class TelegramNotifier:
    def __init__(self, token: str, default_chat_id: str):
        """
        Инициализация уведомлений через Telegram.

        Args:
            token (str): Токен Telegram бота.
            default_chat_id (str): Идентификатор чата по умолчанию.

        Raises:
            ValueError: Если token или default_chat_id некорректны.
            Exception: Если не удалось создать объект Bot.
        """
        if not isinstance(token, str) or not token:
            raise ValueError(f"token должен быть непустой строкой, получено {token}")
        if not isinstance(default_chat_id, str) or not default_chat_id:
            raise ValueError(f"default_chat_id должен быть непустой строкой, получено {default_chat_id}")

        try:
            self.bot = Bot(token=token)
            self.default_chat_id = int(default_chat_id)
        except ValueError:
            raise ValueError(f"Некорректный default_chat_id: {default_chat_id}, должен быть числом")
        except Exception as e:
            logger.error(f"Ошибка создания объекта Bot: {e}")
            raise

        self.max_retries = 5
        self.initial_delay = 2  # Начальная задержка 2 секунды
        self.max_delay = 60  # Максимальная задержка 60 секунд

    def escape_markdown_v2(self, text: Any) -> str:
        """
        Экранирование специальных символов для MarkdownV2.

        Args:
            text (Any): Исходный текст (будет приведен к строке).

        Returns:
            str: Экранированный текст.
        """
        if not isinstance(text, (str, int, float)):
            logger.warning(f"text должен быть строкой или числом, получено {type(text)}, приведено к строке")
        text_str = str(text)
        special_chars = r"_*[]()~`>#+-=|{}.!"
        return "".join([f"\\{c}" if c in special_chars else c for c in text_str])

    async def send_message(self, message: str, chat_id: Optional[int] = None, balance: Optional[float] = None, 
                           positions: Optional[Dict[str, str]] = None, photo: Optional[bytes] = None, 
                           parse_mode: str = "MarkdownV2") -> bool:
        """
        Асинхронная отправка сообщения в Telegram.

        Args:
            message (str): Текст сообщения.
            chat_id (Optional[int]): Идентификатор чата (по умолчанию используется default_chat_id).
            balance (Optional[float]): Баланс для добавления в сообщение (опционально).
            positions (Optional[Dict[str, str]]): Позиции для добавления в сообщение (опционально).
            photo (Optional[bytes]): Байты изображения для отправки (опционально).
            parse_mode (str): Режим форматирования ("MarkdownV2" или другой, по умолчанию "MarkdownV2").

        Returns:
            bool: True, если отправка успешна, иначе False.

        Raises:
            ValueError: Если аргументы имеют некорректный тип или значение.
        """
        if not isinstance(message, str) or not message:
            logger.error(f"message должен быть непустой строкой, получено {message}")
            return False
        if chat_id is not None and not isinstance(chat_id, int):
            logger.error(f"chat_id должен быть целым числом или None, получено {chat_id}")
            return False
        if balance is not None and not isinstance(balance, (int, float)):
            logger.error(f"balance должен быть числом или None, получено {balance}")
            return False
        if positions is not None and not isinstance(positions, dict):
            logger.error(f"positions должен быть словарем или None, получено {positions}")
            return False
        if photo is not None and not isinstance(photo, bytes):
            logger.error(f"photo должен быть байтами или None, получено {type(photo)}")
            return False
        if not isinstance(parse_mode, str):
            logger.error(f"parse_mode должен быть строкой, получено {parse_mode}")
            parse_mode = "MarkdownV2"

        chat_id = chat_id or self.default_chat_id
        full_message = str(message)
        if balance is not None:
            full_message += f"\n**Баланс:** {balance:.2f} USDT"
        if positions:
            full_message += f"\n**Позиции:** {positions}"

        if parse_mode == "MarkdownV2":
            full_message = self.escape_markdown_v2(full_message)

        # Разбиение сообщения на части, если оно превышает 4096 символов
        parts = [full_message[i:i+4093] + "..." for i in range(0, len(full_message), 4093)] if len(full_message) > 4096 else [full_message]
        success = True

        for i, part in enumerate(parts):
            for attempt in range(self.max_retries):
                try:
                    if photo and i == 0:  # Отправляем фото только с первой частью
                        await asyncio.wait_for(
                            self.bot.send_photo(chat_id=chat_id, photo=photo, caption=part, parse_mode=parse_mode),
                            timeout=10
                        )
                    else:
                        await asyncio.wait_for(
                            self.bot.send_message(chat_id=chat_id, text=part, parse_mode=parse_mode),
                            timeout=10
                        )
                    logger.info(f"Сообщение отправлено в чат {chat_id}: {part[:50]}...")
                    break
                except asyncio.TimeoutError:
                    logger.error(f"Таймаут отправки сообщения в чат {chat_id} (попытка {attempt+1}/{self.max_retries})")
                    if attempt == self.max_retries - 1:
                        await self.send_error_message(f"⚠️ Таймаут отправки сообщения после {self.max_retries} попыток", chat_id)
                        success = False
                    else:
                        delay = min(self.initial_delay * (2 ** attempt), self.max_delay)
                        await asyncio.sleep(delay)
                except Exception as e:
                    logger.error(f"Ошибка отправки сообщения в чат {chat_id} (попытка {attempt+1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        error_msg = self.escape_markdown_v2(f"⚠️ Не удалось отправить сообщение после {self.max_retries} попыток: {str(e)}")
                        await self.send_error_message(error_msg, chat_id)
                        success = False
                    else:
                        delay = min(self.initial_delay * (2 ** attempt), self.max_delay)
                        await asyncio.sleep(delay)
        return success

    async def send_error_message(self, error_message: str, chat_id: Optional[int] = None) -> None:
        """
        Асинхронная отправка сообщения об ошибке.

        Args:
            error_message (str): Текст сообщения об ошибке.
            chat_id (Optional[int]): Идентификатор чата (по умолчанию используется default_chat_id).
        """
        if not isinstance(error_message, str) or not error_message:
            logger.error(f"error_message должен быть непустой строкой, получено {error_message}")
            return
        if chat_id is not None and not isinstance(chat_id, int):
            logger.error(f"chat_id должен быть целым числом или None, получено {chat_id}")
            return

        chat_id = chat_id or self.default_chat_id
        parse_mode = "MarkdownV2"
        full_message = self.escape_markdown_v2(error_message)
        try:
            await asyncio.wait_for(
                self.bot.send_message(chat_id=chat_id, text=full_message, parse_mode=parse_mode),
                timeout=10
            )
            logger.info(f"Сообщение об ошибке отправлено в чат {chat_id}")
        except Exception as e:
            logger.error(f"Не удалось отправить сообщение об ошибке в чат {chat_id}: {e}")

    async def close(self) -> None:
        """Закрытие сессии Telegram."""
        try:
            if self.bot.session and not self.bot.session.closed:
                await self.bot.session.close()
                logger.info("Сессия Telegram закрыта")
        except Exception as e:
            logger.error(f"Ошибка закрытия сессии Telegram: {e}")

async def send_async_message(message: str, token: Optional[str] = None, chat_id: Optional[str] = None, 
                            balance: Optional[float] = None, positions: Optional[Dict[str, str]] = None, 
                            photo: Optional[bytes] = None, parse_mode: str = "MarkdownV2") -> bool:
    """
    Глобальная функция для асинхронной отправки сообщения в Telegram.

    Args:
        message (str): Текст сообщения.
        token (Optional[str]): Токен Telegram бота (по умолчанию из конфигурации).
        chat_id (Optional[str]): Идентификатор чата (по умолчанию из конфигурации).
        balance (Optional[float]): Баланс для добавления в сообщение.
        positions (Optional[Dict[str, str]]): Позиции для добавления в сообщение.
        photo (Optional[bytes]): Байты изображения.
        parse_mode (str): Режим форматирования (по умолчанию "MarkdownV2").

    Returns:
        bool: True, если отправка успешна, иначе False.
    """
    if not isinstance(message, str) or not message:
        logger.error(f"message должен быть непустой строкой, получено {message}")
        return False
    if token is not None and not isinstance(token, str):
        logger.error(f"token должен быть строкой или None, получено {token}")
        return False
    if chat_id is not None and not isinstance(chat_id, str):
        logger.error(f"chat_id должен быть строкой или None, получено {chat_id}")
        return False
    if balance is not None and not isinstance(balance, (int, float)):
        logger.error(f"balance должен быть числом или None, получено {balance}")
        return False
    if positions is not None and not isinstance(positions, dict):
        logger.error(f"positions должен быть словарем или None, получено {positions}")
        return False
    if photo is not None and not isinstance(photo, bytes):
        logger.error(f"photo должен быть байтами или None, получено {type(photo)}")
        return False
    if not isinstance(parse_mode, str):
        logger.error(f"parse_mode должен быть строкой, получено {parse_mode}")
        parse_mode = "MarkdownV2"

    token = token or TELEGRAM_TOKEN
    chat_id = chat_id or TELEGRAM_CHAT_ID
    try:
        notifier = TelegramNotifier(token=token, default_chat_id=chat_id)
        result = await notifier.send_message(message, chat_id, balance, positions, photo, parse_mode)
        await notifier.close()
        return result
    except ValueError as e:
        logger.error(f"Ошибка инициализации Telegram: {e}")
        return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка в send_async_message: {e}")
        return False

async def send_strategy_update(strategy: str, token: Optional[str] = None, chat_id: Optional[str] = None) -> None:
    """
    Отправка уведомления об изменении стратегии.

    Args:
        strategy (str): Название стратегии.
        token (Optional[str]): Токен Telegram бота (по умолчанию из конфигурации).
        chat_id (Optional[str]): Идентификатор чата (по умолчанию из конфигурации).
    """
    if not isinstance(strategy, str) or not strategy:
        logger.error(f"strategy должен быть непустой строкой, получено {strategy}")
        return
    message = f"✅ Выбрана новая стратегия: {strategy}"
    await send_async_message(message, token, chat_id)

async def send_indicator_update(indicator: str, enabled: bool, token: Optional[str] = None, 
                               chat_id: Optional[str] = None) -> None:
    """
    Отправка уведомления об изменении индикатора.

    Args:
        indicator (str): Название индикатора.
        enabled (bool): Статус индикатора (включён/выключен).
        token (Optional[str]): Токен Telegram бота (по умолчанию из конфигурации).
        chat_id (Optional[str]): Идентификатор чата (по умолчанию из конфигурации).
    """
    if not isinstance(indicator, str) or not indicator:
        logger.error(f"indicator должен быть непустой строкой, получено {indicator}")
        return
    if not isinstance(enabled, bool):
        logger.error(f"enabled должен быть булевым значением, получено {enabled}")
        return
    status = "включён" if enabled else "выключен"
    message = f"⚙️ Индикатор {indicator} {status}"
    await send_async_message(message, token, chat_id)

if __name__ == "__main__":
    async def test():
        await send_async_message("Тестовое сообщение")
    asyncio.run(test())