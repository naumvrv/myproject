# telegram_handler.py
import telebot
import asyncio
import pandas as pd
from typing import Optional, Dict, List, Union, Any
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import aiohttp
import logging

# Попытка импорта зависимостей с базовым логгером
try:
    from logging_setup import setup_logging
    logger = setup_logging("telegram_handler_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("telegram_handler")
    logger.warning(f"Не удалось импортировать logging_setup: {e}")

try:
    from config_loader import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS, MAX_DAILY_LOSS, MAX_WEEKLY_LOSS, STRATEGIES
    from telegram_utils import send_async_message
    from feature_engineer import FeatureEngineer
    from grok3_analyze import grok3_analyze
    from generate_report import generate_report
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    async def send_async_message(msg: str, **kwargs) -> bool:
        logger.warning(f"Telegram уведомления отключены: {msg}")
        return False
    raise SystemExit(1)

class TelegramHandler:
    def __init__(self, executor: Optional[Any], exchange: Optional[Any], webhook_url: Optional[str] = None):
        """
        Инициализация обработчика Telegram.

        Args:
            executor (Optional[Any]): Объект TradeExecutor (опционально).
            exchange (Optional[Any]): Объект биржи (опционально).
            webhook_url (Optional[str]): URL для Webhook (по умолчанию None, требуется реальный URL!).

        Raises:
            ValueError: Если TELEGRAM_TOKEN или TELEGRAM_CHAT_ID некорректны.
            Exception: Если не удалось создать объект бота.
        """
        if not isinstance(TELEGRAM_TOKEN, str) or not TELEGRAM_TOKEN:
            raise ValueError(f"TELEGRAM_TOKEN должен быть непустой строкой, получено {TELEGRAM_TOKEN}")
        if not isinstance(TELEGRAM_CHAT_ID, str) or not TELEGRAM_CHAT_ID:
            raise ValueError(f"TELEGRAM_CHAT_ID должен быть непустой строкой, получено {TELEGRAM_CHAT_ID}")
        if webhook_url is not None and not isinstance(webhook_url, str):
            logger.error(f"webhook_url должен быть строкой или None, получено {type(webhook_url)}")
            webhook_url = None

        self.executor = executor
        self.exchange = exchange
        try:
            self.telegram_bot = telebot.TeleBot(TELEGRAM_TOKEN)
        except Exception as e:
            logger.error(f"Ошибка создания объекта Telegram бота: {e}", exc_info=True)
            raise
        self.paused = False
        try:
            self.authorized_chat_id = int(TELEGRAM_CHAT_ID)
        except ValueError:
            raise ValueError(f"Некорректный TELEGRAM_CHAT_ID: {TELEGRAM_CHAT_ID}, должен быть числом")
        self.loop = asyncio.get_event_loop()
        self.webhook_url = webhook_url or None  # Требуется реальный URL для Webhook
        self.start_time: float = 0.0
        self.last_balance: Optional[float] = None
        self.setup_handlers()
        if self.webhook_url:
            self.set_webhook()

    def set_webhook(self) -> None:
        """Установка Webhook для Telegram бота."""
        if not self.webhook_url:
            logger.warning("Webhook URL не указан, пропуск установки")
            return
        try:
            self.telegram_bot.remove_webhook()
            self.telegram_bot.set_webhook(url=self.webhook_url)
            logger.info(f"Webhook установлен: {self.webhook_url}")
        except Exception as e:
            logger.error(f"Ошибка установки Webhook: {e}", exc_info=True)

    def setup_handlers(self) -> None:
        """Настройка обработчиков команд и callback-запросов."""
        @self.telegram_bot.message_handler(commands=["start", "menu"])
        def send_menu(message):
            if not isinstance(message.chat.id, int):
                logger.error(f"message.chat.id должен быть целым числом, получено {type(message.chat.id)}")
                return
            if message.chat.id != self.authorized_chat_id:
                self.telegram_bot.reply_to(message, "⛔ Вы не авторизованы для управления ботом.")
                return
            asyncio.run_coroutine_threadsafe(self.send_main_menu(message.chat.id, None), self.loop)

        @self.telegram_bot.callback_query_handler(func=lambda call: True)
        def callback_handler(call):
            if not isinstance(call.message.chat.id, int):
                logger.error(f"call.message.chat.id должен быть целым числом, получено {type(call.message.chat.id)}")
                return
            if call.message.chat.id != self.authorized_chat_id:
                self.telegram_bot.answer_callback_query(call.id, "⛔ Вы не авторизованы.")
                return
            asyncio.run_coroutine_threadsafe(self.handle_callback(call), self.loop)

    async def handle_callback(self, call: telebot.types.CallbackQuery) -> None:
        """Обработка callback-запросов от Telegram.

        Args:
            call (telebot.types.CallbackQuery): Объект callback-запроса.
        """
        if not isinstance(call, telebot.types.CallbackQuery):
            logger.error(f"call должен быть экземпляром CallbackQuery, получено {type(call)}")
            return
        chat_id = call.message.chat.id
        message_id = call.message.message_id
        data = call.data
        
        if not isinstance(data, str):
            logger.error(f"data должен быть строкой, получено {type(data)}")
            return

        try:
            if data == "status":
                await self._handle_status(chat_id, message_id)
            elif data == "pause":
                await self._handle_pause(chat_id, message_id)
            elif data == "resume":
                await self._handle_resume(chat_id, message_id)
            elif data == "set_leverage":
                await self._handle_set_leverage_menu(chat_id, message_id)
            elif data.startswith("leverage_"):
                await self._handle_set_leverage(chat_id, message_id, data)
            elif data == "report":
                await self._handle_report_menu(chat_id, message_id)
            elif data.startswith("report_"):
                await self._handle_report(chat_id, message_id, data)
            elif data == "predict":
                await self._handle_predict(chat_id, message_id)
            elif data == "balance":
                await self._handle_balance(chat_id, message_id)
            elif data == "close_all":
                await self._handle_close_all(chat_id, message_id)
            elif data == "restart":
                await self._handle_restart(chat_id, message_id)
            elif data == "graphs":
                await self._handle_graphs(chat_id, message_id)
            elif data == "logs":
                await self._handle_logs(chat_id, message_id)
            elif data == "back":
                await self.send_main_menu(chat_id, message_id)
            elif data == "adjust_amount":
                await self._handle_adjust_amount_menu(chat_id, message_id)
            elif data in ["increase_amount", "decrease_amount"]:
                await self._handle_adjust_amount(chat_id, message_id, data)
            elif data == "set_strategy":
                await self._handle_set_strategy_menu(chat_id, message_id)
            elif data.startswith("strategy_"):
                await self._handle_set_strategy(chat_id, message_id, data)
            elif data == "manage_indicators":
                await self._handle_manage_indicators_menu(chat_id, message_id)
            elif data.startswith("ind_"):
                await self._handle_manage_indicators(chat_id, message_id, data)
        except Exception as e:
            logger.error(f"Ошибка в обработке callback: {e}", exc_info=True)
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=f"Ошибка: {e}")
            await send_async_message(f"⚠️ Ошибка в Telegram: {e}")

    async def _handle_status(self, chat_id: int, message_id: int) -> None:
        """Обработка команды 'status'."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        if not self.executor or not self.exchange:
            logger.error("Executor или Exchange не инициализированы")
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Ошибка: Executor или Exchange не инициализированы")
            return

        balance = await self.fetch_balance_with_retry()
        symbol = list(SYMBOLS.keys())[0]
        price = self.executor.entry_prices.get(symbol, 0) if self.executor.positions.get(symbol) else 0
        ohlcv = await self.fetch_ohlcv_with_retry(symbol)
        additional_data = {"funding_rate": 0, "open_interest": 0, "spread": 0}
        
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            features = FeatureEngineer().prepare_features({symbol: df}, [symbol], {symbol: additional_data})[symbol][2]
            current_price = df["close"].iloc[-1]
            atr = features["atr"].iloc[-1] if "atr" in features else 0
            rsi = features["rsi"].iloc[-1] if "rsi" in features else 50
        else:
            current_price, atr, rsi = price, 0, 50

        msg = (
            f"Статус: {'Пауза' if self.paused else 'Работает'}\n"
            f"Позиции: {self.executor.positions}\n"
            f"Текущая цена: {current_price:.2f} USDT\n"
            f"ATR: {atr:.2f}\n"
            f"RSI: {rsi:.2f}\n"
            f"Дневной профит: {self.executor.daily_profit:.2f} USDT\n"
            f"Недельный профит: {self.executor.weekly_profit:.2f} USDT\n"
            f"Баланс: {balance:.2f} USDT\n"
            f"Время работы: {int(asyncio.get_event_loop().time() - self.start_time)} сек"
        )
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=msg)

    async def _handle_pause(self, chat_id: int, message_id: int) -> None:
        """Обработка команды 'pause'."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        self.paused = True
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Бот приостановлен.")

    async def _handle_resume(self, chat_id: int, message_id: int) -> None:
        """Обработка команды 'resume'."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        self.paused = False
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Бот возобновил работу.")

    async def _handle_set_leverage_menu(self, chat_id: int, message_id: int) -> None:
        """Обработка меню установки плеча."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        markup = InlineKeyboardMarkup()
        markup.add(
            InlineKeyboardButton("5x", callback_data="leverage_5"),
            InlineKeyboardButton("10x", callback_data="leverage_10"),
            InlineKeyboardButton("20x", callback_data="leverage_20"),
            InlineKeyboardButton("50x", callback_data="leverage_50"),
            InlineKeyboardButton("Назад", callback_data="back")
        )
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Выберите плечо:", reply_markup=markup)

    async def _handle_set_leverage(self, chat_id: int, message_id: int, data: str) -> None:
        """Обработка установки конкретного значения плеча."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int) or not isinstance(data, str):
            logger.error(f"chat_id, message_id и data должны быть корректными, получено {chat_id}, {message_id}, {data}")
            return
        if not self.executor:
            logger.error("Executor не инициализирован")
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Ошибка: Executor не инициализирован")
            return

        try:
            leverage = int(data.split("_")[1])
            self.executor.leverage = leverage
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=f"Плечо установлено: {leverage}x")
        except (IndexError, ValueError) as e:
            logger.error(f"Ошибка установки плеча: {e}", exc_info=True)
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=f"Ошибка установки плеча: {e}")

    async def _handle_report_menu(self, chat_id: int, message_id: int) -> None:
        """Обработка меню отчёта."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        markup = InlineKeyboardMarkup()
        markup.add(
            InlineKeyboardButton("За час", callback_data="report_1h"),
            InlineKeyboardButton("За день", callback_data="report_1d"),
            InlineKeyboardButton("За неделю", callback_data="report_1w"),
            InlineKeyboardButton("Всё время", callback_data="report_all"),
            InlineKeyboardButton("Назад", callback_data="back")
        )
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Выберите период для отчёта:", reply_markup=markup)

    async def _handle_report(self, chat_id: int, message_id: int, data: str) -> None:
        """Обработка генерации отчёта."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int) or not isinstance(data, str):
            logger.error(f"chat_id, message_id и data должны быть корректными, получено {chat_id}, {message_id}, {data}")
            return
        try:
            period = data.split("_")[1]
            report = await generate_report(period=period)
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=report)
        except (IndexError, ValueError) as e:
            logger.error(f"Ошибка генерации отчета: {e}", exc_info=True)
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=f"Ошибка генерации отчета: {e}")

    async def _handle_predict(self, chat_id: int, message_id: int) -> None:
        """Обработка предсказания."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        if not self.exchange:
            logger.error("Exchange не инициализирован")
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Ошибка: Exchange не инициализирован")
            return

        symbol = list(SYMBOLS.keys())[0]
        ohlcv = await self.fetch_ohlcv_with_retry(symbol)
        additional_data = {"funding_rate": 0, "open_interest": 0, "spread": 0}
        
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            features = FeatureEngineer().prepare_features({symbol: df}, [symbol], {symbol: additional_data})[symbol][2]
            grok_analysis = await grok3_analyze(df["close"].iloc[-1], {
                "volatility": features["volatility"].iloc[-1] if "volatility" in features else 0,
                "rsi": features["rsi"].iloc[-1] if "rsi" in features else 50,
                "macd": features["macd"].iloc[-1] if "macd" in features else 0,
                "adx": features["adx"].iloc[-1] if "adx" in features else 25,
                "atr": features["atr"].iloc[-1] if "atr" in features else 0
            })
            msg = (
                f"Предсказание для {symbol}:\n"
                f"Текущая цена: {df['close'].iloc[-1]:.2f} USDT\n"
                f"Sentiment: {grok_analysis['sentiment']}\n"
                f"Уверенность: {grok_analysis['confidence']:.2%}\n"
                f"Предсказанное изменение: {grok_analysis['predicted_change']:.2f}"
            )
        else:
            msg = "Недостаточно данных для предсказания"
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=msg)

    async def _handle_balance(self, chat_id: int, message_id: int) -> None:
        """Обработка запроса баланса."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        balance = await self.fetch_balance_with_retry()
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=f"Текущий баланс: {balance:.2f} USDT")

    async def _handle_close_all(self, chat_id: int, message_id: int) -> None:
        """Обработка закрытия всех позиций."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        if not self.executor or not self.exchange:
            logger.error("Executor или Exchange не инициализированы")
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Ошибка: Executor или Exchange не инициализированы")
            return

        for symbol in SYMBOLS:
            if self.executor.positions.get(symbol):
                side = "sell" if self.executor.positions[symbol] == "long" else "buy"
                await self.executor.close_position(symbol, (await self.exchange.fetch_ticker(symbol))["last"])
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Все позиции закрыты.")

    async def _handle_restart(self, chat_id: int, message_id: int) -> None:
        """Обработка перезапуска."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        self.paused = False
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Бот перезапущен.")

    async def _handle_graphs(self, chat_id: int, message_id: int) -> None:
        """Обработка запроса графиков."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        await send_async_message("Графики отправлены в чат.")
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Запрос отправлен, проверьте чат.")

    async def _handle_logs(self, chat_id: int, message_id: int) -> None:
        """Обработка запроса логов."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        try:
            async with aiofiles.open("trade_log.jsonl", "r") as f:
                lines = await f.readlines()
                last_logs = "\n".join(lines[-5:])
            await send_async_message(f"Последние 5 записей лога:\n{last_logs}")
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Логи отправлены в чат.")
        except Exception as e:
            logger.error(f"Ошибка получения логов: {e}", exc_info=True)
            await send_async_message(f"Ошибка получения логов: {e}")

    async def _handle_adjust_amount_menu(self, chat_id: int, message_id: int) -> None:
        """Обработка меню настройки объёма."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        markup = InlineKeyboardMarkup()
        markup.add(
            InlineKeyboardButton("Увеличить объем", callback_data="increase_amount"),
            InlineKeyboardButton("Уменьшить объем", callback_data="decrease_amount"),
            InlineKeyboardButton("Назад", callback_data="back")
        )
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Выберите действие с объемом:", reply_markup=markup)

    async def _handle_adjust_amount(self, chat_id: int, message_id: int, data: str) -> None:
        """Обработка изменения объёма."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int) or not isinstance(data, str):
            logger.error(f"chat_id, message_id и data должны быть корректными, получено {chat_id}, {message_id}, {data}")
            return
        if not self.executor:
            logger.error("Executor не инициализирован")
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Ошибка: Executor не инициализирован")
            return

        symbol = list(SYMBOLS.keys())[0]
        current_amount = self.executor.amounts.get(symbol, 0)
        new_amount = current_amount * 1.1 if data == "increase_amount" else current_amount * 0.9
        self.executor.amounts[symbol] = new_amount
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=f"Объем для {symbol} изменен: {new_amount:.4f}")

    async def _handle_set_strategy_menu(self, chat_id: int, message_id: int) -> None:
        """Обработка меню выбора стратегии."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        markup = InlineKeyboardMarkup()
        for strategy in STRATEGIES.keys():
            markup.add(InlineKeyboardButton(strategy, callback_data=f"strategy_{strategy}"))
        markup.add(InlineKeyboardButton("Назад", callback_data="back"))
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Выберите стратегию:", reply_markup=markup)

    async def _handle_set_strategy(self, chat_id: int, message_id: int, data: str) -> None:
        """Обработка установки стратегии."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int) or not isinstance(data, str):
            logger.error(f"chat_id, message_id и data должны быть корректными, получено {chat_id}, {message_id}, {data}")
            return
        try:
            strategy = data.split("_")[1]
            from telegram_utils import send_strategy_update  # Импорт внутри функции для избежания циклического импорта
            await send_strategy_update(strategy)
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=f"Стратегия установлена: {strategy}")
        except (IndexError, ValueError) as e:
            logger.error(f"Ошибка установки стратегии: {e}", exc_info=True)
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=f"Ошибка установки стратегии: {e}")

    async def _handle_manage_indicators_menu(self, chat_id: int, message_id: int) -> None:
        """Обработка меню управления индикаторами."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            logger.error(f"chat_id и message_id должны быть целыми числами, получено {chat_id}, {message_id}")
            return
        markup = InlineKeyboardMarkup()
        indicators = ["rsi", "macd", "atr", "volatility"]
        for ind in indicators:
            markup.add(
                InlineKeyboardButton(f"{ind} вкл", callback_data=f"ind_on_{ind}"),
                InlineKeyboardButton(f"{ind} выкл", callback_data=f"ind_off_{ind}")
            )
        markup.add(InlineKeyboardButton("Назад", callback_data="back"))
        self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Управление индикаторами:", reply_markup=markup)

    async def _handle_manage_indicators(self, chat_id: int, message_id: int, data: str) -> None:
        """Обработка управления индикаторами."""
        if not isinstance(chat_id, int) or not isinstance(message_id, int) or not isinstance(data, str):
            logger.error(f"chat_id, message_id и data должны быть корректными, получено {chat_id}, {message_id}, {data}")
            return
        try:
            action, indicator = data.split("_")[1], data.split("_")[2]
            enabled = action == "on"
            from telegram_utils import send_indicator_update  # Импорт внутри функции для избежания циклического импорта
            await send_indicator_update(indicator, enabled)
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, 
                                               text=f"Индикатор {indicator} {'включён' if enabled else 'выключен'}")
        except (IndexError, ValueError) as e:
            logger.error(f"Ошибка управления индикаторами: {e}", exc_info=True)
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=f"Ошибка управления индикаторами: {e}")

    async def send_main_menu(self, chat_id: int, message_id: Optional[int]) -> None:
        """Отправка главного меню.

        Args:
            chat_id (int): Идентификатор чата.
            message_id (Optional[int]): Идентификатор сообщения для редактирования.
        """
        if not isinstance(chat_id, int):
            logger.error(f"chat_id должен быть целым числом, получено {chat_id}")
            return
        if message_id is not None and not isinstance(message_id, int):
            logger.error(f"message_id должен быть целым числом или None, получено {message_id}")
            return

        markup = InlineKeyboardMarkup(row_width=2)
        markup.add(
            InlineKeyboardButton("Статус", callback_data="status"),
            InlineKeyboardButton("Пауза", callback_data="pause"),
            InlineKeyboardButton("Возобновить", callback_data="resume"),
            InlineKeyboardButton("Установить плечо", callback_data="set_leverage"),
            InlineKeyboardButton("Отчет", callback_data="report"),
            InlineKeyboardButton("Предсказание", callback_data="predict"),
            InlineKeyboardButton("Баланс", callback_data="balance"),
            InlineKeyboardButton("Закрыть все позиции", callback_data="close_all"),
            InlineKeyboardButton("Перезапуск", callback_data="restart"),
            InlineKeyboardButton("Графики", callback_data="graphs"),
            InlineKeyboardButton("Логи", callback_data="logs"),
            InlineKeyboardButton("Настроить объем", callback_data="adjust_amount"),
            InlineKeyboardButton("Установить стратегию", callback_data="set_strategy"),
            InlineKeyboardButton("Управление индикаторами", callback_data="manage_indicators")
        )
        if message_id:
            self.telegram_bot.edit_message_text(chat_id=chat_id, message_id=message_id, text="Выберите действие:", reply_markup=markup)
        else:
            self.telegram_bot.send_message(chat_id=chat_id, text="Выберите действие:", reply_markup=markup)

    async def notify_trade(self, action: str, price: float, profit: float, symbol: str) -> None:
        """Уведомление о завершённой сделке.

        Args:
            action (str): Действие.
            price (float): Цена сделки.
            profit (float): Прибыль/убыток.
            symbol (str): Символ.
        """
        if not isinstance(action, str) or not isinstance(price, (int, float)) or not isinstance(profit, (int, float)) or not isinstance(symbol, str):
            logger.error(f"Некорректные параметры notify_trade: action={action}, price={price}, profit={profit}, symbol={symbol}")
            return
        if profit is not None:
            msg = f"✅ Сделка завершена: {action} по {price:.2f} ({symbol}), профит: {profit:.2f} USDT"
            try:
                await send_async_message(msg)
            except Exception as e:
                logger.error(f"Ошибка отправки уведомления о сделке: {e}", exc_info=True)

    async def check_limits(self) -> None:
        """Проверка лимитов убытков."""
        if not self.executor:
            logger.error("Executor не инициализирован")
            return
        try:
            if self.executor.daily_profit <= -MAX_DAILY_LOSS:
                msg = f"⚠️ Дневной лимит убытков достигнут: {self.executor.daily_profit:.2f} USDT"
                await send_async_message(msg)
                self.paused = True
            if self.executor.weekly_profit <= -MAX_WEEKLY_LOSS:
                msg = f"⚠️ Недельный лимит убытков достигнут: {self.executor.weekly_profit:.2f} USDT"
                await send_async_message(msg)
                self.paused = True
        except Exception as e:
            logger.error(f"Ошибка проверки лимитов: {e}", exc_info=True)

    async def fetch_balance_with_retry(self, retries: int = 3, delay: int = 5) -> float:
        """Получение баланса с повторными попытками.

        Args:
            retries (int): Количество попыток (по умолчанию 3).
            delay (int): Задержка между попытками в секундах (по умолчанию 5).

        Returns:
            float: Баланс в USDT.
        """
        if not isinstance(retries, int) or retries < 0:
            logger.error(f"retries должен быть неотрицательным числом, получено {retries}")
            retries = 3
        if not isinstance(delay, int) or delay < 0:
            logger.error(f"delay должен быть неотрицательным числом, получено {delay}")
            delay = 5
        if not self.exchange:
            logger.error("Exchange не инициализирован")
            return self.last_balance if self.last_balance is not None else 10000

        for attempt in range(retries):
            try:
                balance_response = await asyncio.wait_for(self.exchange.fetch_balance(), timeout=10)
                balance = float(balance_response.get("total", {}).get("USDT", 0))
                self.last_balance = balance
                return balance
            except Exception as e:
                logger.error(f"Ошибка получения баланса (попытка {attempt+1}/{retries}): {e}", exc_info=True)
                if attempt == retries - 1:
                    await send_async_message(f"⚠️ Не удалось получить баланс после {retries} попыток")
                    return self.last_balance if self.last_balance is not None else 10000
                await asyncio.sleep(delay)

    async def fetch_ohlcv_with_retry(self, symbol: str, retries: int = 3, delay: int = 5) -> List[List[Union[int, float]]]:
        """Получение OHLCV данных с повторными попытками.

        Args:
            symbol (str): Символ (например, "BTC/USDT:USDT").
            retries (int): Количество попыток (по умолчанию 3).
            delay (int): Задержка между попытками в секундах (по умолчанию 5).

        Returns:
            List[List[Union[int, float]]]: Данные OHLCV или пустой список при ошибке.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получено {symbol}")
            return []
        if not isinstance(retries, int) or retries < 0:
            logger.error(f"retries должен быть неотрицательным числом, получено {retries}")
            retries = 3
        if not isinstance(delay, int) or delay < 0:
            logger.error(f"delay должен быть неотрицательным числом, получено {delay}")
            delay = 5
        if not self.exchange:
            logger.error("Exchange не инициализирован")
            return []

        for attempt in range(retries):
            try:
                ohlcv = await asyncio.wait_for(self.exchange.fetch_ohlcv(symbol, "15m", limit=60), timeout=10)
                return ohlcv
            except Exception as e:
                logger.error(f"Ошибка получения OHLCV для {symbol} (попытка {attempt+1}/{retries}): {e}", exc_info=True)
                if attempt == retries - 1:
                    await send_async_message(f"⚠️ Не удалось получить OHLCV для {symbol} после {retries} попыток")
                    return []
                await asyncio.sleep(delay)

    def set_start_time(self, start_time: float) -> None:
        """Установка времени начала работы.

        Args:
            start_time (float): Время начала в секундах.
        """
        if not isinstance(start_time, (int, float)):
            logger.error(f"start_time должен быть числом, получено {start_time}")
            return
        self.start_time = float(start_time)

    def stop_polling(self) -> None:
        """Остановка polling (если используется)."""
        try:
            self.telegram_bot.stop_polling()
            logger.info("Polling остановлен")
        except Exception as e:
            logger.error(f"Ошибка остановки polling: {e}", exc_info=True)

    async def process_webhook_update(self, update: Dict[str, Any]) -> None:
        """Обработка обновлений через Webhook.

        Args:
            update (Dict[str, Any]): Обновление от Telegram.
        """
        if not isinstance(update, dict):
            logger.error(f"update должен быть словарем, получено {type(update)}")
            return
        try:
            self.telegram_bot.process_new_updates([telebot.types.Update.de_json(update)])
        except Exception as e:
            logger.error(f"Ошибка обработки Webhook обновления: {e}", exc_info=True)

# Пример запуска Webhook
async def start_webhook(handler: "TelegramHandler") -> None:
    """Запуск Webhook сервера.

    Args:
        handler (TelegramHandler): Объект обработчика Telegram.
    """
    if not isinstance(handler, TelegramHandler):
        logger.error(f"handler должен быть экземпляром TelegramHandler, получено {type(handler)}")
        return
    if not handler.webhook_url:
        logger.error("Webhook URL не указан, запуск невозможен")
        return

    async with aiohttp.web.Application() as app:
        async def handle_webhook(request):
            update = await request.json()
            await handler.process_webhook_update(update)
            return aiohttp.web.Response(status=200)
        
        app.router.add_post(f"/{TELEGRAM_TOKEN}", handle_webhook)
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, "0.0.0.0", 8443)  # Используйте ваш порт
        await site.start()
        logger.info("Webhook сервер запущен")
        await asyncio.Future()  # Бесконечный цикл

if __name__ == "__main__":
    handler = TelegramHandler(None, None, "https://your-server.com/bot")  # Замените URL
    asyncio.run(start_webhook(handler))