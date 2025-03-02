# trade_executor.py
import ccxt.async_support as ccxt_async
import pickle
import os
import numpy as np
from typing import Dict, Optional, Union, Callable, Any, Tuple
import aiofiles
import asyncio
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("trade_executor_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("trade_executor")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from config_loader import SYMBOLS, MAX_DAILY_LOSS, MAX_WEEKLY_LOSS, STRATEGIES
    from dynamic_config import LEVERAGE
    from telegram_utils import send_async_message
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    async def send_async_message(msg: str) -> None:
        logger.warning(f"Telegram уведомления отключены: {msg}")
    raise SystemExit(1)

class TradeExecutor:
    def __init__(self, exchange: ccxt_async.Exchange, telegram_handler: Optional[Any] = None):
        """
        Инициализация исполнителя сделок.

        Args:
            exchange (ccxt_async.Exchange): Объект биржи из ccxt.async_support.
            telegram_handler (Optional[Any]): Обработчик Telegram (опционально, по умолчанию None).
        """
        if not isinstance(exchange, ccxt_async.Exchange):
            raise ValueError(f"exchange должен быть объектом ccxt_async.Exchange, получен {type(exchange)}")
        self.exchange = exchange
        self.telegram_handler = telegram_handler
        self.positions: Dict[str, Optional[str]] = {symbol: None for symbol in SYMBOLS}
        self.entry_prices: Dict[str, float] = {symbol: 0 for symbol in SYMBOLS}
        self.amounts: Dict[str, float] = {symbol: SYMBOLS[symbol]["amount"] for symbol in SYMBOLS}
        self.leverage: int = LEVERAGE
        self.daily_profit: float = 0
        self.weekly_profit: float = 0
        self.active_orders: Dict[str, Dict[str, str]] = {}
        self.current_atr: Dict[str, float] = {symbol: 0 for symbol in SYMBOLS}
        self.max_leverage: int = 125
        self.state_file: str = "trade_executor_state.pkl"
        asyncio.create_task(self.load_state())  # Асинхронная загрузка состояния

    async def load_state(self) -> None:
        """Загрузка состояния из файла."""
        if not os.path.exists(self.state_file):
            logger.info(f"Файл состояния {self.state_file} не существует, используется начальное состояние")
            return
        try:
            async with aiofiles.open(self.state_file, "rb") as f:
                state = pickle.loads(await f.read())
            self.positions = state.get("positions", self.positions)
            self.entry_prices = state.get("entry_prices", self.entry_prices)
            self.amounts = state.get("amounts", self.amounts)
            self.daily_profit = state.get("daily_profit", 0)
            self.weekly_profit = state.get("weekly_profit", 0)
            self.active_orders = state.get("active_orders", {})
            logger.info("Состояние TradeExecutor загружено")
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния: {e}")

    async def save_state(self) -> None:
        """Сохранение состояния в файл."""
        state = {
            "positions": self.positions,
            "entry_prices": self.entry_prices,
            "amounts": self.amounts,
            "daily_profit": self.daily_profit,
            "weekly_profit": self.weekly_profit,
            "active_orders": self.active_orders
        }
        try:
            os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
            async with aiofiles.open(self.state_file, "wb") as f:
                await f.write(pickle.dumps(state))
            logger.info("Состояние TradeExecutor сохранено")
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

    async def execute_trade(self, action: str, price: float, grok_analysis: Dict[str, float], 
                            predicted_price: float, log_trade: Callable, 
                            symbol: str = "", atr: float = 0, rsi: float = 50, fred_rate: float = 0, 
                            retries: int = 3, delay: int = 5) -> None:
        """
        Выполнение торговой операции.

        Args:
            action (str): Действие (buy_long, sell_short и т.д.).
            price (float): Цена сделки.
            grok_analysis (Dict[str, float]): Анализ Grok с ключами 'volatility', 'confidence', 'sentiment'.
            predicted_price (float): Предсказанная цена.
            log_trade (Callable): Функция логирования.
            symbol (str): Символ (например, "BTC/USDT:USDT").
            atr (float): Значение ATR (по умолчанию 0).
            rsi (float): Значение RSI (по умолчанию 50).
            fred_rate (float): Ставка FRED (по умолчанию 0).
            retries (int): Количество попыток (по умолчанию 3).
            delay (int): Задержка между попытками в секундах (по умолчанию 5).
        """
        if not isinstance(action, str) or not action:
            logger.error(f"action должен быть непустой строкой, получен {action}")
            return
        if not isinstance(price, (int, float)) or price <= 0:
            logger.error(f"price должен быть положительным числом, получен {price}")
            return
        if not isinstance(grok_analysis, dict) or "volatility" not in grok_analysis:
            logger.error(f"grok_analysis должен быть словарем с ключом 'volatility', получен {grok_analysis}")
            return
        if not isinstance(predicted_price, (int, float)) or predicted_price <= 0:
            logger.error(f"predicted_price должен быть положительным числом, получен {predicted_price}")
            return
        if not callable(log_trade):
            logger.error(f"log_trade должен быть вызываемой функцией, получен {type(log_trade)}")
            return
        if not isinstance(symbol, str) or symbol not in SYMBOLS:
            logger.error(f"symbol должен быть строкой из SYMBOLS, получен {symbol}")
            return
        if not isinstance(atr, (int, float)) or atr < 0:
            logger.error(f"atr должен быть неотрицательным числом, получен {atr}")
            atr = 0
        if not isinstance(rsi, (int, float)):
            logger.error(f"rsi должен быть числом, получен {rsi}")
            rsi = 50
        if not isinstance(fred_rate, (int, float)):
            logger.error(f"fred_rate должен быть числом, получен {fred_rate}")
            fred_rate = 0
        if not isinstance(retries, int) or retries < 0:
            logger.error(f"retries должен быть неотрицательным числом, получен {retries}")
            retries = 3
        if not isinstance(delay, int) or delay < 0:
            logger.error(f"delay должен быть неотрицательным числом, получен {delay}")
            delay = 5

        try:
            strategy = self.select_strategy(predicted_price - price, grok_analysis["volatility"], rsi)
            strategy_params = STRATEGIES.get(strategy, STRATEGIES["trend"])
            leverage = min(self.calculate_dynamic_leverage(symbol, atr, grok_analysis["volatility"]), self.max_leverage)
            amount = self.amounts[symbol] * leverage
            current_position = self.positions[symbol]
            logger.info(f"Размер позиции для {symbol}: {amount}, leverage={leverage}, стратегия: {strategy}")

            order_book = await self.exchange.fetch_order_book(symbol, limit=10)
            liquidity = sum(bid[1] for bid in order_book["bids"][:5]) + sum(ask[1] for ask in order_book["asks"][:5])
            if liquidity < amount * 2:
                logger.warning(f"Недостаточная ликвидность для {symbol}: {liquidity} < {amount * 2}")
                await log_trade(symbol, action, price, None)
                return

            if current_position == "long" and action == "buy_long" or current_position == "short" and action == "sell_short":
                logger.info(f"Действие {action} не выполнено: уже открыта позиция {current_position}")
                await log_trade(symbol, action, price, None)
                return

            if action in ["increase_amount", "decrease_amount"]:
                await self._adjust_amount(symbol, action, price, log_trade)
                return

            if action in ["grid_buy", "grid_sell"]:
                await self.execute_grid_trade(action, price, symbol, amount, leverage, strategy_params, grok_analysis, log_trade)
                return

            if current_position:
                await self.close_position(symbol, price)

            await self._execute_market_trade(action, symbol, price, amount, leverage, atr, rsi, grok_analysis, strategy_params, log_trade)

        except Exception as e:
            logger.error(f"Ошибка выполнения сделки {action} для {symbol}: {e}")
            await send_async_message(f"⚠️ Ошибка выполнения сделки: {e}")

    async def _adjust_amount(self, symbol: str, action: str, price: float, log_trade: Callable) -> None:
        """Изменение объёма позиции."""
        if not symbol in SYMBOLS or not action in ["increase_amount", "decrease_amount"]:
            logger.error(f"Некорректные параметры: symbol={symbol}, action={action}")
            return
        try:
            current_amount = self.amounts[symbol]
            new_amount = current_amount * 1.1 if action == "increase_amount" else current_amount * 0.9
            self.amounts[symbol] = new_amount
            logger.info(f"Обновлен объем для {symbol}: {new_amount}")
            await self.save_state()
            await log_trade(symbol, action, price, None)
        except Exception as e:
            logger.error(f"Ошибка изменения объема для {symbol}: {e}")

    async def execute_grid_trade(self, action: str, price: float, symbol: str, amount: float, 
                                 leverage: int, strategy_params: Dict[str, float], 
                                 grok_analysis: Dict[str, float], log_trade: Callable) -> None:
        """Реализация грид-стратегии."""
        if action not in ["grid_buy", "grid_sell"]:
            logger.error(f"Некорректное действие для грид-стратегии: {action}")
            return
        try:
            grid_levels = 5
            step = grok_analysis.get("volatility", 0) * price * 0.5
            side = "buy" if action == "grid_buy" else "sell"
            pos_side = "long" if action == "grid_buy" else "short"
            for i in range(grid_levels):
                grid_price = price - step * (i + 1) if action == "grid_buy" else price + step * (i + 1)
                order_amount = amount / grid_levels
                order = await self.exchange.create_order(
                    symbol, "limit", side, order_amount, grid_price,
                    params={"tdMode": "cross", "leverage": leverage, "posSide": pos_side}
                )
                logger.info(f"Грид-ордер создан: {order['id']} для {symbol} на {grid_price}")
            self.positions[symbol] = pos_side
            self.entry_prices[symbol] = price
            await log_trade(symbol, action, price, None)
            await self.save_state()
        except Exception as e:
            logger.error(f"Ошибка грид-стратегии для {symbol}: {e}")

    async def _execute_market_trade(self, action: str, symbol: str, price: float, amount: float, 
                                    leverage: int, atr: float, rsi: float, grok_analysis: Dict[str, float], 
                                    strategy_params: Dict[str, float], log_trade: Callable) -> None:
        """Выполнение рыночной сделки."""
        if action not in ["buy_long", "sell_short"]:
            logger.error(f"Некорректное действие для рыночной сделки: {action}")
            return
        side = "buy" if action == "buy_long" else "sell"
        pos_side = "long" if action == "buy_long" else "short"
        for attempt in range(3):
            try:
                order = await self.exchange.create_order(
                    symbol, "market", side, amount,
                    params={"tdMode": "cross", "leverage": leverage, "posSide": pos_side}
                )
                self.positions[symbol] = pos_side
                self.entry_prices[symbol] = price
                self.current_atr[symbol] = atr
                logger.info(f"Ордер создан: {order['id']} для {symbol}, details={order}")
                sl_price, tp_price = self.calculate_sl_tp(price, atr, grok_analysis, strategy_params)
                msg = (
                    f"📈 {action.upper()} по {price:.2f} ({symbol})\n"
                    f"SL: {sl_price:.2f}\nTP: {tp_price:.2f}\nRSI: {rsi:.2f}\n"
                    f"Предсказание: {grok_analysis.get('predicted_change', 0):.2f}"
                )
                await send_async_message(msg)
                await log_trade(symbol, action, price, None)
                await self.set_stop_loss_take_profit(symbol, sl_price, tp_price, amount, side, pos_side)
                await self.save_state()
                break
            except ccxt.RateLimitExceeded as e:
                logger.error(f"Превышен лимит запросов для {symbol}: {e}")
                await asyncio.sleep(5 * (attempt + 1))
            except Exception as e:
                logger.error(f"Ошибка создания ордера для {symbol} (попытка {attempt+1}/3): {e}")
                if attempt == 2:
                    await send_async_message(f"⚠️ Не удалось создать ордер для {symbol} после 3 попыток: {e}")
                else:
                    await asyncio.sleep(5)

    def select_strategy(self, predicted_change: float, volatility: float, rsi: float) -> str:
        """Динамический выбор стратегии."""
        if not isinstance(predicted_change, (int, float)) or not isinstance(volatility, (int, float)) or not isinstance(rsi, (int, float)):
            logger.error(f"Некорректные параметры стратегии: predicted_change={predicted_change}, volatility={volatility}, rsi={rsi}")
            return "trend"
        try:
            if abs(predicted_change) > 0.01 and volatility > 0.02:
                return "trend"
            elif volatility > 0.03:
                return "scalping"
            elif rsi > 70 or rsi < 30:
                return "mean_reversion"
            return "countertrend"
        except Exception as e:
            logger.error(f"Ошибка выбора стратегии: {e}")
            return "trend"

    def calculate_dynamic_leverage(self, symbol: str, atr: float, volatility: float) -> float:
        """Расчёт динамического плеча."""
        if not isinstance(symbol, str) or not symbol in SYMBOLS:
            logger.error(f"symbol должен быть строкой из SYMBOLS, получен {symbol}")
            return self.leverage
        if not isinstance(atr, (int, float)) or atr < 0:
            logger.error(f"atr должен быть неотрицательным числом, получен {atr}")
            atr = 0
        if not isinstance(volatility, (int, float)) or volatility < 0:
            logger.error(f"volatility должен быть неотрицательным числом, получен {volatility}")
            volatility = 0
        try:
            base_leverage = self.leverage
            volatility_factor = max(0.1, min(10, 1 / (volatility + 0.001)))
            atr_factor = max(0.5, min(2, 1000 / (atr + 1)))
            risk_factor = max(0.2, 1 - (abs(self.daily_profit) / MAX_DAILY_LOSS if self.daily_profit < 0 else 0))
            corr_factor = self.calculate_correlation_risk()
            return min(base_leverage * volatility_factor * atr_factor * risk_factor * corr_factor, self.max_leverage)
        except Exception as e:
            logger.error(f"Ошибка расчета динамического плеча для {symbol}: {e}")
            return self.leverage

    def calculate_correlation_risk(self) -> float:
        """Расчёт корреляционного риска."""
        try:
            positions = {s: p for s, p in self.positions.items() if p}
            if len(positions) < 2:
                return 1.0
            return 0.9  # Упрощённое снижение риска
        except Exception as e:
            logger.error(f"Ошибка расчета корреляции: {e}")
            return 1.0

    def calculate_sl_tp(self, price: float, atr: float, grok_analysis: Dict[str, float], 
                        strategy_params: Dict[str, float]) -> Tuple[float, float]:
        """Расчёт уровней Stop Loss и Take Profit."""
        if not isinstance(price, (int, float)) or price <= 0:
            logger.error(f"price должен быть положительным числом, получен {price}")
            return price, price
        if not isinstance(atr, (int, float)) or atr < 0:
            logger.error(f"atr должен быть неотрицательным числом, получен {atr}")
            atr = 0
        if not isinstance(grok_analysis, dict):
            logger.error(f"grok_analysis должен быть словарем, получен {grok_analysis}")
            grok_analysis = {"sentiment": "neutral", "confidence": 0.5, "volatility": 0}
        if not isinstance(strategy_params, dict):
            logger.error(f"strategy_params должен быть словарем, получен {strategy_params}")
            strategy_params = STRATEGIES["trend"]

        try:
            sentiment_factor = (1 + grok_analysis.get("confidence", 0.5) if grok_analysis.get("sentiment", "neutral") == "bullish" 
                               else 1 - grok_analysis.get("confidence", 0.5))
            volatility_factor = 1 + grok_analysis.get("volatility", 0) * 0.5
            profit_factor = max(0.5, 1 - (self.daily_profit / MAX_DAILY_LOSS if self.daily_profit > 0 else 0))
            sl_distance = atr * strategy_params.get("stop_loss_factor", 1.0) * sentiment_factor * volatility_factor * profit_factor
            tp_distance = atr * strategy_params.get("take_profit_factor", 1.5) * sentiment_factor * volatility_factor / profit_factor
            sl_price = price - sl_distance if grok_analysis.get("sentiment", "neutral") == "bullish" else price + sl_distance
            tp_price = price + tp_distance if grok_analysis.get("sentiment", "neutral") == "bullish" else price - tp_distance
            return float(sl_price), float(tp_price)
        except Exception as e:
            logger.error(f"Ошибка расчета SL/TP: {e}")
            return price, price

    async def set_stop_loss_take_profit(self, symbol: str, sl_price: float, tp_price: float, 
                                        amount: float, side: str, pos_side: str, retries: int = 3, delay: int = 5) -> None:
        """Установка Stop Loss и Take Profit."""
        if not symbol in SYMBOLS or not side in ["buy", "sell"] or not pos_side in ["long", "short"]:
            logger.error(f"Некорректные параметры: symbol={symbol}, side={side}, pos_side={pos_side}")
            return
        if not isinstance(sl_price, (int, float)) or not isinstance(tp_price, (int, float)) or not isinstance(amount, (int, float)):
            logger.error(f"sl_price, tp_price и amount должны быть числами: sl_price={sl_price}, tp_price={tp_price}, amount={amount}")
            return
        try:
            opposite_side = "sell" if side == "buy" else "buy"
            for attempt in range(retries):
                try:
                    sl_order = await self.exchange.create_order(
                        symbol, "stop_market", opposite_side, amount,
                        params={"stopPrice": sl_price, "tdMode": "cross", "reduceOnly": True, "posSide": pos_side}
                    )
                    tp_order = await self.exchange.create_order(
                        symbol, "take_profit_market", opposite_side, amount,
                        params={"stopPrice": tp_price, "tdMode": "cross", "reduceOnly": True, "posSide": pos_side}
                    )
                    self.active_orders[symbol] = {"sl": sl_order["id"], "tp": tp_order["id"]}
                    logger.info(f"SL установлен на {sl_price} и TP на {tp_price} для {symbol}")
                    await self.save_state()
                    break
                except Exception as e:
                    logger.error(f"Ошибка установки SL/TP для {symbol} (попытка {attempt+1}/{retries}): {e}")
                    if attempt == retries - 1:
                        await send_async_message(f"⚠️ Не удалось установить SL/TP для {symbol}: {e}")
                    else:
                        await asyncio.sleep(delay)
        except Exception as e:
            logger.error(f"Ошибка установки SL/TP для {symbol}: {e}")

    async def close_position(self, symbol: str, current_price: float, retries: int = 3, delay: int = 5) -> None:
        """Закрытие позиции."""
        if not symbol in SYMBOLS or not isinstance(current_price, (int, float)):
            logger.error(f"Некорректные параметры: symbol={symbol}, current_price={current_price}")
            return
        try:
            if self.positions[symbol] is None:
                return
            side = "sell" if self.positions[symbol] == "long" else "buy"
            pos_side = self.positions[symbol]
            position_info = await self.exchange.fetch_position(symbol)
            amount = abs(float(position_info["info"]["pos"])) if isinstance(position_info.get("info", {}), dict) and "pos" in position_info["info"] else self.amounts[symbol]
            
            for attempt in range(retries):
                try:
                    order = await self.exchange.create_order(
                        symbol, "market", side, amount,
                        params={"tdMode": "cross", "reduceOnly": True, "posSide": pos_side}
                    )
                    profit = ((current_price - self.entry_prices[symbol]) * amount * self.leverage 
                             if side == "sell" else 
                             (self.entry_prices[symbol] - current_price) * amount * self.leverage)
                    self.daily_profit += profit
                    self.weekly_profit += profit
                    self.positions[symbol] = None
                    self.entry_prices[symbol] = 0
                    logger.info(f"Позиция {symbol} закрыта по {current_price}, профит: {profit:.2f}")
                    if self.telegram_handler:
                        await self.telegram_handler.notify_trade(
                            f"close_{'long' if side == 'sell' else 'short'}", current_price, profit, symbol
                        )
                    await self.cancel_orders(symbol)
                    if hasattr(self.telegram_handler, 'check_limits'):
                        await self.telegram_handler.check_limits()
                    await self.save_state()
                    break
                except Exception as e:
                    logger.error(f"Ошибка закрытия позиции для {symbol} (попытка {attempt+1}/{retries}): {e}")
                    if attempt == retries - 1:
                        await send_async_message(f"⚠️ Не удалось закрыть позицию для {symbol}: {e}")
                    else:
                        await asyncio.sleep(delay)
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции для {symbol}: {e}")

    async def cancel_orders(self, symbol: str) -> None:
        """Отмена всех активных ордеров для символа."""
        if not symbol in SYMBOLS:
            logger.error(f"symbol должен быть строкой из SYMBOLS, получен {symbol}")
            return
        try:
            if symbol in self.active_orders:
                for order_type, order_id in self.active_orders[symbol].items():
                    try:
                        await self.exchange.cancel_order(order_id, symbol)
                    except Exception as e:
                        logger.error(f"Ошибка отмены ордера {order_type} для {symbol}: {e}")
                del self.active_orders[symbol]
                logger.info(f"Ордера SL/TP для {symbol} отменены")
                await self.save_state()
        except Exception as e:
            logger.error(f"Ошибка отмены ордеров для {symbol}: {e}")

    async def check_open_positions(self, symbol: str, current_price: float) -> None:
        """Проверка открытых позиций и их закрытие при необходимости."""
        if not symbol in SYMBOLS or not isinstance(current_price, (int, float)):
            logger.error(f"Некорректные параметры: symbol={symbol}, current_price={current_price}")
            return
        try:
            if self.positions[symbol]:
                for order_type, order_id in self.active_orders.get(symbol, {}).items():
                    order_status = await self.exchange.fetch_order(order_id, symbol)
                    if order_status.get("status") == "closed":
                        await self.close_position(symbol, current_price)
                        return

                position_info = await self.exchange.fetch_position(symbol)
                if position_info and isinstance(position_info, dict):
                    atr = self.current_atr[symbol]
                    sl_price = (self.entry_prices[symbol] - (2 * atr) if self.positions[symbol] == "long" else 
                               self.entry_prices[symbol] + (2 * atr))
                    if (self.positions[symbol] == "long" and current_price <= sl_price) or \
                       (self.positions[symbol] == "short" and current_price >= sl_price):
                        await self.close_position(symbol, current_price)
        except Exception as e:
            logger.error(f"Ошибка проверки открытых позиций для {symbol}: {e}")

    async def monitor_api_limits(self) -> None:
        """Мониторинг лимитов API."""
        try:
            rate_limit_info = await self.exchange.fetch_status()
            if not isinstance(rate_limit_info, dict):
                logger.error(f"Некорректный ответ fetch_status: {rate_limit_info}")
                return
            remaining_requests = rate_limit_info.get("rateLimitRemaining", {}).get("requests", 0)
            if not isinstance(remaining_requests, (int, float)):
                logger.error(f"Некорректное значение remaining_requests: {remaining_requests}")
                return
            if remaining_requests < 10:
                await send_async_message(f"⚠️ Осталось мало запросов к API: {remaining_requests}")
            logger.info(f"Остаток запросов к API: {remaining_requests}")
        except Exception as e:
            logger.error(f"Ошибка мониторинга лимитов API: {e}")

    async def simulate_trade(self, action: str, price: float, grok_analysis: Dict[str, float], 
                             predicted_price: float, symbol: str, atr: float, rsi: float, 
                             commission: float) -> Optional[float]:
        """Симуляция сделки для бэктестинга."""
        if not action in ["buy_long", "sell_short"] or not symbol in SYMBOLS:
            logger.error(f"Некорректные параметры: action={action}, symbol={symbol}")
            return None
        if not isinstance(price, (int, float)) or price <= 0 or not isinstance(predicted_price, (int, float)) or predicted_price <= 0:
            logger.error(f"price и predicted_price должны быть положительными числами: price={price}, predicted_price={predicted_price}")
            return None
        if not isinstance(grok_analysis, dict) or "volatility" not in grok_analysis:
            logger.error(f"grok_analysis должен быть словарем с ключом 'volatility', получен {grok_analysis}")
            return None
        if not isinstance(atr, (int, float)) or atr < 0 or not isinstance(rsi, (int, float)) or not isinstance(commission, (int, float)) or commission < 0:
            logger.error(f"atr, rsi и commission должны быть неотрицательными числами: atr={atr}, rsi={rsi}, commission={commission}")
            return None

        try:
            leverage = self.calculate_dynamic_leverage(symbol, atr, grok_analysis["volatility"])
            amount = self.amounts[symbol] * leverage
            if action == "buy_long":
                profit = (predicted_price - price) * amount * leverage
            elif action == "sell_short":
                profit = (price - predicted_price) * amount * leverage
            profit -= profit * commission
            logger.info(f"Симуляция сделки для {symbol}: {action}, профит={profit:.2f}")
            return float(profit)
        except Exception as e:
            logger.error(f"Ошибка симуляции сделки для {symbol}: {e}")
            return None

if __name__ == "__main__":
    async def test():
        exchange = ccxt_async.okx({"enableRateLimit": True})
        executor = TradeExecutor(exchange, None)
        await executor.execute_trade(
            "buy_long", 30000, {"volatility": 0.02, "confidence": 0.8, "sentiment": "bullish"}, 
            30500, lambda s, a, p, pr: logger.info(f"Logged: {s}, {a}, {p}, {pr}"), 
            symbol="BTC/USDT:USDT", atr=1.5, rsi=60
        )
    asyncio.run(test())