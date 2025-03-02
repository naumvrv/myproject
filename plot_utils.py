# plot_utils.py
import plotly.graph_objs as go
import pandas as pd
from typing import Dict, List, Union, Optional, Dict as DictType
from datetime import datetime, timedelta
import aiofiles
import os
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("plot_utils_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("plot_utils")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from telegram_utils import send_async_message
except ImportError as e:
    logger.error(f"Не удалось импортировать telegram_utils: {e}")
    async def send_async_message(message: str, photo: Optional[bytes] = None) -> None:
        logger.warning(f"Telegram уведомления отключены: {message}")

class PlotUtils:
    @staticmethod
    def create_price_plot(data: Union[pd.DataFrame, List[Dict]], symbol: str, 
                          period: str = "all") -> go.Figure:
        """
        Создание графика цены.

        Args:
            data (Union[pd.DataFrame, List[Dict]]): Данные с ценами (DataFrame или список словарей).
            symbol (str): Символ (например, "BTC/USDT:USDT").
            period (str): Период для отображения ("1h", "1d", "1w", "all"; по умолчанию "all").

        Returns:
            go.Figure: Объект Figure с графиком цены.
        """
        if not isinstance(data, (pd.DataFrame, list)):
            logger.error(f"data должен быть pd.DataFrame или списком, получен {type(data)}")
            return go.Figure()
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return go.Figure()
        if not isinstance(period, str) or period not in ["1h", "1d", "1w", "all"]:
            logger.error(f"period должен быть одним из ['1h', '1d', '1w', 'all'], получен {period}")
            period = "all"

        try:
            df = pd.DataFrame(data) if isinstance(data, list) else data.copy()
            if df.empty:
                logger.warning(f"Нет данных для построения графика цены для {symbol}")
                return go.Figure()
            if "timestamp" not in df.columns:
                logger.error(f"Отсутствует столбец 'timestamp' в данных для {symbol}")
                return go.Figure()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            prices = df["close"] if "close" in df.columns else df.get("price")
            if prices is None:
                logger.error(f"Отсутствует столбец 'close' или 'price' в данных для {symbol}")
                return go.Figure()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"], y=prices, mode="lines", name=f"{symbol} Price"))
            fig.update_layout(
                title=f"Price ({symbol}, {period})",
                xaxis_title="Time",
                yaxis_title="Price (USDT)",
                template="plotly_dark"
            )
            return fig
        except Exception as e:
            logger.error(f"Ошибка создания графика цены для {symbol}: {e}")
            return go.Figure()

    @staticmethod
    def create_profit_plot(data: Union[pd.DataFrame, List[Dict]], symbol: str, 
                           period: str = "all") -> go.Figure:
        """
        Создание графика профита.

        Args:
            data (Union[pd.DataFrame, List[Dict]]): Данные с профитом (DataFrame или список словарей).
            symbol (str): Символ.
            period (str): Период для отображения ("1h", '1d', '1w', 'all'; по умолчанию "all").

        Returns:
            go.Figure: Объект Figure с графиком профита.
        """
        if not isinstance(data, (pd.DataFrame, list)):
            logger.error(f"data должен быть pd.DataFrame или списком, получен {type(data)}")
            return go.Figure()
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return go.Figure()
        if not isinstance(period, str) or period not in ["1h", "1d", "1w", "all"]:
            logger.error(f"period должен быть одним из ['1h', '1d', '1w', 'all'], получен {period}")
            period = "all"

        try:
            df = pd.DataFrame(data) if isinstance(data, list) else data.copy()
            if df.empty:
                logger.warning(f"Нет данных для построения графика профита для {symbol}")
                return go.Figure()
            if "timestamp" not in df.columns or "profit" not in df.columns:
                logger.error(f"Отсутствуют обязательные столбцы 'timestamp' или 'profit' в данных для {symbol}")
                return go.Figure()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            profits = df["profit"].fillna(0)
            cumulative_profit = profits.cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"], y=profits, mode="lines", name="Profit per Trade"))
            fig.add_trace(go.Scatter(x=df["timestamp"], y=cumulative_profit, mode="lines", name="Cumulative Profit"))
            fig.update_layout(
                title=f"Profit ({symbol}, {period})",
                xaxis_title="Time",
                yaxis_title="Profit (USDT)",
                template="plotly_dark"
            )
            return fig
        except Exception as e:
            logger.error(f"Ошибка создания графика профита для {symbol}: {e}")
            return go.Figure()

    @staticmethod
    def create_rsi_plot(data: Union[pd.DataFrame, List[Dict]], symbol: str, 
                        period: str = "all") -> go.Figure:
        """
        Создание графика RSI.

        Args:
            data (Union[pd.DataFrame, List[Dict]]): Данные с RSI (DataFrame или список словарей).
            symbol (str): Символ.
            period (str): Период для отображения ("1h", "1d", "1w", "all"; по умолчанию "all").

        Returns:
            go.Figure: Объект Figure с графиком RSI.
        """
        if not isinstance(data, (pd.DataFrame, list)):
            logger.error(f"data должен быть pd.DataFrame или списком, получен {type(data)}")
            return go.Figure()
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return go.Figure()
        if not isinstance(period, str) or period not in ["1h", "1d", "1w", "all"]:
            logger.error(f"period должен быть одним из ['1h', '1d', '1w', 'all'], получен {period}")
            period = "all"

        try:
            df = pd.DataFrame(data) if isinstance(data, list) else data.copy()
            if df.empty:
                logger.warning(f"Нет данных для построения графика RSI для {symbol}")
                return go.Figure()
            if "timestamp" not in df.columns:
                logger.error(f"Отсутствует столбец 'timestamp' в данных для {symbol}")
                return go.Figure()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            rsi = df["rsi"] if "rsi" in df.columns else pd.Series([50] * len(df))
            if rsi.empty:
                logger.warning(f"Отсутствует столбец 'rsi' в данных для {symbol}, используется значение по умолчанию 50")
                rsi = pd.Series([50] * len(df))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"], y=rsi, mode="lines", name="RSI"))
            fig.update_layout(
                title=f"RSI ({symbol}, {period})",
                xaxis_title="Time",
                yaxis_title="RSI",
                template="plotly_dark"
            )
            return fig
        except Exception as e:
            logger.error(f"Ошибка создания графика RSI для {symbol}: {e}")
            return go.Figure()

    @staticmethod
    def create_volatility_plot(data: Union[pd.DataFrame, List[Dict]], symbol: str, 
                               period: str = "all") -> go.Figure:
        """
        Создание графика волатильности.

        Args:
            data (Union[pd.DataFrame, List[Dict]]): Данные с волатильностью (DataFrame или список словарей).
            symbol (str): Символ.
            period (str): Период для отображения ("1h", "1d", "1w", "all"; по умолчанию "all").

        Returns:
            go.Figure: Объект Figure с графиком волатильности.
        """
        if not isinstance(data, (pd.DataFrame, list)):
            logger.error(f"data должен быть pd.DataFrame или списком, получен {type(data)}")
            return go.Figure()
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return go.Figure()
        if not isinstance(period, str) or period not in ["1h", "1d", "1w", "all"]:
            logger.error(f"period должен быть одним из ['1h', '1d', '1w', 'all'], получен {period}")
            period = "all"

        try:
            df = pd.DataFrame(data) if isinstance(data, list) else data.copy()
            if df.empty:
                logger.warning(f"Нет данных для построения графика волатильности для {symbol}")
                return go.Figure()
            if "timestamp" not in df.columns:
                logger.error(f"Отсутствует столбец 'timestamp' в данных для {symbol}")
                return go.Figure()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            volatility = df["volatility"] if "volatility" in df.columns else pd.Series([0] * len(df))
            if volatility.empty:
                logger.warning(f"Отсутствует столбец 'volatility' в данных для {symbol}, используется значение по умолчанию 0")
                volatility = pd.Series([0] * len(df))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"], y=volatility, mode="lines", name="Volatility"))
            fig.update_layout(
                title=f"Volatility ({symbol}, {period})",
                xaxis_title="Time",
                yaxis_title="Volatility",
                template="plotly_dark"
            )
            return fig
        except Exception as e:
            logger.error(f"Ошибка создания графика волатильности для {symbol}: {e}")
            return go.Figure()

    @staticmethod
    async def generate_plots(data: Union[pd.DataFrame, List[Dict]], symbol: str, period: str = "all", 
                             send_to_telegram: bool = True) -> Dict[str, go.Figure]:
        """
        Генерация всех графиков и их отправка в Telegram.

        Args:
            data (Union[pd.DataFrame, List[Dict]]): Данные для построения графиков.
            symbol (str): Символ.
            period (str): Период для отображения ("1h", "1d", "1w", "all"; по умолчанию "all").
            send_to_telegram (bool): Отправлять ли графики в Telegram (по умолчанию True).

        Returns:
            Dict[str, go.Figure]: Словарь с объектами Figure для каждого графика.

        Notes:
            Требуется установка пакета `kaleido` для сохранения графиков в файл (`pip install -U kaleido`).
        """
        if not isinstance(data, (pd.DataFrame, list)):
            logger.error(f"data должен быть pd.DataFrame или списком, получен {type(data)}")
            return {}
        if not isinstance(symbol, str) or not symbol:
            logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
            return {}
        if not isinstance(period, str) or period not in ["1h", "1d", "1w", "all"]:
            logger.error(f"period должен быть одним из ['1h', '1d', '1w', 'all'], получен {period}")
            period = "all"
        if not isinstance(send_to_telegram, bool):
            logger.error(f"send_to_telegram должен быть булевым значением, получен {send_to_telegram}")
            send_to_telegram = True

        try:
            df = pd.DataFrame(data) if isinstance(data, list) else data.copy()
            if df.empty:
                logger.warning(f"Нет данных для построения графиков для {symbol}")
                return {}
            if "timestamp" not in df.columns:
                logger.error(f"Отсутствует столбец 'timestamp' в данных для {symbol}")
                return {}
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Фильтрация по периоду
            if period != "all":
                time_delta = {"1h": timedelta(hours=1), "1d": timedelta(days=1), "1w": timedelta(weeks=1), 
                              "1m": timedelta(days=30)}
                cutoff = datetime.now() - time_delta[period]
                df = df[df["timestamp"] >= cutoff]

            if df.empty:
                logger.info(f"Нет данных для {symbol} за период {period}")
                return {}

            # Генерация графиков
            plots = {
                "price": PlotUtils.create_price_plot(df, symbol, period),
                "profit": PlotUtils.create_profit_plot(df, symbol, period),
                "rsi": PlotUtils.create_rsi_plot(df, symbol, period),
                "volatility": PlotUtils.create_volatility_plot(df, symbol, period)
            }

            # Сохранение и отправка в Telegram
            if send_to_telegram:
                try:
                    import kaleido  # Проверка наличия kaleido
                    output_file = f"plot_{symbol.replace('/', '_')}_{period}_{int(datetime.now().timestamp())}.png"
                    plots["profit"].write_image(output_file)
                    async with aiofiles.open(output_file, "rb") as photo:
                        await send_async_message(f"Графики за период {period} для {symbol}", photo=await photo.read())
                    logger.info(f"Графики отправлены в Telegram для {symbol}, период: {period}")
                    os.remove(output_file)  # Удаляем файл после отправки
                except ImportError:
                    logger.error("Пакет 'kaleido' не установлен. Установите его с помощью 'pip install -U kaleido'")
                except Exception as e:
                    logger.error(f"Ошибка сохранения или отправки графика в Telegram: {e}")

            return plots
        except Exception as e:
            logger.error(f"Ошибка генерации графиков для {symbol}: {e}")
            await send_async_message(f"⚠️ Ошибка генерации графиков для {symbol}: {e}")
            return {}

# Экспорт функций для удобства
create_price_plot = PlotUtils.create_price_plot
create_profit_plot = PlotUtils.create_profit_plot
create_rsi_plot = PlotUtils.create_rsi_plot
create_volatility_plot = PlotUtils.create_volatility_plot
generate_plots = PlotUtils.generate_plots

if __name__ == "__main__":
    # Тестовый пример
    test_data = [
        {"timestamp": "2023-01-01T00:00:00", "close": 100, "profit": 10, "rsi": 60, "volatility": 0.02},
        {"timestamp": "2023-01-01T00:15:00", "close": 101, "profit": -5, "rsi": 65, "volatility": 0.03}
    ]
    import asyncio
    plots = asyncio.run(generate_plots(test_data, "BTC/USDT:USDT", "1h", send_to_telegram=False))
    logger.info(f"Сгенерированы графики: {list(plots.keys())}")