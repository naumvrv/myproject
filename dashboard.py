# dashboard.py
from dash import Dash, dcc, html, Output, Input
import asyncio
import aiofiles
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import websockets
import plotly.graph_objs as go
import pandas as pd
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("dashboard_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dashboard")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from config_loader import LOG_FILE, DASH_PORT, SYMBOLS, STRATEGIES
    from cache_manager import get_cached_data, save_cached_data
    from plot_utils import generate_plots
    from telegram_utils import send_async_message
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    raise SystemExit(1)

app = Dash(__name__)
app.layout = html.Div([
    html.H1("Trading Bot Dashboard"),
    dcc.Dropdown(id="period-dropdown", options=[
        {"label": "1 час", "value": "1h"},
        {"label": "1 день", "value": "1d"},
        {"label": "1 неделя", "value": "1w"},
        {"label": "Все время", "value": "all"}
    ], value="1d"),
    dcc.Dropdown(id="symbol-dropdown", options=[{"label": s, "value": s} for s in SYMBOLS.keys()], value="BTC/USDT:USDT"),
    dcc.Dropdown(id="strategy-dropdown", options=[{"label": s, "value": s} for s in STRATEGIES.keys()], value="trend"),
    dcc.DatePickerRange(id="date-picker", min_date_allowed=datetime(2020, 1, 1), max_date_allowed=datetime.now()),
    dcc.Graph(id="price-graph"),
    dcc.Graph(id="profit-graph"),
    dcc.Graph(id="rsi-graph"),
    dcc.Graph(id="volatility-graph"),
    html.Button("Pause", id="pause-button", n_clicks=0),
    html.Button("Resume", id="resume-button", n_clicks=0),
    dcc.Input(id="leverage-input", type="number", placeholder="Set Leverage", min=1, max=125),
    html.Button("Set Leverage", id="set-leverage-button", n_clicks=0),
    dcc.Interval(id="interval-component", interval=10 * 1000, n_intervals=0),
    dcc.WebSocket(id="ws", url=f"ws://localhost:{DASH_PORT}/ws")
])

async def fetch_logs(log_file: str = LOG_FILE) -> List[Dict]:
    """Асинхронное чтение логов из файла.

    Args:
        log_file (str): Путь к файлу логов.

    Returns:
        List[Dict]: Список словарей с данными логов.
    """
    if not isinstance(log_file, str) or not log_file:
        logger.error(f"log_file должен быть непустой строкой, получен {type(log_file)}")
        return []
    try:
        if not os.path.exists(log_file):
            error_msg = f"Файл логов {log_file} не существует"
            logger.warning(error_msg)
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
        if not logs:
            logger.info(f"Файл логов {log_file} пуст")
        return logs
    except Exception as e:
        logger.error(f"Ошибка чтения логов ({log_file}): {e}")
        await send_async_message(f"⚠️ Ошибка чтения логов: {e}")
        return []

# Поскольку Dash не поддерживает асинхронные коллбэки, делаем функцию синхронной
@app.callback(
    [Output("price-graph", "figure"), Output("profit-graph", "figure"),
     Output("rsi-graph", "figure"), Output("volatility-graph", "figure")],
    [Input("interval-component", "n_intervals"), Input("period-dropdown", "value"),
     Input("symbol-dropdown", "value"), Input("date-picker", "start_date"), Input("date-picker", "end_date"),
     Input("ws", "message")]
)
def update_graphs(n: int, period: str, symbol: str, start_date: Optional[str], 
                  end_date: Optional[str], ws_message: Optional[Dict]) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Обновление графиков на дашборде.

    Args:
        n (int): Количество интервалов обновления.
        period (str): Период фильтрации данных.
        symbol (str): Символ для отображения.
        start_date (Optional[str]): Начальная дата фильтрации.
        end_date (Optional[str]): Конечная дата фильтрации.
        ws_message (Optional[Dict]): Сообщение от WebSocket.

    Returns:
        Tuple[go.Figure, ...]: Кортеж с четырьмя графиками.
    """
    if not isinstance(n, int) or n < 0:
        logger.error(f"n должен быть неотрицательным числом, получен {n}")
        return [go.Figure()] * 4
    if not isinstance(period, str) or not period:
        logger.error(f"period должен быть непустой строкой, получен {period}")
        return [go.Figure()] * 4
    if not isinstance(symbol, str) or not symbol:
        logger.error(f"symbol должен быть непустой строкой, получен {symbol}")
        return [go.Figure()] * 4

    try:
        # Вызов асинхронной функции через asyncio.run() для совместимости с Dash
        logs = asyncio.run(fetch_logs())
        if not logs:
            logger.info("Нет данных в логах для визуализации")
            return [go.Figure()] * 4

        filtered_logs = [log for log in logs if log.get("symbol") == symbol]
        if not filtered_logs:
            logger.info(f"Нет данных для {symbol}")
            return [go.Figure()] * 4

        df = pd.DataFrame(filtered_logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["price"] = df["price"].fillna(0)
        df["profit"] = df["profit"].fillna(0)
        # Безопасное извлечение данных из grok_analysis
        df["rsi"] = df["grok_analysis"].apply(lambda x: x.get("rsi", 50) if isinstance(x, dict) and "rsi" in x else 50)
        df["volatility"] = df["grok_analysis"].apply(lambda x: x.get("volatility", 0) if isinstance(x, dict) and "volatility" in x else 0)

        # Фильтрация по дате или периоду
        if start_date and end_date:
            try:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                filt = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            except ValueError as e:
                logger.error(f"Ошибка преобразования дат: {e}")
                return [go.Figure()] * 4
        elif period != "all":
            time_delta = {"1h": timedelta(hours=1), "1d": timedelta(days=1), "1w": timedelta(weeks=1)}
            cutoff = datetime.now() - time_delta.get(period, timedelta(days=1))
            filt = df["timestamp"] >= cutoff
        else:
            filt = [True] * len(df)

        df = df[filt]
        if df.empty:
            logger.info(f"Нет данных для {symbol} за период {period}")
            return [go.Figure()] * 4

        plots = asyncio.run(generate_plots(df, symbol, period, send_to_telegram=False))
        logger.info(f"Графики обновлены для {symbol}, период: {period}")
        return (plots["price"], plots["profit"], plots["rsi"], plots["volatility"])
    except Exception as e:
        logger.error(f"Ошибка обновления графиков: {e}")
        asyncio.run(send_async_message(f"⚠️ Ошибка обновления графиков: {e}"))
        return [go.Figure()] * 4

@app.callback(
    Output("pause-button", "n_clicks"),
    Input("pause-button", "n_clicks")
)
def pause_bot(n_clicks: int) -> int:
    """Приостановка бота.

    Args:
        n_clicks (int): Количество нажатий кнопки.

    Returns:
        int: Сброс счетчика нажатий.
    """
    if not isinstance(n_clicks, int) or n_clicks < 0:
        logger.error(f"n_clicks должен быть неотрицательным числом, получен {n_clicks}")
        return 0
    if n_clicks:
        asyncio.run(save_cached_data("bot_status", "paused", ttl=3600))
        asyncio.run(send_async_message("✅ Бот приостановлен через дашборд"))
        logger.info("Бот приостановлен через дашборд")
    return 0

@app.callback(
    Output("resume-button", "n_clicks"),
    Input("resume-button", "n_clicks")
)
def resume_bot(n_clicks: int) -> int:
    """Возобновление работы бота.

    Args:
        n_clicks (int): Количество нажатий кнопки.

    Returns:
        int: Сброс счетчика нажатий.
    """
    if not isinstance(n_clicks, int) or n_clicks < 0:
        logger.error(f"n_clicks должен быть неотрицательным числом, получен {n_clicks}")
        return 0
    if n_clicks:
        asyncio.run(save_cached_data("bot_status", "running", ttl=3600))
        asyncio.run(send_async_message("✅ Бот возобновил работу через дашборд"))
        logger.info("Бот возобновил работу через дашборд")
    return 0

@app.callback(
    Output("leverage-input", "value"),
    [Input("set-leverage-button", "n_clicks"), Input("leverage-input", "value")]
)
def set_leverage(n_clicks: int, leverage: Optional[float]) -> Optional[float]:
    """Установка плеча.

    Args:
        n_clicks (int): Количество нажатий кнопки.
        leverage (Optional[float]): Значение плеча.

    Returns:
        Optional[float]: Установленное значение плеча.
    """
    if not isinstance(n_clicks, int) or n_clicks < 0:
        logger.error(f"n_clicks должен быть неотрицательным числом, получен {n_clicks}")
        return leverage
    if leverage is not None and (not isinstance(leverage, (int, float)) or leverage < 1 or leverage > 125):
        logger.error(f"leverage должен быть числом от 1 до 125, получен {leverage}")
        return leverage
    if n_clicks and leverage:
        asyncio.run(save_cached_data("leverage", leverage, ttl=3600))
        asyncio.run(send_async_message(f"✅ Плечо установлено на {leverage}x через дашборд"))
        logger.info(f"Плечо установлено на {leverage}x через дашборд")
    return leverage

@app.callback(
    Output("strategy-dropdown", "value"),
    Input("strategy-dropdown", "value")
)
def set_strategy(strategy: str) -> str:
    """Установка стратегии.

    Args:
        strategy (str): Название стратегии.

    Returns:
        str: Установленная стратегия.
    """
    if not isinstance(strategy, str) or not strategy:
        logger.error(f"strategy должна быть непустой строкой, получен {strategy}")
        return "trend"  # Значение по умолчанию
    if strategy:
        asyncio.run(save_cached_data("strategy", strategy, ttl=3600))
        asyncio.run(send_async_message(f"✅ Установлена стратегия: {strategy}"))
        logger.info(f"Стратегия установлена: {strategy}")
    return strategy

# Глобальная переменная для управления WebSocket-сервером
websocket_server_running = True

async def websocket_server():
    """Запуск WebSocket сервера для обновлений в реальном времени."""
    global websocket_server_running
    try:
        server = await websockets.serve(handle_websocket, "localhost", DASH_PORT)
        logger.info(f"WebSocket сервер запущен на порту {DASH_PORT}")
        while websocket_server_running:
            await asyncio.sleep(1)
        await server.close()
    except Exception as e:
        logger.error(f"Ошибка запуска WebSocket сервера: {e}")
        await send_async_message(f"⚠️ Ошибка запуска WebSocket: {e}")

async def handle_websocket(websocket, path: str):
    """Обработка WebSocket соединений.

    Args:
        websocket: WebSocket соединение.
        path (str): Путь соединения.
    """
    try:
        while True:
            logs = await fetch_logs()
            if logs:
                latest_log = logs[-1]
                await websocket.send(json.dumps(latest_log))
            await asyncio.sleep(10)
    except websockets.ConnectionClosed:
        logger.info("WebSocket соединение закрыто")
    except Exception as e:
        logger.error(f"Ошибка WebSocket: {e}")

async def run_dashboard():
    """Запуск дашборда."""
    global websocket_server_running
    try:
        asyncio.create_task(websocket_server())
        app.run_server(debug=False, port=DASH_PORT)
        logger.info(f"Дашборд запущен на порту {DASH_PORT}")
    except Exception as e:
        error_msg = f"Ошибка запуска дашборда: {e}"
        logger.error(error_msg)
        await send_async_message(error_msg)
    finally:
        websocket_server_running = False  # Остановка WebSocket при завершении

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_dashboard())
    except KeyboardInterrupt:
        global websocket_server_running
        websocket_server_running = False
        loop.close()
        logger.info("Дашборд остановлен")
    except Exception as e:
        logger.error(f"Ошибка при запуске дашборда: {e}")