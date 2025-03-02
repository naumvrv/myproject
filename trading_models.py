# trading_models.py
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from typing import Optional, Tuple, Any, Union
from collections import deque
import random
import time
import asyncio
import os
import pickle
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("trading_models_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("trading_models")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from config_loader import RL_GAMMA, RL_LEARNING_RATE, RL_EPSILON, RL_EPSILON_MIN, RL_EPSILON_DECAY, SYMBOLS
    from telegram_utils import send_async_message
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    async def send_async_message(msg: str) -> None:
        logger.warning(f"Telegram уведомления отключены: {msg}")
    raise SystemExit(1)

class TradingModels:
    def __init__(self, lookback: int = 60, input_features: int = 15, model_path: str = "models/transformer_model"):
        """
        Инициализация моделей торговли.

        Args:
            lookback (int): Количество свечей для анализа (по умолчанию 60).
            input_features (int): Количество входных признаков (по умолчанию 15).
            model_path (str): Путь для сохранения моделей (по умолчанию "models/transformer_model").
        """
        if not isinstance(lookback, int) or lookback <= 0:
            raise ValueError(f"lookback должен быть положительным целым числом, получено {lookback}")
        if not isinstance(input_features, int) or input_features <= 0:
            raise ValueError(f"input_features должен быть положительным целым числом, получено {input_features}")
        if not isinstance(model_path, str) or not model_path:
            raise ValueError(f"model_path должен быть непустой строкой, получено {model_path}")

        self.lookback = lookback
        self.input_features = input_features
        self.memory = deque(maxlen=100000)
        self.batch_size = 16
        self.amounts = {symbol: SYMBOLS[symbol]["amount"] for symbol in SYMBOLS}
        self.leverage = 10
        self.last_trained = time.time()
        self.device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        logger.info(f"Устройство для моделей: {self.device.split(':')[1]}")
        self.model_path = model_path
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        self.epsilon = RL_EPSILON * 2
        self.epsilon_min = RL_EPSILON_MIN
        self.epsilon_decay = 0.99
        self.gamma = RL_GAMMA
        self.learning_rate = RL_LEARNING_RATE
        self.action_space = 8  # hold, buy_long, sell_short, increase_amount, decrease_amount, grid_buy, grid_sell, arbitrage
        self._build_models()

    def _build_models(self) -> None:
        """Создание моделей LSTM, Transformer и DQN."""
        with tf.device(self.device):
            self.lstm_model = Sequential([
                Input(shape=(self.lookback, self.input_features)),
                LSTM(128, return_sequences=True),
                Dropout(0.3),
                LSTM(64),
                Dropout(0.3),
                Dense(32, activation="relu"),
                Dense(1)
            ])
            self.lstm_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")

            inputs = Input(shape=(self.lookback, self.input_features))
            x = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
            x = LayerNormalization(epsilon=1e-6)(x)
            x = GlobalAveragePooling1D()(x)
            x = Dense(128, activation="relu")(x)
            x = Dropout(0.3)(x)
            x = Dense(64, activation="relu")(x)
            outputs = Dense(1)(x)
            self.transformer_model = Model(inputs=inputs, outputs=outputs)
            self.transformer_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")

            self.dqn_model = Sequential([
                Input(shape=(self.lookback, self.input_features)),
                Dense(128, activation="relu"),
                Dropout(0.3),
                LSTM(64),
                Dense(32, activation="relu"),
                Dense(self.action_space, activation="linear")
            ])
            self.dqn_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")

            for model, name in [(self.lstm_model, "lstm"), (self.transformer_model, "transformer"), (self.dqn_model, "dqn")]:
                self._load_model_weights(model, name)

    def _load_model_weights(self, model: Union[Sequential, Model], name: str) -> None:
        """Загрузка весов модели из файла.

        Args:
            model (Union[Sequential, Model]): Модель для загрузки весов.
            name (str): Имя модели (lstm, transformer, dqn).
        """
        if not isinstance(model, (Sequential, Model)):
            logger.error(f"model должен быть экземпляром Sequential или Model, получено {type(model)}")
            return
        if not isinstance(name, str) or not name:
            logger.error(f"name должен быть непустой строкой, получено {name}")
            return

        weight_file = f"{self.model_path}_{name}.weights.h5"
        config_file = f"{self.model_path}_{name}_config.pkl"
        if os.path.exists(weight_file) and os.path.exists(config_file):
            try:
                with open(config_file, "rb") as f:
                    config = pickle.load(f)
                if config.get("lookback") != self.lookback or config.get("input_features") != self.input_features:
                    logger.warning(f"Несоответствие конфигурации {name.upper()} модели: lookback={self.lookback}, input_features={self.input_features}")
                else:
                    model.load_weights(weight_file)
                    logger.info(f"Загружены веса модели {name.upper()}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить веса модели {name.upper()} из-за ошибки: {e}")
                logger.info(f"Создаётся новая модель {name.upper()}")

    async def train_model(self, X: np.ndarray, y: np.ndarray, volatility: Optional[float] = None) -> None:
        """Асинхронное обучение моделей.

        Args:
            X (np.ndarray): Входные данные (тензор признаков).
            y (np.ndarray): Целевые значения (изменения цен).
            volatility (Optional[float]): Уровень волатильности (по умолчанию None).
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            logger.error(f"X и y должны быть экземплярами np.ndarray, получено {type(X)} и {type(y)}")
            return
        if X.shape[0] != y.shape[0]:
            logger.error(f"Количество образцов в X и y не совпадает: {X.shape[0]} != {y.shape[0]}")
            return
        min_data_size = self.batch_size * 2
        if X.shape[0] < min_data_size or X.shape[1] != self.lookback or X.shape[2] != self.input_features:
            logger.error(f"Некорректная форма входных данных: {X.shape}, ожидается (>={min_data_size}, {self.lookback}, {self.input_features})")
            return
        if volatility is not None and not isinstance(volatility, (int, float)) or volatility < 0:
            logger.error(f"volatility должен быть неотрицательным числом или None, получено {volatility}")
            volatility = None

        try:
            split_idx = int(X.shape[0] * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            with tf.device(self.device):
                self.lstm_model.fit(X_train, y_train, epochs=5, batch_size=self.batch_size, 
                                    validation_data=(X_val, y_val), verbose=0)
                lstm_val_loss = self.lstm_model.evaluate(X_val, y_val, verbose=0)
                logger.info(f"LSTM валидационная ошибка: {lstm_val_loss}")

                self.transformer_model.fit(X_train, y_train, epochs=5, batch_size=self.batch_size, 
                                          validation_data=(X_val, y_val), verbose=0)
                transformer_val_loss = self.transformer_model.evaluate(X_val, y_val, verbose=0)
                logger.info(f"Transformer валидационная ошибка: {transformer_val_loss}")

                if min(lstm_val_loss, transformer_val_loss) > 0.15:
                    logger.warning(f"Высокая ошибка валидации: {min(lstm_val_loss, transformer_val_loss)}")
                    await send_async_message(f"⚠️ Высокая ошибка валидации модели: {min(lstm_val_loss, transformer_val_loss)}")

                self._save_models()
                self.last_trained = time.time()

        except Exception as e:
            logger.error(f"Ошибка обучения моделей: {e}")
            await send_async_message(f"⚠️ Ошибка обучения моделей: {e}")

    def _save_models(self) -> None:
        """Сохранение всех моделей."""
        for model, name in [(self.lstm_model, "lstm"), (self.transformer_model, "transformer"), (self.dqn_model, "dqn")]:
            try:
                model.save_weights(f"{self.model_path}_{name}.weights.h5")
                with open(f"{self.model_path}_{name}_config.pkl", "wb") as f:
                    pickle.dump({"lookback": self.lookback, "input_features": self.input_features}, f)
                logger.info(f"Модель {name.upper()} сохранена")
            except Exception as e:
                logger.error(f"Ошибка сохранения модели {name.upper()}: {e}")
        logger.info("Все модели успешно сохранены")

    def predict(self, state: np.ndarray, volatility: Optional[float] = None) -> np.ndarray:
        """
        Предсказание изменения цены.

        Args:
            state (np.ndarray): Состояние (тензор признаков).
            volatility (Optional[float]): Уровень волатильности (по умолчанию None).

        Returns:
            np.ndarray: Предсказанное изменение цены.
        """
        if not isinstance(state, np.ndarray) or state.shape != (1, self.lookback, self.input_features):
            logger.error(f"state должен быть np.ndarray формы (1, {self.lookback}, {self.input_features}), получено {state.shape if isinstance(state, np.ndarray) else type(state)}")
            return np.array([0])
        if volatility is not None and (not isinstance(volatility, (int, float)) or volatility < 0):
            logger.error(f"volatility должен быть неотрицательным числом или None, получено {volatility}")
            volatility = None

        try:
            with tf.device(self.device):
                lstm_pred = self.lstm_model.predict(state, verbose=0)
                transformer_pred = self.transformer_model.predict(state, verbose=0)
                volatility_weight = min(0.5, max(0.1, (volatility or 0) * 2))
                avg_pred = lstm_pred * (1 - volatility_weight) + transformer_pred * volatility_weight
                return avg_pred

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return np.array([0])

    def get_action(self, state: np.ndarray, volatility: Optional[float] = None, sentiment: str = "neutral") -> int:
        """
        Получение действия для DQN модели.

        Args:
            state (np.ndarray): Состояние (тензор признаков).
            volatility (Optional[float]): Уровень волатильности (по умолчанию None).
            sentiment (str): Сентимент рынка (bullish, bearish, neutral; по умолчанию "neutral").

        Returns:
            int: Индекс действия.
        """
        if not isinstance(state, np.ndarray) or state.shape != (self.lookback, self.input_features):
            logger.error(f"state должен быть np.ndarray формы ({self.lookback}, {self.input_features}), получено {state.shape if isinstance(state, np.ndarray) else type(state)}")
            return 0
        if volatility is not None and (not isinstance(volatility, (int, float)) or volatility < 0):
            logger.error(f"volatility должен быть неотрицательным числом или None, получено {volatility}")
            volatility = None
        if not isinstance(sentiment, str) or sentiment not in ["bullish", "bearish", "neutral"]:
            logger.error(f"sentiment должен быть одним из ['bullish', 'bearish', 'neutral'], получено {sentiment}")
            sentiment = "neutral"

        try:
            state_expanded = np.expand_dims(state, axis=0)
            with tf.device(self.device):
                q_values = self.dqn_model.predict(state_expanded, verbose=0)[0]

            sentiment_factor = {"bullish": 0.2, "bearish": -0.2, "neutral": 0}.get(sentiment, 0)
            volatility_adj = min(0.3, max(0, (volatility or 0) * 1.5))
            q_values[1] += sentiment_factor + volatility_adj  # buy_long
            q_values[2] -= sentiment_factor - volatility_adj  # sell_short
            q_values[5] += volatility_adj  # grid_buy
            q_values[6] -= volatility_adj  # grid_sell

            if random.random() < self.epsilon:
                action = random.randint(0, self.action_space - 1)
                logger.info(f"Случайное действие выбрано (epsilon={self.epsilon}): {action}")
            else:
                action = int(np.argmax(q_values))
                logger.info(f"Действие выбрано по Q-значению: {action}, Q-значения: {q_values}")

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return action

        except Exception as e:
            logger.error(f"Ошибка получения действия: {e}")
            return 0

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Сохранение перехода в память.

        Args:
            state (np.ndarray): Текущее состояние.
            action (int): Выбранное действие.
            reward (float): Полученная награда.
            next_state (np.ndarray): Следующее состояние.
            done (bool): Завершено ли состояние.
        """
        if not isinstance(state, np.ndarray) or state.shape != (self.lookback, self.input_features):
            logger.error(f"state должен быть np.ndarray формы ({self.lookback}, {self.input_features}), получено {state.shape if isinstance(state, np.ndarray) else type(state)}")
            return
        if not isinstance(action, int) or action < 0 or action >= self.action_space:
            logger.error(f"action должен быть целым числом от 0 до {self.action_space - 1}, получено {action}")
            return
        if not isinstance(reward, (int, float)):
            logger.error(f"reward должен быть числом, получено {reward}")
            return
        if not isinstance(next_state, np.ndarray) or next_state.shape != (self.lookback, self.input_features):
            logger.error(f"next_state должен быть np.ndarray формы ({self.lookback}, {self.input_features}), получено {next_state.shape if isinstance(next_state, np.ndarray) else type(next_state)}")
            return
        if not isinstance(done, bool):
            logger.error(f"done должен быть булевым значением, получено {done}")
            return

        try:
            self.memory.append((state, action, reward, next_state, done))
        except Exception as e:
            logger.error(f"Ошибка сохранения перехода: {e}")

    def replay(self) -> None:
        """Обновление DQN модели через воспроизведение опыта."""
        try:
            if len(self.memory) < self.batch_size:
                return

            batch = random.sample(self.memory, self.batch_size)
            states = np.array([t[0] for t in batch])
            actions = np.array([t[1] for t in batch])
            rewards = np.array([t[2] for t in batch])
            next_states = np.array([t[3] for t in batch])
            dones = np.array([t[4] for t in batch])

            with tf.device(self.device):
                targets = self.dqn_model.predict(states, verbose=0)
                next_q_values = self.dqn_model.predict(next_states, verbose=0)
                for i in range(self.batch_size):
                    if dones[i]:
                        targets[i, actions[i]] = rewards[i]
                    else:
                        targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
                self.dqn_model.fit(states, targets, epochs=1, verbose=0)
                logger.info("DQN модель обновлена через replay")

        except Exception as e:
            logger.error(f"Ошибка воспроизведения опыта: {e}")
            asyncio.run_coroutine_threadsafe(send_async_message(f"⚠️ Ошибка воспроизведения опыта: {e}"), asyncio.get_event_loop())

    def update_amount(self, symbol: str, amount: float) -> None:
        """Обновление объёма для символа.

        Args:
            symbol (str): Символ (например, "BTC/USDT:USDT").
            amount (float): Новый объем.
        """
        if not isinstance(symbol, str) or symbol not in SYMBOLS:
            logger.error(f"symbol должен быть строкой из SYMBOLS, получено {symbol}")
            return
        if not isinstance(amount, (int, float)) or amount <= 0:
            logger.error(f"amount должен быть положительным числом, получено {amount}")
            return

        try:
            self.amounts[symbol] = max(0.01, min(10, amount))
            logger.info(f"Обновлен объем для {symbol}: {self.amounts[symbol]}")
        except Exception as e:
            logger.error(f"Ошибка обновления объема: {e}")

    async def should_retrain(self) -> bool:
        """Проверка необходимости переобучения.

        Returns:
            bool: True, если нужно переобучить модели.
        """
        try:
            return (time.time() - self.last_trained > 15 * 60) or (len(self.memory) > 500)
        except Exception as e:
            logger.error(f"Ошибка проверки необходимости переобучения: {e}")
            return False

if __name__ == "__main__":
    async def test():
        models = TradingModels()
        X = np.random.rand(100, 60, 15)
        y = np.random.rand(100)
        await models.train_model(X, y)
    asyncio.run(test())