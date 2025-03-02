# fine_tune_distilbert.py
import asyncio
import os
import platform
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer, PreTrainedModel, Trainer, TrainingArguments
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
import pandas as pd
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("fine_tune_distilbert_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fine_tune_distilbert")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from telegram_utils import send_async_message
    from news_fetchers import fetch_rss_news, fetch_newsapi_sentiment, fetch_reddit_sentiment
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    async def send_async_message(msg: str) -> None:
        logger.warning(f"Telegram уведомления отключены: {msg}")
    raise SystemExit(1)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Отключаем oneDNN предупреждения

class SentimentModel(ABC):
    """Абстрактный базовый класс для моделей анализа сентимента."""
    
    @abstractmethod
    def train(self, dataset: Dataset) -> None:
        """Обучение модели на данных.

        Args:
            dataset (Dataset): Датасет для обучения.
        """
        pass
    
    @abstractmethod
    def save(self, model_dir: str) -> None:
        """Сохранение модели.

        Args:
            model_dir (str): Путь для сохранения модели.
        """
        pass
    
    @abstractmethod
    def analyze_text(self, text: str) -> int:
        """Анализ текста и возврат сентимента (0: негатив, 1: нейтрал, 2: позитив).

        Args:
            text (str): Текст для анализа.

        Returns:
            int: Сентимент текста.
        """
        pass

class DistilBertSentimentModel(SentimentModel):
    def __init__(self, model_path: str = "fine_tuned_distilbert"):
        """
        Инициализация модели DistilBERT для анализа сентимента.

        Args:
            model_path (str): Путь для сохранения/загрузки модели (по умолчанию "fine_tuned_distilbert").
        """
        if not isinstance(model_path, str) or not model_path:
            logger.error(f"model_path должен быть непустой строкой, получен {model_path}")
            model_path = "fine_tuned_distilbert"
        self.model_path = model_path
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            logger.info("Загружена дообученная модель DistilBERT")
        except Exception as e:
            logger.warning(f"Ошибка загрузки дообученной модели: {e}. Используется базовая модель.")
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
        
        self.training_args = TrainingArguments(
            output_dir=f"{model_path}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir="fine_tuned_distilbert_logs",
            logging_steps=5,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
        )
        self.trainer = None

    def tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, Any]:
        """Токенизация текстов для обучения.

        Args:
            examples (Dict[str, List[str]]): Словарь с текстами для токенизации.

        Returns:
            Dict[str, Any]: Токенизированные данные.
        """
        return self.tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    def train(self, dataset: Dataset) -> None:
        """Обучение модели на данных.

        Args:
            dataset (Dataset): Датасет для обучения.
        """
        if not isinstance(dataset, Dataset):
            logger.error(f"dataset должен быть экземпляром Dataset, получен {type(dataset)}")
            raise ValueError("dataset должен быть экземпляром Dataset")
        try:
            tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["sentence"])
            tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
            tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=tokenized_dataset.select(range(int(len(tokenized_dataset) * 0.2)))
            )
            self.trainer.train()
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            raise

    def save(self, model_dir: str) -> None:
        """Сохранение модели и токенизатора.

        Args:
            model_dir (str): Путь для сохранения модели.
        """
        if not isinstance(model_dir, str) or not model_dir:
            logger.error(f"model_dir должен быть непустой строкой, получен {model_dir}")
            return
        try:
            self.model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)
            logger.info(f"Модель сохранена в '{model_dir}'")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели в {model_dir}: {e}")

    def analyze_text(self, text: str) -> int:
        """Анализ текста и возврат сентимента.

        Args:
            text (str): Текст для анализа.

        Returns:
            int: Сентимент текста (0: негатив, 1: нейтрал, 2: позитив).
        """
        if not isinstance(text, str) or not text:
            logger.error(f"text должен быть непустой строкой, получен {text}")
            return 1  # Нейтральный по умолчанию
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = logits.argmax().item()
            return prediction  # 0: негатив, 1: нейтрал, 2: позитив
        except Exception as e:
            logger.error(f"Ошибка анализа текста: {e}")
            return 1  # Нейтральный по умолчанию

async def collect_news_data(analyzer: SentimentModel) -> List[Dict[str, Any]]:
    """
    Сбор данных из новостных источников для дообучения.

    Args:
        analyzer (SentimentModel): Экземпляр модели для анализа сентимента.

    Returns:
        List[Dict[str, Any]]: Список словарей с новостями и их сентиментами.
    """
    if not isinstance(analyzer, SentimentModel):
        logger.error(f"analyzer должен быть экземпляром SentimentModel, получен {type(analyzer)}")
        return []
    try:
        news_sentiment, news_count = await fetch_rss_news(analyzer)
        api_sentiment, api_count = await fetch_newsapi_sentiment(analyzer)
        reddit_sentiment, reddit_count = await fetch_reddit_sentiment(analyzer)
        
        news_data = []
        if isinstance(news_count, int) and news_count > 0 and isinstance(news_sentiment, (int, float)):
            news_data.extend([{"sentence": f"Sample RSS news {i}", "label": 2 if news_sentiment > 0 else 0 if news_sentiment < 0 else 1} 
                             for i in range(min(news_count, 50))])
        if isinstance(api_count, int) and api_count > 0 and isinstance(api_sentiment, (int, float)):
            news_data.extend([{"sentence": f"Sample NewsAPI {i}", "label": 2 if api_sentiment > 0 else 0 if api_sentiment < 0 else 1} 
                             for i in range(min(api_count, 50))])
        if isinstance(reddit_count, int) and reddit_count > 0 and isinstance(reddit_sentiment, (int, float)):
            news_data.extend([{"sentence": f"Sample Reddit {i}", "label": 2 if reddit_sentiment > 0 else 0 if reddit_sentiment < 0 else 1} 
                             for i in range(min(reddit_count, 50))])
        
        return news_data
    except Exception as e:
        logger.error(f"Ошибка сбора новостных данных: {e}")
        return []

async def fine_tune_model(model: SentimentModel, new_data: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    Дообучение модели на новых данных.

    Args:
        model (SentimentModel): Экземпляр модели сентимента.
        new_data (Optional[List[Dict[str, Any]]]): Дополнительные данные для дообучения (по умолчанию None).
    """
    if not isinstance(model, SentimentModel):
        logger.error(f"model должен быть экземпляром SentimentModel, получен {type(model)}")
        return
    if new_data is not None and (not isinstance(new_data, list) or not all(isinstance(d, dict) and "sentence" in d and "label" in d for d in new_data)):
        logger.error(f"new_data должен быть списком словарей с ключами 'sentence' и 'label', получен {new_data}")
        new_data = None

    try:
        logger.info("Начало дообучения модели...")
        # Базовый датасет как заглушка
        dataset = Dataset.from_pandas(pd.DataFrame({
            "sentence": ["Positive example", "Negative example", "Neutral example"],
            "label": [2, 0, 1]
        }))
        logger.warning("Используется заглушка базового датасета. Замените на реальные данные для полноценного обучения.")
        
        news_data = await collect_news_data(model)
        if new_data:
            news_data.extend(new_data)

        if len(news_data) < 50:
            logger.warning(f"Недостаточно данных для дообучения: {len(news_data)} < 50, использую только базовый датасет")
        else:
            new_df = pd.DataFrame(news_data)
            dataset = Dataset.from_pandas(new_df)
            logger.info(f"Добавлено {len(news_data)} новых записей для дообучения")

        model.train(dataset)
        model.save("fine_tuned_distilbert")
        await send_async_message("✅ Модель успешно дообучена")
    except Exception as e:
        logger.error(f"Ошибка обучения: {e}")
        await send_async_message(f"⚠️ Ошибка дообучения модели: {e}")

async def auto_fine_tune(model: SentimentModel) -> None:
    """
    Автоматическое ежедневное дообучение модели.

    Args:
        model (SentimentModel): Экземпляр модели сентимента.
    """
    if not isinstance(model, SentimentModel):
        logger.error(f"model должен быть экземпляром SentimentModel, получен {type(model)}")
        return
    while True:
        try:
            now = datetime.now()
            next_run = datetime(now.year, now.month, now.day, 0, 0) + timedelta(days=1)
            sleep_seconds = (next_run - now).total_seconds()
            await asyncio.sleep(sleep_seconds)
            await fine_tune_model(model)
            logger.info("Ежедневное дообучение выполнено")
        except Exception as e:
            logger.error(f"Ошибка в цикле автоматического дообучения: {e}")
            await asyncio.sleep(3600)  # Задержка 1 час перед следующей попыткой

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    model = DistilBertSentimentModel()
    try:
        loop.run_until_complete(fine_tune_model(model))
        loop.create_task(auto_fine_tune(model))
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(model.save("fine_tuned_distilbert_interrupt"))
        loop.close()
        logger.info("Программа завершена по сигналу прерывания")
    except Exception as e:
        logger.error(f"Ошибка при запуске программы: {e}")
        loop.close()