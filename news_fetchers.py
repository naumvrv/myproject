# news_fetchers.py
import asyncio
import aiohttp
import feedparser
import asyncpraw
from typing import List, Tuple, Any
import logging

# Попытка импорта зависимостей с базовым логгером в случае сбоя
try:
    from logging_setup import setup_logging
    logger = setup_logging("news_fetchers_log.txt")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("news_fetchers")
    logger.warning(f"Не удалось импортировать logging_setup: {e}. Используется базовый логгер.")

try:
    from config_loader import NEWSAPI_KEY, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, SYMBOLS
    from cache_manager import get_cached_data, save_cached_data
    from fine_tune_distilbert import SentimentModel
except ImportError as e:
    logger.error(f"Не удалось импортировать зависимости: {e}")
    raise SystemExit(1)

RSS_FEEDS_DEFAULT = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptopanic.com/news/rss/",
]

async def fetch_rss_feeds() -> List[str]:
    """
    Получение списка RSS-лент.

    Returns:
        List[str]: Список URL RSS-лент.
    """
    return RSS_FEEDS_DEFAULT

async def fetch_rss_news(analyzer: SentimentModel, interval_minutes: int = 15) -> Tuple[float, int]:
    """
    Асинхронное получение новостей из RSS-лент и анализ их сентимента.

    Args:
        analyzer (SentimentModel): Модель для анализа сентимента.
        interval_minutes (int): Интервал кэширования в минутах (по умолчанию 15).

    Returns:
        Tuple[float, int]: Кортеж (суммарный сентимент, количество новостей).
    """
    if not isinstance(analyzer, SentimentModel):
        logger.error(f"analyzer должен быть экземпляром SentimentModel, получен {type(analyzer)}")
        return 0.0, 0
    if not isinstance(interval_minutes, int) or interval_minutes <= 0:
        logger.error(f"interval_minutes должен быть положительным числом, получен {interval_minutes}")
        interval_minutes = 15

    cache_key = f"rss_news_sentiment:{interval_minutes}"
    cached_data = await get_cached_data(cache_key)
    if cached_data and isinstance(cached_data, dict) and "sentiment" in cached_data and "count" in cached_data:
        logger.info(f"Использованы кэшированные данные RSS: {cached_data}")
        return cached_data["sentiment"], cached_data["count"]

    news_sentiment = 0.0
    news_count = 0
    keywords = ["bitcoin", "crypto", "cryptocurrency", "market", "volatility", "economy", "blockchain"] + [
        s.split("/")[0].lower() for s in SYMBOLS if isinstance(s, str)
    ]
    rss_feeds = await fetch_rss_feeds()
    
    async with aiohttp.ClientSession() as session:
        for feed_url in rss_feeds:
            for attempt in range(3):
                try:
                    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        feed_content = await response.text()
                        feed = feedparser.parse(feed_content)
                        if feed.bozo:
                            logger.warning(f"Некорректный RSS {feed_url}")
                            break
                        for entry in feed.entries[:20]:
                            text = f"{entry.get('title', '')} {entry.get('description', '')}".lower()
                            if any(keyword in text for keyword in keywords):
                                score = analyzer.analyze_text(text)
                                weight = 1 + len(entry.get("links", [])) * 0.1
                                news_sentiment += score * weight
                                news_count += 1
                        break  # Успешный запрос, выходим из цикла попыток
                except Exception as e:
                    logger.error(f"Ошибка парсинга RSS {feed_url} (попытка {attempt+1}/3): {e}")
                    if attempt == 2:
                        logger.error(f"Не удалось получить RSS {feed_url} после 3 попыток")
                    await asyncio.sleep(5 * (attempt + 1))
    
    if news_count < 5:
        logger.warning(f"Получено недостаточно новостей из RSS: {news_count}")
    avg_sentiment = news_sentiment / max(news_count, 1) if news_count > 0 else 0.0
    result = {"sentiment": avg_sentiment, "count": news_count}
    await save_cached_data(cache_key, result, ttl=interval_minutes * 60)
    return avg_sentiment, news_count

async def fetch_newsapi_sentiment(analyzer: SentimentModel, interval_minutes: int = 15) -> Tuple[float, int]:
    """
    Асинхронное получение новостей из NewsAPI и анализ их сентимента.

    Args:
        analyzer (SentimentModel): Модель для анализа сентимента.
        interval_minutes (int): Интервал кэширования в минутах (по умолчанию 15).

    Returns:
        Tuple[float, int]: Кортеж (суммарный сентимент, количество новостей).
    """
    if not isinstance(analyzer, SentimentModel):
        logger.error(f"analyzer должен быть экземпляром SentimentModel, получен {type(analyzer)}")
        return 0.0, 0
    if not isinstance(interval_minutes, int) or interval_minutes <= 0:
        logger.error(f"interval_minutes должен быть положительным числом, получен {interval_minutes}")
        interval_minutes = 15

    cache_key = f"newsapi_sentiment:{interval_minutes}"
    cached_data = await get_cached_data(cache_key)
    if cached_data and isinstance(cached_data, dict) and "sentiment" in cached_data and "count" in cached_data:
        logger.info(f"Использованы кэшированные данные NewsAPI: {cached_data}")
        return cached_data["sentiment"], cached_data["count"]

    async with aiohttp.ClientSession() as session:
        url = f"https://newsapi.org/v2/everything?q=bitcoin+OR+crypto&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
        for attempt in range(3):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        raise ValueError(f"Ошибка NewsAPI: {response.status}")
                    data = await response.json()
                    if not isinstance(data, dict) or "articles" not in data:
                        raise ValueError(f"Некорректный ответ NewsAPI: {data}")
                    articles = data["articles"][:10]
                    sentiment_sum = 0.0
                    count = 0
                    for article in articles:
                        if not isinstance(article, dict):
                            continue
                        text = f"{article.get('title', '')} {article.get('description', '') or ''}".lower()
                        score = analyzer.analyze_text(text)
                        source_id = article.get("source", {}).get("id")
                        weight = 1.2 if source_id else 1.0  # Упрощенная логика веса
                        sentiment_sum += score * weight
                        count += 1
                    avg_sentiment = sentiment_sum / max(count, 1) if count > 0 else 0.0
                    logger.info(f"Сентимент из NewsAPI: {avg_sentiment}, обработано статей: {count}")
                    result = {"sentiment": avg_sentiment, "count": count}
                    await save_cached_data(cache_key, result, ttl=interval_minutes * 60)
                    return avg_sentiment, count
            except Exception as e:
                logger.error(f"Ошибка анализа NewsAPI (попытка {attempt+1}/3): {e}")
                if attempt == 2:
                    return 0.0, 0
                await asyncio.sleep(5 * (attempt + 1))
    return 0.0, 0

async def fetch_reddit_sentiment(analyzer: SentimentModel, interval_minutes: int = 15) -> Tuple[float, int]:
    """
    Асинхронное получение постов из Reddit и анализ их сентимента.

    Args:
        analyzer (SentimentModel): Модель для анализа сентимента.
        interval_minutes (int): Интервал кэширования в минутах (по умолчанию 15).

    Returns:
        Tuple[float, int]: Кортеж (суммарный сентимент, количество постов).
    """
    if not isinstance(analyzer, SentimentModel):
        logger.error(f"analyzer должен быть экземпляром SentimentModel, получен {type(analyzer)}")
        return 0.0, 0
    if not isinstance(interval_minutes, int) or interval_minutes <= 0:
        logger.error(f"interval_minutes должен быть положительным числом, получен {interval_minutes}")
        interval_minutes = 15

    cache_key = f"reddit_sentiment:{interval_minutes}"
    cached_data = await get_cached_data(cache_key)
    if cached_data and isinstance(cached_data, dict) and "sentiment" in cached_data and "count" in cached_data:
        logger.info(f"Использованы кэшированные данные Reddit: {cached_data}")
        return cached_data["sentiment"], cached_data["count"]

    try:
        reddit = asyncpraw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        sentiment_sum = 0.0
        count = 0
        keywords = ["bitcoin", "crypto", "cryptocurrency", "market", "volatility", "economy", "blockchain"] + [
            s.split("/")[0].lower() for s in SYMBOLS if isinstance(s, str)
        ]
        
        subreddit = await reddit.subreddit("CryptoCurrency+Bitcoin")
        async for submission in subreddit.hot(limit=20):
            try:
                text = f"{submission.title} {submission.selftext or ''}".lower()
                if any(keyword in text for keyword in keywords):
                    score = analyzer.analyze_text(text)
                    weight = 1 + max(submission.score, 0) * 0.01  # Защита от отрицательных значений
                    sentiment_sum += score * weight
                    count += 1
            except AttributeError as e:
                logger.warning(f"Некорректный пост Reddit: {e}")
                continue
        avg_sentiment = sentiment_sum / max(count, 1) if count > 0 else 0.0
        logger.info(f"Сентимент из Reddit: {avg_sentiment}, обработано постов: {count}")
        result = {"sentiment": avg_sentiment, "count": count}
        await save_cached_data(cache_key, result, ttl=interval_minutes * 60)
        await reddit.close()
        return avg_sentiment, count
    except Exception as e:
        logger.error(f"Ошибка анализа Reddit: {e}")
        await reddit.close()
        return 0.0, 0

if __name__ == "__main__":
    from fine_tune_distilbert import DistilBertSentimentModel
    async def test():
        analyzer = DistilBertSentimentModel()
        rss_sentiment, rss_count = await fetch_rss_news(analyzer)
        logger.info(f"RSS: Sentiment={rss_sentiment}, Count={rss_count}")
        newsapi_sentiment, newsapi_count = await fetch_newsapi_sentiment(analyzer)
        logger.info(f"NewsAPI: Sentiment={newsapi_sentiment}, Count={newsapi_count}")
        reddit_sentiment, reddit_count = await fetch_reddit_sentiment(analyzer)
        logger.info(f"Reddit: Sentiment={reddit_sentiment}, Count={reddit_count}")
    asyncio.run(test())