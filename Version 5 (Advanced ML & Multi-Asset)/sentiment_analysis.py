"""Sentiment analysis module for news and social media."""

import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from data_structures import SentimentData, NewsArticle, SocialPost
from config import SENTIMENT_AVAILABLE, NEWS_API_AVAILABLE, TWITTER_AVAILABLE, TradingConfig

if SENTIMENT_AVAILABLE:
    from textblob import TextBlob
    import nltk

if NEWS_API_AVAILABLE:
    from newsapi import NewsApiClient

if TWITTER_AVAILABLE:
    import tweepy

class SentimentAnalyzer:
    """Analyzes sentiment from various sources"""

    def __init__(self, news_api_key: Optional[str] = None,
                 twitter_bearer_token: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        # Initialize APIs if available
        self.news_client = None
        self.twitter_client = None

        if NEWS_API_AVAILABLE and news_api_key:
            try:
                self.news_client = NewsApiClient(api_key=news_api_key)
                self.logger.info("News API initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize News API: {e}")

        if TWITTER_AVAILABLE and twitter_bearer_token:
            try:
                self.twitter_client = tweepy.Client(bearer_token=twitter_bearer_token)
                self.logger.info("Twitter API initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Twitter API: {e}")

        # Download NLTK data if needed
        if SENTIMENT_AVAILABLE:
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                self.logger.info("Downloading NLTK vader_lexicon...")
                nltk.download('vader_lexicon', quiet=True)

    def analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        if not SENTIMENT_AVAILABLE or not text:
            return 0.0

        try:
            blob = TextBlob(text)
            # TextBlob returns polarity from -1 to 1
            return blob.sentiment.polarity
        except Exception as e:
            self.logger.error(f"Error analyzing text sentiment: {e}")
            return 0.0

    def fetch_news_sentiment(self, symbol: str, lookback_hours: int = 24) -> Tuple[List[NewsArticle], float]:
        """Fetch and analyze news sentiment"""
        articles = []
        avg_sentiment = 0.0

        if not self.news_client:
            self.logger.warning("News API not available")
            return articles, avg_sentiment

        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(hours=lookback_hours)

            # Fetch news
            company_name = self._get_company_name(symbol)
            query = f"{symbol} OR {company_name}"

            response = self.news_client.get_everything(
                q=query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=20
            )

            if response['status'] == 'ok':
                sentiments = []

                for article_data in response['articles']:
                    # Analyze sentiment of title and description
                    title = article_data.get('title', '')
                    description = article_data.get('description', '')
                    content = f"{title}. {description}"

                    sentiment = self.analyze_text_sentiment(content)
                    sentiments.append(sentiment)

                    article = NewsArticle(
                        title=title,
                        description=description,
                        source=article_data.get('source', {}).get('name', 'Unknown'),
                        url=article_data.get('url', ''),
                        published_at=datetime.fromisoformat(
                            article_data.get('publishedAt', '').replace('Z', '+00:00')
                        ),
                        sentiment=sentiment
                    )
                    articles.append(article)

                # Calculate average sentiment
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)

                self.logger.info(f"Fetched {len(articles)} news articles for {symbol}, avg sentiment: {avg_sentiment:.3f}")

        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {e}")

        return articles, avg_sentiment

    def fetch_social_sentiment(self, symbol: str, max_posts: int = 100) -> Tuple[List[SocialPost], float]:
        """Fetch and analyze social media sentiment"""
        posts = []
        avg_sentiment = 0.0

        if not self.twitter_client:
            self.logger.warning("Twitter API not available")
            return posts, avg_sentiment

        try:
            # Search for tweets about the symbol
            query = f"${symbol} OR #{symbol}"

            response = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=min(max_posts, 100),
                tweet_fields=['created_at', 'public_metrics', 'author_id']
            )

            if response.data:
                sentiments = []

                for tweet in response.data:
                    text = tweet.text
                    sentiment = self.analyze_text_sentiment(text)
                    sentiments.append(sentiment)

                    # Calculate engagement (likes + retweets)
                    metrics = tweet.public_metrics
                    engagement = metrics.get('like_count', 0) + metrics.get('retweet_count', 0)

                    post = SocialPost(
                        text=text,
                        source='twitter',
                        author=str(tweet.author_id),
                        timestamp=tweet.created_at,
                        engagement=engagement,
                        sentiment=sentiment
                    )
                    posts.append(post)

                # Weighted average by engagement
                if sentiments:
                    total_engagement = sum(p.engagement for p in posts)
                    if total_engagement > 0:
                        weighted_sum = sum(p.sentiment * p.engagement for p in posts)
                        avg_sentiment = weighted_sum / total_engagement
                    else:
                        avg_sentiment = sum(sentiments) / len(sentiments)

                self.logger.info(f"Fetched {len(posts)} social posts for {symbol}, avg sentiment: {avg_sentiment:.3f}")

        except Exception as e:
            self.logger.error(f"Error fetching social sentiment for {symbol}: {e}")

        return posts, avg_sentiment

    def get_comprehensive_sentiment(self, symbol: str) -> SentimentData:
        """Get comprehensive sentiment analysis combining all sources"""

        # Fetch news sentiment
        news_articles, news_sentiment = self.fetch_news_sentiment(
            symbol, TradingConfig.NEWS_LOOKBACK_HOURS
        )

        # Fetch social sentiment
        social_posts, social_sentiment = self.fetch_social_sentiment(symbol)

        # Combine sentiments with weights
        news_weight = 0.6
        social_weight = 0.4

        combined_sentiment = (news_sentiment * news_weight + social_sentiment * social_weight)

        sources = []
        if self.news_client:
            sources.append('news_api')
        if self.twitter_client:
            sources.append('twitter')

        sentiment_data = SentimentData(
            symbol=symbol,
            news_sentiment=news_sentiment,
            social_sentiment=social_sentiment,
            combined_sentiment=combined_sentiment,
            news_count=len(news_articles),
            social_mentions=len(social_posts),
            sources=sources
        )

        return sentiment_data

    def get_sentiment_signal(self, sentiment_data: SentimentData) -> str:
        """Convert sentiment data to trading signal"""
        sentiment = sentiment_data.combined_sentiment

        # Require minimum data points
        min_sources = 5
        total_sources = sentiment_data.news_count + sentiment_data.social_mentions

        if total_sources < min_sources:
            return "HOLD"  # Not enough data

        threshold = TradingConfig.SENTIMENT_THRESHOLD

        if sentiment > threshold:
            return "BUY"
        elif sentiment < -threshold:
            return "SELL"
        else:
            return "HOLD"

    def get_sentiment_confidence(self, sentiment_data: SentimentData) -> float:
        """Calculate confidence based on sentiment data quality"""
        # More sources = higher confidence
        source_factor = min(1.0, (sentiment_data.news_count + sentiment_data.social_mentions) / 50)

        # Stronger sentiment = higher confidence
        sentiment_factor = min(1.0, abs(sentiment_data.combined_sentiment) * 2)

        confidence = (source_factor * 0.5 + sentiment_factor * 0.5)
        return confidence

    def _get_company_name(self, symbol: str) -> str:
        """Map symbol to company name for better news search"""
        # Basic mapping - extend as needed
        company_map = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google Alphabet',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'NVDA': 'Nvidia',
            'META': 'Meta Facebook',
            'NFLX': 'Netflix',
            'AMD': 'AMD',
            'CRM': 'Salesforce'
        }
        return company_map.get(symbol, symbol)

    def analyze_sentiment_trend(self, symbol: str, days: int = 7) -> Dict[str, float]:
        """Analyze sentiment trend over time"""
        # This would fetch historical sentiment data
        # For now, return current sentiment
        sentiment_data = self.get_comprehensive_sentiment(symbol)

        return {
            'current': sentiment_data.combined_sentiment,
            'trend': 0.0,  # Would calculate trend from historical data
            'volatility': 0.0  # Would calculate from historical data
        }

class FakeSentimentAnalyzer:
    """Fallback analyzer when sentiment libraries not available"""

    def __init__(self, news_api_key: Optional[str] = None,
                 twitter_bearer_token: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Sentiment analysis libraries not available. Using fake analyzer.")

    def get_comprehensive_sentiment(self, symbol: str) -> SentimentData:
        """Return neutral sentiment"""
        return SentimentData(
            symbol=symbol,
            news_sentiment=0.0,
            social_sentiment=0.0,
            combined_sentiment=0.0,
            news_count=0,
            social_mentions=0,
            sources=[]
        )

    def get_sentiment_signal(self, sentiment_data: SentimentData) -> str:
        return "HOLD"

    def get_sentiment_confidence(self, sentiment_data: SentimentData) -> float:
        return 0.0

# Factory function
def create_sentiment_analyzer(news_api_key: Optional[str] = None,
                              twitter_bearer_token: Optional[str] = None):
    """Create appropriate sentiment analyzer based on available libraries"""
    if SENTIMENT_AVAILABLE:
        return SentimentAnalyzer(news_api_key, twitter_bearer_token)
    else:
        return FakeSentimentAnalyzer(news_api_key, twitter_bearer_token)
