"""Enhanced Version 5 Trading Strategies integrating all new features."""

import pandas as pd
import logging
from typing import List, Optional, Dict
from data_structures import TradeSignal, OptionsSignal
from indicators import AdvancedIndicators
from ml_predictor import MLPredictor
from deep_learning import DeepLearningPredictor
from multi_timeframe import MultiTimeframeAnalyzer
from sentiment_analysis import SentimentAnalyzer, create_sentiment_analyzer
from options_trading import OptionsAnalyzer
from config import TradingConfig

class EnhancedTradingStrategiesV5:
    """Version 5 strategies with deep learning, multi-timeframe, sentiment, and options"""

    def __init__(self, ml_predictor: MLPredictor,
                 news_api_key: Optional[str] = None,
                 twitter_bearer_token: Optional[str] = None):
        self.ml_predictor = ml_predictor
        self.logger = logging.getLogger(__name__)

        # NEW V5: Initialize advanced components
        self.dl_predictor = DeepLearningPredictor()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.sentiment_analyzer = create_sentiment_analyzer(news_api_key, twitter_bearer_token)
        self.options_analyzer = OptionsAnalyzer()

        self.logger.info("Enhanced V5 trading strategies initialized")

    def enhanced_momentum_strategy(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Enhanced momentum with all V5 features"""
        if len(data) < 50:
            return TradeSignal("", "HOLD", 0.0, 0.0, 0)

        current = data.iloc[-1]

        # Traditional momentum indicators
        trend_bullish = (
            current['Close'] > current['SMA_20'] and
            current['SMA_20'] > current['SMA_50'] and
            current['EMA_12'] > current['EMA_26']
        )

        momentum_bullish = (
            current['RSI'] > 50 and current['RSI'] < 80 and
            current['MACD'] > current['MACD_Signal'] and
            current['Volume_Ratio'] > 1.2
        )

        # NEW V5: Deep learning prediction
        dl_prediction = self.dl_predictor.predict(data, symbol)
        dl_bullish = dl_prediction.ensemble_prediction > 0.01
        dl_confidence = dl_prediction.confidence

        # NEW V5: Sentiment analysis
        sentiment_data = self.sentiment_analyzer.get_comprehensive_sentiment(symbol)
        sentiment_signal = self.sentiment_analyzer.get_sentiment_signal(sentiment_data)
        sentiment_bullish = sentiment_signal == "BUY"
        sentiment_confidence = self.sentiment_analyzer.get_sentiment_confidence(sentiment_data)

        # Combine all signals
        buy_score = sum([trend_bullish, momentum_bullish, dl_bullish, sentiment_bullish])

        if buy_score >= 3:  # Require strong confluence
            # Calculate confidence weighted by all sources
            confidence = (
                0.3 * (buy_score / 4) +  # Traditional indicators
                0.4 * dl_confidence +      # Deep learning
                0.3 * sentiment_confidence # Sentiment
            )

            confidence = min(0.9, confidence)

            stop_loss = current['Close'] - (current['ATR'] * 2)
            take_profit = current['Close'] + (current['ATR'] * 3)

            return TradeSignal(
                symbol=symbol,
                action="BUY",
                confidence=confidence,
                price=current['Close'],
                quantity=0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy="enhanced_momentum_v5",
                metadata={
                    'traditional_score': buy_score - int(dl_bullish) - int(sentiment_bullish),
                    'dl_prediction': dl_prediction.ensemble_prediction,
                    'sentiment': sentiment_data.combined_sentiment
                }
            )

        return TradeSignal(symbol, "HOLD", 0.0, current['Close'], 0)

    def multi_timeframe_strategy(self, symbol: str) -> Optional[TradeSignal]:
        """Strategy based on multi-timeframe analysis"""
        # Fetch multi-timeframe data
        mtf_data = self.mtf_analyzer.fetch_multi_timeframe_data(symbol)

        if mtf_data is None:
            return None

        # Generate signal from multi-timeframe analysis
        signal = self.mtf_analyzer.generate_mtf_signal(mtf_data)

        if signal and signal.action != "HOLD":
            # Enhance with higher timeframe context
            context = self.mtf_analyzer.get_higher_timeframe_context(mtf_data)

            # Adjust confidence based on context
            if context['daily_trend'] == 'BULLISH' and signal.action == 'BUY':
                signal.confidence = min(0.95, signal.confidence * 1.2)
            elif context['daily_trend'] == 'BEARISH' and signal.action == 'SELL':
                signal.confidence = min(0.95, signal.confidence * 1.2)

            signal.metadata['higher_tf_context'] = context

        return signal

    def sentiment_driven_strategy(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Strategy primarily based on sentiment analysis"""
        if len(data) < 20:
            return TradeSignal("", "HOLD", 0.0, 0.0, 0)

        current = data.iloc[-1]

        # Get comprehensive sentiment
        sentiment_data = self.sentiment_analyzer.get_comprehensive_sentiment(symbol)

        # Require strong sentiment and sufficient data
        if sentiment_data.news_count + sentiment_data.social_mentions < 10:
            return TradeSignal(symbol, "HOLD", 0.0, current['Close'], 0)

        sentiment_signal = self.sentiment_analyzer.get_sentiment_signal(sentiment_data)

        if sentiment_signal == "HOLD":
            return TradeSignal(symbol, "HOLD", 0.0, current['Close'], 0)

        # Confirm with technical indicators
        tech_confirmation = False
        if sentiment_signal == "BUY":
            tech_confirmation = current['RSI'] < 70 and current['Close'] > current['SMA_20']
        elif sentiment_signal == "SELL":
            tech_confirmation = current['RSI'] > 30 and current['Close'] < current['SMA_20']

        if tech_confirmation:
            confidence = self.sentiment_analyzer.get_sentiment_confidence(sentiment_data)
            confidence = min(0.85, confidence)

            if sentiment_signal == "BUY":
                return TradeSignal(
                    symbol=symbol,
                    action="BUY",
                    confidence=confidence,
                    price=current['Close'],
                    quantity=0,
                    stop_loss=current['Close'] - (current['ATR'] * 2),
                    take_profit=current['Close'] + (current['ATR'] * 3),
                    strategy="sentiment_driven_v5",
                    metadata={
                        'sentiment': sentiment_data.combined_sentiment,
                        'news_count': sentiment_data.news_count,
                        'social_mentions': sentiment_data.social_mentions
                    }
                )

        return TradeSignal(symbol, "HOLD", 0.0, current['Close'], 0)

    def options_strategy(self, data: pd.DataFrame, symbol: str) -> Optional[OptionsSignal]:
        """Generate options trading signals"""
        if len(data) < 50:
            return None

        # Determine market outlook
        trend = AdvancedIndicators.get_trend_direction(data)

        # Map trend to options strategy
        market_outlook_map = {
            'BULLISH': 'BULLISH',
            'BEARISH': 'BEARISH',
            'SIDEWAYS': 'NEUTRAL'
        }

        market_outlook = market_outlook_map.get(trend, 'NEUTRAL')

        # Generate options signal
        options_signal = self.options_analyzer.generate_options_signal(
            symbol, data, market_outlook
        )

        return options_signal

    def hybrid_strategy(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Hybrid strategy combining all V5 features"""
        if len(data) < 60:
            return TradeSignal("", "HOLD", 0.0, 0.0, 0)

        current = data.iloc[-1]

        # 1. Traditional technical analysis
        tech_score = 0
        if current['Close'] > current['SMA_50']:
            tech_score += 1
        if current['RSI'] > 50 and current['RSI'] < 70:
            tech_score += 1
        if current['MACD'] > current['MACD_Signal']:
            tech_score += 1

        # 2. Machine learning (traditional)
        ml_up_prob, ml_confidence = self.ml_predictor.predict_direction(data)
        ml_score = 1 if ml_up_prob > 0.6 else 0

        # 3. Deep learning
        dl_prediction = self.dl_predictor.predict(data, symbol)
        dl_score = 1 if dl_prediction.ensemble_prediction > 0.01 else 0

        # 4. Multi-timeframe
        mtf_signal = self.multi_timeframe_strategy(symbol)
        mtf_score = 1 if mtf_signal and mtf_signal.action == "BUY" else 0

        # 5. Sentiment
        sentiment_data = self.sentiment_analyzer.get_comprehensive_sentiment(symbol)
        sentiment_score = 1 if sentiment_data.combined_sentiment > 0.1 else 0

        # Combined decision
        total_score = tech_score + ml_score + dl_score + mtf_score + sentiment_score
        max_score = 7  # 3 tech + 1 ml + 1 dl + 1 mtf + 1 sentiment

        if total_score >= 5:  # Require strong consensus
            # Weighted confidence
            confidence = (
                (tech_score / 3) * 0.2 +
                ml_confidence * 0.2 +
                dl_prediction.confidence * 0.3 +
                (mtf_score) * 0.15 +
                (sentiment_score) * 0.15
            )

            confidence = min(0.95, confidence)

            return TradeSignal(
                symbol=symbol,
                action="BUY",
                confidence=confidence,
                price=current['Close'],
                quantity=0,
                stop_loss=current['Close'] - (current['ATR'] * 2.5),
                take_profit=current['Close'] + (current['ATR'] * 4),
                strategy="hybrid_v5",
                metadata={
                    'tech_score': tech_score,
                    'ml_score': ml_score,
                    'dl_score': dl_score,
                    'mtf_score': mtf_score,
                    'sentiment_score': sentiment_score,
                    'total_score': total_score,
                    'max_score': max_score
                }
            )

        return TradeSignal(symbol, "HOLD", 0.0, current['Close'], 0)

    def generate_all_signals(self, data: pd.DataFrame, symbol: str) -> List[TradeSignal]:
        """Generate signals from all available strategies"""
        signals = []

        # Enhanced momentum
        momentum_signal = self.enhanced_momentum_strategy(data, symbol)
        if momentum_signal.action != "HOLD":
            signals.append(momentum_signal)

        # Multi-timeframe
        mtf_signal = self.multi_timeframe_strategy(symbol)
        if mtf_signal and mtf_signal.action != "HOLD":
            signals.append(mtf_signal)

        # Sentiment driven
        sentiment_signal = self.sentiment_driven_strategy(data, symbol)
        if sentiment_signal.action != "HOLD":
            signals.append(sentiment_signal)

        # Hybrid (combines all)
        hybrid_signal = self.hybrid_strategy(data, symbol)
        if hybrid_signal.action != "HOLD":
            signals.append(hybrid_signal)

        return signals

    def get_best_signal(self, signals: List[TradeSignal]) -> Optional[TradeSignal]:
        """Select the best signal from multiple strategies"""
        if not signals:
            return None

        if len(signals) == 1:
            return signals[0]

        # Weight by confidence and strategy type
        best_signal = max(signals, key=lambda s: s.confidence)

        # Prefer hybrid strategy if confidence is similar
        hybrid_signals = [s for s in signals if s.strategy == "hybrid_v5"]
        if hybrid_signals:
            hybrid_best = max(hybrid_signals, key=lambda s: s.confidence)
            if hybrid_best.confidence >= best_signal.confidence * 0.9:
                return hybrid_best

        return best_signal
