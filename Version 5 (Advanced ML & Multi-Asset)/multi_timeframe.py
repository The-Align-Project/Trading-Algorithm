"""Multi-timeframe analysis module for comprehensive market analysis."""

import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from data_structures import MultiTimeframeData, TradeSignal
from config import YF_AVAILABLE, TradingConfig

if YF_AVAILABLE:
    import yfinance as yf

class MultiTimeframeAnalyzer:
    """Analyzes price action across multiple timeframes"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.timeframes = TradingConfig.TIMEFRAMES
        self.cache = {}  # Cache for recent data

    def fetch_multi_timeframe_data(self, symbol: str) -> Optional[MultiTimeframeData]:
        """Fetch data for all timeframes"""
        if not YF_AVAILABLE:
            self.logger.warning("yfinance not available. Cannot fetch multi-timeframe data.")
            return None

        try:
            timeframe_data = {}

            for timeframe in self.timeframes:
                period, interval = self._map_timeframe_to_yfinance(timeframe)

                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)

                if data.empty:
                    self.logger.warning(f"No data for {symbol} at {timeframe}")
                    continue

                timeframe_data[timeframe] = data

            if not timeframe_data:
                return None

            return MultiTimeframeData(
                symbol=symbol,
                timeframes=timeframe_data
            )

        except Exception as e:
            self.logger.error(f"Error fetching multi-timeframe data for {symbol}: {e}")
            return None

    def _map_timeframe_to_yfinance(self, timeframe: str) -> tuple:
        """Map our timeframe notation to yfinance parameters"""
        mapping = {
            '1m': ('1d', '1m'),
            '5m': ('5d', '5m'),
            '15m': ('5d', '15m'),
            '1h': ('1mo', '1h'),
            '1d': ('1y', '1d')
        }
        return mapping.get(timeframe, ('1y', '1d'))

    def analyze_trend_alignment(self, mtf_data: MultiTimeframeData) -> Dict[str, str]:
        """Analyze if trends align across timeframes"""
        trends = {}

        for timeframe, data in mtf_data.timeframes.items():
            if len(data) < 50:
                trends[timeframe] = 'NEUTRAL'
                continue

            current = data.iloc[-1]

            # Calculate SMAs if not present
            if 'SMA_20' not in data.columns:
                data['SMA_20'] = data['Close'].rolling(20).mean()
            if 'SMA_50' not in data.columns:
                data['SMA_50'] = data['Close'].rolling(50).mean()

            # Determine trend
            if current['Close'] > current['SMA_20'] > current['SMA_50']:
                trends[timeframe] = 'BULLISH'
            elif current['Close'] < current['SMA_20'] < current['SMA_50']:
                trends[timeframe] = 'BEARISH'
            else:
                trends[timeframe] = 'NEUTRAL'

        return trends

    def get_trend_strength_score(self, mtf_data: MultiTimeframeData) -> float:
        """Calculate trend strength across timeframes (0-1)"""
        trends = self.analyze_trend_alignment(mtf_data)

        if not trends:
            return 0.0

        bullish_count = sum(1 for t in trends.values() if t == 'BULLISH')
        bearish_count = sum(1 for t in trends.values() if t == 'BEARISH')
        total = len(trends)

        # Positive score for bullish, negative for bearish
        if bullish_count > bearish_count:
            return bullish_count / total
        elif bearish_count > bullish_count:
            return -(bearish_count / total)
        else:
            return 0.0

    def detect_confluence_zones(self, mtf_data: MultiTimeframeData) -> Dict[str, List[float]]:
        """Detect support/resistance zones with multi-timeframe confluence"""
        support_levels = []
        resistance_levels = []

        for timeframe, data in mtf_data.timeframes.items():
            if len(data) < 20:
                continue

            # Find recent highs and lows
            high_roll = data['High'].rolling(20)
            low_roll = data['Low'].rolling(20)

            recent_high = high_roll.max().iloc[-1]
            recent_low = low_roll.min().iloc[-1]

            resistance_levels.append(recent_high)
            support_levels.append(recent_low)

        # Cluster levels that are close to each other
        clustered_resistance = self._cluster_levels(resistance_levels)
        clustered_support = self._cluster_levels(support_levels)

        return {
            'support': clustered_support,
            'resistance': clustered_resistance
        }

    def _cluster_levels(self, levels: List[float], threshold_pct: float = 0.02) -> List[float]:
        """Cluster price levels that are close together"""
        if not levels:
            return []

        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold_pct:
                current_cluster.append(level)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))

        return clusters

    def generate_mtf_signal(self, mtf_data: MultiTimeframeData) -> Optional[TradeSignal]:
        """Generate trading signal based on multi-timeframe analysis"""
        trends = self.analyze_trend_alignment(mtf_data)
        trend_strength = self.get_trend_strength_score(mtf_data)

        # Get primary timeframe data
        primary_tf = TradingConfig.PRIMARY_TIMEFRAME
        primary_data = mtf_data.get_timeframe(primary_tf)

        if primary_data is None or len(primary_data) == 0:
            return None

        current_price = primary_data['Close'].iloc[-1]

        # Strong bullish signal - most timeframes aligned
        if trend_strength > 0.6:
            # Calculate stop loss and take profit
            atr = primary_data.get('ATR', pd.Series([current_price * 0.02])).iloc[-1]

            return TradeSignal(
                symbol=mtf_data.symbol,
                action="BUY",
                confidence=abs(trend_strength),
                price=current_price,
                quantity=0,
                stop_loss=current_price - (atr * 2),
                take_profit=current_price + (atr * 3),
                strategy="multi_timeframe",
                metadata={
                    'trends': trends,
                    'trend_strength': trend_strength,
                    'timeframes_analyzed': len(trends)
                }
            )

        # Strong bearish signal
        elif trend_strength < -0.6:
            return TradeSignal(
                symbol=mtf_data.symbol,
                action="SELL",
                confidence=abs(trend_strength),
                price=current_price,
                quantity=0,
                strategy="multi_timeframe",
                metadata={
                    'trends': trends,
                    'trend_strength': trend_strength,
                    'timeframes_analyzed': len(trends)
                }
            )

        return None

    def get_higher_timeframe_context(self, mtf_data: MultiTimeframeData) -> Dict:
        """Get context from higher timeframes for decision making"""
        context = {
            'daily_trend': 'NEUTRAL',
            'hourly_trend': 'NEUTRAL',
            'near_resistance': False,
            'near_support': False
        }

        # Daily trend
        if '1d' in mtf_data.timeframes:
            daily_data = mtf_data.timeframes['1d']
            if len(daily_data) >= 50:
                current = daily_data.iloc[-1]
                if 'SMA_20' in daily_data.columns and 'SMA_50' in daily_data.columns:
                    if current['Close'] > current['SMA_20'] > current['SMA_50']:
                        context['daily_trend'] = 'BULLISH'
                    elif current['Close'] < current['SMA_20'] < current['SMA_50']:
                        context['daily_trend'] = 'BEARISH'

        # Hourly trend
        if '1h' in mtf_data.timeframes:
            hourly_data = mtf_data.timeframes['1h']
            if len(hourly_data) >= 20:
                current = hourly_data.iloc[-1]
                if 'SMA_20' not in hourly_data.columns:
                    hourly_data['SMA_20'] = hourly_data['Close'].rolling(20).mean()

                if current['Close'] > current['SMA_20']:
                    context['hourly_trend'] = 'BULLISH'
                elif current['Close'] < current['SMA_20']:
                    context['hourly_trend'] = 'BEARISH'

        # Check if near support/resistance
        confluence_zones = self.detect_confluence_zones(mtf_data)
        primary_price = mtf_data.timeframes[TradingConfig.PRIMARY_TIMEFRAME]['Close'].iloc[-1]

        for resistance in confluence_zones['resistance']:
            if abs(primary_price - resistance) / primary_price < 0.01:  # Within 1%
                context['near_resistance'] = True
                break

        for support in confluence_zones['support']:
            if abs(primary_price - support) / primary_price < 0.01:  # Within 1%
                context['near_support'] = True
                break

        return context
