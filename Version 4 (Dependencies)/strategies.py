"""Trading strategies for the algorithm."""

import pandas as pd
import logging
from typing import List

from data_structures import TradeSignal
from ml_predictor import MLPredictor
from indicators import AdvancedIndicators
from config import TradingConfig

class AdvancedTradingStrategies:
    """Collection of advanced trading strategies"""
    
    def __init__(self, ml_predictor: MLPredictor):
        self.ml_predictor = ml_predictor
        self.logger = logging.getLogger(__name__)
    
    def momentum_breakout_strategy(self, data: pd.DataFrame) -> TradeSignal:
        """Enhanced momentum breakout strategy"""
        if len(data) < 50:
            return TradeSignal("", "HOLD", 0.0, 0.0, 0)
        
        current = data.iloc[-1]
        symbol = getattr(data, 'symbol', 'UNKNOWN')
        
        # Trend confirmation
        trend_bullish = (
            current['Close'] > current['SMA_20'] and
            current['SMA_20'] > current['SMA_50'] and
            current['EMA_12'] > current['EMA_26']
        )
        
        trend_bearish = (
            current['Close'] < current['SMA_20'] and
            current['SMA_20'] < current['SMA_50'] and
            current['EMA_12'] < current['EMA_26']
        )
        
        # Momentum conditions
        momentum_bullish = (
            current['RSI'] > 50 and current['RSI'] < 80 and
            current['MACD'] > current['MACD_Signal'] and
            current['Volume_Ratio'] > 1.2
        )
        
        momentum_bearish = (
            current['RSI'] < 50 and current['RSI'] > 20 and
            current['MACD'] < current['MACD_Signal']
        )
        
        # Volatility filter
        volatility_acceptable = current['ATR_Norm'] < 0.05  # Not too volatile
        
        # ML prediction
        ml_up_prob, ml_confidence = self.ml_predictor.predict_direction(data)
        ml_bullish = ml_up_prob > 0.6
        ml_bearish = ml_up_prob < 0.4
        
        # Combined signal
        buy_score = sum([trend_bullish, momentum_bullish, ml_bullish, volatility_acceptable])
        sell_score = sum([trend_bearish, momentum_bearish, ml_bearish])
        
        if buy_score >= 3:
            confidence = min(0.8, (buy_score / 4) * (1 + ml_confidence))
            stop_loss = current['Close'] - (current['ATR'] * 2)
            take_profit = current['Close'] + (current['ATR'] * 3)
            
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                confidence=confidence,
                price=current['Close'],
                quantity=0,  # Will be calculated later
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy="momentum_breakout"
            )
        
        elif sell_score >= 2:
            confidence = min(0.7, (sell_score / 3) * (1 + ml_confidence))
            return TradeSignal(
                symbol=symbol,
                action="SELL",
                confidence=confidence,
                price=current['Close'],
                quantity=0,
                strategy="momentum_breakout"
            )
        
        return TradeSignal(symbol, "HOLD", 0.0, current['Close'], 0)
    
    def mean_reversion_strategy(self, data: pd.DataFrame) -> TradeSignal:
        """Mean reversion strategy for oversold/overbought conditions"""
        if len(data) < 30:
            return TradeSignal("", "HOLD", 0.0, 0.0, 0)
        
        current = data.iloc[-1]
        symbol = getattr(data, 'symbol', 'UNKNOWN')
        
        # Oversold conditions
        oversold = AdvancedIndicators.is_oversold(data)
        
        # Overbought conditions
        overbought = AdvancedIndicators.is_overbought(data)
        
        # Trend context (prefer counter-trend in ranging markets)
        weak_trend = True  # Default to weak trend when ADX not available
        if 'ADX' in current and not pd.isna(current['ADX']):
            weak_trend = current['ADX'] < 25
        
        if oversold and weak_trend:
            confidence = min(0.7, (30 - current['RSI']) / 30 + 0.3)
            take_profit = current['BB_Middle']  # Target middle Bollinger Band
            stop_loss = current['Close'] - (current['ATR'] * 1.5)
            
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                confidence=confidence,
                price=current['Close'],
                quantity=0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy="mean_reversion"
            )
        
        elif overbought and weak_trend:
            confidence = min(0.7, (current['RSI'] - 70) / 30 + 0.3)
            return TradeSignal(
                symbol=symbol,
                action="SELL",
                confidence=confidence,
                price=current['Close'],
                quantity=0,
                strategy="mean_reversion"
            )
        
        return TradeSignal(symbol, "HOLD", 0.0, current['Close'], 0)
    
    def trend_following_strategy(self, data: pd.DataFrame) -> TradeSignal:
        """Trend following strategy using multiple timeframes"""
        if len(data) < 100:
            return TradeSignal("", "HOLD", 0.0, 0.0, 0)
        
        current = data.iloc[-1]
        symbol = getattr(data, 'symbol', 'UNKNOWN')
        
        # Trend direction
        trend = AdvancedIndicators.get_trend_direction(data)
        
        # Trend strength
        trend_strength = current.get('ADX', 25)
        strong_trend = trend_strength > 25
        
        # Entry conditions for bullish trend
        if trend == "BULLISH" and strong_trend:
            # Look for pullback to moving average
            pullback_buy = (
                current['Close'] > current['SMA_50'] and
                current['Close'] < current['SMA_20'] and
                current['RSI'] < 60
            )
            
            if pullback_buy:
                confidence = min(0.75, trend_strength / 50 + 0.25)
                stop_loss = current['SMA_50']
                take_profit = current['Close'] + (current['ATR'] * 4)
                
                return TradeSignal(
                    symbol=symbol,
                    action="BUY",
                    confidence=confidence,
                    price=current['Close'],
                    quantity=0,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy="trend_following"
                )
        
        # Entry conditions for bearish trend
        elif trend == "BEARISH" and strong_trend:
            # Look for rally to moving average
            pullback_sell = (
                current['Close'] < current['SMA_50'] and
                current['Close'] > current['SMA_20'] and
                current['RSI'] > 40
            )
            
            if pullback_sell:
                confidence = min(0.75, trend_strength / 50 + 0.25)
                return TradeSignal(
                    symbol=symbol,
                    action="SELL",
                    confidence=confidence,
                    price=current['Close'],
                    quantity=0,
                    strategy="trend_following"
                )
        
        return TradeSignal(symbol, "HOLD", 0.0, current['Close'], 0)
    
    def breakout_strategy(self, data: pd.DataFrame) -> TradeSignal:
        """Breakout strategy using Bollinger Bands and volume"""
        if len(data) < 20:
            return TradeSignal("", "HOLD", 0.0, 0.0, 0)
        
        current = data.iloc[-1]
        previous = data.iloc[-2]
        symbol = getattr(data, 'symbol', 'UNKNOWN')
        
        # Bollinger Band breakout
        bb_breakout_up = (
            previous['Close'] <= previous['BB_Upper'] and
            current['Close'] > current['BB_Upper'] and
            current['Volume_Ratio'] > 1.5
        )
        
        bb_breakout_down = (
            previous['Close'] >= previous['BB_Lower'] and
            current['Close'] < current['BB_Lower'] and
            current['Volume_Ratio'] > 1.2
        )
        
        # Volatility expansion
        volatility_expanding = current['BB_Width'] > data['BB_Width'].rolling(10).mean().iloc[-1]
        
        if bb_breakout_up and volatility_expanding:
            confidence = min(0.8, current['Volume_Ratio'] / 3)
            stop_loss = current['BB_Upper'] - (current['ATR'] * 1.5)
            take_profit = current['Close'] + (current['ATR'] * 3)
            
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                confidence=confidence,
                price=current['Close'],
                quantity=0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy="breakout"
            )
        
        elif bb_breakout_down and volatility_expanding:
            confidence = min(0.7, current['Volume_Ratio'] / 3)
            return TradeSignal(
                symbol=symbol,
                action="SELL",
                confidence=confidence,
                price=current['Close'],
                quantity=0,
                strategy="breakout"
            )
        
        return TradeSignal(symbol, "HOLD", 0.0, current['Close'], 0)
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate signals from all strategies and return the best ones"""
        signals = []
        
        # Get signals from all strategies
        momentum_signal = self.momentum_breakout_strategy(data)
        if momentum_signal.action != "HOLD":
            signals.append(momentum_signal)
        
        mean_reversion_signal = self.mean_reversion_strategy(data)
        if mean_reversion_signal.action != "HOLD":
            signals.append(mean_reversion_signal)
        
        trend_signal = self.trend_following_strategy(data)
        if trend_signal.action != "HOLD":
            signals.append(trend_signal)
        
        breakout_signal = self.breakout_strategy(data)
        if breakout_signal.action != "HOLD":
            signals.append(breakout_signal)
        
        return signals
    
    def create_ensemble_signal(self, signals: List[TradeSignal]) -> TradeSignal:
        """Create ensemble signal from multiple strategies"""
        if not signals:
            return TradeSignal("", "HOLD", 0.0, 0.0, 0)
        
        if len(signals) == 1:
            return signals[0]
        
        # Weight by confidence and create ensemble
        buy_signals = [s for s in signals if s.action == "BUY"]
        sell_signals = [s for s in signals if s.action == "SELL"]
        
        buy_weight = sum(s.confidence for s in buy_signals)
        sell_weight = sum(s.confidence for s in sell_signals)
        
        symbol = signals[0].symbol
        price = signals[0].price
        
        if buy_weight > sell_weight and buy_weight > 0.7:
            best_buy = max(buy_signals, key=lambda x: x.confidence)
            ensemble_confidence = min(0.9, buy_weight / len(signals))
            
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                confidence=ensemble_confidence,
                price=price,
                quantity=0,
                stop_loss=best_buy.stop_loss,
                take_profit=best_buy.take_profit,
                strategy="ensemble"
            )
        
        elif sell_weight > buy_weight and sell_weight > 0.7:
            best_sell = max(sell_signals, key=lambda x: x.confidence)
            ensemble_confidence = min(0.9, sell_weight / len(signals))
            
            return TradeSignal(
                symbol=symbol,
                action="SELL",
                confidence=ensemble_confidence,
                price=price,
                quantity=0,
                strategy="ensemble"
            )
        
        return TradeSignal(symbol, "HOLD", 0.0, price, 0)