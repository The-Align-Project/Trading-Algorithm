"""Technical indicators for the trading algorithm."""

import pandas as pd
import numpy as np
from config import TALIB_AVAILABLE, TradingConfig

if TALIB_AVAILABLE:
    import talib

class SimpleIndicators:
    """Simplified technical indicators when TA-Lib is not available"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD indicator"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: int = 2):
        """Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent

class AdvancedIndicators:
    """Advanced technical indicators using TA-Lib or fallback to simple indicators"""
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = data.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        if TALIB_AVAILABLE:
            # Use TA-Lib if available
            close_values = close.values
            high_values = high.values
            low_values = low.values
            volume_values = volume.values
            
            # Price-based indicators
            df['SMA_10'] = talib.SMA(close_values, timeperiod=TradingConfig.SMA_SHORT)
            df['SMA_20'] = talib.SMA(close_values, timeperiod=TradingConfig.SMA_MEDIUM)
            df['SMA_50'] = talib.SMA(close_values, timeperiod=TradingConfig.SMA_LONG)
            df['SMA_200'] = talib.SMA(close_values, timeperiod=TradingConfig.SMA_VERY_LONG)
            
            df['EMA_12'] = talib.EMA(close_values, timeperiod=TradingConfig.EMA_FAST)
            df['EMA_26'] = talib.EMA(close_values, timeperiod=TradingConfig.EMA_SLOW)
            df['EMA_50'] = talib.EMA(close_values, timeperiod=TradingConfig.SMA_LONG)
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(close_values)
            
            # RSI variants
            df['RSI'] = talib.RSI(close_values, timeperiod=TradingConfig.RSI_WINDOW)
            df['RSI_2'] = talib.RSI(close_values, timeperiod=2)
            df['RSI_50'] = talib.RSI(close_values, timeperiod=50)
            
            # Stochastic
            df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high_values, low_values, close_values)
            
            # Williams %R
            df['WILLR'] = talib.WILLR(high_values, low_values, close_values)
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
                close_values, timeperiod=TradingConfig.BB_WINDOW, nbdevup=TradingConfig.BB_STD_DEV, 
                nbdevdn=TradingConfig.BB_STD_DEV
            )
            
            # ATR
            df['ATR'] = talib.ATR(high_values, low_values, close_values, timeperiod=TradingConfig.ATR_WINDOW)
            
            # Momentum indicators
            df['MOM'] = talib.MOM(close_values, timeperiod=10)
            df['ROC'] = talib.ROC(close_values, timeperiod=10)
            df['CCI'] = talib.CCI(high_values, low_values, close_values, timeperiod=14)
            
            # Volume indicators
            df['OBV'] = talib.OBV(close_values, volume_values)
            df['AD'] = talib.AD(high_values, low_values, close_values, volume_values)
            
            # ADX
            df['ADX'] = talib.ADX(high_values, low_values, close_values, timeperiod=14)
            df['DI_Plus'] = talib.PLUS_DI(high_values, low_values, close_values, timeperiod=14)
            df['DI_Minus'] = talib.MINUS_DI(high_values, low_values, close_values, timeperiod=14)
            
        else:
            # Use simple indicators as fallback
            # Price-based indicators
            df['SMA_10'] = SimpleIndicators.sma(close, TradingConfig.SMA_SHORT)
            df['SMA_20'] = SimpleIndicators.sma(close, TradingConfig.SMA_MEDIUM)
            df['SMA_50'] = SimpleIndicators.sma(close, TradingConfig.SMA_LONG)
            df['SMA_200'] = SimpleIndicators.sma(close, TradingConfig.SMA_VERY_LONG)
            
            df['EMA_12'] = SimpleIndicators.ema(close, TradingConfig.EMA_FAST)
            df['EMA_26'] = SimpleIndicators.ema(close, TradingConfig.EMA_SLOW)
            df['EMA_50'] = SimpleIndicators.ema(close, TradingConfig.SMA_LONG)
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = SimpleIndicators.macd(close)
            
            # RSI
            df['RSI'] = SimpleIndicators.rsi(close, TradingConfig.RSI_WINDOW)
            df['RSI_2'] = SimpleIndicators.rsi(close, 2)
            df['RSI_50'] = SimpleIndicators.rsi(close, 50)
            
            # Stochastic
            df['STOCH_K'], df['STOCH_D'] = SimpleIndicators.stochastic(high, low, close)
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = SimpleIndicators.bollinger_bands(
                close, TradingConfig.BB_WINDOW, TradingConfig.BB_STD_DEV
            )
            
            # ATR
            df['ATR'] = SimpleIndicators.atr(high, low, close, TradingConfig.ATR_WINDOW)
            
            # Simplified momentum indicators
            df['MOM'] = close.pct_change(10)
            df['ROC'] = close.pct_change(10)
            
            # Default values for indicators not implemented in simple version
            df['WILLR'] = -50  # Neutral value
            df['CCI'] = 0
            df['OBV'] = volume.cumsum()
            df['AD'] = 0
            df['ADX'] = 25  # Moderate trend strength
            df['DI_Plus'] = 25
            df['DI_Minus'] = 25
        
        # Common calculations regardless of TA-Lib availability
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['ATR_Norm'] = df['ATR'] / close
        
        # Custom indicators
        df['Price_Change'] = close.pct_change()
        df['Volatility_20'] = df['Price_Change'].rolling(20).std()
        df['Volume_Ratio'] = volume / SimpleIndicators.sma(volume, 20)
        
        # Support and Resistance levels
        df['Pivot'] = (high + low + close) / 3
        df['R1'] = 2 * df['Pivot'] - low
        df['S1'] = 2 * df['Pivot'] - high
        
        # Price ratios for ML features
        df['SMA_Ratio'] = close / df['SMA_20']
        df['EMA_Ratio'] = df['EMA_12'] / df['EMA_26']
        
        # Trend strength
        df['Trend_Strength'] = abs(df['SMA_20'] - df['SMA_50']) / df['SMA_50']
        
        return df
    
    @staticmethod
    def get_trend_direction(data: pd.DataFrame) -> str:
        """Determine overall trend direction"""
        if len(data) < 20:
            return "SIDEWAYS"
        
        current = data.iloc[-1]
        
        # Multiple timeframe trend analysis
        short_trend = current['SMA_10'] > current['SMA_20']
        medium_trend = current['SMA_20'] > current['SMA_50']
        long_trend = current['SMA_50'] > current['SMA_200'] if 'SMA_200' in current else True
        
        bullish_signals = sum([short_trend, medium_trend, long_trend])
        
        if bullish_signals >= 2:
            return "BULLISH"
        elif bullish_signals <= 1:
            return "BEARISH"
        else:
            return "SIDEWAYS"
    
    @staticmethod
    def is_oversold(data: pd.DataFrame) -> bool:
        """Check if the asset is oversold"""
        if len(data) == 0:
            return False
        
        current = data.iloc[-1]
        return (
            current.get('RSI', 50) < 30 and
            current.get('STOCH_K', 50) < 20 and
            current.get('BB_Position', 0.5) < 0.2
        )
    
    @staticmethod
    def is_overbought(data: pd.DataFrame) -> bool:
        """Check if the asset is overbought"""
        if len(data) == 0:
            return False
        
        current = data.iloc[-1]
        return (
            current.get('RSI', 50) > 70 and
            current.get('STOCH_K', 50) > 80 and
            current.get('BB_Position', 0.5) > 0.8
        )