import pandas as pd
import numpy as np
import warnings
import threading
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime, timedelta
import time
import json

warnings.filterwarnings('ignore')

# Try importing required packages with graceful fallbacks
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: yfinance import error: {e}")
    print("Installing websockets might fix this: pip install websockets>=11.0")
    YF_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available. Some statistical functions may not work.")
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. ML predictions disabled.")
    SKLEARN_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    print("Warning: TA-Lib not available. Using simplified technical indicators.")
    TALIB_AVAILABLE = False

# Alpaca imports
try:
    from alpaca_trade_api import REST, Stream
    from alpaca_trade_api.common import URL
    ALPACA_AVAILABLE = True
except ImportError:
    print("Warning: Alpaca Trade API not installed. Run: pip install alpaca-trade-api")
    ALPACA_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class TradeSignal:
    """Enhanced trade signal with confidence and metadata"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    price: float
    quantity: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = ""
    timestamp: datetime = None
    metadata: dict = None

@dataclass
class RiskMetrics:
    """Risk management metrics"""
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    beta: float = 0.0
    correlation_spy: float = 0.0

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
            df['SMA_10'] = talib.SMA(close_values, timeperiod=10)
            df['SMA_20'] = talib.SMA(close_values, timeperiod=20)
            df['SMA_50'] = talib.SMA(close_values, timeperiod=50)
            df['SMA_200'] = talib.SMA(close_values, timeperiod=200)
            
            df['EMA_12'] = talib.EMA(close_values, timeperiod=12)
            df['EMA_26'] = talib.EMA(close_values, timeperiod=26)
            df['EMA_50'] = talib.EMA(close_values, timeperiod=50)
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(close_values)
            
            # RSI variants
            df['RSI'] = talib.RSI(close_values, timeperiod=14)
            df['RSI_2'] = talib.RSI(close_values, timeperiod=2)
            df['RSI_50'] = talib.RSI(close_values, timeperiod=50)
            
            # Stochastic
            df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high_values, low_values, close_values)
            
            # Williams %R
            df['WILLR'] = talib.WILLR(high_values, low_values, close_values)
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(close_values)
            
            # ATR
            df['ATR'] = talib.ATR(high_values, low_values, close_values, timeperiod=14)
            
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
            df['SMA_10'] = SimpleIndicators.sma(close, 10)
            df['SMA_20'] = SimpleIndicators.sma(close, 20)
            df['SMA_50'] = SimpleIndicators.sma(close, 50)
            df['SMA_200'] = SimpleIndicators.sma(close, 200)
            
            df['EMA_12'] = SimpleIndicators.ema(close, 12)
            df['EMA_26'] = SimpleIndicators.ema(close, 26)
            df['EMA_50'] = SimpleIndicators.ema(close, 50)
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = SimpleIndicators.macd(close)
            
            # RSI
            df['RSI'] = SimpleIndicators.rsi(close, 14)
            df['RSI_2'] = SimpleIndicators.rsi(close, 2)
            df['RSI_50'] = SimpleIndicators.rsi(close, 50)
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = SimpleIndicators.bollinger_bands(close)
            
            # ATR
            df['ATR'] = SimpleIndicators.atr(high, low, close)
            
            # Simplified momentum indicators
            df['MOM'] = close.pct_change(10)
            df['ROC'] = close.pct_change(10)
            
            # Default values for indicators not implemented in simple version
            df['STOCH_K'] = 50  # Neutral value
            df['STOCH_D'] = 50
            df['WILLR'] = -50
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
        
        return df

class MLPredictor:
    """Machine Learning predictor for price movements"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.is_trained = False
        
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model"""
        if not SKLEARN_AVAILABLE:
            return np.array([])
            
        features = [
            'RSI', 'MACD', 'MACD_Signal', 'BB_Position', 'BB_Width',
            'Volume_Ratio', 'Price_Change', 'Volatility_20', 'ATR_Norm'
        ]
        
        # Add price ratios
        data['SMA_Ratio'] = data['Close'] / data['SMA_20']
        data['EMA_Ratio'] = data['EMA_12'] / data['EMA_26']
        features.extend(['SMA_Ratio', 'EMA_Ratio'])
        
        # Add lagged features
        for lag in [1, 2, 3]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
            features.extend([f'Close_Lag_{lag}', f'Volume_Lag_{lag}'])
        
        # Filter features that exist in the data
        available_features = [f for f in features if f in data.columns]
        
        return data[available_features].dropna()
    
    def train_model(self, data: pd.DataFrame) -> bool:
        """Train the ML model"""
        if not SKLEARN_AVAILABLE:
            return False
            
        try:
            features_df = self.prepare_features(data)
            if len(features_df) < 100:
                return False
            
            # Create target (future price direction)
            target = (data['Close'].shift(-1) > data['Close']).astype(int)
            target = target.iloc[:-1]  # Remove last NaN
            
            # Align features and target
            min_len = min(len(features_df), len(target))
            X = features_df.iloc[:min_len].values
            y = target.iloc[:min_len].values
            
            # Remove any remaining NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = GradientBoostingClassifier(
                n_estimators=50,  # Reduced for faster training
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            self.feature_importance = self.model.feature_importances_
            self.is_trained = True
            
            return True
        
        except Exception as e:
            logging.error(f"Error training ML model: {e}")
            return False
    
    def predict_direction(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Predict price direction probability"""
        if not SKLEARN_AVAILABLE or not self.is_trained or self.model is None:
            return 0.5, 0.0  # Neutral prediction
        
        try:
            features_df = self.prepare_features(data)
            if len(features_df) == 0:
                return 0.5, 0.0
            
            # Use last row for prediction
            X = features_df.iloc[-1:].values
            if np.isnan(X).any():
                return 0.5, 0.0
            
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Return probability of upward movement and confidence
            up_prob = probabilities[1] if len(probabilities) > 1 else 0.5
            confidence = abs(up_prob - 0.5) * 2  # Convert to 0-1 scale
            
            return up_prob, confidence
        
        except Exception as e:
            logging.error(f"Error in ML prediction: {e}")
            return 0.5, 0.0

class AdvancedTradingStrategies:
    """Collection of advanced trading strategies"""
    
    def __init__(self, ml_predictor: MLPredictor):
        self.ml_predictor = ml_predictor
    
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
        oversold = (
            current['RSI'] < 30 and
            current['BB_Position'] < 0.1  # Near lower Bollinger Band
        )
        
        # Overbought conditions
        overbought = (
            current['RSI'] > 70 and
            current['BB_Position'] > 0.9  # Near upper Bollinger Band
        )
        
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

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.max_position_size = 0.1  # Max 10% per position
        self.max_daily_loss = 0.02  # Max 2% daily loss
        self.current_drawdown = 0.0
        
        # Risk metrics tracking
        self.daily_returns = []
        self.portfolio_values = []
        self.peak_value = initial_capital
        
    def calculate_position_size(self, signal: TradeSignal, portfolio_value: float, 
                              volatility: float) -> int:
        """Calculate optimal position size using simplified Kelly Criterion"""
        
        # Simplified Kelly Criterion
        win_prob = signal.confidence
        avg_win = 0.02  # Assume 2% average win
        avg_loss = 0.01  # Assume 1% average loss
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Volatility adjustment
        vol_adjustment = min(1.0, 0.02 / max(volatility, 0.01))  # Target 2% volatility
        
        # Portfolio heat (total risk exposure)
        portfolio_heat_adjustment = max(0.5, 1.0 - self.current_drawdown)
        
        # Combined position size
        optimal_fraction = kelly_fraction * vol_adjustment * portfolio_heat_adjustment
        optimal_fraction = min(optimal_fraction, self.max_position_size)
        
        position_value = portfolio_value * optimal_fraction
        quantity = int(position_value / signal.price)
        
        return max(0, quantity)
    
    def update_risk_metrics(self, portfolio_value: float) -> RiskMetrics:
        """Update and calculate risk metrics"""
        self.portfolio_values.append(portfolio_value)
        
        if len(self.portfolio_values) > 1:
            daily_return = (portfolio_value / self.portfolio_values[-2]) - 1
            self.daily_returns.append(daily_return)
        
        # Update peak and drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        
        # Calculate metrics
        if len(self.daily_returns) < 30:
            return RiskMetrics()
        
        returns_series = pd.Series(self.daily_returns)
        
        # Sharpe Ratio (assuming risk-free rate of 2%)
        excess_returns = returns_series - (0.02 / 252)  # Daily risk-free rate
        sharpe_ratio = excess_returns.mean() / max(excess_returns.std(), 0.001) * np.sqrt(252)
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        if len(downside_returns) > 0:
            sortino_ratio = returns_series.mean() / max(downside_returns.std(), 0.001) * np.sqrt(252)
        else:
            sortino_ratio = sharpe_ratio
        
        # Value at Risk (95%)
        var_95 = returns_series.quantile(0.05) if len(returns_series) >= 20 else 0
        
        return RiskMetrics(
            max_drawdown=self.current_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95
        )
    
    def should_stop_trading(self, current_loss: float, portfolio_value: float) -> bool:
        """Determine if trading should be stopped due to risk limits"""
        daily_loss_pct = current_loss / max(portfolio_value, 1)
        
        return (
            daily_loss_pct > self.max_daily_loss or
            self.current_drawdown > 0.15  # Max 15% drawdown
        )

class SimplifiedDataFetcher:
    """Simplified data fetcher with alternative to yfinance"""
    
    @staticmethod
    def generate_sample_data(symbol: str, days: int = 252) -> pd.DataFrame:
        """Generate realistic sample stock data for testing"""
        np.random.seed(hash(symbol) % 2**32)  # Consistent data for same symbol
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic price movement
        returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
        price = 100  # Starting price
        prices = [price]
        
        for ret in returns[1:]:
            price *= (1 + ret)
            prices.append(price)
        
        # Generate OHLCV data
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            high = close_price * (1 + abs(np.random.normal(0, 0.01)))
            low = close_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else close_price
            volume = int(np.random.normal(1000000, 500000))
            
            data.append({
                'Open': open_price,
                'High': max(high, open_price, close_price),
                'Low': min(low, open_price, close_price),
                'Close': close_price,
                'Volume': max(volume, 10000)
            })
        
        df = pd.DataFrame(data, index=dates)
        df.symbol = symbol
        return df
    
    @staticmethod
    def fetch_data(symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch data with fallback to sample data"""
        try:
            if YF_AVAILABLE:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval="1d")
                if not data.empty:
                    data.symbol = symbol
                    return data
            
            # Fallback to sample data
            days_map = {"1y": 252, "3mo": 66, "6mo": 126, "2y": 504}
            days = days_map.get(period, 252)
            
            print(f"Using sample data for {symbol} (yfinance not available)")
            return SimplifiedDataFetcher.generate_sample_data(symbol, days)
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return SimplifiedDataFetcher.generate_sample_data(symbol, 252)

class UltimateAlpacaTrader:
    """Ultimate advanced trading algorithm with simplified dependencies"""
    
    def __init__(self, api_key=None, secret_key=None, paper=True, initial_capital=100000):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.initial_capital = initial_capital
        
        # Initialize components
        self.ml_predictor = MLPredictor()
        self.strategies = AdvancedTradingStrategies(self.ml_predictor)
        self.risk_manager = RiskManager(initial_capital)
        self.data_fetcher = SimplifiedDataFetcher()
        
        # Trading state
        self.positions = {}
        self.orders = {}
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.trade_history = []
        
        # Performance tracking
        self.performance_metrics = {}
        self.last_rebalance = datetime.now()
        
        # Initialize Alpaca connection
        self._initialize_alpaca()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _initialize_alpaca(self):
        """Initialize Alpaca API connection"""
        if ALPACA_AVAILABLE and self.api_key and self.secret_key:
            try:
                base_url = URL('https://paper-api.alpaca.markets') if self.paper else URL('https://api.alpaca.markets')
                self.api = REST(self.api_key, self.secret_key, base_url, api_version='v2')
                
                # Test connection
                account = self.api.get_account()
                self.alpaca_connected = True
                self.logger.info(f"✅ Connected to Alpaca {'Paper' if self.paper else 'Live'} Trading")
                self.logger.info(f"Account Status: {account.status}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to Alpaca: {e}")
                self.alpaca_connected = False
                self.api = None
        else:
            self.alpaca_connected = False
            self.api = None
            self.logger.info("🎮 Running in simulation mode")
    
    def fetch_data_parallel(self, symbols: List[str], period="3mo") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols in parallel"""
        data_dict = {}
        
        def fetch_symbol(symbol):
            try:
                if self.alpaca_connected:
                    # Use Alpaca for real-time data
                    end = datetime.now()
                    start = end - timedelta(days=90 if period == "3mo" else 365)
                    
                    bars = self.api.get_bars(
                        symbol,
                        timeframe='1Day',
                        start=start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        end=end.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        adjustment='raw'
                    ).df
                    
                    if not bars.empty:
                        bars = bars.rename(columns={
                            'open': 'Open', 'high': 'High', 'low': 'Low',
                            'close': 'Close', 'volume': 'Volume'
                        })
                        bars.symbol = symbol
                        return symbol, bars
                
                # Fallback to data fetcher (yfinance or sample data)
                data = self.data_fetcher.fetch_data(symbol, period)
                if not data.empty:
                    data.symbol = symbol
                    return symbol, data
                    
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                return symbol, self.data_fetcher.generate_sample_data(symbol)
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(fetch_symbol, symbol): symbol for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol, data = future.result()
                if data is not None:
                    data_dict[symbol] = data
        
        return data_dict
    
    def analyze_symbol(self, symbol: str, data: pd.DataFrame) -> List[TradeSignal]:
        """Analyze a symbol and generate signals from multiple strategies"""
        try:
            # Calculate indicators
            data_with_indicators = AdvancedIndicators.calculate_all_indicators(data)
            
            # Train ML model if not trained and if sklearn is available
            if SKLEARN_AVAILABLE and not self.ml_predictor.is_trained:
                self.ml_predictor.train_model(data_with_indicators)
            
            # Generate signals from different strategies
            signals = []
            
            # Momentum breakout
            signal1 = self.strategies.momentum_breakout_strategy(data_with_indicators)
            if signal1.action != "HOLD":
                signals.append(signal1)
            
            # Mean reversion
            signal2 = self.strategies.mean_reversion_strategy(data_with_indicators)
            if signal2.action != "HOLD":
                signals.append(signal2)
            
            # Ensemble signal (combine strategies)
            if len(signals) > 1:
                # Weight by confidence and create ensemble
                buy_weight = sum(s.confidence for s in signals if s.action == "BUY")
                sell_weight = sum(s.confidence for s in signals if s.action == "SELL")
                
                if buy_weight > sell_weight and buy_weight > 0.7:
                    best_buy = max([s for s in signals if s.action == "BUY"], key=lambda x: x.confidence)
                    best_buy.confidence = min(0.9, buy_weight / len(signals))
                    best_buy.strategy = "ensemble"
                    return [best_buy]
                elif sell_weight > buy_weight and sell_weight > 0.7:
                    best_sell = max([s for s in signals if s.action == "SELL"], key=lambda x: x.confidence)
                    best_sell.confidence = min(0.9, sell_weight / len(signals))
                    best_sell.strategy = "ensemble"
                    return [best_sell]
            
            # Return best single signal
            return [max(signals, key=lambda x: x.confidence)] if signals else []
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return []
    
    def execute_signal(self, signal: TradeSignal) -> bool:
        """Execute trading signal with advanced order management"""
        try:
            # Calculate position size
            volatility = 0.02  # Default volatility
            quantity = self.risk_manager.calculate_position_size(
                signal, self.portfolio_value, volatility
            )
            
            if quantity == 0:
                self.logger.info(f"Position size too small for {signal.symbol}")
                return False
            
            signal.quantity = quantity
            
            # Check risk limits
            if self.risk_manager.should_stop_trading(0, self.portfolio_value):
                self.logger.warning("Risk limits reached, stopping trading")
                return False
            
            # Execute order
            if self.alpaca_connected:
                return self._execute_alpaca_order(signal)
            else:
                return self._execute_simulated_order(signal)
                
        except Exception as e:
            self.logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return False
    
    def _execute_alpaca_order(self, signal: TradeSignal) -> bool:
        """Execute order via Alpaca API"""
        try:
            order_data = {
                'symbol': signal.symbol,
                'qty': signal.quantity,
                'side': signal.action.lower(),
                'type': 'market',
                'time_in_force': 'day'
            }
            
            # Add bracket orders if available
            if signal.stop_loss and signal.take_profit and signal.action == "BUY":
                order_data['order_class'] = 'bracket'
                order_data['stop_loss'] = {'stop_price': round(signal.stop_loss, 2)}
                order_data['take_profit'] = {'limit_price': round(signal.take_profit, 2)}
            
            order = self.api.submit_order(**order_data)
            self.orders[signal.symbol] = order.id
            
            self.logger.info(f"✅ Order executed: {signal.action} {signal.quantity} {signal.symbol} @ ${signal.price:.2f}")
            self._record_trade(signal, executed=True)
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing Alpaca order: {e}")
            return False
    
    def _execute_simulated_order(self, signal: TradeSignal) -> bool:
        """Execute simulated order"""
        try:
            cost = signal.quantity * signal.price
            
            if signal.action == "BUY":
                if cost <= self.cash:
                    self.positions[signal.symbol] = {
                        'quantity': signal.quantity,
                        'entry_price': signal.price,
                        'current_price': signal.price,
                        'unrealized_pnl': 0,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'strategy': signal.strategy,
                        'entry_time': datetime.now()
                    }
                    
                    self.cash -= cost
                    self.logger.info(f"🎮 SIMULATED BUY: {signal.quantity} {signal.symbol} @ ${signal.price:.2f}")
                    self._record_trade(signal, executed=True)
                    return True
                else:
                    self.logger.warning(f"Insufficient cash for {signal.symbol}")
                    return False
            
            elif signal.action == "SELL" and signal.symbol in self.positions:
                position = self.positions[signal.symbol]
                proceeds = signal.quantity * signal.price
                cost_basis = position['quantity'] * position['entry_price']
                pnl = proceeds - cost_basis
                
                self.cash += proceeds
                self.logger.info(f"🎮 SIMULATED SELL: {signal.quantity} {signal.symbol} @ ${signal.price:.2f}")
                self.logger.info(f"   P&L: ${pnl:.2f} ({pnl/cost_basis*100:.2f}%)")
                
                # Remove position
                del self.positions[signal.symbol]
                self._record_trade(signal, executed=True, pnl=pnl)
                return True
                
        except Exception as e:
            self.logger.error(f"Error in simulated order: {e}")
            return False
        
        return False
    
    def _record_trade(self, signal: TradeSignal, executed: bool, pnl: float = 0):
        """Record trade in history"""
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': signal.symbol,
            'action': signal.action,
            'quantity': signal.quantity,
            'price': signal.price,
            'confidence': signal.confidence,
            'strategy': signal.strategy,
            'executed': executed,
            'pnl': pnl
        }
        self.trade_history.append(trade_record)
    
    def update_positions(self) -> Dict[str, float]:
        """Update position values and check stop losses/take profits"""
        current_prices = {}
        
        if not self.positions:
            return current_prices
        
        # Get current prices for all positions
        symbols = list(self.positions.keys())
        
        try:
            for symbol in symbols:
                if self.alpaca_connected:
                    try:
                        quote = self.api.get_latest_quote(symbol)
                        current_price = (quote.bid_price + quote.ask_price) / 2
                    except:
                        # Fallback to data fetcher
                        data = self.data_fetcher.fetch_data(symbol, "1d")
                        current_price = data['Close'].iloc[-1] if not data.empty else self.positions[symbol]['current_price']
                else:
                    # Simulation mode - get latest data
                    data = self.data_fetcher.fetch_data(symbol, "1d")
                    current_price = data['Close'].iloc[-1] if not data.empty else self.positions[symbol]['current_price']
                
                current_prices[symbol] = current_price
                
                # Update position
                position = self.positions[symbol]
                position['current_price'] = current_price
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
                
                # Check stop loss and take profit
                if position.get('stop_loss') and current_price <= position['stop_loss']:
                    self.logger.info(f"🛑 Stop loss triggered for {symbol} at ${current_price:.2f}")
                    signal = TradeSignal(symbol, "SELL", 1.0, current_price, position['quantity'])
                    self.execute_signal(signal)
                    
                elif position.get('take_profit') and current_price >= position['take_profit']:
                    self.logger.info(f"🎯 Take profit triggered for {symbol} at ${current_price:.2f}")
                    signal = TradeSignal(symbol, "SELL", 1.0, current_price, position['quantity'])
                    self.execute_signal(signal)
        
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
        
        # Update portfolio value
        position_values = sum(pos['quantity'] * pos['current_price'] for pos in self.positions.values())
        self.portfolio_value = self.cash + position_values
        
        return current_prices
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            if len(self.trade_history) < 5:
                return {}
            
            # Convert trade history to DataFrame
            df = pd.DataFrame(self.trade_history)
            executed_trades = df[df['executed'] == True]
            
            if len(executed_trades) == 0:
                return {}
            
            # Calculate metrics
            total_trades = len(executed_trades)
            profitable_trades = len(executed_trades[executed_trades['pnl'] > 0])
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = executed_trades['pnl'].sum()
            avg_win = executed_trades[executed_trades['pnl'] > 0]['pnl'].mean() if profitable_trades > 0 else 0
            avg_loss = executed_trades[executed_trades['pnl'] < 0]['pnl'].mean() if (total_trades - profitable_trades) > 0 else 0
            
            # Risk metrics
            risk_metrics = self.risk_manager.update_risk_metrics(self.portfolio_value)
            
            # Strategy breakdown
            strategy_performance = executed_trades.groupby('strategy')['pnl'].agg(['sum', 'count', 'mean']).to_dict()
            
            metrics = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss < 0 else float('inf'),
                'current_portfolio_value': self.portfolio_value,
                'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'max_drawdown': risk_metrics.max_drawdown,
                'strategy_performance': strategy_performance
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def run_live_trading(self, symbols: List[str], check_interval: int = 300):
        """Run live trading with multiple symbols"""
        self.logger.info(f"🚀 Starting Ultimate Trading Algorithm")
        self.logger.info(f"Symbols: {symbols}")
        self.logger.info(f"Check interval: {check_interval} seconds")
        self.logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Print availability status
        self.logger.info(f"📊 Data Source: {'yfinance' if YF_AVAILABLE else 'Sample Data'}")
        self.logger.info(f"🤖 ML Predictor: {'Enabled' if SKLEARN_AVAILABLE else 'Disabled'}")
        self.logger.info(f"📈 Technical Analysis: {'TA-Lib' if TALIB_AVAILABLE else 'Simplified'}")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                start_time = time.time()
                
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Trading Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Update existing positions
                current_prices = self.update_positions()
                
                # Fetch data for all symbols
                data_dict = self.fetch_data_parallel(symbols, period="3mo")
                self.logger.info(f"📊 Fetched data for {len(data_dict)}/{len(symbols)} symbols")
                
                # Analyze symbols and generate signals
                all_signals = []
                
                for symbol, data in data_dict.items():
                    if data is not None and len(data) > 50:
                        signals = self.analyze_symbol(symbol, data)
                        all_signals.extend(signals)
                
                # Sort signals by confidence
                all_signals.sort(key=lambda x: x.confidence, reverse=True)
                
                self.logger.info(f"🎯 Generated {len(all_signals)} signals")
                
                # Execute top signals (limit to 2 per iteration)
                executed_count = 0
                for signal in all_signals[:3]:
                    if signal.confidence > 0.6:  # Minimum confidence threshold
                        if self.execute_signal(signal):
                            executed_count += 1
                        
                        # Limit executions per iteration
                        if executed_count >= 2:
                            break
                
                # Performance reporting (every 5 iterations)
                if iteration % 5 == 0:
                    metrics = self.calculate_performance_metrics()
                    if metrics:
                        self.logger.info(f"\n📈 PERFORMANCE METRICS:")
                        self.logger.info(f"   Portfolio Value: ${metrics.get('current_portfolio_value', 0):,.2f}")
                        self.logger.info(f"   Total Return: {metrics.get('total_return', 0)*100:.2f}%")
                        self.logger.info(f"   Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
                        self.logger.info(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                        self.logger.info(f"   Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
                        self.logger.info(f"   Total Trades: {metrics.get('total_trades', 0)}")
                
                # Current status
                self.logger.info(f"\n💰 Current Status:")
                self.logger.info(f"   Portfolio Value: ${self.portfolio_value:,.2f}")
                self.logger.info(f"   Cash: ${self.cash:,.2f}")
                self.logger.info(f"   Active Positions: {len(self.positions)}")
                
                if self.positions:
                    self.logger.info(f"   Position Details:")
                    for symbol, pos in self.positions.items():
                        pnl_pct = (pos['unrealized_pnl'] / (pos['quantity'] * pos['entry_price'])) * 100
                        self.logger.info(f"     {symbol}: {pos['quantity']} shares, P&L: ${pos['unrealized_pnl']:,.2f} ({pnl_pct:+.1f}%)")
                
                # Calculate sleep time
                elapsed_time = time.time() - start_time
                sleep_time = max(0, check_interval - elapsed_time)
                
                self.logger.info(f"⏱️ Iteration completed in {elapsed_time:.1f}s. Next check in {sleep_time:.0f}s")
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            self.logger.info("🛑 Trading stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Error in trading loop: {e}")
        finally:
            self._cleanup()
    
    def run_backtest(self, symbols: List[str], start_date: str = None, end_date: str = None):
        """Run comprehensive backtest"""
        self.logger.info(f"🔬 Starting Backtest for {symbols}")
        
        # Determine period
        if start_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date) if end_date else datetime.now()
            period_days = (end - start).days
            
            if period_days > 365:
                period = "2y"
            elif period_days > 90:
                period = "1y"
            else:
                period = "3mo"
        else:
            period = "1y"
        
        data_dict = self.fetch_data_parallel(symbols, period=period)
        
        # Backtest results
        backtest_results = {}
        overall_trades = []
        
        for symbol, data in data_dict.items():
            if data is None or len(data) < 100:
                continue
                
            self.logger.info(f"Backtesting {symbol}...")
            
            # Calculate indicators
            data_with_indicators = AdvancedIndicators.calculate_all_indicators(data)
            
            # Train ML model
            if SKLEARN_AVAILABLE:
                self.ml_predictor.train_model(data_with_indicators)
            
            # Simulate trading
            capital = self.initial_capital / len(symbols)  # Allocate capital evenly
            positions = []
            trades = []
            
            # Walk through data
            for i in range(50, len(data_with_indicators)):
                current_data = data_with_indicators.iloc[:i+1]
                current_row = current_data.iloc[-1]
                
                # Generate signals
                signals = self.analyze_symbol(symbol, current_data)
                
                if signals:
                    signal = signals[0]  # Take best signal
                    
                    # Buy signal
                    if signal.action == "BUY" and len(positions) == 0 and signal.confidence > 0.6:
                        position_size = min(100, int(capital * 0.1 / signal.price))
                        if position_size > 0:
                            entry = {
                                'date': current_row.name,
                                'price': signal.price,
                                'quantity': position_size,
                                'stop_loss': signal.stop_loss,
                                'take_profit': signal.take_profit,
                                'strategy': signal.strategy
                            }
                            positions.append(entry)
                            capital -= position_size * signal.price
                    
                    # Sell signal
                    elif signal.action == "SELL" and positions and signal.confidence > 0.5:
                        for pos in positions:
                            exit_price = signal.price
                            pnl = (exit_price - pos['price']) * pos['quantity']
                            capital += pos['quantity'] * exit_price
                            
                            trade = {
                                'symbol': symbol,
                                'entry_date': pos['date'],
                                'exit_date': current_row.name,
                                'entry_price': pos['price'],
                                'exit_price': exit_price,
                                'quantity': pos['quantity'],
                                'pnl': pnl,
                                'return': pnl / (pos['price'] * pos['quantity']),
                                'strategy': pos['strategy'],
                                'days_held': (current_row.name - pos['date']).days
                            }
                            trades.append(trade)
                            overall_trades.append(trade)
                        
                        positions = []
                
                # Check stop losses and take profits
                for pos in positions[:]:
                    current_price = current_row['Close']
                    
                    if pos['stop_loss'] and current_price <= pos['stop_loss']:
                        # Stop loss
                        exit_price = pos['stop_loss']
                        pnl = (exit_price - pos['price']) * pos['quantity']
                        capital += pos['quantity'] * exit_price
                        
                        trade = {
                            'symbol': symbol,
                            'entry_date': pos['date'],
                            'exit_date': current_row.name,
                            'entry_price': pos['price'],
                            'exit_price': exit_price,
                            'quantity': pos['quantity'],
                            'pnl': pnl,
                            'return': pnl / (pos['price'] * pos['quantity']),
                            'strategy': pos['strategy'] + '_SL',
                            'days_held': (current_row.name - pos['date']).days
                        }
                        trades.append(trade)
                        overall_trades.append(trade)
                        positions.remove(pos)
                    
                    elif pos['take_profit'] and current_price >= pos['take_profit']:
                        # Take profit
                        exit_price = pos['take_profit']
                        pnl = (exit_price - pos['price']) * pos['quantity']
                        capital += pos['quantity'] * exit_price
                        
                        trade = {
                            'symbol': symbol,
                            'entry_date': pos['date'],
                            'exit_date': current_row.name,
                            'entry_price': pos['price'],
                            'exit_price': exit_price,
                            'quantity': pos['quantity'],
                            'pnl': pnl,
                            'return': pnl / (pos['price'] * pos['quantity']),
                            'strategy': pos['strategy'] + '_TP',
                            'days_held': (current_row.name - pos['date']).days
                        }
                        trades.append(trade)
                        overall_trades.append(trade)
                        positions.remove(pos)
            
            # Close remaining positions
            if positions:
                final_price = data_with_indicators.iloc[-1]['Close']
                for pos in positions:
                    pnl = (final_price - pos['price']) * pos['quantity']
                    capital += pos['quantity'] * final_price
                    
                    trade = {
                        'symbol': symbol,
                        'entry_date': pos['date'],
                        'exit_date': data_with_indicators.index[-1],
                        'entry_price': pos['price'],
                        'exit_price': final_price,
                        'quantity': pos['quantity'],
                        'pnl': pnl,
                        'return': pnl / (pos['price'] * pos['quantity']),
                        'strategy': pos['strategy'] + '_FINAL',
                        'days_held': (data_with_indicators.index[-1] - pos['date']).days
                    }
                    trades.append(trade)
                    overall_trades.append(trade)
            
            # Calculate results for this symbol
            if trades:
                trades_df = pd.DataFrame(trades)
                
                initial_symbol_capital = self.initial_capital / len(symbols)
                total_return = (capital - initial_symbol_capital) / initial_symbol_capital
                win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
                avg_return = trades_df['return'].mean()
                
                backtest_results[symbol] = {
                    'total_trades': len(trades),
                    'win_rate': win_rate,
                    'total_return': total_return,
                    'avg_return_per_trade': avg_return,
                    'best_trade': trades_df['return'].max(),
                    'worst_trade': trades_df['return'].min(),
                    'avg_days_held': trades_df['days_held'].mean(),
                    'final_capital': capital
                }
        
        # Print results
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🔬 BACKTEST RESULTS")
        self.logger.info(f"{'='*60}")
        
        total_final_capital = sum(results['final_capital'] for results in backtest_results.values())
        
        for symbol, results in backtest_results.items():
            self.logger.info(f"\n📊 {symbol}:")
            self.logger.info(f"   Total Trades: {results['total_trades']}")
            self.logger.info(f"   Win Rate: {results['win_rate']*100:.1f}%")
            self.logger.info(f"   Total Return: {results['total_return']*100:.2f}%")
            self.logger.info(f"   Best/Worst Trade: {results['best_trade']*100:.2f}% / {results['worst_trade']*100:.2f}%")
            self.logger.info(f"   Final Capital: ${results['final_capital']:,.2f}")
        
        # Overall performance
        if total_final_capital > 0:
            overall_return = (total_final_capital - self.initial_capital) / self.initial_capital
            self.logger.info(f"\n🎯 OVERALL PERFORMANCE:")
            self.logger.info(f"   Initial Capital: ${self.initial_capital:,.2f}")
            self.logger.info(f"   Final Capital: ${total_final_capital:,.2f}")
            self.logger.info(f"   Total Return: {overall_return*100:.2f}%")
            self.logger.info(f"   Total Trades: {len(overall_trades)}")
            
            if overall_trades:
                overall_df = pd.DataFrame(overall_trades)
                overall_win_rate = len(overall_df[overall_df['pnl'] > 0]) / len(overall_df)
                self.logger.info(f"   Overall Win Rate: {overall_win_rate*100:.1f}%")
        
        return backtest_results
    
    def _cleanup(self):
        """Clean up resources"""
        self.logger.info("🧹 Cleaning up resources...")
        
        # Cancel any open orders
        if self.alpaca_connected:
            try:
                orders = self.api.list_orders(status='open')
                for order in orders:
                    self.api.cancel_order(order.id)
                    self.logger.info(f"Cancelled order: {order.symbol}")
            except:
                pass
        
        # Final performance summary
        final_metrics = self.calculate_performance_metrics()
        if final_metrics:
            self.logger.info(f"\n📈 FINAL PERFORMANCE SUMMARY:")
            self.logger.info(f"   Final Portfolio Value: ${final_metrics.get('current_portfolio_value', 0):,.2f}")
            self.logger.info(f"   Total Return: {final_metrics.get('total_return', 0)*100:.2f}%")
            self.logger.info(f"   Total Trades: {final_metrics.get('total_trades', 0)}")
            self.logger.info(f"   Win Rate: {final_metrics.get('win_rate', 0)*100:.1f}%")
        
        self.logger.info("✅ Cleanup completed")

def install_dependencies():
    """Helper function to install missing dependencies"""
    import subprocess
    import sys
    
    packages_to_install = []
    
    if not YF_AVAILABLE:
        packages_to_install.extend(['yfinance', 'websockets>=11.0'])
    if not SKLEARN_AVAILABLE:
        packages_to_install.append('scikit-learn')
    if not SCIPY_AVAILABLE:
        packages_to_install.append('scipy')
    if not TALIB_AVAILABLE:
        print("Note: TA-Lib installation requires additional steps:")
        print("Windows: pip install TA-Lib")
        print("macOS: brew install ta-lib && pip install TA-Lib")
        print("Linux: sudo apt-get install libta-lib-dev && pip install TA-Lib")
    if not ALPACA_AVAILABLE:
        packages_to_install.append('alpaca-trade-api')
    
    if packages_to_install:
        print(f"Installing missing packages: {', '.join(packages_to_install)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages_to_install)
            print("✅ Dependencies installed successfully!")
            print("Please restart the script to use the newly installed packages.")
        except subprocess.CalledProcessError:
            print("❌ Failed to install some packages. Please install manually:")
            for pkg in packages_to_install:
                print(f"   pip install {pkg}")
    else:
        print("✅ All dependencies are available!")

def main():
    """Main function to run the Ultimate Trading Algorithm"""
    print("🚀 Ultimate Advanced Trading Algorithm v2.0")
    print("=" * 50)
    print("Available Dependencies:")
    print(f"   📊 yfinance: {'✅' if YF_AVAILABLE else '❌'}")
    print(f"   🤖 scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
    print(f"   📈 TA-Lib: {'✅' if TALIB_AVAILABLE else '❌'}")
    print(f"   📉 SciPy: {'✅' if SCIPY_AVAILABLE else '❌'}")
    print(f"   🏦 Alpaca API: {'✅' if ALPACA_AVAILABLE else '❌'}")
    print()
    
    if not any([YF_AVAILABLE, SKLEARN_AVAILABLE, TALIB_AVAILABLE, SCIPY_AVAILABLE, ALPACA_AVAILABLE]):
        print("⚠️  Most dependencies are missing. The algorithm will run with simplified features.")
        install_choice = input("Would you like to install missing dependencies? (y/n): ").strip().lower()
        if install_choice == 'y':
            install_dependencies()
            return
    
    print("Select Mode:")
    print("1. Live Trading with Alpaca")
    print("2. Simulation Mode")
    print("3. Backtest Mode")
    print("4. Install Dependencies")
    
    choice = input("\nSelect mode (1-4): ").strip()
    
    if choice == "4":
        install_dependencies()
        return
    
    # Get API credentials if needed
    api_key = None
    secret_key = None
    
    if choice == "1":
        print("\n🔑 Enter Alpaca API Credentials:")
        api_key = input("API Key: ").strip()
        secret_key = input("Secret Key: ").strip()
        
        if not api_key or not secret_key:
            print("❌ Invalid credentials. Switching to simulation mode.")
            choice = "2"
    
    # Initialize trader
    trader = UltimateAlpacaTrader(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,  # Set to False for live trading
        initial_capital=100000
    )
    
    # Get symbol selection
    print("\nSelect Watchlist:")
    print("1. Tech Stocks (AAPL, MSFT, GOOGL, TSLA, NVDA)")
    print("2. Blue Chips (AAPL, MSFT, JNJ, PG, KO)")
    print("3. Growth Stocks (TSLA, NVDA, AMD, CRM, NFLX)")
    print("4. Custom (enter your own)")
    
    watchlist_choice = input("Select watchlist (1-4): ").strip()
    
    if watchlist_choice == "1":
        watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    elif watchlist_choice == "2":
        watchlist = ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO']
    elif watchlist_choice == "3":
        watchlist = ['TSLA', 'NVDA', 'AMD', 'CRM', 'NFLX']
    else:
        symbols_input = input("Enter symbols separated by commas (e.g., AAPL,MSFT,GOOGL): ").strip()
        watchlist = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
        if not watchlist:
            watchlist = ['AAPL', 'MSFT', 'GOOGL']  # Default
    
    print(f"\n📋 Selected symbols: {watchlist}")
    
    try:
        if choice in ["1", "2"]:
            # Live/Simulation Trading
            print(f"\n🎯 Starting {'Live' if choice == '1' else 'Simulation'} Trading")
            
            # Get check interval
            interval_input = input("Check interval in seconds (default: 300): ").strip()
            check_interval = int(interval_input) if interval_input.isdigit() else 300
            
            trader.run_live_trading(
                symbols=watchlist,
                check_interval=check_interval
            )
        
        elif choice == "3":
            # Backtest Mode
            print("\n🔬 Starting Backtest")
            print("Enter date range (YYYY-MM-DD format) or press Enter for default (1 year)")
            start_date = input("Start date (optional): ").strip()
            end_date = input("End date (optional): ").strip()
            
            if not start_date:
                start_date = None
                end_date = None
            elif start_date and not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            trader.run_backtest(
                symbols=watchlist,
                start_date=start_date,
                end_date=end_date
            )
        
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Main error: {e}", exc_info=True)
    finally:
        try:
            trader._cleanup()
        except:
            pass

if __name__ == "__main__":
    # Print startup information
    print("🔧 Checking dependencies...")
    
    missing_core = []
    if not YF_AVAILABLE and not ALPACA_AVAILABLE:
        missing_core.append("data source (yfinance or alpaca)")
    
    if missing_core:
        print("⚠️  Critical dependencies missing:")
        for missing in missing_core:
            print(f"   - {missing}")
        print("\nThe algorithm will use sample data for demonstration.")
        print("For real trading, please install the required packages.\n")
    
    main()