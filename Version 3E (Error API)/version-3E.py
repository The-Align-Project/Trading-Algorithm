import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
import warnings
import threading
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import talib
import asyncio

warnings.filterwarnings('ignore')

# Alpaca imports
try:
    from alpaca_trade_api import REST, Stream
    from alpaca_trade_api.common import URL
    ALPACA_AVAILABLE = True
except ImportError:
    print("Alpaca Trade API not installed. Run: pip install alpaca-trade-api")
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

class AdvancedIndicators:
    """Advanced technical indicators using TA-Lib"""
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = data.copy()
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # Price-based indicators
        df['SMA_10'] = talib.SMA(close, timeperiod=10)
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        df['SMA_200'] = talib.SMA(close, timeperiod=200)
        
        df['EMA_12'] = talib.EMA(close, timeperiod=12)
        df['EMA_26'] = talib.EMA(close, timeperiod=26)
        df['EMA_50'] = talib.EMA(close, timeperiod=50)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(close)
        
        # RSI variants
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['RSI_2'] = talib.RSI(close, timeperiod=2)
        df['RSI_50'] = talib.RSI(close, timeperiod=50)
        
        # Stochastic
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close)
        
        # Williams %R
        df['WILLR'] = talib.WILLR(high, low, close)
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(close)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR and volatility
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['ATR_Norm'] = df['ATR'] / close
        
        # Momentum indicators
        df['MOM'] = talib.MOM(close, timeperiod=10)
        df['ROC'] = talib.ROC(close, timeperiod=10)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        
        # Volume indicators
        df['OBV'] = talib.OBV(close, volume)
        df['AD'] = talib.AD(high, low, close, volume)
        df['ADOSC'] = talib.ADOSC(high, low, close, volume)
        
        # Pattern recognition
        df['DOJI'] = talib.CDLDOJI(df['Open'], high, low, close)
        df['HAMMER'] = talib.CDLHAMMER(df['Open'], high, low, close)
        df['ENGULFING'] = talib.CDLENGULFING(df['Open'], high, low, close)
        
        # Custom indicators
        df['Price_Change'] = close / np.roll(close, 1) - 1
        df['Volatility_20'] = df['Price_Change'].rolling(20).std()
        df['Volume_Ratio'] = volume / talib.SMA(volume, timeperiod=20)
        
        # Support and Resistance levels
        df['Pivot'] = (high + low + close) / 3
        df['R1'] = 2 * df['Pivot'] - low
        df['S1'] = 2 * df['Pivot'] - high
        
        # Trend strength
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['DI_Plus'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['DI_Minus'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        return df

class MLPredictor:
    """Machine Learning predictor for price movements"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.is_trained = False
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model"""
        features = [
            'RSI', 'MACD', 'MACD_Signal', 'BB_Position', 'BB_Width',
            'STOCH_K', 'STOCH_D', 'WILLR', 'CCI', 'ADX', 'ATR_Norm',
            'Volume_Ratio', 'Price_Change', 'Volatility_20'
        ]
        
        # Add price ratios
        data['SMA_Ratio'] = data['Close'] / data['SMA_20']
        data['EMA_Ratio'] = data['EMA_12'] / data['EMA_26']
        features.extend(['SMA_Ratio', 'EMA_Ratio'])
        
        # Add lagged features
        for lag in [1, 2, 3, 5]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
            features.extend([f'Close_Lag_{lag}', f'Volume_Lag_{lag}'])
        
        return data[features].dropna()
    
    def train_model(self, data: pd.DataFrame) -> bool:
        """Train the ML model"""
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
                n_estimators=100,
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
        if not self.is_trained or self.model is None:
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
            current['Close'] > current['SMA_20'] > current['SMA_50'] and
            current['EMA_12'] > current['EMA_26'] and
            current['ADX'] > 25  # Strong trend
        )
        
        trend_bearish = (
            current['Close'] < current['SMA_20'] < current['SMA_50'] and
            current['EMA_12'] < current['EMA_26'] and
            current['ADX'] > 25
        )
        
        # Momentum conditions
        momentum_bullish = (
            current['RSI'] > 50 and current['RSI'] < 80 and
            current['MACD'] > current['MACD_Signal'] and
            current['STOCH_K'] > current['STOCH_D'] and
            current['Volume_Ratio'] > 1.2
        )
        
        momentum_bearish = (
            current['RSI'] < 50 and current['RSI'] > 20 and
            current['MACD'] < current['MACD_Signal'] and
            current['STOCH_K'] < current['STOCH_D']
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
            current['BB_Position'] < 0.1 and  # Near lower Bollinger Band
            current['STOCH_K'] < 20
        )
        
        # Overbought conditions
        overbought = (
            current['RSI'] > 70 and
            current['BB_Position'] > 0.9 and  # Near upper Bollinger Band
            current['STOCH_K'] > 80
        )
        
        # Trend context (prefer counter-trend in ranging markets)
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
    
    def breakout_pattern_strategy(self, data: pd.DataFrame) -> TradeSignal:
        """Pattern-based breakout strategy"""
        if len(data) < 20:
            return TradeSignal("", "HOLD", 0.0, 0.0, 0)
        
        current = data.iloc[-1]
        recent = data.iloc[-20:]  # Last 20 periods
        symbol = getattr(data, 'symbol', 'UNKNOWN')
        
        # Calculate support and resistance
        resistance = recent['High'].max()
        support = recent['Low'].min()
        
        # Breakout conditions
        resistance_breakout = (
            current['Close'] > resistance and
            current['Volume_Ratio'] > 1.5 and  # High volume
            current['Close'] > current['SMA_20']
        )
        
        support_breakdown = (
            current['Close'] < support and
            current['Volume_Ratio'] > 1.5 and
            current['Close'] < current['SMA_20']
        )
        
        # Pattern confirmation
        bullish_pattern = current['HAMMER'] > 0 or current['ENGULFING'] > 0
        bearish_pattern = current['HAMMER'] < 0 or current['ENGULFING'] < 0
        
        if resistance_breakout or bullish_pattern:
            confidence = 0.6 if resistance_breakout else 0.5
            if bullish_pattern:
                confidence += 0.2
            
            take_profit = current['Close'] + (resistance - support)
            stop_loss = resistance * 0.98  # Just below breakout level
            
            return TradeSignal(
                symbol=symbol,
                action="BUY",
                confidence=min(confidence, 0.8),
                price=current['Close'],
                quantity=0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy="breakout_pattern"
            )
        
        elif support_breakdown or bearish_pattern:
            confidence = 0.6 if support_breakdown else 0.5
            if bearish_pattern:
                confidence += 0.2
            
            return TradeSignal(
                symbol=symbol,
                action="SELL",
                confidence=min(confidence, 0.8),
                price=current['Close'],
                quantity=0,
                strategy="breakout_pattern"
            )
        
        return TradeSignal(symbol, "HOLD", 0.0, current['Close'], 0)

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.max_position_size = 0.1  # Max 10% per position
        self.max_daily_loss = 0.02  # Max 2% daily loss
        self.max_sector_exposure = 0.3  # Max 30% per sector
        self.correlation_limit = 0.7  # Max correlation between positions
        self.current_drawdown = 0.0
        
        # Risk metrics tracking
        self.daily_returns = []
        self.portfolio_values = []
        self.peak_value = initial_capital
        
    def calculate_position_size(self, signal: TradeSignal, portfolio_value: float, 
                              volatility: float, correlation_matrix: pd.DataFrame = None) -> int:
        """Calculate optimal position size using Kelly Criterion and risk limits"""
        
        # Kelly Criterion
        win_prob = signal.confidence
        avg_win = 0.02  # Assume 2% average win
        avg_loss = 0.01  # Assume 1% average loss
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Volatility adjustment
        vol_adjustment = min(1.0, 0.02 / volatility)  # Target 2% volatility
        
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
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        if len(downside_returns) > 0:
            sortino_ratio = returns_series.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = sharpe_ratio
        
        # Value at Risk (95%)
        var_95 = returns_series.quantile(0.05)
        
        return RiskMetrics(
            max_drawdown=self.current_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95
        )
    
    def should_stop_trading(self, current_loss: float, portfolio_value: float) -> bool:
        """Determine if trading should be stopped due to risk limits"""
        daily_loss_pct = current_loss / portfolio_value
        
        return (
            daily_loss_pct > self.max_daily_loss or
            self.current_drawdown > 0.15  # Max 15% drawdown
        )

class UltimateAlpacaTrader:
    """Ultimate advanced trading algorithm"""
    
    def __init__(self, api_key=None, secret_key=None, paper=True, initial_capital=100000):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.initial_capital = initial_capital
        
        # Initialize components
        self.ml_predictor = MLPredictor()
        self.strategies = AdvancedTradingStrategies(self.ml_predictor)
        self.risk_manager = RiskManager(initial_capital)
        
        # Trading state
        self.positions = {}
        self.orders = {}
        self.portfolio_value = initial_capital
        self.trade_history = []
        
        # Performance tracking
        self.performance_metrics = {}
        self.last_rebalance = datetime.now()
        
        # Threading
        self.data_queue = queue.Queue()
        self.signal_queue = queue.Queue()
        self.running = False
        
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
                
                # Fallback to yfinance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval="1d")
                if not data.empty:
                    data.symbol = symbol
                    return symbol, data
                    
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                return symbol, None
        
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
            
            # Train ML model if not trained
            if not self.ml_predictor.is_trained:
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
            
            # Breakout patterns
            signal3 = self.strategies.breakout_pattern_strategy(data_with_indicators)
            if signal3.action != "HOLD":
                signals.append(signal3)
            
            # Ensemble signal (combine strategies)
            if len(signals) > 1:
                # Weight by confidence and create ensemble
                total_confidence = sum(s.confidence for s in signals)
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
                if cost <= self.portfolio_value * 0.95:  # Keep 5% cash
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
                    
                    self.portfolio_value -= cost
                    self.logger.info(f"🎮 SIMULATED BUY: {signal.quantity} {signal.symbol} @ ${signal.price:.2f}")
                    self._record_trade(signal, executed=True)
                    return True
                else:
                    self.logger.warning(f"Insufficient capital for {signal.symbol}")
                    return False
            
            elif signal.action == "SELL" and signal.symbol in self.positions:
                position = self.positions[signal.symbol]
                proceeds = signal.quantity * signal.price
                cost_basis = position['quantity'] * position['entry_price']
                pnl = proceeds - cost_basis
                
                self.portfolio_value += proceeds
                self.logger.info(f"🎮 SIMULATED SELL: {signal.quantity} {signal.symbol} @ ${signal.price:.2f}")
                self.logger.info(f"   P&L: ${pnl:.2f} ({pnl/cost_basis*100:.2f}%)")
                
                # Remove position
                del self.positions[signal.symbol]
                self._record_trade(signal, executed=True, pnl=pnl)
                return True
                
        except Exception as e:
            self.logger.error(f"Error in simulated order: {e}")
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
        total_unrealized_pnl = 0
        
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
                        # Fallback to yfinance
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(period="1d", interval="1m")
                        current_price = data['Close'].iloc[-1] if not data.empty else self.positions[symbol]['current_price']
                else:
                    # Simulation mode - use yfinance
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1d", interval="1m")
                    current_price = data['Close'].iloc[-1] if not data.empty else self.positions[symbol]['current_price']
                
                current_prices[symbol] = current_price
                
                # Update position
                position = self.positions[symbol]
                position['current_price'] = current_price
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
                total_unrealized_pnl += position['unrealized_pnl']
                
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
        self.portfolio_value = self.portfolio_value + position_values if not self.alpaca_connected else self._get_account_value()
        
        return current_prices
    
    def _get_account_value(self) -> float:
        """Get account value from Alpaca"""
        try:
            if self.alpaca_connected:
                account = self.api.get_account()
                return float(account.portfolio_value)
            return self.portfolio_value
        except:
            return self.portfolio_value
    
    def rebalance_portfolio(self):
        """Rebalance portfolio based on risk metrics"""
        try:
            # Calculate current allocation
            total_value = sum(pos['quantity'] * pos['current_price'] for pos in self.positions.values())
            
            if total_value == 0:
                return
            
            # Check for overweight positions
            for symbol, position in list(self.positions.items()):
                position_weight = (position['quantity'] * position['current_price']) / total_value
                
                # Reduce position if too large
                if position_weight > self.risk_manager.max_position_size * 1.5:
                    reduce_quantity = int(position['quantity'] * 0.25)  # Reduce by 25%
                    if reduce_quantity > 0:
                        signal = TradeSignal(symbol, "SELL", 0.8, position['current_price'], reduce_quantity)
                        self.execute_signal(signal)
                        self.logger.info(f"🔄 Rebalanced {symbol}: reduced by {reduce_quantity} shares")
            
            self.last_rebalance = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error in portfolio rebalancing: {e}")
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            if len(self.trade_history) < 10:
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
        
        self.running = True
        iteration = 0
        
        try:
            while self.running:
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
                
                # Execute top signals (limit to 3 per iteration)
                executed_count = 0
                for signal in all_signals[:3]:
                    if signal.confidence > 0.6:  # Minimum confidence threshold
                        if self.execute_signal(signal):
                            executed_count += 1
                        
                        # Limit executions per iteration
                        if executed_count >= 2:
                            break
                
                # Portfolio rebalancing (every 10 iterations)
                if iteration % 10 == 0:
                    self.rebalance_portfolio()
                
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
                self.logger.info(f"   Active Positions: {len(self.positions)}")
                self.logger.info(f"   Cash: ${self.portfolio_value - sum(pos['quantity'] * pos['current_price'] for pos in self.positions.values()):,.2f}")
                
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
            self.running = False
            self._cleanup()
    
    def run_backtest(self, symbols: List[str], start_date: str = None, end_date: str = None):
        """Run comprehensive backtest"""
        self.logger.info(f"🔬 Starting Backtest for {symbols}")
        
        # Fetch historical data
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
        
        # Backtest each symbol
        backtest_results = {}
        
        for symbol, data in data_dict.items():
            if data is None or len(data) < 100:
                continue
                
            self.logger.info(f"Backtesting {symbol}...")
            
            # Reset for each symbol
            symbol_capital = self.initial_capital / len(symbols)  # Allocate capital evenly
            symbol_positions = []
            symbol_trades = []
            
            # Calculate indicators
            data_with_indicators = AdvancedIndicators.calculate_all_indicators(data)
            
            # Train ML model
            self.ml_predictor.train_model(data_with_indicators)
            
            # Walk through data
            for i in range(50, len(data_with_indicators)):
                current_data = data_with_indicators.iloc[:i+1]
                current_row = current_data.iloc[-1]
                
                # Generate signals
                signals = self.analyze_symbol(symbol, current_data)
                
                if signals:
                    signal = signals[0]  # Take best signal
                    
                    # Calculate position size
                    if signal.action == "BUY" and len(symbol_positions) == 0:
                        position_size = min(100, int(symbol_capital * 0.1 / signal.price))
                        if position_size > 0:
                            entry = {
                                'date': current_row.name,
                                'price': signal.price,
                                'quantity': position_size,
                                'stop_loss': signal.stop_loss,
                                'take_profit': signal.take_profit,
                                'strategy': signal.strategy
                            }
                            symbol_positions.append(entry)
                            symbol_capital -= position_size * signal.price
                    
                    elif signal.action == "SELL" and symbol_positions:
                        # Close all positions
                        for pos in symbol_positions:
                            exit_price = signal.price
                            pnl = (exit_price - pos['price']) * pos['quantity']
                            symbol_capital += pos['quantity'] * exit_price
                            
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
                            symbol_trades.append(trade)
                        
                        symbol_positions = []
                
                # Check stop losses and take profits
                for pos in symbol_positions[:]:
                    if pos['stop_loss'] and current_row['Close'] <= pos['stop_loss']:
                        # Stop loss hit
                        exit_price = pos['stop_loss']
                        pnl = (exit_price - pos['price']) * pos['quantity']
                        symbol_capital += pos['quantity'] * exit_price
                        
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
                        symbol_trades.append(trade)
                        symbol_positions.remove(pos)
                    
                    elif pos['take_profit'] and current_row['Close'] >= pos['take_profit']:
                        # Take profit hit
                        exit_price = pos['take_profit']
                        pnl = (exit_price - pos['price']) * pos['quantity']
                        symbol_capital += pos['quantity'] * exit_price
                        
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
                        symbol_trades.append(trade)
                        symbol_positions.remove(pos)
            
            # Close remaining positions at end
            if symbol_positions:
                final_price = data_with_indicators.iloc[-1]['Close']
                for pos in symbol_positions:
                    pnl = (final_price - pos['price']) * pos['quantity']
                    symbol_capital += pos['quantity'] * final_price
                    
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
                    symbol_trades.append(trade)
            
            # Calculate results for this symbol
            if symbol_trades:
                trades_df = pd.DataFrame(symbol_trades)
                
                total_return = (symbol_capital - (self.initial_capital / len(symbols))) / (self.initial_capital / len(symbols))
                win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
                avg_return = trades_df['return'].mean()
                max_return = trades_df['return'].max()
                min_return = trades_df['return'].min()
                avg_days_held = trades_df['days_held'].mean()
                
                backtest_results[symbol] = {
                    'total_trades': len(symbol_trades),
                    'win_rate': win_rate,
                    'total_return': total_return,
                    'avg_return_per_trade': avg_return,
                    'best_trade': max_return,
                    'worst_trade': min_return,
                    'avg_days_held': avg_days_held,
                    'final_capital': symbol_capital,
                    'strategy_breakdown': trades_df.groupby('strategy')['pnl'].agg(['count', 'sum', 'mean']).to_dict()
                }
        
        # Print backtest results
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🔬 BACKTEST RESULTS")
        self.logger.info(f"{'='*60}")
        
        total_final_capital = 0
        for symbol, results in backtest_results.items():
            total_final_capital += results['final_capital']
            
            self.logger.info(f"\n📊 {symbol}:")
            self.logger.info(f"   Total Trades: {results['total_trades']}")
            self.logger.info(f"   Win Rate: {results['win_rate']*100:.1f}%")
            self.logger.info(f"   Total Return: {results['total_return']*100:.2f}%")
            self.logger.info(f"   Avg Return/Trade: {results['avg_return_per_trade']*100:.2f}%")
            self.logger.info(f"   Best Trade: {results['best_trade']*100:.2f}%")
            self.logger.info(f"   Worst Trade: {results['worst_trade']*100:.2f}%")
            self.logger.info(f"   Avg Days Held: {results['avg_days_held']:.1f}")
            self.logger.info(f"   Final Capital: ${results['final_capital']:,.2f}")
        
        # Overall results
        overall_return = (total_final_capital - self.initial_capital) / self.initial_capital
        self.logger.info(f"\n🎯 OVERALL PERFORMANCE:")
        self.logger.info(f"   Initial Capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"   Final Capital: ${total_final_capital:,.2f}")
        self.logger.info(f"   Total Return: {overall_return*100:.2f}%")
        
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
            self.logger.info(f"   Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.2f}")
        
        self.logger.info("✅ Cleanup completed")

def main():
    """Main function to run the Ultimate Trading Algorithm"""
    print("🚀 Ultimate Advanced Trading Algorithm")
    print("=" * 50)
    print("1. Live Trading with Alpaca")
    print("2. Simulation Mode")
    print("3. Backtest Mode")
    
    choice = input("\nSelect mode (1-3): ").strip()
    
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
    
    # Define watchlist (can be customized)
    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    
    try:
        if choice in ["1", "2"]:
            # Live/Simulation Trading
            print(f"\n🎯 Starting {'Live' if choice == '1' else 'Simulation'} Trading")
            trader.run_live_trading(
                symbols=watchlist,
                check_interval=300  # 5 minutes
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
    finally:
        trader._cleanup()

if __name__ == "__main__":
    # Install required packages
    required_packages = [
        "pandas", "numpy", "yfinance", "scikit-learn", 
        "TA-Lib", "alpaca-trade-api"
    ]
    
    print("📦 Required packages:", ", ".join(required_packages))
    print("Install with: pip install " + " ".join(required_packages))
    print("Note: TA-Lib may require additional setup on some systems")
    print()
    
    main()