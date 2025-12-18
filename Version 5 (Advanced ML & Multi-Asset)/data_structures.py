"""Enhanced data structures and models for Version 5 Trading Algorithm."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List
from enum import Enum

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

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

# NEW V5: Options-related structures
class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class OptionAction(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class OptionLeg:
    """Represents a single leg of an options strategy"""
    option_type: OptionType
    action: OptionAction
    strike: float
    expiration: datetime
    quantity: int
    premium: float = 0.0

@dataclass
class OptionsSignal:
    """Signal for options trading"""
    symbol: str  # Underlying symbol
    strategy_name: str  # e.g., 'long_call', 'iron_condor'
    legs: List[OptionLeg]
    confidence: float
    max_risk: float
    max_profit: float
    breakeven_points: List[float]
    timestamp: datetime = None
    metadata: dict = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class OptionsPosition:
    """Represents an open options position"""
    symbol: str
    strategy_name: str
    legs: List[OptionLeg]
    entry_time: datetime
    current_value: float
    initial_cost: float
    max_risk: float
    max_profit: float

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        return self.current_value - self.initial_cost

    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized P&L as percentage"""
        if self.initial_cost == 0:
            return 0.0
        return (self.current_value - self.initial_cost) / abs(self.initial_cost) * 100

# NEW V5: Multi-timeframe data structure
@dataclass
class MultiTimeframeData:
    """Container for data across multiple timeframes"""
    symbol: str
    timeframes: Dict[str, 'pd.DataFrame']  # timeframe -> data
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def get_timeframe(self, timeframe: str):
        """Get data for specific timeframe"""
        return self.timeframes.get(timeframe)

# NEW V5: Sentiment analysis structures
@dataclass
class SentimentData:
    """Sentiment analysis data"""
    symbol: str
    news_sentiment: float  # -1.0 to 1.0
    social_sentiment: float  # -1.0 to 1.0
    combined_sentiment: float  # -1.0 to 1.0
    news_count: int
    social_mentions: int
    timestamp: datetime = None
    sources: List[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.sources is None:
            self.sources = []

@dataclass
class NewsArticle:
    """Individual news article"""
    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    sentiment: float  # -1.0 to 1.0

@dataclass
class SocialPost:
    """Social media post"""
    text: str
    source: str  # 'twitter', 'reddit', etc.
    author: str
    timestamp: datetime
    engagement: int  # likes, retweets, etc.
    sentiment: float  # -1.0 to 1.0

# Existing structures from V4
@dataclass
class RiskMetrics:
    """Risk management metrics"""
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    beta: float = 0.0
    correlation_spy: float = 0.0

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = ""

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized P&L as percentage"""
        return (self.current_price - self.entry_price) / self.entry_price * 100

    @property
    def market_value(self) -> float:
        """Calculate current market value"""
        return self.current_price * abs(self.quantity)

@dataclass
class TradeRecord:
    """Record of a completed trade"""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    confidence: float
    strategy: str
    executed: bool
    pnl: float = 0.0
    commission: float = 0.0

    @property
    def gross_pnl(self) -> float:
        """Gross P&L before commissions"""
        return self.pnl + self.commission

@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state"""
    timestamp: datetime
    cash: float
    positions_value: float
    total_value: float
    positions_count: int
    daily_pnl: float = 0.0
    options_value: float = 0.0  # NEW V5

    @property
    def cash_percentage(self) -> float:
        """Cash as percentage of total portfolio"""
        return self.cash / self.total_value * 100 if self.total_value > 0 else 0

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_portfolio_value: float = 0.0
    strategy_performance: dict = None

    def __post_init__(self):
        if self.strategy_performance is None:
            self.strategy_performance = {}

# NEW V5: Deep Learning model outputs
@dataclass
class DeepLearningPrediction:
    """Output from deep learning models"""
    symbol: str
    model_type: str  # 'LSTM', 'Transformer', 'Hybrid'
    prediction: float  # Predicted price or direction
    confidence: float  # Model confidence 0-1
    price_targets: Dict[str, float]  # 'short', 'medium', 'long' term
    volatility_forecast: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ModelEnsemblePrediction:
    """Combined prediction from multiple models"""
    symbol: str
    lstm_prediction: Optional[DeepLearningPrediction]
    transformer_prediction: Optional[DeepLearningPrediction]
    ml_prediction: float  # From traditional ML
    ensemble_prediction: float  # Weighted combination
    confidence: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
