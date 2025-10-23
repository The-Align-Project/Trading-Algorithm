"""Data structures and models for the trading algorithm."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

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