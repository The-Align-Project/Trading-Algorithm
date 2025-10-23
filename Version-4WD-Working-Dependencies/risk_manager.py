"""Fixed risk management system for the trading algorithm."""

import pandas as pd
import numpy as np
import logging
from typing import List

from data_structures import TradeSignal, RiskMetrics
from config import TradingConfig

class RiskManager:
    """Advanced risk management system with proper initialization"""
    
    def __init__(self, initial_capital: float = TradingConfig.DEFAULT_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.max_position_size = TradingConfig.MAX_POSITION_SIZE
        self.max_daily_loss = TradingConfig.MAX_DAILY_LOSS
        self.max_drawdown = TradingConfig.MAX_DRAWDOWN
        self.current_drawdown = 0.0
        
        # Risk metrics tracking
        self.daily_returns = []
        self.portfolio_values = [initial_capital]  # Start with initial capital
        self.peak_value = initial_capital
        
        # Track daily losses
        self.daily_loss = 0.0
        self.last_portfolio_value = initial_capital
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🛡️ Risk Manager initialized with ${initial_capital:,.2f} capital")
    
    def calculate_position_size(self, signal: TradeSignal, portfolio_value: float, 
                              volatility: float) -> int:
        """Calculate optimal position size using simplified Kelly Criterion"""
        
        # Ensure minimum portfolio value
        if portfolio_value <= 0:
            return 0
        
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
        quantity = int(position_value / max(signal.price, 0.01))  # Avoid division by zero
        
        return max(0, quantity)
    
    def calculate_stop_loss(self, signal: TradeSignal, atr: float, 
                          risk_per_trade: float = 0.01) -> float:
        """Calculate stop loss based on ATR and risk per trade"""
        if pd.isna(atr) or atr <= 0:
            atr = signal.price * 0.02  # Default 2% ATR
            
        if signal.action == "BUY":
            # Use 2x ATR or risk-based stop loss, whichever is closer
            atr_stop = signal.price - (atr * 2)
            risk_stop = signal.price * (1 - risk_per_trade)
            return max(atr_stop, risk_stop, signal.price * 0.95)  # At least 5% stop
        
        elif signal.action == "SELL":
            atr_stop = signal.price + (atr * 2)
            risk_stop = signal.price * (1 + risk_per_trade)
            return min(atr_stop, risk_stop, signal.price * 1.05)  # At least 5% stop
        
        return signal.price
    
    def calculate_position_risk(self, signal: TradeSignal, quantity: int) -> float:
        """Calculate risk for a single position"""
        if signal.stop_loss is None or pd.isna(signal.stop_loss):
            return 0.02  # Default 2% risk
        
        if signal.action == "BUY":
            risk = (signal.price - signal.stop_loss) / max(signal.price, 0.01)
        else:
            risk = (signal.stop_loss - signal.price) / max(signal.price, 0.01)
        
        return abs(risk)
    
    def update_risk_metrics(self, portfolio_value: float) -> RiskMetrics:
        """Update and calculate risk metrics"""
        # Ensure portfolio value is valid
        if portfolio_value <= 0:
            portfolio_value = self.initial_capital
        
        self.portfolio_values.append(portfolio_value)
        
        # Calculate daily return
        if len(self.portfolio_values) > 1:
            previous_value = self.portfolio_values[-2]
            daily_return = (portfolio_value / max(previous_value, 0.01)) - 1
            self.daily_returns.append(daily_return)
            
            # Update daily loss
            self.daily_loss = portfolio_value - previous_value if previous_value > 0 else 0
        
        # Update peak and drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        self.current_drawdown = (self.peak_value - portfolio_value) / max(self.peak_value, 1)
        
        # Calculate metrics only if we have sufficient data
        if len(self.daily_returns) < 10:
            return RiskMetrics(max_drawdown=self.current_drawdown)
        
        try:
            returns_series = pd.Series(self.daily_returns)
            
            # Remove any infinite or NaN values
            returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(returns_series) < 5:
                return RiskMetrics(max_drawdown=self.current_drawdown)
            
            # Sharpe Ratio (assuming risk-free rate of 2%)
            excess_returns = returns_series - (0.02 / 252)  # Daily risk-free rate
            returns_std = excess_returns.std()
            
            if returns_std > 0:
                sharpe_ratio = excess_returns.mean() / returns_std * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns_series[returns_series < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    sortino_ratio = returns_series.mean() / downside_std * np.sqrt(252)
                else:
                    sortino_ratio = sharpe_ratio
            else:
                sortino_ratio = sharpe_ratio if sharpe_ratio > 0 else 0
            
            # Value at Risk (95%)
            var_95 = returns_series.quantile(0.05) if len(returns_series) >= 20 else 0
            
            return RiskMetrics(
                max_drawdown=self.current_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                var_95=var_95
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(max_drawdown=self.current_drawdown)
    
    def should_stop_trading(self, current_loss: float, portfolio_value: float) -> bool:
        """Determine if trading should be stopped due to risk limits"""
        # Don't stop trading immediately on first run
        if len(self.portfolio_values) <= 1:
            return False
    
        if portfolio_value <= 0:
            return False
    
        # Calculate daily loss percentage from initial capital, not previous day
        daily_loss_pct = (self.initial_capital - portfolio_value) / self.initial_capital
    
        # Only stop if losses are really severe (much more lenient)
        severe_daily_loss = daily_loss_pct > 0.25  # 25% total loss
        severe_drawdown = self.current_drawdown > 0.30  # 30% drawdown
    
        if severe_daily_loss or severe_drawdown:
            self.logger.warning(f"🚨 Severe risk limits reached:")
            self.logger.warning(f"   Total loss: {daily_loss_pct*100:.2f}% (limit: 25.00%)")
            self.logger.warning(f"   Drawdown: {self.current_drawdown*100:.2f}% (limit: 30.00%)")
            return True
    
        return False
    
    def validate_signal(self, signal: TradeSignal, portfolio_value: float, 
                       current_positions: dict) -> bool:
        """Validate if a signal meets risk management criteria"""
        
        # Check minimum confidence
        if signal.confidence < TradingConfig.MIN_CONFIDENCE:
            self.logger.debug(f"Signal rejected: low confidence ({signal.confidence:.2f})")
            return False
        
        # Don't reject signals for existing positions - allow position management
        # if signal.symbol in current_positions:
        #     self.logger.info(f"Signal rejected: already have position in {signal.symbol}")
        #     return False
        
        # Ensure portfolio value is positive
        if portfolio_value <= 0:
            self.logger.warning("Signal rejected: invalid portfolio value")
            return False
        
        # Calculate proposed position size
        volatility = 0.02  # Default volatility
        quantity = self.calculate_position_size(signal, portfolio_value, volatility)
        
        if quantity == 0:
            self.logger.debug(f"Signal rejected: position size too small for {signal.symbol}")
            return False
        
        # Check position value doesn't exceed limits
        if signal.price > 0:
            position_value = quantity * signal.price
            position_pct = position_value / portfolio_value
            
            if position_pct > self.max_position_size * 2:  # Be more lenient
                self.logger.info(f"Signal rejected: position size too large ({position_pct:.1%})")
                return False
        
        return True
    
    def adjust_stop_loss(self, symbol: str, current_price: float, 
                        position: dict, trailing_pct: float = 0.05) -> float:
        """Adjust stop loss for trailing stop"""
        if not position.get('stop_loss') or current_price <= 0:
            return None
        
        current_stop = position['stop_loss']
        
        try:
            if position['quantity'] > 0:  # Long position
                # Trail stop loss upward
                new_stop = current_price * (1 - trailing_pct)
                return max(current_stop, new_stop)
            else:  # Short position
                # Trail stop loss downward
                new_stop = current_price * (1 + trailing_pct)
                return min(current_stop, new_stop)
        except (KeyError, TypeError, ValueError):
            return current_stop
    
    def check_position_exit(self, symbol: str, current_price: float, 
                           position: dict) -> bool:
        """Check if position should be exited based on risk management"""
        
        if current_price <= 0:
            return False
        
        try:
            # Check stop loss
            if position.get('stop_loss'):
                stop_loss = position['stop_loss']
                if position['quantity'] > 0 and current_price <= stop_loss:
                    self.logger.info(f"🛑 Stop loss triggered for {symbol}: ${current_price:.2f} <= ${stop_loss:.2f}")
                    return True
                elif position['quantity'] < 0 and current_price >= stop_loss:
                    self.logger.info(f"🛑 Stop loss triggered for {symbol}: ${current_price:.2f} >= ${stop_loss:.2f}")
                    return True
            
            # Check take profit
            if position.get('take_profit'):
                take_profit = position['take_profit']
                if position['quantity'] > 0 and current_price >= take_profit:
                    self.logger.info(f"🎯 Take profit triggered for {symbol}: ${current_price:.2f} >= ${take_profit:.2f}")
                    return True
                elif position['quantity'] < 0 and current_price <= take_profit:
                    self.logger.info(f"🎯 Take profit triggered for {symbol}: ${current_price:.2f} <= ${take_profit:.2f}")
                    return True
            
            # Check maximum holding period (30 days)
            if 'entry_time' in position:
                from datetime import datetime, timedelta
                holding_period = datetime.now() - position['entry_time']
                if holding_period > timedelta(days=30):
                    self.logger.info(f"⏰ Maximum holding period reached for {symbol}")
                    return True
            
        except (KeyError, TypeError, ValueError) as e:
            self.logger.error(f"Error checking exit conditions for {symbol}: {e}")
        
        return False
    
    def get_risk_summary(self) -> dict:
        """Get comprehensive risk summary"""
        current_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        risk_metrics = self.update_risk_metrics(current_value)
        
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': risk_metrics.max_drawdown,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'sortino_ratio': risk_metrics.sortino_ratio,
            'var_95': risk_metrics.var_95,
            'portfolio_values_count': len(self.portfolio_values),
            'daily_returns_count': len(self.daily_returns),
            'peak_value': self.peak_value,
            'current_value': current_value,
            'total_return': (current_value / self.initial_capital - 1) if self.initial_capital > 0 else 0
        }