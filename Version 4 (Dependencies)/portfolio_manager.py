"""Portfolio management and tracking for the trading algorithm."""

import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional

from data_structures import Position, TradeRecord, PortfolioSnapshot, PerformanceMetrics
from config import TradingConfig

class PortfolioManager:
    """Manage portfolio positions and track performance"""
    
    def __init__(self, initial_capital: float = TradingConfig.DEFAULT_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital  # Don't override this immediately
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.portfolio_history: List[PortfolioSnapshot] = []

        self.logger = logging.getLogger(__name__)
    
    def sync_with_alpaca_account(self, account_cash: float):
        """Sync portfolio with Alpaca account - call this after initialization"""
        if account_cash > 0:
            self.cash = account_cash
            # Don't change initial_capital - keep it for return calculations
            self.logger.info(f"Synced portfolio cash with Alpaca: ${account_cash:,.2f}")
    
    def add_position(self, symbol: str, quantity: int, entry_price: float, 
                    stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                    strategy: str = "") -> bool:
        """Add a new position to the portfolio"""
        try:
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=strategy
            )
            
            self.positions[symbol] = position
            
            # Update cash
            cost = abs(quantity) * entry_price
            self.cash -= cost
            
            self.logger.info(f"Added position: {quantity} shares of {symbol} at ${entry_price:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding position for {symbol}: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, strategy: str = "") -> Optional[TradeRecord]:
        """Close an existing position"""
        if symbol not in self.positions:
            self.logger.warning(f"No position found for {symbol}")
            return None
        
        try:
            position = self.positions[symbol]
            
            # Calculate P&L
            if position.quantity > 0:  # Long position
                pnl = (exit_price - position.entry_price) * position.quantity
            else:  # Short position
                pnl = (position.entry_price - exit_price) * abs(position.quantity)
            
            # Update cash
            proceeds = abs(position.quantity) * exit_price
            self.cash += proceeds
            
            # Create trade record
            trade_record = TradeRecord(
                timestamp=datetime.now(),
                symbol=symbol,
                action="SELL" if position.quantity > 0 else "COVER",
                quantity=abs(position.quantity),
                price=exit_price,
                confidence=1.0,  # Exit trades have full confidence
                strategy=strategy or position.strategy,
                executed=True,
                pnl=pnl
            )
            
            self.trade_history.append(trade_record)
            
            # Remove position
            del self.positions[symbol]
            
            self.logger.info(f"Closed position: {symbol} at ${exit_price:.2f}, P&L: ${pnl:.2f}")
            return trade_record
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return None
    
    def update_position_prices(self, current_prices: Dict[str, float]) -> Dict[str, Position]:
        """Update current prices for all positions"""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.current_price = current_prices[symbol]
        
        return self.positions
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_positions_summary(self) -> Dict:
        """Get summary of current positions"""
        if not self.positions:
            return {}
        
        total_value = 0
        total_pnl = 0
        long_positions = 0
        short_positions = 0
        
        positions_detail = []
        
        for symbol, position in self.positions.items():
            total_value += position.market_value
            total_pnl += position.unrealized_pnl
            
            if position.quantity > 0:
                long_positions += 1
            else:
                short_positions += 1
            
            positions_detail.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_percent,
                'days_held': (datetime.now() - position.entry_time).days,
                'strategy': position.strategy
            })
        
        return {
            'total_positions': len(self.positions),
            'long_positions': long_positions,
            'short_positions': short_positions,
            'total_market_value': total_value,
            'total_unrealized_pnl': total_pnl,
            'positions_detail': positions_detail
        }
    
    def record_trade(self, symbol: str, action: str, quantity: int, price: float,
                    confidence: float, strategy: str, executed: bool = True) -> TradeRecord:
        """Record a trade in the history"""
        trade_record = TradeRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            confidence=confidence,
            strategy=strategy,
            executed=executed
        )
        
        self.trade_history.append(trade_record)
        return trade_record
    
    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a snapshot of current portfolio state"""
        portfolio_value = self.get_portfolio_value()
        positions_value = portfolio_value - self.cash
        
        # Calculate daily P&L if we have previous snapshots
        daily_pnl = 0.0
        if self.portfolio_history:
            previous_value = self.portfolio_history[-1].total_value
            daily_pnl = portfolio_value - previous_value
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            cash=self.cash,
            positions_value=positions_value,
            total_value=portfolio_value,
            positions_count=len(self.positions),
            daily_pnl=daily_pnl
        )
        
        self.portfolio_history.append(snapshot)
        return snapshot
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.trade_history:
            return PerformanceMetrics()
        
        try:
            # Filter executed trades
            executed_trades = [trade for trade in self.trade_history if trade.executed]
            
            if not executed_trades:
                return PerformanceMetrics()
            
            # Basic metrics
            total_trades = len(executed_trades)
            profitable_trades = len([trade for trade in executed_trades if trade.pnl > 0])
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(trade.pnl for trade in executed_trades)
            winning_trades = [trade for trade in executed_trades if trade.pnl > 0]
            losing_trades = [trade for trade in executed_trades if trade.pnl < 0]
            
            avg_win = sum(trade.pnl for trade in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(trade.pnl for trade in losing_trades) / len(losing_trades) if losing_trades else 0
            
            # Profit factor
            gross_profit = sum(trade.pnl for trade in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(trade.pnl for trade in losing_trades)) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Portfolio metrics
            current_value = self.get_portfolio_value()
            total_return = (current_value - self.initial_capital) / self.initial_capital
            
            # Strategy breakdown
            strategy_performance = {}
            trades_df = pd.DataFrame([{
                'strategy': trade.strategy,
                'pnl': trade.pnl,
                'executed': trade.executed
            } for trade in executed_trades])
            
            if not trades_df.empty:
                strategy_stats = trades_df.groupby('strategy')['pnl'].agg(['sum', 'count', 'mean']).to_dict()
                strategy_performance = {
                    'sum': strategy_stats.get('sum', {}),
                    'count': strategy_stats.get('count', {}),
                    'mean': strategy_stats.get('mean', {})
                }
            
            return PerformanceMetrics(
                total_trades=total_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_return=total_return,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                current_portfolio_value=current_value,
                strategy_performance=strategy_performance
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics()
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.trade_history:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trade_history:
            trades_data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'action': trade.action,
                'quantity': trade.quantity,
                'price': trade.price,
                'confidence': trade.confidence,
                'strategy': trade.strategy,
                'executed': trade.executed,
                'pnl': trade.pnl,
                'commission': trade.commission
            })
        
        return pd.DataFrame(trades_data)
    
    def get_portfolio_history_df(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        if not self.portfolio_history:
            return pd.DataFrame()
        
        history_data = []
        for snapshot in self.portfolio_history:
            history_data.append({
                'timestamp': snapshot.timestamp,
                'cash': snapshot.cash,
                'positions_value': snapshot.positions_value,
                'total_value': snapshot.total_value,
                'positions_count': snapshot.positions_count,
                'daily_pnl': snapshot.daily_pnl,
                'cash_percentage': snapshot.cash_percentage
            })
        
        return pd.DataFrame(history_data)
    
    def export_performance_report(self, filename: str = None) -> str:
        """Export detailed performance report"""
        if not filename:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        metrics = self.calculate_performance_metrics()
        positions_summary = self.get_positions_summary()
        
        report = []
        report.append("=" * 60)
        report.append("PORTFOLIO PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("PORTFOLIO OVERVIEW:")
        report.append(f"  Initial Capital: ${self.initial_capital:,.2f}")
        report.append(f"  Current Value: ${metrics.current_portfolio_value:,.2f}")
        report.append(f"  Cash: ${self.cash:,.2f}")
        report.append(f"  Total Return: {metrics.total_return*100:.2f}%")
        report.append("")
        
        report.append("TRADING PERFORMANCE:")
        report.append(f"  Total Trades: {metrics.total_trades}")
        report.append(f"  Win Rate: {metrics.win_rate*100:.1f}%")
        report.append(f"  Total P&L: ${metrics.total_pnl:,.2f}")
        report.append(f"  Average Win: ${metrics.avg_win:.2f}")
        report.append(f"  Average Loss: ${metrics.avg_loss:.2f}")
        report.append(f"  Profit Factor: {metrics.profit_factor:.2f}")
        report.append("")
        
        if positions_summary:
            report.append("CURRENT POSITIONS:")
            report.append(f"  Total Positions: {positions_summary['total_positions']}")
            report.append(f"  Long Positions: {positions_summary['long_positions']}")
            report.append(f"  Short Positions: {positions_summary['short_positions']}")
            report.append(f"  Unrealized P&L: ${positions_summary['total_unrealized_pnl']:,.2f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        try:
            with open(filename, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Performance report exported to {filename}")
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
        
        return report_text