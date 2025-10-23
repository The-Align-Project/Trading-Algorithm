"""Backtesting engine for the trading algorithm."""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from config import TradingConfig
from data_structures import TradeSignal, TradeRecord
from data_fetcher import SimplifiedDataFetcher
from indicators import AdvancedIndicators
from ml_predictor import MLPredictor
from strategies import AdvancedTradingStrategies
from risk_manager import RiskManager

class BacktestEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, initial_capital: float = TradingConfig.DEFAULT_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_fetcher = SimplifiedDataFetcher()
        self.ml_predictor = MLPredictor()
        self.strategies = AdvancedTradingStrategies(self.ml_predictor)
        self.risk_manager = RiskManager(initial_capital)
    
    def run_backtest(self, symbols: List[str], start_date: str = None, 
                    end_date: str = None, commission: float = 0.0) -> Dict:
        """Run comprehensive backtest on multiple symbols"""
        
        self.logger.info(f"🔬 Starting backtest for symbols: {symbols}")
        
        # Determine data period
        period = self._determine_period(start_date, end_date)
        
        # Fetch data for all symbols
        data_dict = self.data_fetcher.fetch_data_parallel(symbols, period=period)
        
        # Filter out symbols with insufficient data
        valid_data = {symbol: data for symbol, data in data_dict.items() 
                     if data is not None and len(data) >= 100}
        
        if not valid_data:
            self.logger.error("No valid data for backtesting")
            return {}
        
        self.logger.info(f"Backtesting {len(valid_data)} symbols with sufficient data")
        
        # Run backtest for each symbol
        symbol_results = {}
        overall_trades = []
        
        for symbol, data in valid_data.items():
            self.logger.info(f"Backtesting {symbol}...")
            
            symbol_result, symbol_trades = self._backtest_symbol(symbol, data, commission)
            symbol_results[symbol] = symbol_result
            overall_trades.extend(symbol_trades)
        
        # Calculate overall results
        overall_results = self._calculate_overall_results(symbol_results, overall_trades)
        
        # Generate detailed report
        self._generate_backtest_report(symbol_results, overall_results)
        
        return {
            'overall_results': overall_results,
            'symbol_results': symbol_results,
            'total_trades': len(overall_trades),
            'symbols_tested': list(valid_data.keys())
        }
    
    def _determine_period(self, start_date: str = None, end_date: str = None) -> str:
        """Determine appropriate data period"""
        if start_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date) if end_date else datetime.now()
            period_days = (end - start).days
            
            if period_days > 365:
                return "2y"
            elif period_days > 90:
                return "1y"
            else:
                return "3mo"
        else:
            return "1y"
    
    def _backtest_symbol(self, symbol: str, data: pd.DataFrame, 
                        commission: float) -> Tuple[Dict, List[Dict]]:
        """Backtest a single symbol"""
        
        # Calculate technical indicators
        data_with_indicators = AdvancedIndicators.calculate_all_indicators(data)
        
        # Train ML model on first portion of data
        train_size = min(200, len(data_with_indicators) // 2)
        train_data = data_with_indicators.iloc[:train_size]
        self.ml_predictor.train_model(train_data)
        
        # Initialize backtest state
        capital = self.initial_capital / len([symbol])  # Allocated capital for this symbol
        positions = []
        trades = []
        equity_curve = []
        
        # Walk through data day by day
        for i in range(50, len(data_with_indicators)):
            current_date = data_with_indicators.index[i]
            current_data = data_with_indicators.iloc[:i+1]
            current_row = current_data.iloc[-1]
            
            # Record equity
            position_value = sum(pos['quantity'] * current_row['Close'] for pos in positions)
            total_equity = capital + position_value
            equity_curve.append({
                'date': current_date,
                'equity': total_equity,
                'cash': capital,
                'positions_value': position_value
            })
            
            # Generate trading signals
            signals = self.strategies.generate_signals(current_data)
            
            if signals:
                best_signal = max(signals, key=lambda x: x.confidence)
                
                # Execute buy signals
                if (best_signal.action == "BUY" and 
                    len(positions) == 0 and 
                    best_signal.confidence > TradingConfig.MIN_CONFIDENCE):
                    
                    # Calculate position size (simplified)
                    max_position_value = capital * 0.95  # Use 95% of capital
                    position_size = min(100, int(max_position_value / best_signal.price))
                    
                    if position_size > 0:
                        cost = position_size * best_signal.price + commission
                        
                        if cost <= capital:
                            # Create position
                            position = {
                                'symbol': symbol,
                                'quantity': position_size,
                                'entry_price': best_signal.price,
                                'entry_date': current_date,
                                'stop_loss': best_signal.stop_loss,
                                'take_profit': best_signal.take_profit,
                                'strategy': best_signal.strategy
                            }
                            positions.append(position)
                            capital -= cost
                
                # Execute sell signals
                elif (best_signal.action == "SELL" and 
                      positions and 
                      best_signal.confidence > 0.5):
                    
                    # Close all positions
                    for pos in positions:
                        exit_price = best_signal.price
                        proceeds = pos['quantity'] * exit_price - commission
                        pnl = proceeds - (pos['quantity'] * pos['entry_price'])
                        capital += proceeds
                        
                        # Record trade
                        trade = self._create_trade_record(pos, exit_price, pnl, 
                                                        current_date, best_signal.strategy)
                        trades.append(trade)
                    
                    positions = []
            
            # Check stop losses and take profits
            for pos in positions[:]:  # Copy list to allow modification during iteration
                current_price = current_row['Close']
                exit_triggered = False
                exit_reason = ""
                
                if pos.get('stop_loss') and current_price <= pos['stop_loss']:
                    exit_triggered = True
                    exit_reason = "stop_loss"
                    exit_price = pos['stop_loss']
                    
                elif pos.get('take_profit') and current_price >= pos['take_profit']:
                    exit_triggered = True
                    exit_reason = "take_profit"
                    exit_price = pos['take_profit']
                
                if exit_triggered:
                    proceeds = pos['quantity'] * exit_price - commission
                    pnl = proceeds - (pos['quantity'] * pos['entry_price'])
                    capital += proceeds
                    
                    # Record trade
                    trade = self._create_trade_record(pos, exit_price, pnl, 
                                                    current_date, pos['strategy'] + f"_{exit_reason}")
                    trades.append(trade)
                    positions.remove(pos)
        
        # Close remaining positions at final price
        if positions:
            final_price = data_with_indicators.iloc[-1]['Close']
            final_date = data_with_indicators.index[-1]
            
            for pos in positions:
                proceeds = pos['quantity'] * final_price - commission
                pnl = proceeds - (pos['quantity'] * pos['entry_price'])
                capital += proceeds
                
                trade = self._create_trade_record(pos, final_price, pnl, 
                                                final_date, pos['strategy'] + "_final")
                trades.append(trade)
        
        # Calculate symbol results
        symbol_result = self._calculate_symbol_results(symbol, capital, trades, equity_curve)
        
        return symbol_result, trades
    
    def _create_trade_record(self, position: Dict, exit_price: float, pnl: float,
                           exit_date: pd.Timestamp, strategy: str) -> Dict:
        """Create a trade record dictionary"""
        return {
            'symbol': position['symbol'],
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'return_pct': pnl / (position['quantity'] * position['entry_price']),
            'strategy': strategy,
            'days_held': (exit_date - position['entry_date']).days
        }
    
    def _calculate_symbol_results(self, symbol: str, final_capital: float,
                                 trades: List[Dict], equity_curve: List[Dict]) -> Dict:
        """Calculate results for a single symbol"""
        
        initial_capital = self.initial_capital
        
        if not trades:
            return {
                'symbol': symbol,
                'total_trades': 0,
                'final_capital': final_capital,
                'total_return': (final_capital - initial_capital) / initial_capital,
                'win_rate': 0,
                'avg_return_per_trade': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winning_trades) / total_trades
        
        total_return = (final_capital - initial_capital) / initial_capital
        avg_return_per_trade = trades_df['return_pct'].mean()
        best_trade = trades_df['return_pct'].max()
        worst_trade = trades_df['return_pct'].min()
        
        # Calculate drawdown from equity curve
        equity_df = pd.DataFrame(equity_curve)
        if len(equity_df) > 0:
            equity_series = equity_df['equity']
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = drawdown.min()
            
            # Calculate Sharpe ratio
            equity_returns = equity_series.pct_change().dropna()
            if len(equity_returns) > 0 and equity_returns.std() > 0:
                sharpe_ratio = equity_returns.mean() / equity_returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0
        
        return {
            'symbol': symbol,
            'total_trades': total_trades,
            'final_capital': final_capital,
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return_per_trade,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_days_held': trades_df['days_held'].mean(),
            'max_drawdown': abs(max_drawdown),
            'sharpe_ratio': sharpe_ratio
        }
    
    def _calculate_overall_results(self, symbol_results: Dict, 
                                  overall_trades: List[Dict]) -> Dict:
        """Calculate overall backtest results"""
        
        if not symbol_results:
            return {}
        
        # Aggregate results
        total_final_capital = sum(result['final_capital'] for result in symbol_results.values())
        total_trades = sum(result['total_trades'] for result in symbol_results.values())
        
        # Calculate weighted averages
        capitals = [result['final_capital'] for result in symbol_results.values()]
        returns = [result['total_return'] for result in symbol_results.values()]
        win_rates = [result['win_rate'] for result in symbol_results.values() if result['total_trades'] > 0]
        
        avg_return = np.mean(returns) if returns else 0
        avg_win_rate = np.mean(win_rates) if win_rates else 0
        
        overall_return = (total_final_capital - self.initial_capital) / self.initial_capital
        
        # Calculate additional metrics from trades
        if overall_trades:
            trades_df = pd.DataFrame(overall_trades)
            profit_factor = self._calculate_profit_factor(trades_df)
            avg_trade_return = trades_df['return_pct'].mean()
        else:
            profit_factor = 0
            avg_trade_return = 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': total_final_capital,
            'total_return': overall_return,
            'total_trades': total_trades,
            'symbols_traded': len(symbol_results),
            'avg_return_per_symbol': avg_return,
            'avg_win_rate': avg_win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return
        }
    
    def _calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate profit factor"""
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _generate_backtest_report(self, symbol_results: Dict, overall_results: Dict):
        """Generate and log comprehensive backtest report"""
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🔬 BACKTEST RESULTS")
        self.logger.info(f"{'='*60}")
        
        # Overall performance
        if overall_results:
            self.logger.info(f"\n🎯 OVERALL PERFORMANCE:")
            self.logger.info(f"   Initial Capital: ${overall_results['initial_capital']:,.2f}")
            self.logger.info(f"   Final Capital: ${overall_results['final_capital']:,.2f}")
            self.logger.info(f"   Total Return: {overall_results['total_return']*100:.2f}%")
            self.logger.info(f"   Total Trades: {overall_results['total_trades']}")
            self.logger.info(f"   Symbols Traded: {overall_results['symbols_traded']}")
            self.logger.info(f"   Average Win Rate: {overall_results['avg_win_rate']*100:.1f}%")
            self.logger.info(f"   Profit Factor: {overall_results.get('profit_factor', 0):.2f}")
        
        # Individual symbol performance
        for symbol, results in symbol_results.items():
            self.logger.info(f"\n📊 {symbol}:")
            self.logger.info(f"   Total Trades: {results['total_trades']}")
            self.logger.info(f"   Win Rate: {results['win_rate']*100:.1f}%")
            self.logger.info(f"   Total Return: {results['total_return']*100:.2f}%")
            self.logger.info(f"   Best/Worst Trade: {results['best_trade']*100:.2f}% / {results['worst_trade']*100:.2f}%")
            self.logger.info(f"   Max Drawdown: {results['max_drawdown']*100:.2f}%")
            self.logger.info(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            self.logger.info(f"   Final Capital: ${results['final_capital']:,.2f}")
    
    def export_backtest_results(self, results: Dict, filename: str = None) -> str:
        """Export backtest results to file"""
        import os
        
        # Create results directory if it doesn't exist
        os.makedirs('results/backtest', exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/backtest/backtest_results_{timestamp}.txt"
        
        try:
            report_lines = []
            report_lines.append("BACKTEST RESULTS REPORT")
            report_lines.append("=" * 50)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Add overall results
            if 'overall_results' in results:
                overall = results['overall_results']
                report_lines.append("OVERALL PERFORMANCE:")
                for key, value in overall.items():
                    if isinstance(value, float):
                        if 'return' in key.lower():
                            report_lines.append(f"  {key}: {value*100:.2f}%")
                        else:
                            report_lines.append(f"  {key}: {value:.2f}")
                    else:
                        report_lines.append(f"  {key}: {value}")
                report_lines.append("")
            
            # Add individual symbol results
            if 'symbol_results' in results:
                report_lines.append("INDIVIDUAL SYMBOL RESULTS:")
                for symbol, symbol_result in results['symbol_results'].items():
                    report_lines.append(f"\n{symbol}:")
                    for key, value in symbol_result.items():
                        if key != 'symbol':
                            if isinstance(value, float):
                                if 'return' in key.lower() or 'rate' in key.lower():
                                    report_lines.append(f"    {key}: {value*100:.2f}%")
                                else:
                                    report_lines.append(f"    {key}: {value:.2f}")
                            else:
                                report_lines.append(f"    {key}: {value}")
            
            report_text = "\n".join(report_lines)
            
            with open(filename, 'w') as f:
                f.write(report_text)
            
            self.logger.info(f"Backtest results exported to {filename}")
            return report_text
            
        except Exception as e:
            self.logger.error(f"Error exporting backtest results: {e}")
            return ""