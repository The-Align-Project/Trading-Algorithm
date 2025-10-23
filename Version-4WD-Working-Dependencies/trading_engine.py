"""Main trading engine that coordinates all components."""

import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Create a dummy class for type hints
    class pd:
        class DataFrame:
            pass

from config import (
    setup_logging, ALPACA_AVAILABLE, TradingConfig,
    is_market_open, get_time_until_market_open, get_market_status_message
)
from data_structures import TradeSignal
from data_fetcher import SimplifiedDataFetcher, AlpacaDataFetcher
from indicators import AdvancedIndicators
from ml_predictor import MLPredictor
from strategies import AdvancedTradingStrategies
from risk_manager import RiskManager
from portfolio_manager import PortfolioManager

if ALPACA_AVAILABLE:
    from alpaca_trade_api import REST
    from alpaca_trade_api.common import URL

class TradingEngine:
    """Main trading engine that coordinates all components"""
    
    def __init__(self, api_key=None, secret_key=None, paper=True, 
                 initial_capital=TradingConfig.DEFAULT_INITIAL_CAPITAL):
        # Setup logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.initial_capital = initial_capital
        
        # Initialize core components
        self.ml_predictor = MLPredictor()
        self.strategies = AdvancedTradingStrategies(self.ml_predictor)
        self.risk_manager = RiskManager(initial_capital)
        self.portfolio_manager = PortfolioManager(initial_capital)
        
        # Initialize data fetcher
        self._initialize_data_fetcher()
        
        # Initialize Alpaca connection
        self._initialize_alpaca()
        
        # Trading state
        self.is_trading = False
        self.iteration_count = 0
    
    def _initialize_data_fetcher(self):
        """Initialize appropriate data fetcher"""
        if ALPACA_AVAILABLE and self.api_key and self.secret_key:
            # Will be set after Alpaca initialization
            self.data_fetcher = None
        else:
            self.data_fetcher = SimplifiedDataFetcher()
    
    def _initialize_alpaca(self):
        """Initialize Alpaca API connection"""
        if ALPACA_AVAILABLE and self.api_key and self.secret_key:
            try:
                base_url = URL('https://paper-api.alpaca.markets') if self.paper else URL('https://api.alpaca.markets')
                self.api = REST(self.api_key, self.secret_key, base_url, api_version='v2')

                # Test connection
                account = self.api.get_account()
                self.alpaca_connected = True

                # Initialize Alpaca data fetcher
                self.data_fetcher = AlpacaDataFetcher(self.api)

                self.logger.info(f"✅ Connected to Alpaca {'Paper' if self.paper else 'Live'} Trading")
                self.logger.info(f"Account Status: {account.status}")

                # Sync portfolio with Alpaca account info properly
                try:
                    account_cash = float(account.cash)
                    if account_cash > 0 and hasattr(self.portfolio_manager, 'sync_with_alpaca_account'):
                        self.portfolio_manager.sync_with_alpaca_account(account_cash)
                    else:
                        self.portfolio_manager.cash = account_cash
                        self.logger.info(f"Updated cash from Alpaca: ${account_cash:,.2f}")
                except Exception as e:
                    self.logger.warning(f"Could not sync Alpaca account: {e}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to Alpaca: {e}")
                self.alpaca_connected = False
                self.api = None
                self.data_fetcher = SimplifiedDataFetcher()
        else:
            self.alpaca_connected = False
            self.api = None
            if not self.data_fetcher:
                self.data_fetcher = SimplifiedDataFetcher()
            self.logger.info("🎮 Running in simulation mode")
    
    def analyze_symbol(self, symbol: str, data) -> List[TradeSignal]:
        """Analyze a symbol and generate trading signals"""
        try:
            # Calculate technical indicators
            data_with_indicators = AdvancedIndicators.calculate_all_indicators(data)
            
            # Set symbol as attribute (for strategies to access)
            data_with_indicators.attrs['symbol'] = symbol
            
            # Train ML model if needed
            if not self.ml_predictor.is_trained:
                self.ml_predictor.train_model(data_with_indicators)
            
            # Generate signals from strategies - pass symbol explicitly
            signals = self.strategies.generate_signals(data_with_indicators, symbol=symbol)
            
            # Create ensemble signal if multiple strategies agree
            if len(signals) > 1:
                ensemble_signal = self.strategies.create_ensemble_signal(signals)
                if ensemble_signal.action != "HOLD":
                    return [ensemble_signal]
            
            # Return best individual signal
            if signals:
                return [max(signals, key=lambda x: x.confidence)]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return []
    
    def execute_signal(self, signal: TradeSignal) -> bool:
        """Execute a trading signal"""
        try:
            # Validate signal with risk manager
            current_positions = {pos.symbol: pos for pos in self.portfolio_manager.positions.values()}
            portfolio_value = self.portfolio_manager.get_portfolio_value()
            
            if not self.risk_manager.validate_signal(signal, portfolio_value, current_positions):
                return False
            
            # Calculate position size
            volatility = 0.02  # Default volatility
            quantity = self.risk_manager.calculate_position_size(signal, portfolio_value, volatility)
            
            if quantity == 0:
                self.logger.info(f"Position size too small for {signal.symbol}")
                return False
            
            signal.quantity = quantity
            
            # Execute order
            if self.alpaca_connected:
                success = self._execute_alpaca_order(signal)
            else:
                success = self._execute_simulated_order(signal)
            
            # Record trade
            self.portfolio_manager.record_trade(
                symbol=signal.symbol,
                action=signal.action,
                quantity=signal.quantity,
                price=signal.price,
                confidence=signal.confidence,
                strategy=signal.strategy,
                executed=success
            )
            
            return success
            
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
            
            self.logger.info(f"✅ Order submitted: {signal.action} {signal.quantity} {signal.symbol} @ ${signal.price:.2f}")
            
            # For market orders, wait briefly and check if filled
            if order_data['type'] == 'market':
                import time
                time.sleep(0.5)  # Brief wait for fill
                
                try:
                    order_status = self.api.get_order(order.id)
                    
                    if order_status.status == 'filled':
                        filled_price = float(order_status.filled_avg_price)
                        self.logger.info(f"✅ Order FILLED: {signal.symbol} @ ${filled_price:.2f}")
                        
                        # Add position to portfolio manager only if filled
                        if signal.action == "BUY":
                            self.portfolio_manager.add_position(
                                symbol=signal.symbol,
                                quantity=signal.quantity,
                                entry_price=filled_price,
                                stop_loss=signal.stop_loss,
                                take_profit=signal.take_profit,
                                strategy=signal.strategy
                            )
                            
                            # Immediately update current price for accurate P&L
                            current_price = self.data_fetcher.get_current_price(signal.symbol)
                            if current_price > 0 and signal.symbol in self.portfolio_manager.positions:
                                self.portfolio_manager.positions[signal.symbol].current_price = current_price
                        return True
                    else:
                        self.logger.warning(f"⚠️ Order not filled yet: {signal.symbol} (status: {order_status.status})")
                        # Don't add to portfolio manager yet
                        return True  # Still count as successful submission
                        
                except Exception as e:
                    self.logger.warning(f"Could not verify fill status: {e}")
                    # Assume filled for market orders in paper trading
                    if signal.action == "BUY":
                        self.portfolio_manager.add_position(
                            symbol=signal.symbol,
                            quantity=signal.quantity,
                            entry_price=signal.price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            strategy=signal.strategy
                        )
                    return True
            else:
                # For limit/bracket orders, don't add to portfolio yet
                self.logger.info(f"⏳ Bracket order pending: {signal.symbol}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error executing Alpaca order: {e}")
            return False
    
    def _execute_simulated_order(self, signal: TradeSignal) -> bool:
        """Execute simulated order"""
        try:
            cost = signal.quantity * signal.price
            
            if signal.action == "BUY":
                if cost <= self.portfolio_manager.cash:
                    success = self.portfolio_manager.add_position(
                        symbol=signal.symbol,
                        quantity=signal.quantity,
                        entry_price=signal.price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        strategy=signal.strategy
                    )
                    
                    if success:
                        self.logger.info(f"🎮 SIMULATED BUY: {signal.quantity} {signal.symbol} @ ${signal.price:.2f}")
                    return success
                else:
                    self.logger.warning(f"Insufficient cash for {signal.symbol}")
                    return False
            
            elif signal.action == "SELL" and signal.symbol in self.portfolio_manager.positions:
                trade_record = self.portfolio_manager.close_position(
                    symbol=signal.symbol,
                    exit_price=signal.price,
                    strategy=signal.strategy
                )
                
                if trade_record:
                    self.logger.info(f"🎮 SIMULATED SELL: {signal.quantity} {signal.symbol} @ ${signal.price:.2f}")
                    self.logger.info(f"   P&L: ${trade_record.pnl:.2f}")
                    return True
                
        except Exception as e:
            self.logger.error(f"Error in simulated order: {e}")
            return False
        
        return False
    
    def sync_portfolio_with_alpaca(self):
        """Sync portfolio manager positions with actual Alpaca positions"""
        if not self.alpaca_connected:
            return
        
        try:
            # Get actual positions from Alpaca
            alpaca_positions = self.api.list_positions()
            alpaca_symbols = {pos.symbol for pos in alpaca_positions}
            
            # Get portfolio manager positions
            portfolio_symbols = set(self.portfolio_manager.positions.keys())
            
            # Find discrepancies
            missing_from_portfolio = alpaca_symbols - portfolio_symbols
            missing_from_alpaca = portfolio_symbols - alpaca_symbols
            
            # Add positions that exist in Alpaca but not in portfolio manager
            for pos in alpaca_positions:
                if pos.symbol in missing_from_portfolio:
                    qty = int(pos.qty)
                    avg_price = float(pos.avg_entry_price)
                    current_price = float(pos.current_price)
                    
                    self.portfolio_manager.add_position(
                        symbol=pos.symbol,
                        quantity=qty,
                        entry_price=avg_price,
                        strategy="synced_from_alpaca"
                    )
                    
                    # Update current price immediately after adding
                    if pos.symbol in self.portfolio_manager.positions:
                        self.portfolio_manager.positions[pos.symbol].current_price = current_price
                    
                    self.logger.info(f"🔄 Synced position from Alpaca: {pos.symbol} ({qty} shares @ ${avg_price:.2f}, current: ${current_price:.2f})")
            
            # Remove positions from portfolio manager that don't exist in Alpaca
            for symbol in missing_from_alpaca:
                if symbol in self.portfolio_manager.positions:
                    pos = self.portfolio_manager.positions[symbol]
                    del self.portfolio_manager.positions[symbol]
                    # Restore cash (position was never really filled)
                    self.portfolio_manager.cash += pos.quantity * pos.entry_price
                    self.logger.warning(f"⚠️ Removed ghost position: {symbol} (not in Alpaca)")
            
        except Exception as e:
            self.logger.error(f"Error syncing with Alpaca: {e}")
    
    def update_positions(self, symbols: List[str]) -> Dict[str, float]:
        """Update position prices and check exit conditions"""
        current_prices = {}
        
        # Sync with Alpaca first to ensure accuracy
        self.sync_portfolio_with_alpaca()
        
        if not self.portfolio_manager.positions:
            return current_prices
        
        try:
            # Get current prices for all position symbols
            position_symbols = list(self.portfolio_manager.positions.keys())
            
            for symbol in position_symbols:
                current_price = self.data_fetcher.get_current_price(symbol)
                current_prices[symbol] = current_price
            
            # Update portfolio manager with current prices
            self.portfolio_manager.update_position_prices(current_prices)
            
            # Check exit conditions for each position
            positions_to_close = []
            for symbol, position in self.portfolio_manager.positions.items():
                current_price = current_prices.get(symbol, position.current_price)
                
                # Convert position to dict format for risk manager
                position_dict = {
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': current_price,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit,
                    'entry_time': position.entry_time
                }
                
                if self.risk_manager.check_position_exit(symbol, current_price, position_dict):
                    positions_to_close.append((symbol, current_price))
            
            # Close positions that need to be closed
            for symbol, exit_price in positions_to_close:
                self.portfolio_manager.close_position(symbol, exit_price, strategy="exit_condition")
        
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
        
        return current_prices
    
    def run_single_iteration(self, symbols: List[str]) -> Dict:
        """Run a single trading iteration"""
        iteration_start = time.time()
        results = {
            'signals_generated': 0,
            'signals_executed': 0,
            'positions_updated': 0,
            'errors': []
        }
        
        try:
            # Update existing positions
            current_prices = self.update_positions(symbols)
            results['positions_updated'] = len(current_prices)
            
            # Fetch data for all symbols
            data_dict = self.data_fetcher.fetch_data_parallel(symbols, period="3mo")
            
            # Filter for valid data with more lenient requirements
            valid_data = {symbol: data for symbol, data in data_dict.items() 
                         if data is not None and len(data) >= 30}  # Reduced from 50 to 30
            
            self.logger.info(f"📊 Fetched data for {len(valid_data)}/{len(symbols)} symbols")
            
            # If no valid data, just continue without stopping
            if not valid_data:
                self.logger.warning("No sufficient data for analysis, skipping this iteration")
                return results
            
            # Analyze each symbol and generate signals
            all_signals = []
            for symbol, data in valid_data.items():
                try:
                    # Analyze symbol and get signals
                    signals = self.analyze_symbol(symbol, data)
                    if signals:
                        all_signals.extend(signals)
                        results['signals_generated'] += len(signals)
                        
                except Exception as e:
                    error_msg = f"Error analyzing {symbol}: {e}"
                    self.logger.warning(error_msg)
                    results['errors'].append(error_msg)
            
            # Filter signals by confidence threshold
            high_confidence_signals = [
                s for s in all_signals 
                if s.confidence >= TradingConfig.MIN_CONFIDENCE and s.action != "HOLD"
            ]
            
            # Get current positions to avoid buying stocks we already own
            current_position_symbols = set(self.portfolio_manager.positions.keys())
            
            # Filter out BUY signals for stocks we already own (unless we want to add to position)
            # Keep SELL signals for stocks we own
            filtered_signals = []
            for signal in high_confidence_signals:
                if signal.action == "BUY" and signal.symbol in current_position_symbols:
                    self.logger.debug(f"Skipping BUY signal for {signal.symbol} - already have position")
                    continue
                elif signal.action == "SELL" and signal.symbol not in current_position_symbols:
                    self.logger.debug(f"Skipping SELL signal for {signal.symbol} - no position to sell")
                    continue
                else:
                    filtered_signals.append(signal)
            
            if filtered_signals:
                self.logger.info(f"🎯 Found {len(filtered_signals)} high-confidence signals (filtered from {len(high_confidence_signals)})")
                
                # Sort signals by confidence (highest first)
                filtered_signals.sort(key=lambda x: x.confidence, reverse=True)
                
                # Execute top signals (limit executions per iteration)
                executed_count = 0
                max_executions = TradingConfig.MAX_EXECUTIONS_PER_ITERATION
                
                for signal in filtered_signals:
                    if executed_count >= max_executions:
                        self.logger.info(f"⏸️ Reached max executions ({max_executions}) for this iteration")
                        break
                    
                    # Execute the signal
                    if self.execute_signal(signal):
                        executed_count += 1
                        results['signals_executed'] += 1
                        self.logger.info(f"✅ Executed {signal.action} signal for {signal.symbol} (confidence: {signal.confidence:.2f})")
                    else:
                        self.logger.info(f"❌ Failed to execute {signal.action} signal for {signal.symbol}")
            else:
                self.logger.info("📊 No high-confidence signals generated this iteration")
            
        except Exception as e:
            error_msg = f"Error in trading iteration: {e}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
        
        results['iteration_time'] = time.time() - iteration_start
        return results
    
    def run_live_trading(self, symbols: List[str], check_interval: int = TradingConfig.DEFAULT_CHECK_INTERVAL):
        """Run live trading with specified symbols"""
        self.is_trading = True
        self.logger.info(f"🚀 Starting Live Trading Engine (24/7 Mode)")
        self.logger.info(f"Symbols: {symbols}")
        self.logger.info(f"Check interval: {check_interval} seconds")
        self.logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Validate symbols
        valid_symbols = self.data_fetcher.validate_symbols(symbols)
        if len(valid_symbols) != len(symbols):
            self.logger.warning(f"Some symbols invalid. Using: {valid_symbols}")
            symbols = valid_symbols
        
        if not symbols:
            self.logger.error("No valid symbols to trade")
            return
        
        try:
            while self.is_trading:
                # Check market hours
                if not is_market_open():
                    status_msg = get_market_status_message()
                    self.logger.info(f"\n{'='*50}")
                    self.logger.info(status_msg)
                    
                    seconds_until_open = get_time_until_market_open()
                    
                    # If market opens soon (within check_interval), wait until then
                    if seconds_until_open <= check_interval:
                        self.logger.info(f"⏰ Waiting {int(seconds_until_open)}s until market opens...")
                        time.sleep(seconds_until_open)
                        continue
                    else:
                        # Otherwise, sleep for check_interval and check again
                        hours = int(seconds_until_open // 3600)
                        minutes = int((seconds_until_open % 3600) // 60)
                        self.logger.info(f"💤 Sleeping for {check_interval}s (Market opens in ~{hours}h {minutes}m)")
                        time.sleep(check_interval)
                        continue
                
                # Market is open - proceed with trading
                self.iteration_count += 1
                
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Trading Iteration #{self.iteration_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(get_market_status_message())
                
                # Run single iteration
                results = self.run_single_iteration(symbols)
                
                # Log iteration results
                self.logger.info(f"🎯 Signals generated: {results['signals_generated']}")
                self.logger.info(f"✅ Signals executed: {results['signals_executed']}")
                self.logger.info(f"📊 Positions updated: {results['positions_updated']}")
                
                # Performance reporting every 5 iterations
                if self.iteration_count % 5 == 0:
                    self._log_performance_metrics()
                
                # Current status
                self._log_current_status()
                
                # Check if we should stop trading due to risk limits
                portfolio_value = self.portfolio_manager.get_portfolio_value()
                if self.risk_manager.should_stop_trading(0, portfolio_value):
                    self.logger.warning("⚠️ Risk limits reached, stopping trading")
                    break
                
                # Calculate sleep time
                elapsed_time = results['iteration_time']
                sleep_time = max(0, check_interval - elapsed_time)
                
                self.logger.info(f"⏱️ Iteration completed in {elapsed_time:.1f}s. Next check in {sleep_time:.0f}s")
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            self.logger.info("🛑 Trading stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Critical error in trading loop: {e}")
        finally:
            self.is_trading = False
            self._cleanup()
    
    def _log_performance_metrics(self):
        """Log performance metrics"""
        try:
            metrics = self.portfolio_manager.calculate_performance_metrics()
            risk_summary = self.risk_manager.get_risk_summary()
            
            self.logger.info(f"\n📈 PERFORMANCE METRICS:")
            self.logger.info(f"   Portfolio Value: ${metrics.current_portfolio_value:,.2f}")
            self.logger.info(f"   Total Return: {metrics.total_return*100:.2f}%")
            self.logger.info(f"   Win Rate: {metrics.win_rate*100:.1f}%")
            self.logger.info(f"   Profit Factor: {metrics.profit_factor:.2f}")
            self.logger.info(f"   Total Trades: {metrics.total_trades}")
            
            if risk_summary.get('sharpe_ratio'):
                self.logger.info(f"   Sharpe Ratio: {risk_summary['sharpe_ratio']:.2f}")
                self.logger.info(f"   Max Drawdown: {risk_summary['max_drawdown']*100:.2f}%")
            
        except Exception as e:
            self.logger.error(f"Error logging performance metrics: {e}")
    
    def _log_current_status(self):
        """Log current portfolio status"""
        try:
            portfolio_value = self.portfolio_manager.get_portfolio_value()
            positions_summary = self.portfolio_manager.get_positions_summary()
            
            self.logger.info(f"\n💰 CURRENT STATUS:")
            self.logger.info(f"   Portfolio Value: ${portfolio_value:,.2f}")
            self.logger.info(f"   Cash: ${self.portfolio_manager.cash:,.2f}")
            self.logger.info(f"   Active Positions: {len(self.portfolio_manager.positions)}")
            
            if positions_summary.get('positions_detail'):
                self.logger.info(f"   Position Details:")
                for pos_detail in positions_summary['positions_detail'][:5]:  # Show top 5
                    symbol = pos_detail['symbol']
                    pnl = pos_detail['unrealized_pnl']
                    pnl_pct = pos_detail['unrealized_pnl_pct']
                    quantity = pos_detail['quantity']
                    self.logger.info(f"     {symbol}: {quantity} shares, P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Error logging current status: {e}")
    
    def _cleanup(self):
        """Clean up resources and generate final report"""
        self.logger.info("🧹 Cleaning up resources...")
        
        # First, cancel all pending orders
        if self.alpaca_connected:
            try:
                orders = self.api.list_orders(status='open')
                if orders:
                    self.logger.info(f"⚠️ Cancelling {len(orders)} pending orders...")
                    for order in orders:
                        self.api.cancel_order(order.id)
                        self.logger.info(f"❌ Cancelled order: {order.symbol}")
            except Exception as e:
                self.logger.error(f"Error cancelling orders: {e}")
        
        # Then close actual filled positions (from Alpaca, not portfolio manager)
        if self.alpaca_connected:
            try:
                # Get actual positions from Alpaca
                positions = self.api.list_positions()
                
                if positions:
                    self.logger.info(f"⚠️ Closing {len(positions)} actual positions...")
                    
                    for position in positions:
                        symbol = position.symbol
                        qty = abs(int(position.qty))
                        current_price = float(position.current_price)
                        
                        # Submit market order to close
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='sell' if float(position.qty) > 0 else 'buy',
                            type='market',
                            time_in_force='day'
                        )
                        
                        self.logger.info(f"🔴 Submitted close order: {symbol} ({qty} shares @ ~${current_price:.2f})")
                        
                        # Update portfolio manager if position exists there
                        if symbol in self.portfolio_manager.positions:
                            self.portfolio_manager.close_position(symbol, current_price, strategy="cleanup")
                    
                    self.logger.info("✅ All position close orders submitted")
                else:
                    self.logger.info("ℹ️ No actual positions to close")
                    
            except Exception as e:
                self.logger.error(f"Error closing positions: {e}")
        else:
            # Simulated mode - close portfolio manager positions
            if self.portfolio_manager.positions:
                symbols_to_close = list(self.portfolio_manager.positions.keys())
                for symbol in symbols_to_close:
                    position = self.portfolio_manager.positions[symbol]
                    self.portfolio_manager.close_position(symbol, position.current_price, strategy="cleanup")
                    self.logger.info(f"🔴 Closed simulated position: {symbol}")
        
        # Generate final performance report
        try:
            report = self.portfolio_manager.export_performance_report()
            self.logger.info("📊 Final performance report generated")
        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")
        
        self.logger.info("✅ Cleanup completed")
    
    def stop_trading(self):
        """Stop the trading engine"""
        self.is_trading = False
        self.logger.info("🛑 Trading engine stop requested")
    
    def get_status(self) -> Dict:
        """Get current status of the trading engine"""
        portfolio_value = self.portfolio_manager.get_portfolio_value()
        positions_summary = self.portfolio_manager.get_positions_summary()
        metrics = self.portfolio_manager.calculate_performance_metrics()
        
        return {
            'is_trading': self.is_trading,
            'iteration_count': self.iteration_count,
            'portfolio_value': portfolio_value,
            'cash': self.portfolio_manager.cash,
            'positions_count': len(self.portfolio_manager.positions),
            'total_trades': metrics.total_trades,
            'win_rate': metrics.win_rate,
            'total_return': metrics.total_return,
            'alpaca_connected': self.alpaca_connected,
            'ml_trained': self.ml_predictor.is_trained
        }