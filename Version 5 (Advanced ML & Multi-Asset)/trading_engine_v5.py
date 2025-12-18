"""Enhanced Version 5 Trading Engine with all advanced features."""

import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

from config import (
    ALPACA_AVAILABLE, setup_logging, TradingConfig,
    is_market_open, get_market_status_message
)
from data_fetcher import SimplifiedDataFetcher, AlpacaDataFetcher
from indicators import AdvancedIndicators
from ml_predictor import MLPredictor
from strategies_v5 import EnhancedTradingStrategiesV5
from risk_manager import RiskManager
from portfolio_manager import PortfolioManager
from data_structures import TradeSignal

if ALPACA_AVAILABLE:
    from alpaca_trade_api import REST

class TradingEngineV5:
    """Enhanced Version 5 Trading Engine with Deep Learning, Multi-timeframe, Options, and Sentiment"""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None,
                 paper: bool = True, initial_capital: float = 100000,
                 news_api_key: Optional[str] = None,
                 twitter_bearer_token: Optional[str] = None):
        """Initialize Version 5 Trading Engine"""

        setup_logging()
        self.logger = logging.getLogger(__name__)

        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.initial_capital = initial_capital

        # Initialize Alpaca API if credentials provided
        self.api = None
        if api_key and secret_key and ALPACA_AVAILABLE:
            try:
                base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
                self.api = REST(api_key, secret_key, base_url, api_version='v2')

                # Get account info
                account = self.api.get_account()
                self.logger.info(f"✅ Connected to Alpaca ({('Paper' if paper else 'Live')} Trading)")
                self.logger.info(f"   Account: ${float(account.cash):,.2f} cash, ${float(account.portfolio_value):,.2f} total")
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to Alpaca: {e}")
                self.api = None

        # Initialize components
        if self.api:
            self.data_fetcher = AlpacaDataFetcher(self.api)
        else:
            self.data_fetcher = SimplifiedDataFetcher()

        self.ml_predictor = MLPredictor()
        self.strategies = EnhancedTradingStrategiesV5(
            self.ml_predictor,
            news_api_key=news_api_key,
            twitter_bearer_token=twitter_bearer_token
        )
        self.risk_manager = RiskManager(initial_capital)
        self.portfolio_manager = PortfolioManager(initial_capital)

        # Sync with Alpaca account if connected
        if self.api:
            try:
                account = self.api.get_account()
                self.portfolio_manager.sync_with_alpaca_account(float(account.cash))
            except Exception as e:
                self.logger.warning(f"Could not sync with Alpaca account: {e}")

        self.is_running = False

        self.logger.info("🚀 Version 5 Trading Engine initialized")
        self.logger.info(f"   Initial Capital: ${initial_capital:,.2f}")
        self.logger.info(f"   Mode: {'Live API' if self.api else 'Simulation'}")

    def run_live_trading(self, symbols: List[str], check_interval: int = 300):
        """Run live trading loop"""
        self.is_running = True
        self.logger.info(f"▶️  Starting live trading for {len(symbols)} symbols")
        self.logger.info(f"   Symbols: {symbols}")
        self.logger.info(f"   Check Interval: {check_interval}s")

        iteration = 0

        try:
            while self.is_running:
                iteration += 1
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"🔄 ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"{'='*70}")

                # Check market status
                market_status = get_market_status_message()
                self.logger.info(f"📊 {market_status}")

                # Run single iteration
                results = self.run_single_iteration(symbols)

                # Display iteration summary
                self.display_iteration_summary(results)

                # Check risk limits
                portfolio_value = self.portfolio_manager.get_portfolio_value()
                if self.risk_manager.should_stop_trading(0, portfolio_value):
                    self.logger.warning("🛑 Risk limits breached. Stopping trading.")
                    break

                # Wait for next iteration
                self.logger.info(f"\n⏳ Waiting {check_interval}s until next iteration...")
                time.sleep(check_interval)

        except KeyboardInterrupt:
            self.logger.info("\n🛑 Trading stopped by user")
        finally:
            self.stop_trading()

    def run_single_iteration(self, symbols: List[str]) -> Dict:
        """Run a single trading iteration"""
        results = {
            'timestamp': datetime.now(),
            'symbols_analyzed': 0,
            'signals_generated': 0,
            'signals_executed': 0,
            'trades': []
        }

        try:
            # Fetch data for all symbols
            self.logger.info(f"\n📊 Fetching data for {len(symbols)} symbols...")
            data_dict = self.data_fetcher.fetch_data_parallel(symbols, period='3mo')

            if not data_dict:
                self.logger.warning("⚠️  No data fetched")
                return results

            results['symbols_analyzed'] = len(data_dict)

            # Analyze each symbol
            all_signals = []

            for symbol, data in data_dict.items():
                if data is None or len(data) < 100:
                    self.logger.warning(f"⚠️  Insufficient data for {symbol}")
                    continue

                self.logger.info(f"\n📈 Analyzing {symbol}...")

                # Calculate technical indicators
                data_with_indicators = AdvancedIndicators.calculate_all_indicators(data)

                # Train/update ML model
                self.ml_predictor.retrain_if_needed(data_with_indicators)

                # Generate signals from all V5 strategies
                signals = self.strategies.generate_all_signals(data_with_indicators, symbol)

                if signals:
                    self.logger.info(f"   Generated {len(signals)} signals for {symbol}")
                    for sig in signals:
                        self.logger.info(f"   - {sig.strategy}: {sig.action} (confidence: {sig.confidence:.2%})")
                    all_signals.extend(signals)
                else:
                    self.logger.info(f"   No signals for {symbol}")

            results['signals_generated'] = len(all_signals)

            if not all_signals:
                self.logger.info("\n💤 No trading signals generated this iteration")
                return results

            # Filter and rank signals
            valid_signals = self.filter_signals(all_signals)

            if not valid_signals:
                self.logger.info("\n⚠️  No valid signals after filtering")
                return results

            # Execute top signals (limited per iteration)
            executed_count = self.execute_signals(valid_signals[:TradingConfig.MAX_EXECUTIONS_PER_ITERATION])
            results['signals_executed'] = executed_count

            # Update portfolio
            self.update_portfolio_prices(symbols)

            # Check for exits
            self.check_position_exits()

        except Exception as e:
            self.logger.error(f"❌ Error in iteration: {e}")
            import traceback
            traceback.print_exc()

        return results

    def filter_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Filter and validate signals"""
        valid_signals = []
        portfolio_value = self.portfolio_manager.get_portfolio_value()
        current_positions = {pos: True for pos in self.portfolio_manager.positions.keys()}

        for signal in signals:
            # Validate with risk manager
            if self.risk_manager.validate_signal(signal, portfolio_value, current_positions):
                valid_signals.append(signal)
            else:
                self.logger.debug(f"Signal rejected by risk manager: {signal.symbol} {signal.action}")

        # Sort by confidence
        valid_signals.sort(key=lambda x: x.confidence, reverse=True)

        return valid_signals

    def execute_signals(self, signals: List[TradeSignal]) -> int:
        """Execute trading signals"""
        executed_count = 0

        self.logger.info(f"\n🎯 Executing top {len(signals)} signals...")

        for signal in signals:
            try:
                if signal.action == "BUY":
                    success = self.execute_buy(signal)
                elif signal.action == "SELL":
                    success = self.execute_sell(signal)
                else:
                    continue

                if success:
                    executed_count += 1

            except Exception as e:
                self.logger.error(f"❌ Error executing signal for {signal.symbol}: {e}")

        return executed_count

    def execute_buy(self, signal: TradeSignal) -> bool:
        """Execute buy order"""
        try:
            portfolio_value = self.portfolio_manager.get_portfolio_value()

            # Calculate position size
            volatility = signal.metadata.get('volatility', 0.02) if signal.metadata else 0.02
            quantity = self.risk_manager.calculate_position_size(signal, portfolio_value, volatility)

            if quantity <= 0:
                self.logger.warning(f"⚠️  Position size too small for {signal.symbol}")
                return False

            # Calculate stop loss if not provided
            if signal.stop_loss is None:
                atr = signal.metadata.get('atr', signal.price * 0.02) if signal.metadata else signal.price * 0.02
                signal.stop_loss = self.risk_manager.calculate_stop_loss(signal, atr)

            self.logger.info(f"\n🔵 BUY ORDER: {signal.symbol}")
            self.logger.info(f"   Quantity: {quantity}")
            self.logger.info(f"   Price: ${signal.price:.2f}")
            self.logger.info(f"   Confidence: {signal.confidence:.1%}")
            self.logger.info(f"   Strategy: {signal.strategy}")
            self.logger.info(f"   Stop Loss: ${signal.stop_loss:.2f}")
            if signal.take_profit:
                self.logger.info(f"   Take Profit: ${signal.take_profit:.2f}")

            # Execute with Alpaca or simulate
            if self.api:
                try:
                    order = self.api.submit_order(
                        symbol=signal.symbol,
                        qty=quantity,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    self.logger.info(f"   ✅ Order submitted: {order.id}")
                except Exception as e:
                    self.logger.error(f"   ❌ Alpaca order failed: {e}")
                    # Continue with simulation

            # Update portfolio
            success = self.portfolio_manager.add_position(
                symbol=signal.symbol,
                quantity=quantity,
                entry_price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy=signal.strategy
            )

            if success:
                # Record trade
                self.portfolio_manager.record_trade(
                    symbol=signal.symbol,
                    action='BUY',
                    quantity=quantity,
                    price=signal.price,
                    confidence=signal.confidence,
                    strategy=signal.strategy,
                    executed=True
                )
                self.logger.info(f"   ✅ Position opened successfully")
                return True

        except Exception as e:
            self.logger.error(f"❌ Error executing buy for {signal.symbol}: {e}")

        return False

    def execute_sell(self, signal: TradeSignal) -> bool:
        """Execute sell order"""
        try:
            # Check if we have a position
            if signal.symbol not in self.portfolio_manager.positions:
                self.logger.warning(f"⚠️  No position to sell for {signal.symbol}")
                return False

            position = self.portfolio_manager.positions[signal.symbol]

            self.logger.info(f"\n🔴 SELL ORDER: {signal.symbol}")
            self.logger.info(f"   Quantity: {position.quantity}")
            self.logger.info(f"   Entry Price: ${position.entry_price:.2f}")
            self.logger.info(f"   Exit Price: ${signal.price:.2f}")
            self.logger.info(f"   P&L: ${(signal.price - position.entry_price) * position.quantity:.2f}")

            # Execute with Alpaca or simulate
            if self.api:
                try:
                    order = self.api.submit_order(
                        symbol=signal.symbol,
                        qty=position.quantity,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    self.logger.info(f"   ✅ Order submitted: {order.id}")
                except Exception as e:
                    self.logger.error(f"   ❌ Alpaca order failed: {e}")

            # Close position
            trade_record = self.portfolio_manager.close_position(
                symbol=signal.symbol,
                exit_price=signal.price,
                strategy=signal.strategy
            )

            if trade_record:
                self.logger.info(f"   ✅ Position closed: P&L ${trade_record.pnl:.2f}")
                return True

        except Exception as e:
            self.logger.error(f"❌ Error executing sell for {signal.symbol}: {e}")

        return False

    def update_portfolio_prices(self, symbols: List[str]):
        """Update current prices for all positions"""
        current_prices = {}

        for symbol in self.portfolio_manager.positions.keys():
            try:
                price = self.data_fetcher.get_current_price(symbol)
                current_prices[symbol] = price
            except Exception as e:
                self.logger.warning(f"⚠️  Could not get price for {symbol}: {e}")

        if current_prices:
            self.portfolio_manager.update_position_prices(current_prices)

    def check_position_exits(self):
        """Check if any positions should be exited"""
        for symbol, position in list(self.portfolio_manager.positions.items()):
            try:
                should_exit = self.risk_manager.check_position_exit(
                    symbol, position.current_price, position.__dict__
                )

                if should_exit:
                    self.logger.info(f"\n⚠️  Exit signal for {symbol}")
                    # Create exit signal
                    exit_signal = TradeSignal(
                        symbol=symbol,
                        action='SELL',
                        confidence=1.0,
                        price=position.current_price,
                        quantity=position.quantity,
                        strategy='risk_exit'
                    )
                    self.execute_sell(exit_signal)

            except Exception as e:
                self.logger.error(f"❌ Error checking exit for {symbol}: {e}")

    def display_iteration_summary(self, results: Dict):
        """Display summary of iteration results"""
        portfolio_value = self.portfolio_manager.get_portfolio_value()
        positions_summary = self.portfolio_manager.get_positions_summary()

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"📊 ITERATION SUMMARY")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Symbols Analyzed: {results['symbols_analyzed']}")
        self.logger.info(f"Signals Generated: {results['signals_generated']}")
        self.logger.info(f"Signals Executed: {results['signals_executed']}")
        self.logger.info(f"\n💼 PORTFOLIO STATUS")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Total Value: ${portfolio_value:,.2f}")
        self.logger.info(f"Cash: ${self.portfolio_manager.cash:,.2f}")
        self.logger.info(f"Return: {((portfolio_value - self.initial_capital) / self.initial_capital * 100):.2f}%")

        if positions_summary:
            self.logger.info(f"\n🎯 POSITIONS ({positions_summary['total_positions']})")
            self.logger.info(f"{'='*70}")
            for pos_detail in positions_summary['positions_detail']:
                pnl_str = f"+${pos_detail['unrealized_pnl']:.2f}" if pos_detail['unrealized_pnl'] >= 0 else f"-${abs(pos_detail['unrealized_pnl']):.2f}"
                self.logger.info(f"{pos_detail['symbol']:6} | {pos_detail['quantity']:4} shares | {pnl_str:12} ({pos_detail['unrealized_pnl_pct']:+.1f}%)")

        self.logger.info(f"{'='*70}\n")

    def stop_trading(self):
        """Stop the trading engine"""
        self.is_running = False
        self.logger.info("\n🛑 Trading engine stopped")

        # Generate final report
        self.generate_final_report()

    def generate_final_report(self):
        """Generate and display final trading report"""
        self.logger.info("\n" + "="*70)
        self.logger.info("📊 FINAL TRADING REPORT")
        self.logger.info("="*70)

        metrics = self.portfolio_manager.calculate_performance_metrics()
        risk_summary = self.risk_manager.get_risk_summary()

        self.logger.info(f"\n💼 PORTFOLIO PERFORMANCE:")
        self.logger.info(f"   Initial Capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"   Final Value: ${metrics.current_portfolio_value:,.2f}")
        self.logger.info(f"   Total Return: {metrics.total_return*100:.2f}%")
        self.logger.info(f"   Total P&L: ${metrics.total_pnl:,.2f}")

        self.logger.info(f"\n📈 TRADING STATISTICS:")
        self.logger.info(f"   Total Trades: {metrics.total_trades}")
        self.logger.info(f"   Win Rate: {metrics.win_rate*100:.1f}%")
        self.logger.info(f"   Average Win: ${metrics.avg_win:.2f}")
        self.logger.info(f"   Average Loss: ${metrics.avg_loss:.2f}")
        self.logger.info(f"   Profit Factor: {metrics.profit_factor:.2f}")

        self.logger.info(f"\n⚠️  RISK METRICS:")
        self.logger.info(f"   Sharpe Ratio: {risk_summary['sharpe_ratio']:.2f}")
        self.logger.info(f"   Max Drawdown: {risk_summary['max_drawdown']*100:.2f}%")

        # Export report
        self.portfolio_manager.export_performance_report()

        self.logger.info("\n" + "="*70)

    def get_status(self) -> Dict:
        """Get current engine status"""
        portfolio_value = self.portfolio_manager.get_portfolio_value()

        return {
            'is_running': self.is_running,
            'portfolio_value': portfolio_value,
            'cash': self.portfolio_manager.cash,
            'positions_count': len(self.portfolio_manager.positions),
            'total_trades': len(self.portfolio_manager.trade_history),
            'total_return': (portfolio_value - self.initial_capital) / self.initial_capital
        }
