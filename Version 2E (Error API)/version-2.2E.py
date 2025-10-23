import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Alpaca imports
try:
    from alpaca_trade_api import REST, Stream
    from alpaca_trade_api.common import URL
    ALPACA_AVAILABLE = True
except ImportError:
    print("Alpaca Trade API not installed. Run: pip install alpaca-trade-api")
    ALPACA_AVAILABLE = False

class AlpacaTradingAlgorithm:
    def __init__(self, api_key=None, secret_key=None, paper=True, initial_capital=10000, risk_per_trade=0.02):
        """
        Initialize the Alpaca trading algorithm
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading (True) or live trading (False)
            initial_capital: Starting capital amount
            risk_per_trade: Risk percentage per trade (default 2%)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        
        # Trading state
        self.positions = {}
        self.trade_history = []
        self.portfolio_value = []
        self.watchlist = []
        self.active_orders = {}
        
        # Initialize Alpaca API
        if ALPACA_AVAILABLE and api_key and secret_key:
            try:
                base_url = URL('https://paper-api.alpaca.markets') if paper else URL('https://api.alpaca.markets')
                self.api = REST(api_key, secret_key, base_url, api_version='v2')
                self.stream = Stream(api_key, secret_key, base_url=base_url, data_feed='iex')
                
                # Test the connection
                account = self.api.get_account()
                self.alpaca_connected = True
                print(f"✅ Connected to Alpaca {'Paper' if paper else 'Live'} Trading")
                print(f"   Account Status: {account.status}")
                print(f"   Buying Power: ${float(account.buying_power):.2f}")
                
            except Exception as e:
                print(f"❌ Failed to connect to Alpaca: {e}")
                print("🔄 Running in simulation mode")
                self.api = None
                self.stream = None
                self.alpaca_connected = False
        else:
            self.api = None
            self.stream = None
            self.alpaca_connected = False
            print("🎮 Running in pure simulation mode")
    
    def get_account_info(self):
        """Get account information from Alpaca"""
        if not self.alpaca_connected:
            return None
        
        try:
            account = self.api.get_account()
            account_info = {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
            }
            
            # These attributes may not exist in all account types
            if hasattr(account, 'day_trade_count'):
                account_info['day_trade_count'] = int(account.day_trade_count)
            if hasattr(account, 'pattern_day_trader'):
                account_info['pattern_day_trader'] = account.pattern_day_trader
                
            return account_info
        except Exception as e:
            print(f"Error getting account info: {e}")
            return None
    
    def get_positions(self):
        """Get current positions from Alpaca"""
        if not self.alpaca_connected:
            return {}
        
        try:
            positions = self.api.list_positions()
            position_dict = {}
            
            for position in positions:
                position_dict[position.symbol] = {
                    'shares': float(position.qty),
                    'market_value': float(position.market_value),
                    'avg_entry_price': float(position.avg_entry_price),
                    'unrealized_pl': float(position.unrealized_pl),
                    'side': position.side
                }
            
            return position_dict
        except Exception as e:
            print(f"Error getting positions: {e}")
            return {}
    
    def get_real_time_quote(self, symbol):
        """Get real-time quote from Alpaca"""
        if not self.alpaca_connected:
            # Fall back to yfinance for simulation
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return {
                    'bid': data['Close'].iloc[-1],
                    'ask': data['Close'].iloc[-1],
                    'last': data['Close'].iloc[-1]
                }
            return None
        
        try:
            quote = self.api.get_latest_quote(symbol)
            return {
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'last': (float(quote.bid_price) + float(quote.ask_price)) / 2
            }
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            return None
    
    def fetch_data(self, symbol, period="1y", interval="1d"):
        """Fetch historical data"""
        try:
            if self.alpaca_connected:
                # Use Alpaca for historical data with proper date formatting
                end = datetime.now()
                if period == "1y":
                    start = end - timedelta(days=365)
                elif period == "3mo":
                    start = end - timedelta(days=90)
                else:
                    start = end - timedelta(days=30)
                
                # Format dates properly for Alpaca API (RFC3339)
                start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
                end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                timeframe = '1Day' if interval == '1d' else '1Hour'
                
                bars = self.api.get_bars(
                    symbol,
                    timeframe=timeframe,
                    start=start_str,
                    end=end_str,
                    adjustment='raw'
                ).df
                
                if bars.empty:
                    print(f"No data returned from Alpaca for {symbol}, falling back to yfinance")
                    raise Exception("Empty data from Alpaca")
                
                # Rename columns to match yfinance format
                bars = bars.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                return bars
            else:
                # Fall back to yfinance
                ticker = yf.Ticker(symbol)
                return ticker.history(period=period, interval=interval)
                
        except Exception as e:
            print(f"Error fetching data from Alpaca for {symbol}: {e}")
            print(f"Falling back to yfinance for {symbol}")
            try:
                # Fall back to yfinance
                ticker = yf.Ticker(symbol)
                return ticker.history(period=period, interval=interval)
            except Exception as yf_error:
                print(f"Error fetching data from yfinance for {symbol}: {yf_error}")
                return None
    
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # ATR for volatility
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        return df
    
    def momentum_strategy_signal(self, data):
        """Generate momentum strategy signal for current data"""
        if len(data) < 50:
            return 'HOLD'
        
        current = data.iloc[-1]
        
        # Buy conditions
        buy_conditions = [
            current['Close'] > current['SMA_20'],
            current['SMA_20'] > current['SMA_50'],
            current['RSI'] > 50 and current['RSI'] < 70,
            current['MACD'] > current['MACD_Signal'],
            current['Volume_Ratio'] > 1.2
        ]
        
        # Sell conditions
        sell_conditions = [
            current['Close'] < current['SMA_20'],
            current['RSI'] > 70,
            current['MACD'] < current['MACD_Signal'],
            current['Close'] < current['BB_Lower']
        ]
        
        if sum(buy_conditions) >= 3:
            return 'BUY'
        elif sum(sell_conditions) >= 2:
            return 'SELL'
        
        return 'HOLD'
    
    def calculate_position_size(self, symbol, price, atr):
        """Calculate position size based on risk management"""
        if self.alpaca_connected:
            account_info = self.get_account_info()
            available_capital = account_info['buying_power'] if account_info else self.current_capital
        else:
            available_capital = self.current_capital
        
        risk_amount = available_capital * self.risk_per_trade
        stop_loss_distance = atr * 2
        
        if stop_loss_distance == 0:
            return 0
        
        position_size = risk_amount / stop_loss_distance
        
        # Don't use more than 10% of available capital
        max_position_value = available_capital * 0.1
        max_shares = max_position_value / price
        
        return min(int(position_size), int(max_shares))
    
    def place_market_order(self, symbol, qty, side, time_in_force='day'):
        """Place market order with Alpaca"""
        if not self.alpaca_connected:
            # Simulate the trade
            current_price = self.get_current_price_simulation(symbol)
            if current_price:
                self.simulate_trade(symbol, qty, side, current_price)
            return None
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force=time_in_force
            )
            
            print(f"✅ Order submitted: {side} {qty} shares of {symbol}")
            return order
        except Exception as e:
            print(f"❌ Error placing order: {e}")
            return None
    
    def place_bracket_order(self, symbol, qty, side, limit_price=None, stop_loss=None, take_profit=None):
        """Place bracket order (entry + stop loss + take profit)"""
        if not self.alpaca_connected:
            # Simulate the bracket order
            current_price = self.get_current_price_simulation(symbol)
            if current_price:
                self.simulate_trade(symbol, qty, side, current_price, stop_loss, take_profit)
            return None
        
        try:
            order_class = 'bracket' if stop_loss and take_profit else 'simple'
            
            order_data = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': 'market' if not limit_price else 'limit',
                'time_in_force': 'day',
                'order_class': order_class
            }
            
            if limit_price:
                order_data['limit_price'] = limit_price
            
            if order_class == 'bracket':
                order_data['stop_loss'] = {'stop_price': stop_loss}
                order_data['take_profit'] = {'limit_price': take_profit}
            
            order = self.api.submit_order(**order_data)
            print(f"✅ Bracket order submitted: {side} {qty} shares of {symbol}")
            return order
        except Exception as e:
            print(f"❌ Error placing bracket order: {e}")
            return None
    
    def get_current_price_simulation(self, symbol):
        """Get current price for simulation"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return data['Close'].iloc[-1]
        except:
            pass
        return None
    
    def simulate_trade(self, symbol, qty, side, price, stop_loss=None, take_profit=None):
        """Simulate trade execution"""
        if side == 'buy':
            cost = qty * price
            if cost <= self.current_capital:
                self.positions[symbol] = {
                    'shares': qty,
                    'avg_entry_price': price,
                    'market_value': qty * price,
                    'unrealized_pl': 0,
                    'side': 'long',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                self.current_capital -= cost
                print(f"🎮 SIMULATED BUY: {qty} shares of {symbol} @ ${price:.2f}")
                if stop_loss:
                    print(f"   Stop Loss: ${stop_loss:.2f}")
                if take_profit:
                    print(f"   Take Profit: ${take_profit:.2f}")
            else:
                print(f"❌ Insufficient capital for {symbol}")
        
        elif side == 'sell' and symbol in self.positions:
            position = self.positions[symbol]
            proceeds = qty * price
            cost_basis = position['shares'] * position['avg_entry_price']
            profit_loss = proceeds - cost_basis
            
            self.current_capital += proceeds
            print(f"🎮 SIMULATED SELL: {qty} shares of {symbol} @ ${price:.2f}")
            print(f"   P&L: ${profit_loss:.2f} ({profit_loss/cost_basis*100:.1f}%)")
            
            # Remove position
            del self.positions[symbol]
    
    def cancel_all_orders(self, symbol=None):
        """Cancel all open orders"""
        if not self.alpaca_connected:
            return
        
        try:
            orders = self.api.list_orders(status='open')
            for order in orders:
                if symbol is None or order.symbol == symbol:
                    self.api.cancel_order(order.id)
                    print(f"Cancelled order: {order.symbol}")
        except Exception as e:
            print(f"Error cancelling orders: {e}")
    
    def get_open_orders(self):
        """Get all open orders"""
        if not self.alpaca_connected:
            return []
        
        try:
            return self.api.list_orders(status='open')
        except Exception as e:
            print(f"Error getting open orders: {e}")
            return []
    
    def run_strategy(self, symbols, strategy='momentum', check_interval=60):
        """Run the trading strategy continuously"""
        print(f"Starting live trading strategy: {strategy}")
        print(f"Watching symbols: {symbols}")
        print(f"Check interval: {check_interval} seconds")
        
        if self.alpaca_connected:
            # Check market hours
            try:
                clock = self.api.get_clock()
                if not clock.is_open:
                    print("Market is closed. Running in simulation mode...")
                    print(f"Market opens at: {clock.next_open}")
            except Exception as e:
                print(f"Could not check market status: {e}")
                print("Continuing with strategy...")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n=== Strategy Check #{iteration} at {current_time} ===")
                
                # Get account info
                account_info = self.get_account_info()
                if account_info:
                    print(f"Portfolio Value: ${account_info['portfolio_value']:.2f}")
                    print(f"Buying Power: ${account_info['buying_power']:.2f}")
                else:
                    print("Running in simulation mode - no account data")
                
                # Check each symbol
                for symbol in symbols:
                    print(f"\nAnalyzing {symbol}...")
                    
                    # Get historical data
                    data = self.fetch_data(symbol, period="3mo", interval="1d")
                    if data is None or len(data) < 50:
                        print(f"Insufficient data for {symbol} (got {len(data) if data is not None else 0} rows)")
                        continue
                    
                    # Calculate indicators
                    try:
                        data = self.calculate_indicators(data)
                        
                        # Check if we have valid indicators
                        if data['ATR'].iloc[-1] == 0 or pd.isna(data['ATR'].iloc[-1]):
                            print(f"Invalid ATR for {symbol}, skipping...")
                            continue
                        
                    except Exception as e:
                        print(f"Error calculating indicators for {symbol}: {e}")
                        continue
                    
                    # Generate signal
                    signal = self.momentum_strategy_signal(data)
                    current_price = data['Close'].iloc[-1]
                    atr = data['ATR'].iloc[-1]
                    rsi = data['RSI'].iloc[-1]
                    
                    print(f"Signal: {signal}, Price: ${current_price:.2f}, ATR: {atr:.2f}, RSI: {rsi:.1f}")
                    
                    # Get current position
                    positions = self.get_positions()
                    current_position = positions.get(symbol)
                    
                    if current_position:
                        print(f"Current position: {current_position['shares']} shares, P&L: ${current_position['unrealized_pl']:.2f}")
                    
                    # Execute trades based on signal
                    if signal == 'BUY' and not current_position:
                        qty = self.calculate_position_size(symbol, current_price, atr)
                        if qty > 0:
                            stop_loss = current_price - (atr * 2)
                            take_profit = current_price + (atr * 3)
                            
                            print(f"📊 BUY Signal: {qty} shares, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
                            
                            order = self.place_bracket_order(
                                symbol=symbol,
                                qty=qty,
                                side='buy',
                                stop_loss=stop_loss,
                                take_profit=take_profit
                            )
                            
                            if order or not self.alpaca_connected:  # Include simulation mode
                                self.active_orders[symbol] = order.id if (order and hasattr(order, 'id')) else 'simulated'
                        else:
                            print(f"⚠️ Position size too small for {symbol}")
                    
                    elif signal == 'SELL' and current_position and current_position['side'] == 'long':
                        # Close position
                        qty = abs(int(current_position['shares']))
                        if qty > 0:
                            print(f"📊 SELL Signal: {qty} shares")
                            order = self.place_market_order(symbol, qty, 'sell')
                            if (order or not self.alpaca_connected) and symbol in self.active_orders:
                                del self.active_orders[symbol]
                    
                    elif signal == 'HOLD':
                        print(f"⏸️ Holding position for {symbol}")
                
                # Update simulation portfolio
                if not self.alpaca_connected:
                    self.update_simulation_portfolio()
                
                print(f"\nActive orders: {len(self.active_orders)}")
                if not self.alpaca_connected:
                    print(f"💰 Current Capital: ${self.current_capital:.2f}")
                    print(f"📈 Portfolio Value: ${self.get_simulation_portfolio_value():.2f}")
                
                print(f"Next check in {check_interval} seconds...")
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\nStopping strategy...")
                break
            except Exception as e:
                print(f"Error in strategy loop: {e}")
                print("Continuing in 30 seconds...")
                time.sleep(30)
    
    def backtest_strategy(self, symbol, strategy='momentum', period="1y"):
        """Backtest the trading strategy (same as before)"""
        print(f"Backtesting {strategy} strategy for {symbol}...")
        
        data = self.fetch_data(symbol, period)
        if data is None:
            return None
        
        data = self.calculate_indicators(data)
        
        # Generate signals for backtesting
        signals = []
        for i in range(len(data)):
            if i < 50:
                signals.append('HOLD')
            else:
                current_data = data.iloc[:i+1]
                signal = self.momentum_strategy_signal(current_data)
                signals.append(signal)
        
        data['Signal'] = signals
        return data
    
    def update_simulation_portfolio(self):
        """Update portfolio values for simulation"""
        for symbol in list(self.positions.keys()):
            current_price = self.get_current_price_simulation(symbol)
            if current_price:
                position = self.positions[symbol]
                position['market_value'] = position['shares'] * current_price
                position['unrealized_pl'] = position['market_value'] - (position['shares'] * position['avg_entry_price'])
                
                # Check stop loss and take profit
                if position.get('stop_loss') and current_price <= position['stop_loss']:
                    print(f"🛑 Stop Loss triggered for {symbol} at ${current_price:.2f}")
                    self.simulate_trade(symbol, position['shares'], 'sell', current_price)
                elif position.get('take_profit') and current_price >= position['take_profit']:
                    print(f"🎯 Take Profit triggered for {symbol} at ${current_price:.2f}")
                    self.simulate_trade(symbol, position['shares'], 'sell', current_price)
    
    def get_simulation_portfolio_value(self):
        """Calculate total portfolio value for simulation"""
        total_value = self.current_capital
        for symbol, position in self.positions.items():
            total_value += position['market_value']
        return total_value


# Configuration and usage example
def main():
    print("=== Alpaca Trading Algorithm Setup ===")
    print("1. Use your Alpaca API credentials")
    print("2. Run in pure simulation mode (no Alpaca needed)")
    
    mode = input("Choose mode (1 or 2): ").strip()
    
    if mode == "1":
        # Get credentials from user
        api_key = input("Enter your Alpaca API Key: ").strip()
        secret_key = input("Enter your Alpaca Secret Key: ").strip()
        
        if not api_key or not secret_key or api_key == "YOUR_API_KEY":
            print("Invalid credentials. Switching to simulation mode.")
            api_key = None
            secret_key = None
    else:
        print("Running in pure simulation mode")
        api_key = None
        secret_key = None
    
    # Initialize trading algorithm
    trader = AlpacaTradingAlgorithm(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,  # Set to False for live trading
        initial_capital=10000,
        risk_per_trade=0.02
    )
    
    # Print account info
    trader.print_summary()
    
    # Symbols to trade
    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    print("\n" + "="*50)
    print("Choose an option:")
    print("1. Run live trading strategy")
    print("2. Run backtest only")
    print("3. Check account status")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        # Run live strategy
        try:
            trader.run_strategy(
                symbols=watchlist,
                strategy='momentum',
                check_interval=300  # Check every 5 minutes
            )
        except KeyboardInterrupt:
            print("Strategy stopped by user")
        finally:
            trader.print_summary()
    
    elif choice == '2':
        # Run backtest
        for symbol in watchlist[:2]:  # Test first 2 symbols
            data = trader.backtest_strategy(symbol, strategy='momentum')
            if data is not None:
                buy_signals = len(data[data['Signal'] == 'BUY'])
                sell_signals = len(data[data['Signal'] == 'SELL'])
                print(f"{symbol}: {buy_signals} BUY, {sell_signals} SELL signals")
    
    elif choice == '3':
        # Check account status
        trader.print_summary()
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    if not ALPACA_AVAILABLE:
        print("To use Alpaca integration, install: pip install alpaca-trade-api")
    else:
        main()