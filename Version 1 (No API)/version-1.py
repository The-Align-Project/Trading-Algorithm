import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingAlgorithm:
    def __init__(self, initial_capital=10000, risk_per_trade=0.02):
        """
        Initialize the trading algorithm
        
        Args:
            initial_capital: Starting capital amount
            risk_per_trade: Risk percentage per trade (default 2%)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.positions = {}
        self.trade_history = []
        self.portfolio_value = []
        
    def fetch_data(self, symbol, period="1y", interval="1d"):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
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
    
    def momentum_strategy(self, data):
        """Momentum-based trading strategy"""
        signals = []
        
        for i in range(len(data)):
            signal = 'HOLD'
            
            if i < 50:  # Need enough data for indicators
                signals.append(signal)
                continue
            
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Buy conditions
            buy_conditions = [
                current['Close'] > current['SMA_20'],  # Price above 20 SMA
                current['SMA_20'] > current['SMA_50'],  # 20 SMA above 50 SMA
                current['RSI'] > 50 and current['RSI'] < 70,  # RSI in momentum zone
                current['MACD'] > current['MACD_Signal'],  # MACD above signal
                current['Volume_Ratio'] > 1.2  # Above average volume
            ]
            
            # Sell conditions
            sell_conditions = [
                current['Close'] < current['SMA_20'],  # Price below 20 SMA
                current['RSI'] > 70,  # Overbought
                current['MACD'] < current['MACD_Signal'],  # MACD below signal
                current['Close'] < current['BB_Lower']  # Below lower Bollinger Band
            ]
            
            if sum(buy_conditions) >= 3:
                signal = 'BUY'
            elif sum(sell_conditions) >= 2:
                signal = 'SELL'
            
            signals.append(signal)
        
        return signals
    
    def mean_reversion_strategy(self, data):
        """Mean reversion trading strategy"""
        signals = []
        
        for i in range(len(data)):
            signal = 'HOLD'
            
            if i < 50:
                signals.append(signal)
                continue
            
            current = data.iloc[i]
            
            # Buy conditions (oversold)
            buy_conditions = [
                current['RSI'] < 30,  # Oversold
                current['Close'] < current['BB_Lower'],  # Below lower Bollinger Band
                current['Close'] < current['SMA_20'] * 0.95,  # Significantly below SMA
            ]
            
            # Sell conditions (overbought)
            sell_conditions = [
                current['RSI'] > 70,  # Overbought
                current['Close'] > current['BB_Upper'],  # Above upper Bollinger Band
                current['Close'] > current['SMA_20'] * 1.05,  # Significantly above SMA
            ]
            
            if sum(buy_conditions) >= 2:
                signal = 'BUY'
            elif sum(sell_conditions) >= 2:
                signal = 'SELL'
            
            signals.append(signal)
        
        return signals
    
    def calculate_position_size(self, price, atr):
        """Calculate position size based on risk management"""
        risk_amount = self.current_capital * self.risk_per_trade
        stop_loss_distance = atr * 2  # 2x ATR for stop loss
        position_size = risk_amount / stop_loss_distance
        
        # Don't use more than 10% of capital for single position
        max_position_value = self.current_capital * 0.1
        max_shares = max_position_value / price
        
        return min(position_size, max_shares)
    
    def execute_trade(self, symbol, signal, price, date, atr):
        """Execute buy/sell orders"""
        if signal == 'BUY' and symbol not in self.positions:
            shares = self.calculate_position_size(price, atr)
            cost = shares * price
            
            if cost <= self.current_capital:
                self.positions[symbol] = {
                    'shares': shares,
                    'entry_price': price,
                    'entry_date': date,
                    'stop_loss': price - (atr * 2),
                    'take_profit': price + (atr * 3)
                }
                
                self.current_capital -= cost
                
                trade = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'date': date,
                    'capital_after': self.current_capital
                }
                self.trade_history.append(trade)
                
        elif signal == 'SELL' and symbol in self.positions:
            position = self.positions[symbol]
            proceeds = position['shares'] * price
            profit_loss = proceeds - (position['shares'] * position['entry_price'])
            
            self.current_capital += proceeds
            
            trade = {
                'symbol': symbol,
                'action': 'SELL',
                'shares': position['shares'],
                'price': price,
                'date': date,
                'profit_loss': profit_loss,
                'capital_after': self.current_capital
            }
            self.trade_history.append(trade)
            
            del self.positions[symbol]
    
    def check_stop_loss_take_profit(self, symbol, current_price, date):
        """Check and execute stop loss or take profit orders"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        if current_price <= position['stop_loss']:
            self.execute_trade(symbol, 'SELL', current_price, date, 0)
            print(f"Stop loss triggered for {symbol} at {current_price}")
            
        elif current_price >= position['take_profit']:
            self.execute_trade(symbol, 'SELL', current_price, date, 0)
            print(f"Take profit triggered for {symbol} at {current_price}")
    
    def backtest_strategy(self, symbol, strategy='momentum', period="1y"):
        """Backtest the trading strategy"""
        print(f"Backtesting {strategy} strategy for {symbol}...")
        
        # Fetch data
        data = self.fetch_data(symbol, period)
        if data is None:
            return None
        
        # Calculate indicators
        data = self.calculate_indicators(data)
        
        # Generate signals
        if strategy == 'momentum':
            signals = self.momentum_strategy(data)
        elif strategy == 'mean_reversion':
            signals = self.mean_reversion_strategy(data)
        else:
            print("Unknown strategy")
            return None
        
        data['Signal'] = signals
        
        # Execute trades
        for i, (date, row) in enumerate(data.iterrows()):
            if i < 50:  # Skip initial period
                continue
            
            # Check stop loss/take profit for existing positions
            self.check_stop_loss_take_profit(symbol, row['Close'], date)
            
            # Execute new trades based on signals
            if row['Signal'] in ['BUY', 'SELL']:
                self.execute_trade(symbol, row['Signal'], row['Close'], date, row['ATR'])
            
            # Calculate portfolio value
            portfolio_value = self.current_capital
            for pos_symbol, position in self.positions.items():
                portfolio_value += position['shares'] * row['Close']
            
            self.portfolio_value.append({
                'date': date,
                'value': portfolio_value
            })
        
        return data
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.portfolio_value:
            return None
        
        portfolio_df = pd.DataFrame(self.portfolio_value)
        portfolio_df.set_index('date', inplace=True)
        
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['value'].pct_change()
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = portfolio_df['daily_return'].mean() * 252 - risk_free_rate
        volatility = portfolio_df['daily_return'].std() * np.sqrt(252)
        sharpe_ratio = excess_returns / volatility if volatility != 0 else 0
        
        # Maximum drawdown
        portfolio_df['cummax'] = portfolio_df['value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['cummax']) / portfolio_df['cummax']
        max_drawdown = portfolio_df['drawdown'].min() * 100
        
        # Win rate
        profitable_trades = [t for t in self.trade_history if t.get('profit_loss', 0) > 0]
        total_trades = len([t for t in self.trade_history if 'profit_loss' in t])
        win_rate = len(profitable_trades) / total_trades * 100 if total_trades > 0 else 0
        
        metrics = {
            'Initial Capital': self.initial_capital,
            'Final Value': final_value,
            'Total Return (%)': round(total_return, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Total Trades': total_trades
        }
        
        return metrics
    
    def print_trades(self):
        """Print trade history"""
        print("\n=== TRADE HISTORY ===")
        for trade in self.trade_history:
            action = trade['action']
            symbol = trade['symbol']
            shares = round(trade['shares'], 2)
            price = round(trade['price'], 2)
            date = trade['date'].strftime('%Y-%m-%d')
            
            if action == 'SELL' and 'profit_loss' in trade:
                pl = round(trade['profit_loss'], 2)
                print(f"{date}: {action} {shares} shares of {symbol} at ${price} | P&L: ${pl}")
            else:
                print(f"{date}: {action} {shares} shares of {symbol} at ${price}")

# Example usage
def main():
    # Initialize the trading algorithm
    trader = TradingAlgorithm(initial_capital=10000, risk_per_trade=0.02)
    
    # Test with a popular stock
    symbol = "AAPL"
    
    # Backtest momentum strategy
    print("Testing Momentum Strategy...")
    trader.backtest_strategy(symbol, strategy='momentum', period="1y")
    
    # Get performance metrics
    metrics = trader.get_performance_metrics()
    
    if metrics:
        print(f"\n=== PERFORMANCE METRICS ({symbol}) ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    
    # Print trade history
    trader.print_trades()
    
    # Reset for mean reversion test
    trader_mr = TradingAlgorithm(initial_capital=10000, risk_per_trade=0.02)
    
    print(f"\n{'='*50}")
    print("Testing Mean Reversion Strategy...")
    trader_mr.backtest_strategy(symbol, strategy='mean_reversion', period="1y")
    
    metrics_mr = trader_mr.get_performance_metrics()
    
    if metrics_mr:
        print(f"\n=== MEAN REVERSION PERFORMANCE ({symbol}) ===")
        for key, value in metrics_mr.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()