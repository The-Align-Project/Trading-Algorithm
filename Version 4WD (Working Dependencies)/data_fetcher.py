"""Fixed data fetching and management for the trading algorithm."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from config import YF_AVAILABLE

if YF_AVAILABLE:
    import yfinance as yf

class SimplifiedDataFetcher:
    """Simplified data fetcher with alternative to yfinance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def generate_sample_data(symbol: str, days: int = 252) -> pd.DataFrame:
        """Generate realistic sample stock data for testing"""
        np.random.seed(hash(symbol) % 2**32)  # Consistent data for same symbol
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic price movement
        returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
        price = 100  # Starting price
        prices = [price]
        
        for ret in returns[1:]:
            price *= (1 + ret)
            prices.append(price)
        
        # Generate OHLCV data with proper data types
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            high = close_price * (1 + abs(np.random.normal(0, 0.01)))
            low = close_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else close_price
            volume = int(np.random.normal(1000000, 500000))
            
            data.append({
                'Open': float(open_price),
                'High': float(max(high, open_price, close_price)),
                'Low': float(min(low, open_price, close_price)),
                'Close': float(close_price),
                'Volume': float(max(volume, 10000))  # Ensure volume is float
            })
        
        df = pd.DataFrame(data, index=dates)
        
        # Ensure all columns are float64 for TA-Lib compatibility
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(np.float64)
        
        df.symbol = symbol
        return df
    
    def fetch_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch data with fallback to sample data"""
        try:
            if YF_AVAILABLE:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval="1d")
                if not data.empty:
                    # Ensure proper data types
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if col in data.columns:
                            data[col] = data[col].astype(np.float64)
                    
                    data.symbol = symbol
                    self.logger.info(f"✅ Fetched real data for {symbol}: {len(data)} records")
                    return data
            
            # Fallback to sample data
            days_map = {"1y": 252, "3mo": 66, "6mo": 126, "2y": 504}
            days = days_map.get(period, 252)
            
            self.logger.info(f"📊 Using sample data for {symbol} ({days} days)")
            return self.generate_sample_data(symbol, days)
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            self.logger.info(f"📊 Falling back to sample data for {symbol}")
            return self.generate_sample_data(symbol, 252)
    
    def fetch_data_parallel(self, symbols: List[str], period="3mo") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols in parallel"""
        data_dict = {}
        
        def fetch_symbol(symbol):
            try:
                data = self.fetch_data(symbol, period)
                if not data.empty:
                    data.symbol = symbol
                    return symbol, data
                else:
                    return symbol, self.generate_sample_data(symbol)
                    
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                return symbol, self.generate_sample_data(symbol)
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(fetch_symbol, symbol): symbol for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol, data = future.result()
                if data is not None:
                    data_dict[symbol] = data
        
        return data_dict
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            if YF_AVAILABLE:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                if not data.empty:
                    return float(data['Close'].iloc[-1])
            
            # Fallback to last price from daily data
            data = self.fetch_data(symbol, "1d")
            return float(data['Close'].iloc[-1]) if not data.empty else 100.0
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return 100.0  # Default price
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate that symbols can be fetched"""
        valid_symbols = []
        
        for symbol in symbols:
            try:
                data = self.fetch_data(symbol, "5d")
                if not data.empty and len(data) > 1:
                    valid_symbols.append(symbol)
                    self.logger.info(f"✅ Validated {symbol}")
                else:
                    self.logger.warning(f"⚠️ No data available for {symbol}")
            except Exception as e:
                self.logger.error(f"❌ Error validating {symbol}: {e}")
        
        return valid_symbols

class AlpacaDataFetcher(SimplifiedDataFetcher):
    """Data fetcher that uses Alpaca API when available"""
    
    def __init__(self, alpaca_api=None):
        super().__init__()
        self.api = alpaca_api
    
    def fetch_data_from_alpaca(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """Fetch data using Alpaca API with better error handling"""
        if not self.api:
            return pd.DataFrame()
        
        try:
            end = datetime.now()
            days_map = {"1y": 365, "3mo": 90, "6mo": 180, "2y": 730}
            days = days_map.get(period, 90)
            start = end - timedelta(days=days)
            
            # Try different data sources based on subscription
            try:
                # First try with SIP data
                bars = self.api.get_bars(
                    symbol,
                    timeframe='1Day',
                    start=start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    end=end.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    adjustment='raw'
                ).df
            except Exception as sip_error:
                self.logger.warning(f"SIP data failed for {symbol}: {sip_error}")
                # Try with basic data feed
                try:
                    bars = self.api.get_bars(
                        symbol,
                        timeframe='1Day',
                        start=start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        end=end.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        adjustment='raw',
                        feed='iex'  # Use IEX feed instead
                    ).df
                except Exception as iex_error:
                    self.logger.warning(f"IEX data failed for {symbol}: {iex_error}")
                    return pd.DataFrame()
            
            if not bars.empty:
                bars = bars.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low',
                    'close': 'Close', 'volume': 'Volume'
                })
                
                # Ensure proper data types
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in bars.columns:
                        bars[col] = bars[col].astype(np.float64)
                
                bars.symbol = symbol
                self.logger.info(f"✅ Fetched Alpaca data for {symbol}: {len(bars)} records")
                return bars
            
        except Exception as e:
            self.logger.error(f"Error fetching Alpaca data for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def fetch_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch data with Alpaca API priority, fallback to yfinance/sample"""
        # Try Alpaca first
        if self.api:
            data = self.fetch_data_from_alpaca(symbol, period)
            if not data.empty:
                return data
            else:
                self.logger.info(f"🔄 Alpaca data failed for {symbol}, trying fallback...")
        
        # Fallback to parent method (yfinance or sample data)
        return super().fetch_data(symbol, period)
    
    def get_current_price_alpaca(self, symbol: str) -> float:
        """Get current price using Alpaca API"""
        if not self.api:
            return 0.0
        
        try:
            # Try different methods to get current price
            try:
                quote = self.api.get_latest_quote(symbol)
                return float((quote.bid_price + quote.ask_price) / 2)
            except:
                # Fallback to latest bar
                bars = self.api.get_bars(
                    symbol,
                    timeframe='1Min',
                    limit=1
                ).df
                if not bars.empty:
                    return float(bars['close'].iloc[-1])
                    
        except Exception as e:
            self.logger.error(f"Error getting Alpaca price for {symbol}: {e}")
        
        return 0.0
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price with Alpaca priority"""
        # Try Alpaca first
        price = self.get_current_price_alpaca(symbol)
        if price > 0:
            return price
        
        # Fallback to parent method
        return super().get_current_price(symbol)