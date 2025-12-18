"""Configuration and dependency management for Version 5 Trading Algorithm."""

import warnings
import logging
from datetime import datetime, time
import pytz

warnings.filterwarnings('ignore')

# Core dependency checks
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: yfinance import error: {e}")
    print("Installing websockets might fix this: pip install websockets>=11.0")
    YF_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available. Some statistical functions may not work.")
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. ML predictions disabled.")
    SKLEARN_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    print("Warning: TA-Lib not available. Using simplified technical indicators.")
    TALIB_AVAILABLE = False

try:
    from alpaca_trade_api import REST, Stream
    from alpaca_trade_api.common import URL
    ALPACA_AVAILABLE = True
except ImportError:
    print("Warning: Alpaca Trade API not installed. Run: pip install alpaca-trade-api")
    ALPACA_AVAILABLE = False

# NEW V5: Deep Learning dependencies
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Deep learning features disabled.")
    print("Install: pip install torch")
    PYTORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: Transformers library not available. Transformer models disabled.")
    print("Install: pip install transformers")
    TRANSFORMERS_AVAILABLE = False

# NEW V5: Sentiment analysis dependencies
try:
    import nltk
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    print("Warning: Sentiment analysis libraries not available.")
    print("Install: pip install nltk textblob")
    SENTIMENT_AVAILABLE = False

try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    print("Warning: Twitter API (tweepy) not available. Social sentiment disabled.")
    print("Install: pip install tweepy")
    TWITTER_AVAILABLE = False

try:
    from newsapi import NewsApiClient
    NEWS_API_AVAILABLE = True
except ImportError:
    print("Warning: News API client not available. News sentiment disabled.")
    print("Install: pip install newsapi-python")
    NEWS_API_AVAILABLE = False

# Setup logging
def setup_logging():
    """Setup logging configuration."""
    import os
    from datetime import datetime

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/trading_bot_v5_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

# Market hours utilities
def is_market_open():
    """Check if US stock market is currently open"""
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)

        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if within trading hours (9:30 AM - 4:00 PM ET)
        market_open = time(9, 30)
        market_close = time(16, 0)
        current_time = now.time()

        return market_open <= current_time <= market_close
    except Exception:
        # If timezone not available, assume market is open to allow trading
        return True

def get_time_until_market_open():
    """Get seconds until market opens. Returns 0 if market is open."""
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)

        if is_market_open():
            return 0

        # Calculate next market open
        days_ahead = 0
        if now.weekday() == 5:  # Saturday
            days_ahead = 2
        elif now.weekday() == 6:  # Sunday
            days_ahead = 1
        elif now.time() >= time(16, 0):  # After close
            days_ahead = 1

        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if days_ahead > 0:
            from datetime import timedelta
            next_open += timedelta(days=days_ahead)

        seconds_until_open = (next_open - now).total_seconds()
        return max(0, seconds_until_open)
    except Exception:
        # If timezone not available, return 0 to allow immediate trading
        return 0

def get_market_status_message():
    """Get a formatted message about market status"""
    if is_market_open():
        return "🟢 Market is OPEN - Trading active"
    else:
        seconds = get_time_until_market_open()
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)

        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)

        if now.weekday() >= 5:
            return f"🔴 Market CLOSED (Weekend) - Opens in {hours}h {minutes}m"
        elif now.time() < time(9, 30):
            return f"🔴 Market CLOSED (Pre-market) - Opens in {hours}h {minutes}m"
        else:
            return f"🔴 Market CLOSED (After-hours) - Opens in {hours}h {minutes}m"

# Trading configurations
class TradingConfig:
    """Trading configuration constants."""

    # Risk management
    MAX_POSITION_SIZE = 0.1  # Max 10% per position
    MAX_DAILY_LOSS = 0.02    # Max 2% daily loss
    MAX_DRAWDOWN = 0.15      # Max 15% drawdown

    # Technical indicators
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_WINDOW = 14

    # Moving averages
    SMA_SHORT = 10
    SMA_MEDIUM = 20
    SMA_LONG = 50
    SMA_VERY_LONG = 200

    EMA_FAST = 12
    EMA_SLOW = 26
    EMA_SIGNAL = 9

    # Bollinger Bands
    BB_WINDOW = 20
    BB_STD_DEV = 2

    # ATR
    ATR_WINDOW = 14

    # ML parameters
    ML_MIN_SAMPLES = 100
    ML_CONFIDENCE_THRESHOLD = 0.6

    # NEW V5: Deep Learning parameters
    DL_SEQUENCE_LENGTH = 60  # 60 timesteps for LSTM
    DL_BATCH_SIZE = 32
    DL_EPOCHS = 50
    DL_LEARNING_RATE = 0.001

    # NEW V5: Multi-timeframe settings
    TIMEFRAMES = ['1m', '5m', '15m', '1h', '1d']  # Multiple timeframes
    PRIMARY_TIMEFRAME = '5m'  # Main trading timeframe

    # NEW V5: Options trading parameters
    OPTIONS_DELTA_MIN = 0.3  # Minimum delta for options
    OPTIONS_DELTA_MAX = 0.7  # Maximum delta for options
    OPTIONS_DTE_MIN = 7      # Minimum days to expiration
    OPTIONS_DTE_MAX = 60     # Maximum days to expiration

    # NEW V5: Sentiment analysis settings
    SENTIMENT_WEIGHT = 0.2   # Weight of sentiment in final decision
    NEWS_LOOKBACK_HOURS = 24  # Look back 24 hours for news
    SENTIMENT_THRESHOLD = 0.1  # Minimum sentiment score to act

    # Trading
    DEFAULT_CHECK_INTERVAL = 300  # seconds
    MIN_CONFIDENCE = 0.6  # Require 60% confidence for trades
    MAX_EXECUTIONS_PER_ITERATION = 2  # Max 2 trades per iteration to prevent overtrading

    # Portfolio
    DEFAULT_INITIAL_CAPITAL = 100000

# Watchlists
WATCHLISTS = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
    'blue_chips': ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO'],
    'growth': ['TSLA', 'NVDA', 'AMD', 'CRM', 'NFLX'],
    'crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD'],
    'default': ['AAPL', 'MSFT', 'GOOGL']
}

# NEW V5: Options strategies templates
OPTIONS_STRATEGIES = {
    'long_call': {
        'description': 'Bullish strategy - buy call option',
        'legs': ['buy_call']
    },
    'long_put': {
        'description': 'Bearish strategy - buy put option',
        'legs': ['buy_put']
    },
    'bull_call_spread': {
        'description': 'Moderately bullish - buy call, sell higher call',
        'legs': ['buy_call', 'sell_call']
    },
    'bear_put_spread': {
        'description': 'Moderately bearish - buy put, sell lower put',
        'legs': ['buy_put', 'sell_put']
    },
    'iron_condor': {
        'description': 'Neutral strategy - profit from low volatility',
        'legs': ['buy_put', 'sell_put', 'sell_call', 'buy_call']
    }
}

def install_dependencies():
    """Helper function to install missing dependencies."""
    import subprocess
    import sys

    packages_to_install = []

    if not YF_AVAILABLE:
        packages_to_install.extend(['yfinance', 'websockets>=11.0'])
    if not SKLEARN_AVAILABLE:
        packages_to_install.append('scikit-learn')
    if not SCIPY_AVAILABLE:
        packages_to_install.append('scipy')
    if not TALIB_AVAILABLE:
        print("Note: TA-Lib installation requires additional steps:")
        print("Windows: pip install TA-Lib")
        print("macOS: brew install ta-lib && pip install TA-Lib")
        print("Linux: sudo apt-get install libta-lib-dev && pip install TA-Lib")
    if not ALPACA_AVAILABLE:
        packages_to_install.append('alpaca-trade-api')

    # NEW V5 dependencies
    if not PYTORCH_AVAILABLE:
        packages_to_install.append('torch')
    if not TRANSFORMERS_AVAILABLE:
        packages_to_install.append('transformers')
    if not SENTIMENT_AVAILABLE:
        packages_to_install.extend(['nltk', 'textblob'])
    if not TWITTER_AVAILABLE:
        packages_to_install.append('tweepy')
    if not NEWS_API_AVAILABLE:
        packages_to_install.append('newsapi-python')

    if packages_to_install:
        print(f"Installing missing packages: {', '.join(packages_to_install)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages_to_install)
            print("✅ Dependencies installed successfully!")
            print("Please restart the script to use the newly installed packages.")
        except subprocess.CalledProcessError:
            print("❌ Failed to install some packages. Please install manually:")
            for pkg in packages_to_install:
                print(f"   pip install {pkg}")
    else:
        print("✅ All dependencies are available!")

def print_dependency_status():
    """Print the status of all dependencies."""
    print("Core Dependencies:")
    print(f"   📊 yfinance: {'✅' if YF_AVAILABLE else '❌'}")
    print(f"   🤖 scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
    print(f"   📈 TA-Lib: {'✅' if TALIB_AVAILABLE else '❌'}")
    print(f"   📉 SciPy: {'✅' if SCIPY_AVAILABLE else '❌'}")
    print(f"   🏦 Alpaca API: {'✅' if ALPACA_AVAILABLE else '❌'}")
    print("\nV5 New Features:")
    print(f"   🧠 PyTorch: {'✅' if PYTORCH_AVAILABLE else '❌'}")
    print(f"   🤖 Transformers: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
    print(f"   💬 Sentiment Analysis: {'✅' if SENTIMENT_AVAILABLE else '❌'}")
    print(f"   🐦 Twitter API: {'✅' if TWITTER_AVAILABLE else '❌'}")
    print(f"   📰 News API: {'✅' if NEWS_API_AVAILABLE else '❌'}")
