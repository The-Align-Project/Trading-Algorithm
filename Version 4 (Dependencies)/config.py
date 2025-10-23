"""Configuration and dependency management for the trading algorithm."""

import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Dependency checks
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

# Setup logging
def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log'),
            logging.StreamHandler()
        ]
    )

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
    
    # Trading
    DEFAULT_CHECK_INTERVAL = 300  # seconds
    MIN_CONFIDENCE = 0.6
    MAX_EXECUTIONS_PER_ITERATION = 2
    
    # Portfolio
    DEFAULT_INITIAL_CAPITAL = 100000

# Watchlists
WATCHLISTS = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
    'blue_chips': ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO'],
    'growth': ['TSLA', 'NVDA', 'AMD', 'CRM', 'NFLX'],
    'default': ['AAPL', 'MSFT', 'GOOGL']
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
    print("Available Dependencies:")
    print(f"   📊 yfinance: {'✅' if YF_AVAILABLE else '❌'}")
    print(f"   🤖 scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
    print(f"   📈 TA-Lib: {'✅' if TALIB_AVAILABLE else '❌'}")
    print(f"   📉 SciPy: {'✅' if SCIPY_AVAILABLE else '❌'}")
    print(f"   🏦 Alpaca API: {'✅' if ALPACA_AVAILABLE else '❌'}")