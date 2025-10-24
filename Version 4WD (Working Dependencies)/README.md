# Ultimate Advanced Trading Algorithm v2.0

A comprehensive, modular algorithmic trading system with machine learning, advanced technical analysis, and robust risk management.

## ✅ Latest Update (October 22, 2025)
**Critical Bug Fixed**: The live trading mode now fully works! The algorithm was missing the signal analysis and execution logic in `run_single_iteration()`. This has been fixed and tested. The algorithm now properly:
- Analyzes all symbols using multiple strategies
- Generates trading signals with confidence scores
- Executes high-confidence trades (≥ 0.6 confidence)
- Limits to 2 executions per iteration to prevent overtrading

## 🚀 Features

- **Multiple Trading Strategies**: Momentum breakout, mean reversion, trend following, and breakout strategies
- **Machine Learning Integration**: Price direction prediction using gradient boosting
- **Advanced Technical Analysis**: 25+ indicators using TA-Lib or simplified fallbacks
- **Robust Risk Management**: Position sizing, stop losses, drawdown protection
- **Portfolio Management**: Real-time tracking, performance metrics, reporting
- **Live Trading Support**: Alpaca API integration with paper trading
- **Comprehensive Backtesting**: Historical strategy testing with detailed analytics
- **Modular Architecture**: Clean, maintainable code structure

## 📁 Project Structure

```
trading_algorithm/
├── main.py                 # Main entry point
├── config.py              # Configuration and dependencies
├── data_structures.py     # Data models and structures
├── data_fetcher.py        # Market data fetching
├── indicators.py          # Technical indicators
├── ml_predictor.py        # Machine learning predictor
├── strategies.py          # Trading strategies
├── risk_manager.py        # Risk management system
├── portfolio_manager.py   # Portfolio tracking
├── trading_engine.py      # Main trading engine
├── backtester.py          # Backtesting engine
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── BUG_FIX_REPORT.md     # Recent bug fix documentation
├── .gitignore            # Git ignore rules
├── results/              # Trading results
│   ├── backtest/        # Backtest results
│   ├── live/            # Live trading results
│   └── simulations/     # Simulation results
└── logs/                 # Log files
```

## 🛠 Installation

### 1. Clone or Download Files

Save all the Python files in a directory called `trading_algorithm/`.

### 2. Install Dependencies

```bash
# Basic dependencies (required)
pip install pandas numpy

# Optional dependencies (recommended)
pip install yfinance websockets>=11.0
pip install scikit-learn scipy
pip install alpaca-trade-api

# TA-Lib (advanced technical indicators)
# Windows:
pip install TA-Lib

# macOS:
brew install ta-lib
pip install TA-Lib

# Linux:
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

### 3. Run the Algorithm

```bash
cd trading_algorithm
python main.py
```

## 🎯 Usage Modes

### 1. Simulation Mode (Recommended for Beginners)
- Uses sample data or real market data
- Simulated orders and portfolio
- Perfect for testing strategies
- No API credentials required

### 2. Live Trading Mode
- Connects to Alpaca API
- Paper trading (recommended) or live trading
- Requires API credentials
- Real-time market data

### 3. Backtest Mode
- Historical strategy testing
- Comprehensive performance analytics
- Multiple symbols and date ranges
- Export results to files

## 📊 Quick Start Examples

### Basic Simulation
```bash
python main.py
# Select: 2 (Simulation Mode)
# Select: 1 (Tech Stocks)
# Use defaults for other settings
```

### Run Demo
```bash
python main.py --demo
```

### Show Help
```bash
python main.py --help
```

## ⚙️ Configuration

### Watchlists
Pre-configured symbol lists:
- **Tech Stocks**: AAPL, MSFT, GOOGL, TSLA, NVDA
- **Blue Chips**: AAPL, MSFT, JNJ, PG, KO  
- **Growth Stocks**: TSLA, NVDA, AMD, CRM, NFLX
- **Custom**: Enter your own symbols

### Risk Management Settings
```python
# In config.py - TradingConfig class
MAX_POSITION_SIZE = 0.1      # Max 10% per position
MAX_DAILY_LOSS = 0.02        # Max 2% daily loss
MAX_DRAWDOWN = 0.15          # Max 15% drawdown
MIN_CONFIDENCE = 0.6         # Minimum signal confidence
```

### Trading Parameters
- **Check Interval**: How often to analyze markets (default: 300 seconds)
- **Initial Capital**: Starting portfolio value (default: $100,000)
- **Commission**: Trading costs per transaction (default: $0)

## 🤖 Machine Learning Features

### Price Prediction Model
- **Algorithm**: Gradient Boosting Classifier
- **Features**: Technical indicators, price ratios, lagged values
- **Target**: Next-day price direction (up/down)
- **Training**: Automatic retraining on new data

### Feature Engineering
- RSI, MACD, Bollinger Bands positions
- Volume ratios and volatility measures
- Moving average ratios
- Lagged price and volume data

## 📈 Trading Strategies

### 1. Momentum Breakout
- Identifies strong trending moves
- Uses trend confirmation + momentum indicators
- ML prediction integration
- Stop loss: 2x ATR, Take profit: 3x ATR

### 2. Mean Reversion  
- Trades oversold/overbought conditions
- Works best in ranging markets
- RSI + Bollinger Band signals
- Target: Return to mean prices

### 3. Trend Following
- Follows established trends
- Multiple timeframe confirmation
- Pullback entries for better risk/reward
- Trend strength filtering

### 4. Breakout Strategy
- Bollinger Band breakouts
- Volume confirmation required
- Volatility expansion filter
- Quick profits on momentum moves

### 5. Ensemble Method
- Combines multiple strategies
- Weighted by confidence scores
- Reduces false signals
- Improved risk-adjusted returns

## 📊 Risk Management

### Position Sizing
- **Kelly Criterion**: Optimal position sizing based on win rate and risk/reward
- **Volatility Adjustment**: Larger positions in less volatile stocks
- **Portfolio Heat**: Reduced sizing during drawdowns
- **Maximum Position**: 10% of portfolio per position

### Stop Loss Methods
- **ATR-Based**: 2x Average True Range below entry
- **Percentage-Based**: Fixed percentage from entry price
- **Trailing Stops**: Follow favorable price movements
- **Time-Based**: Maximum holding period limits

### Portfolio Protection
- **Daily Loss Limit**: Stop trading if 2% daily loss reached
- **Maximum Drawdown**: Stop at 15% peak-to-trough decline
- **Correlation Limits**: Avoid overconcentration in correlated assets
- **Cash Management**: Maintain minimum cash reserves

## 📋 Performance Tracking

### Key Metrics
- **Total Return**: Portfolio appreciation percentage
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits ÷ gross losses
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Trade**: Mean profit/loss per trade

### Reporting Features
- Real-time portfolio monitoring
- Trade-by-trade logging
- Strategy performance breakdown
- Risk metrics dashboard
- Exportable performance reports

## 🔧 Customization

### Adding New Strategies
```python
# In strategies.py
def my_custom_strategy(self, data: pd.DataFrame) -> TradeSignal:
    current = data.iloc[-1]
    
    # Your strategy logic here
    if buy_condition:
        return TradeSignal(
            symbol=symbol,
            action="BUY", 
            confidence=0.8,
            price=current['Close'],
            quantity=0
        )
    
    return TradeSignal(symbol, "HOLD", 0.0, current['Close'], 0)
```

### Adding New Indicators
```python
# In indicators.py - AdvancedIndicators class
@staticmethod
def my_custom_indicator(data: pd.Series, window: int = 14):
    # Your indicator calculation
    return result
```

### Modifying Risk Rules
```python
# In risk_manager.py
def validate_signal(self, signal: TradeSignal, portfolio_value: float, 
                   current_positions: dict) -> bool:
    # Add your custom validation logic
    if my_custom_condition:
        return False
    return True
```

## 🔍 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing packages
pip install -r requirements.txt

# Or install individually
pip install pandas numpy yfinance scikit-learn
```

**2. TA-Lib Installation Issues**
- Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/
- macOS: `brew install ta-lib` first
- Linux: Install libta-lib-dev package first

**3. No Data Available**
- Check internet connection
- Verify symbol names are correct
- Try different symbols or time periods

**4. Alpaca API Issues**
- Verify API credentials
- Check if market is open
- Ensure sufficient buying power

### Debug Mode
```python
# In config.py, change logging level
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Dependencies Overview

### Required (Algorithm runs with simplified features)
- `pandas`: Data manipulation
- `numpy`: Numerical computations

### Optional (Enhanced features)
- `yfinance`: Real market data
- `scikit-learn`: Machine learning
- `TA-Lib`: Advanced technical indicators  
- `scipy`: Statistical functions
- `alpaca-trade-api`: Live trading

The system automatically detects available packages and adapts functionality accordingly. Missing packages result in simplified features rather than crashes.

## ⚠️ Disclaimers

### Trading Risks
- **Past Performance**: Does not guarantee future results
- **Market Risk**: All investments carry risk of loss
- **Strategy Risk**: No strategy works in all market conditions
- **Technology Risk**: System failures can impact trading

### Recommendations
1. **Start with Paper Trading**: Test strategies without real money
2. **Small Position Sizes**: Begin with minimal capital allocation
3. **Continuous Monitoring**: Don't run unattended for extended periods
4. **Strategy Validation**: Backtest thoroughly before live trading
5. **Risk Management**: Never risk more than you can afford to lose

## 🤝 Contributing

### Code Structure Guidelines
- Follow existing naming conventions
- Add docstrings to all functions
- Include error handling
- Write unit tests for new features
- Update documentation

### Suggested Improvements
- Additional trading strategies
- New technical indicators
- Enhanced ML models
- Better visualization tools
- More sophisticated risk models

## 📞 Support

### Getting Help
1. Check this README for common issues
2. Review the code comments and docstrings
3. Test with paper trading first
4. Start with simulation mode to understand the system

### Feature Requests
The modular design makes it easy to add new features:
- New data sources
- Additional strategies  
- Enhanced ML models
- Better reporting tools
- Advanced order types

## 📄 License

This project is for educational and research purposes. Users are responsible for compliance with applicable financial regulations and trading rules.

---

**Happy Trading! 🚀📈**

Remember: The best strategy is the one you understand completely and can execute consistently with proper risk management.