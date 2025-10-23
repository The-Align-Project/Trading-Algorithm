# Algorithmic Trading System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

A comprehensive, modular algorithmic trading system with machine learning integration, advanced technical analysis, and robust risk management capabilities.

## 🎯 Overview

This repository contains a progressive evolution of algorithmic trading systems, from simple implementations to a fully-featured trading bot with ML capabilities, live trading support, and comprehensive backtesting.

### Project Evolution

- **Version 1**: Basic trading logic without API integration
- **Version 2**: API integration (Alpaca)
- **Version 3**: Enhanced features with error handling
- **Version 4**: Modular architecture with dependencies
- **Version 4WD**: Production-ready system with full functionality ✨ (Current)

## ✨ Key Features

### Trading Capabilities
- 🎯 **Multiple Trading Strategies**: Momentum breakout, mean reversion, trend following, and breakout strategies
- 🤖 **Machine Learning Integration**: Gradient boosting for price direction prediction
- 📊 **Advanced Technical Analysis**: 25+ indicators with TA-Lib support
- 💰 **Live Trading**: Alpaca API integration with paper and live trading modes
- 📈 **Backtesting Engine**: Historical strategy validation with detailed analytics
- 🔄 **Simulation Mode**: Risk-free testing with real market data

### Risk Management
- 📉 **Dynamic Position Sizing**: Kelly Criterion and volatility-based sizing
- 🛡️ **Stop Loss Protection**: ATR-based and percentage-based stops
- 📊 **Portfolio Heat Monitoring**: Drawdown protection and exposure limits
- ⚠️ **Daily Loss Limits**: Automatic trading suspension on excessive losses
- 🎲 **Correlation Management**: Avoid overconcentration in correlated assets

### Technical Infrastructure
- 🏗️ **Modular Architecture**: Clean separation of concerns
- 📝 **Comprehensive Logging**: Detailed trade and performance logging
- 📊 **Performance Reporting**: Real-time metrics and exportable reports
- 🔧 **Configurable Settings**: Easy customization via config files
- 🐛 **Error Handling**: Robust fallbacks and graceful degradation

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Clone or download this repository
git clone https://github.com/The-Align-Project/Trading-Algorithm.git
cd "Algorithmic Trading"
```

### Installation

```bash
# Navigate to the working version
cd Version-4WD-Working-Dependencies

# Install required dependencies
pip install pandas numpy

# Install optional dependencies for full features
pip install yfinance scikit-learn scipy alpaca-trade-api

# Install TA-Lib (optional, for advanced indicators)
# macOS:
brew install ta-lib && pip install TA-Lib

# Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/
# Linux:
sudo apt-get install libta-lib-dev && pip install TA-Lib
```

### Running the Algorithm

```bash
# Start the trading system
python main.py

# Run demo mode
python main.py --demo

# Show help
python main.py --help
```

## 📖 Documentation

### Repository Structure

```
Algorithmic Trading/
├── Version 1 (No API)/              # Initial implementation
├── Version 2 (API)/                 # API integration
├── Version 2E (Error API)/          # Error handling improvements
├── Version 2FW (Fully Working API)/ # Stable API version
├── Version 3E (Error API)/          # Enhanced features
├── Version 4 (Dependencies)/        # Modular architecture
└── Version-4WD-Working-Dependencies/ ⭐ CURRENT VERSION
    ├── main.py                      # Main entry point
    ├── config.py                    # Configuration and settings
    ├── trading_engine.py            # Core trading logic
    ├── strategies.py                # Trading strategies
    ├── indicators.py                # Technical indicators
    ├── ml_predictor.py              # ML prediction model
    ├── risk_manager.py              # Risk management
    ├── portfolio_manager.py         # Portfolio tracking
    ├── data_fetcher.py              # Market data retrieval
    ├── backtester.py                # Backtesting engine
    ├── requirements.txt             # Python dependencies
    ├── README.md                    # Detailed documentation
    ├── QUICK_START.md              # Quick start guide
    ├── BUG_FIX_REPORT.md           # Recent fixes
    ├── logs/                        # Log files
    └── results/                     # Trading results
        ├── backtest/                # Backtest results
        ├── live/                    # Live trading results
        └── simulations/             # Simulation results
```

### Usage Modes

#### 1. Simulation Mode (Recommended for Beginners)
```bash
python main.py
# Select: 2 (Simulation Mode)
# Choose watchlist: 1 (Tech Stocks)
```
- Uses real market data
- Simulated order execution
- No API credentials required
- Perfect for learning and testing

#### 2. Live Trading Mode
```bash
python main.py
# Select: 1 (Live Trading)
# Enter Alpaca API credentials
```
- Connects to Alpaca API
- Paper trading or live trading
- Real-time market data
- Actual order execution

#### 3. Backtest Mode
```bash
python main.py
# Select: 3 (Backtest Mode)
```
- Historical strategy testing
- Performance analytics
- Strategy optimization
- Export detailed reports

## 📊 Trading Strategies

### 1. Momentum Breakout Strategy
- Identifies strong trending moves with volume confirmation
- Uses RSI, MACD, and moving average alignment
- ML prediction integration for directional confidence
- Dynamic stop loss (2x ATR) and take profit (3x ATR)

### 2. Mean Reversion Strategy
- Trades oversold/overbought conditions
- RSI + Bollinger Band signals
- Works best in ranging markets
- Targets return to mean prices

### 3. Trend Following Strategy
- Follows established market trends
- Multiple timeframe confirmation
- Pullback entries for better risk/reward
- Trend strength filtering

### 4. Breakout Strategy
- Bollinger Band expansion breakouts
- Volume spike confirmation
- Volatility expansion filter
- Quick profit-taking on momentum

### 5. Ensemble Strategy
- Combines signals from all strategies
- Weighted by confidence scores
- Reduces false signals
- Improved risk-adjusted returns

## 🤖 Machine Learning

### Price Prediction Model
- **Algorithm**: Gradient Boosting Classifier
- **Features**: 20+ technical indicators and price patterns
- **Target**: Next-day price direction (up/down)
- **Retraining**: Automatic on new data

### Feature Engineering
- Technical indicator values (RSI, MACD, BB)
- Price ratios and momentum
- Volume patterns and volatility
- Lagged price and volume data

## ⚙️ Configuration

### Risk Management Settings
```python
# In config.py - TradingConfig class
MAX_POSITION_SIZE = 0.1      # Max 10% per position
MAX_DAILY_LOSS = 0.02        # Max 2% daily loss
MAX_DRAWDOWN = 0.15          # Max 15% drawdown
MIN_CONFIDENCE = 0.6         # Minimum signal confidence (60%)
MAX_EXECUTIONS_PER_ITERATION = 5  # Limit trades per cycle
```

### Watchlists
Pre-configured symbol lists in `config.py`:
- **Tech**: AAPL, MSFT, GOOGL, TSLA, NVDA
- **Blue Chips**: AAPL, MSFT, JNJ, PG, KO
- **Growth**: TSLA, NVDA, AMD, CRM, NFLX

### Trading Parameters
- **Check Interval**: 300 seconds (5 minutes) default
- **Initial Capital**: $100,000 default
- **Commission**: $0 default (configurable)

## 📈 Performance Metrics

### Key Metrics Tracked
- **Total Return**: Portfolio appreciation percentage
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits ÷ gross losses
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Trade**: Mean profit/loss per trade

### Reporting
- Real-time console output
- Timestamped performance reports
- Trade-by-trade logging
- Strategy performance breakdown
- Exportable CSV and text reports

## 🛠️ Customization

### Adding Custom Strategies
Edit `strategies.py` to add your own trading logic:

```python
def my_custom_strategy(self, data: pd.DataFrame) -> TradeSignal:
    # Your strategy logic here
    if buy_condition:
        return TradeSignal(
            symbol=symbol,
            action="BUY",
            confidence=0.8,
            price=current_price,
            quantity=0
        )
    return TradeSignal(symbol, "HOLD", 0.0, current_price, 0)
```

### Adding Custom Indicators
Edit `indicators.py` to create new technical indicators:

```python
@staticmethod
def my_indicator(data: pd.Series, window: int = 14):
    # Your indicator calculation
    return result
```

## 🧪 Testing

### Backtesting
```bash
python main.py
# Select: 3 (Backtest Mode)
# Configure symbols and date range
# Review results in results/backtest/
```

### Paper Trading
1. Sign up at [Alpaca Markets](https://alpaca.markets)
2. Get Paper Trading API credentials
3. Run in Live Trading mode with paper credentials
4. Monitor performance in real-time

## ⚠️ Important Disclaimers

### Trading Risks
- **Past performance does not guarantee future results**
- **All trading involves risk of loss**
- **You may lose some or all of your invested capital**
- **No strategy works in all market conditions**

### Best Practices
1. ✅ Start with paper trading
2. ✅ Test strategies thoroughly via backtesting
3. ✅ Begin with small position sizes
4. ✅ Monitor the system regularly
5. ✅ Never risk more than you can afford to lose
6. ✅ Understand the strategies you're using
7. ✅ Comply with all applicable regulations

### Technology Risks
- System failures can impact trading
- Network connectivity issues may occur
- API rate limits may apply
- Data feed interruptions are possible

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install -r requirements.txt
```

**No Data Available**
- Check internet connection
- Verify symbol names are correct
- Try different date ranges or symbols

**Alpaca API Issues**
- Verify API credentials
- Check if market is open
- Ensure sufficient buying power

**TA-Lib Installation**
- Windows: Download pre-built wheel
- macOS: Install via Homebrew first
- Linux: Install libta-lib-dev package

### Debug Mode
Enable detailed logging in `config.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Dependencies

### Required
- `pandas` - Data manipulation
- `numpy` - Numerical computations

### Optional (Recommended)
- `yfinance` - Real market data
- `scikit-learn` - Machine learning
- `scipy` - Statistical functions
- `TA-Lib` - Advanced technical indicators
- `alpaca-trade-api` - Live trading

The system automatically detects available packages and adapts accordingly.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

### Code Guidelines
- Follow existing code style
- Add docstrings to functions
- Include error handling
- Update README for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Important**: This software is for educational purposes only. Users are responsible for:
- Compliance with financial regulations
- Understanding trading risks
- Proper testing before live trading
- Their own financial decisions

## 🙏 Acknowledgments

- Built with Python and open-source libraries
- Alpaca Markets for trading API
- TA-Lib for technical indicators
- scikit-learn for machine learning

## 📞 Support

- 📖 Read the [detailed documentation](Version-4WD-Working-Dependencies/README.md)
- 🚀 Check the [Quick Start Guide](Version-4WD-Working-Dependencies/QUICK_START.md)
- 🐛 Review [Bug Fix Reports](Version-4WD-Working-Dependencies/BUG_FIX_REPORT.md)
- 💬 Submit issues via GitHub

## 🗺️ Roadmap

### Planned Features
- [ ] Additional trading strategies
- [ ] Enhanced ML models (LSTM, Transformer)
- [ ] Multi-asset support (crypto, forex)
- [ ] Web dashboard for monitoring
- [ ] Advanced portfolio optimization
- [ ] Sentiment analysis integration
- [ ] Options trading support

---

**⚡ Current Version: 4WD (Working Dependencies)**  
**📅 Last Updated: October 2025**  
**👨‍💻 Maintained by: The Align Project**

---

**Happy Trading! 🚀📈**

*Remember: The best strategy is the one you understand completely and can execute consistently with proper risk management.*
