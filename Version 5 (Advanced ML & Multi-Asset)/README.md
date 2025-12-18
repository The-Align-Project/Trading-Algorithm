# 🚀 Ultimate Trading Algorithm - Version 5.0

## Advanced ML | Multi-Timeframe | Options | Sentiment Analysis

A comprehensive, institutional-grade algorithmic trading system featuring deep learning, multi-timeframe analysis, options trading, and sentiment analysis.

---

## 🎯 What's New in Version 5

### 1. 🧠 Deep Learning Integration
- **LSTM Neural Networks**: Sequence-based price prediction
- **Transformer Models**: Attention-based pattern recognition
- **Ensemble Predictions**: Combined model outputs for higher accuracy
- **Auto-Training**: Automatic model retraining on new data

### 2. ⏱️ Multi-Timeframe Analysis
- **5 Timeframes**: Simultaneous analysis across 1m, 5m, 15m, 1h, 1d
- **Trend Confluence**: Detect alignment across timeframes
- **Support/Resistance**: Multi-timeframe zone identification
- **Higher Timeframe Context**: Use daily/hourly trends to filter trades

### 3. 📊 Options Trading
- **Black-Scholes Pricing**: Accurate options valuation
- **Greeks Calculation**: Delta, Gamma, Theta, Vega
- **Strategy Library**: Calls, Puts, Spreads, Iron Condor
- **Risk Analysis**: Max risk, max profit, breakeven points

### 4. 💬 Sentiment Analysis
- **News Sentiment**: Financial news from major sources
- **Social Media**: Twitter sentiment analysis
- **Combined Scoring**: Weighted sentiment from multiple sources
- **Engagement Weighting**: Higher weight for popular posts

---

## 📁 Project Structure

```
Version 5/
├── main.py                     # Main entry point
├── config.py                   # Configuration & dependencies
├── data_structures.py          # Data models
├── data_fetcher.py            # Market data retrieval
├── indicators.py              # Technical indicators
├── ml_predictor.py            # Traditional ML (Gradient Boosting)
├── deep_learning.py           # 🆕 LSTM & Transformer models
├── multi_timeframe.py         # 🆕 Multi-timeframe analysis
├── options_trading.py         # 🆕 Options strategies & pricing
├── sentiment_analysis.py      # 🆕 News & social sentiment
├── strategies_v5.py           # 🆕 Enhanced V5 strategies
├── risk_manager.py            # Risk management
├── portfolio_manager.py       # Portfolio tracking
├── backtester.py             # Backtesting engine
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## 🛠 Installation

### Option 1: Quick Install (Recommended)
```bash
# Clone or download Version 5 folder
cd "Version 5 (Advanced ML & Multi-Asset)"

# Install all dependencies
pip install -r requirements.txt

# Run the algorithm
python main.py
```

### Option 2: Step-by-Step Install

1. **Install Core Dependencies** (Required):
```bash
pip install pandas numpy yfinance scikit-learn scipy pytz websockets
```

2. **Install Deep Learning** (Recommended):
```bash
pip install torch transformers
```

3. **Install Sentiment Analysis** (Optional):
```bash
pip install nltk textblob newsapi-python tweepy
```

4. **Install Trading API** (For Live Trading):
```bash
pip install alpaca-trade-api
```

5. **Install TA-Lib** (Optional, for advanced indicators):
```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Windows
# Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/
pip install TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl

# Linux
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

---

## 🎮 Usage

### Quick Start

1. **Launch the Algorithm**:
```bash
python main.py
```

2. **Select Mode** from the menu:
   - **Option 2** (Simulation) - Best for beginners
   - **Option 3** (Backtest) - Test on historical data
   - **Option 4** (DL Training) - Train your models
   - **Option 1** (Live Trading) - When you're ready

### Mode Descriptions

#### 1. Live Trading Mode
- Connects to Alpaca broker
- Paper trading (recommended) or live trading
- Real-time data and automated execution
- Requires Alpaca API credentials

```python
# You'll need:
- Alpaca API Key
- Alpaca Secret Key
(Get free paper trading account at alpaca.markets)
```

#### 2. Simulation Mode
- Test strategies without API
- Uses real or sample data
- Perfect for learning and testing
- No credentials required

#### 3. Backtest Mode
- Historical performance testing
- Date range selection
- Comprehensive metrics
- Export detailed reports

#### 4. Deep Learning Training
- Train LSTM models
- Train Transformer models
- Configure epochs
- Save trained models for later use

#### 5. Options Strategy Analysis
- Analyze single symbols
- Get strategy recommendations
- View risk/reward profiles
- Calculate Greeks

---

## 📊 Trading Strategies

### 1. Enhanced Momentum (V5)
Combines traditional technical analysis with deep learning and sentiment.

**Components**:
- Traditional: SMA, EMA, MACD, RSI
- Deep Learning: LSTM + Transformer predictions
- Sentiment: News + Social media
- Confidence: Weighted from all sources

**Best For**: Trending markets, high momentum stocks

### 2. Multi-Timeframe Strategy
Analyzes price action across multiple timeframes simultaneously.

**Features**:
- Trend alignment across 5 timeframes
- Confluence zone detection
- Higher timeframe filtering
- Dynamic confidence adjustment

**Best For**: Strong trends, clear market direction

### 3. Sentiment-Driven Strategy
Primarily based on news and social media sentiment.

**Requirements**:
- News API key (optional)
- Twitter Bearer Token (optional)
- Minimum 10 data sources

**Best For**: Earnings, news events, viral stocks

### 4. Hybrid Strategy (Recommended)
Combines ALL Version 5 features for maximum accuracy.

**Scoring System**:
- Technical Analysis: 3 points
- Traditional ML: 1 point
- Deep Learning: 1 point
- Multi-Timeframe: 1 point
- Sentiment: 1 point
- **Total**: 7 points (requires 5+ for signal)

**Best For**: Most market conditions, highest accuracy

### 5. Options Strategies
Multiple options strategies based on market outlook.

**Strategies Available**:
- **Bullish**: Bull Call Spread
- **Bearish**: Bear Put Spread
- **Neutral**: Iron Condor
- **Single**: Long Call/Put

---

## ⚙️ Configuration

### config.py Settings

```python
# Risk Management
MAX_POSITION_SIZE = 0.1      # Max 10% per position
MAX_DAILY_LOSS = 0.02        # Max 2% daily loss
MAX_DRAWDOWN = 0.15          # Max 15% drawdown

# Deep Learning
DL_SEQUENCE_LENGTH = 60      # 60 timesteps for LSTM
DL_BATCH_SIZE = 32
DL_EPOCHS = 50
DL_LEARNING_RATE = 0.001

# Multi-Timeframe
TIMEFRAMES = ['1m', '5m', '15m', '1h', '1d']
PRIMARY_TIMEFRAME = '5m'

# Options Trading
OPTIONS_DELTA_MIN = 0.3
OPTIONS_DELTA_MAX = 0.7
OPTIONS_DTE_MIN = 7          # Min days to expiration
OPTIONS_DTE_MAX = 60         # Max days to expiration

# Sentiment Analysis
SENTIMENT_WEIGHT = 0.2
NEWS_LOOKBACK_HOURS = 24
SENTIMENT_THRESHOLD = 0.1

# Trading
MIN_CONFIDENCE = 0.6
MAX_EXECUTIONS_PER_ITERATION = 2
DEFAULT_CHECK_INTERVAL = 300  # seconds
```

---

## 🔑 API Keys (Optional)

### Alpaca (For Live/Paper Trading)
1. Sign up at [alpaca.markets](https://alpaca.markets)
2. Create a paper trading account (free)
3. Get API Key and Secret Key
4. Enter when prompted in Live Trading mode

### News API (For News Sentiment)
1. Sign up at [newsapi.org](https://newsapi.org)
2. Get free API key (100 requests/day)
3. Enter when prompted or set in code

### Twitter API (For Social Sentiment)
1. Apply at [developer.twitter.com](https://developer.twitter.com)
2. Get Bearer Token
3. Enter when prompted or set in code

**Note**: All APIs are optional. The system works without them using fallback methods.

---

## 📈 Performance Metrics

Version 5 tracks comprehensive performance metrics:

### Portfolio Metrics
- Total Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor

### Trade Metrics
- Total Trades
- Average Win/Loss
- Winning Trades %
- Strategy Performance Breakdown

### Risk Metrics
- Value at Risk (VaR 95%)
- Portfolio Heat
- Position Correlation
- Daily Loss Tracking

---

## 🧪 Testing & Validation

### Backtesting
Test strategies on historical data:

```bash
python main.py
# Select Option 3: Backtest Mode
# Choose symbols and date range
# Get comprehensive report
```

**Sample Output**:
```
Backtest Results (AAPL, 1 year):
- Total Return: 45.2%
- Sharpe Ratio: 2.1
- Win Rate: 58.3%
- Max Drawdown: -12.4%
- Total Trades: 47
```

### Paper Trading
Before live trading, test with paper money:

```bash
python main.py
# Select Option 1: Live Trading
# Use Paper Trading account
# Monitor for 1-2 weeks
# Review performance before going live
```

---

## ⚠️ Risk Warnings

### Important Disclaimers

1. **Past Performance ≠ Future Results**
   - Historical results do not guarantee future performance
   - Markets change and strategies may become less effective

2. **Market Risk**
   - All trading involves risk of loss
   - You can lose more than your initial investment (especially with options)
   - Never invest more than you can afford to lose

3. **Technology Risk**
   - System failures can impact trading
   - Internet connectivity issues
   - API outages
   - Model errors

4. **Strategy Risk**
   - No strategy works in all market conditions
   - Drawdowns are inevitable
   - Overfitting can occur in backtests

### Risk Management Best Practices

1. **Start Small**
   - Begin with paper trading
   - Use small position sizes
   - Gradually increase exposure

2. **Diversify**
   - Trade multiple symbols
   - Use multiple strategies
   - Don't put all capital in one position

3. **Monitor Actively**
   - Check positions daily
   - Review performance weekly
   - Adjust strategies as needed

4. **Set Limits**
   - Use stop losses
   - Respect daily loss limits
   - Take profits at targets

5. **Continuous Learning**
   - Study market conditions
   - Review trade history
   - Improve strategies based on results

---

## 🤝 Support & Contributing

### Getting Help

1. **Check Documentation**
   - This README
   - In-app help (Option 7)
   - Code comments

2. **Common Issues**
   - See TROUBLESHOOTING.md
   - Check logs/ directory
   - Review error messages

3. **Report Bugs**
   - Create GitHub issue
   - Include error logs
   - Describe steps to reproduce

### Feature Requests

Have ideas for Version 6? We'd love to hear them!

Potential V6 features:
- Reinforcement learning
- Portfolio optimization algorithms
- Advanced order types
- Risk parity allocation
- Volatility forecasting
- Custom strategy builder

---

## 📝 License

This project is for educational and research purposes.

**Important**:
- Users are responsible for compliance with financial regulations
- Trading rules and restrictions vary by jurisdiction
- Consult with financial and legal advisors before live trading

---

## 🎓 Learn More

### Recommended Reading
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Algorithmic Trading" - Ernest P. Chan
- "Options as a Strategic Investment" - Lawrence G. McMillan

### Online Resources
- Alpaca Docs: https://alpaca.markets/docs
- TA-Lib Documentation: https://mrjbq7.github.io/ta-lib/
- PyTorch Tutorials: https://pytorch.org/tutorials/

---

## 🙏 Acknowledgments

Version 5 builds upon:
- Version 4WD: Working Dependencies architecture
- Version 3E: ML integration and advanced indicators
- Version 2FW: Stable API integration
- Version 1: Core trading logic

Special thanks to the open-source community for:
- PyTorch (Deep Learning)
- scikit-learn (Traditional ML)
- pandas/numpy (Data handling)
- yfinance (Market data)
- TA-Lib (Technical indicators)

---

## 📞 Contact

For questions, suggestions, or collaborations:
- GitHub Issues: [Report bugs and request features]
- Documentation: Check VERSION_5_GUIDE.md for detailed technical docs

---

**Happy Trading! 🚀📈**

*Remember: The best strategy is the one you understand completely and can execute consistently with proper risk management.*

---

## Version History

- **V5.0** (Current) - Deep Learning, Multi-Timeframe, Options, Sentiment
- **V4WD** - Working Dependencies, Modular Architecture
- **V4** - Modular Refactor
- **V3E** - ML Integration, Advanced Indicators
- **V2FW** - Fully Working API
- **V2** - API Integration
- **V1** - Basic Trading Logic
