# Quick Start Guide

## Installation & Setup

### 1. Install Python Dependencies
```bash
pip install pandas numpy yfinance scikit-learn scipy alpaca-trade-api
```

### 2. Optional: Install TA-Lib
```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Windows - download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/
pip install TA_Lib-0.4.XX-cpXX-cpXX-winXX.whl

# Linux
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

## Running the Algorithm

### Simulation Mode (Recommended First)
```bash
python main.py
# Select: 2 (Simulation Mode)
# Select: 1 (Tech Stocks)
# Initial capital: 100000 (or press Enter for default)
# Check interval: 300 (or press Enter for default)
```

The algorithm will:
- Fetch real market data for AAPL, MSFT, GOOGL, TSLA, NVDA
- Analyze each symbol every 5 minutes
- Generate trading signals when conditions are met
- Execute simulated trades
- Display performance metrics

### Live Trading Mode (Paper Trading)
```bash
python main.py
# Select: 1 (Live Trading)
# Enter your Alpaca Paper Trading API credentials
# Select watchlist
# Configure capital and interval
```

**Get Alpaca Paper Trading Credentials:**
1. Sign up at https://alpaca.markets
2. Go to Paper Trading section
3. Generate API Key and Secret Key
4. The algorithm uses paper trading by default (no real money)

### Backtest Mode
```bash
python main.py
# Select: 3 (Backtest Mode)
# Select watchlist
# Enter date range (or press Enter for 1 year)
# Configure capital and commission
```

## Understanding the Output

### Console Logs
```
==================================================
Trading Iteration #1 - 2025-10-22 14:30:00
==================================================
📊 Fetched data for 5/5 symbols
🎯 Found 2 high-confidence signals
✅ Executed BUY signal for AAPL (confidence: 0.75)
✅ Executed BUY signal for MSFT (confidence: 0.68)
📊 Positions updated: 2

💰 CURRENT STATUS:
   Portfolio Value: $100,450.00
   Cash: $89,200.00
   Active Positions: 2
```

### What the Logs Mean
- **Signals generated**: How many strategies identified potential trades
- **Signals executed**: How many trades were actually made
- **Confidence**: Signal strength (0.0 to 1.0, algorithm requires ≥ 0.6)
- **Portfolio Value**: Total account value (cash + positions)

### Performance Reports
Automatically saved to `results/live/performance_report_TIMESTAMP.txt`

Includes:
- Total return percentage
- Win rate
- Profit factor
- Sharpe ratio
- Maximum drawdown
- Trade-by-trade history

## Configuration Tips

### Adjust Risk Tolerance
Edit `config.py` - `TradingConfig` class:
```python
MAX_POSITION_SIZE = 0.05    # More conservative: 5% per position
MAX_DAILY_LOSS = 0.01       # Stop if 1% daily loss
MIN_CONFIDENCE = 0.7        # Require higher confidence (fewer trades)
```

### Change Check Interval
- Shorter (60-120 seconds): More frequent checks, more data usage
- Longer (600-900 seconds): Less frequent, more conservative

### Customize Watchlist
```bash
python main.py
# Select: 4 (Custom)
# Enter: NVDA,AMD,INTC,MU,QCOM
```

## Troubleshooting

### "No signals generated"
**This is normal!** The algorithm is conservative and waits for:
- Strong technical indicator alignment
- High ML prediction confidence
- Favorable risk/reward ratios
- Appropriate market conditions

Not every iteration will generate trades.

### "Failed to execute signal"
Possible reasons:
- Insufficient cash
- Risk limits reached (max positions, drawdown limits)
- Position size too small

### "No data available"
- Check internet connection
- Verify symbols are valid (use uppercase)
- Try different symbols or time period

### Import errors
```bash
# Install missing package
pip install <package-name>

# Or install all at once
pip install -r requirements.txt
```

## Best Practices

### Starting Out
1. **Run simulation mode first** to understand the system
2. **Start with small capital** even in paper trading
3. **Monitor for a few hours** before leaving it unattended
4. **Review the logs** to understand signal generation

### Risk Management
1. Never risk more than you can afford to lose
2. Start with conservative settings
3. Use paper trading to test strategies
4. Monitor drawdown and daily losses
5. Don't override risk limits

### Strategy Optimization
1. **Backtest first** before live trading
2. **Test different watchlists** to find what works
3. **Adjust confidence threshold** based on results
4. **Review trade history** to identify patterns
5. **Keep performance records** for analysis

## Next Steps

### After Your First Run
1. Review `results/live/performance_report_TIMESTAMP.txt`
2. Check trade execution in console logs
3. Analyze which strategies generated signals
4. Adjust configuration if needed
5. Run backtest to validate strategy

### Advanced Usage
- Modify strategies in `strategies.py`
- Add custom indicators in `indicators.py`
- Adjust risk rules in `risk_manager.py`
- Customize ML features in `ml_predictor.py`

## Support & Documentation

- **Main README**: `README.md` - Complete documentation
- **Bug Fix Report**: `BUG_FIX_REPORT.md` - Recent fixes
- **Code Comments**: All files have detailed docstrings
- **Results**: Check `results/` folder for output files

## Important Reminders

⚠️ **This is for educational purposes**
- Past performance doesn't guarantee future results
- All trading involves risk of loss
- Test thoroughly before using real money
- Monitor the system regularly
- Understand the strategies you're using

✅ **The algorithm is working correctly when:**
- It fetches data without errors
- It generates occasional signals (not every iteration)
- It respects risk limits
- It logs all actions clearly

🚀 **Happy Trading!**

Remember: The best results come from understanding the system, using proper risk management, and being patient with high-confidence opportunities.
