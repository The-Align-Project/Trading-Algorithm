"""Main entry point for Version 5 Advanced Trading Algorithm."""

import sys
from datetime import datetime

from config import (
    print_dependency_status, install_dependencies, WATCHLISTS,
    YF_AVAILABLE, SKLEARN_AVAILABLE, TALIB_AVAILABLE,
    SCIPY_AVAILABLE, ALPACA_AVAILABLE, PYTORCH_AVAILABLE,
    SENTIMENT_AVAILABLE, setup_logging
)

# Setup logging first
setup_logging()

def main():
    """Main function to run Version 5 Trading Algorithm"""
    print("=" * 70)
    print("🚀 ULTIMATE TRADING ALGORITHM VERSION 5.0")
    print("   Advanced ML | Multi-Timeframe | Options | Sentiment Analysis")
    print("=" * 70)
    print()

    # Print dependency status
    print_dependency_status()
    print()

    # Check for critical dependencies
    if not any([YF_AVAILABLE, SKLEARN_AVAILABLE]):
        print("⚠️  Critical dependencies missing!")
        install_choice = input("Would you like to install missing dependencies? (y/n): ").strip().lower()
        if install_choice == 'y':
            install_dependencies()
            return

    # Main menu
    print("=" * 70)
    print("SELECT MODE:")
    print("=" * 70)
    print("1. Live Trading with Alpaca (Paper/Live)")
    print("2. Simulation Mode (Test with Real/Sample Data)")
    print("3. Backtest Mode (Historical Performance Testing)")
    print("4. Deep Learning Training Mode (Train LSTM/Transformer)")
    print("5. Options Strategy Analysis")
    print("6. Install/Update Dependencies")
    print("7. View Help & Documentation")
    print("8. Exit")
    print("=" * 70)

    choice = input("\nSelect mode (1-8): ").strip()

    if choice == "6":
        install_dependencies()
        return
    elif choice == "7":
        show_help()
        return
    elif choice == "8":
        print("👋 Goodbye!")
        return

    # Get configuration based on mode
    if choice == "1":
        run_live_trading()
    elif choice == "2":
        run_simulation()
    elif choice == "3":
        run_backtest()
    elif choice == "4":
        run_dl_training()
    elif choice == "5":
        run_options_analysis()
    else:
        print("❌ Invalid choice")

def run_live_trading():
    """Run live trading mode"""
    print("\n" + "=" * 70)
    print("🎯 LIVE TRADING MODE")
    print("=" * 70)

    if not ALPACA_AVAILABLE:
        print("❌ Alpaca API not available. Please install: pip install alpaca-trade-api")
        return

    print("\n🔑 Enter Alpaca API Credentials:")
    api_key = input("API Key: ").strip()
    secret_key = input("Secret Key: ").strip()

    if not api_key or not secret_key:
        print("❌ Invalid credentials. Exiting.")
        return

    # Get watchlist
    symbols = get_watchlist()

    # Get initial capital
    capital = get_initial_capital()

    # Get API keys for sentiment (optional)
    print("\n📰 Sentiment Analysis (Optional - press Enter to skip):")
    news_api_key = input("News API Key: ").strip() or None
    twitter_token = input("Twitter Bearer Token: ").strip() or None

    print("\n🚀 Starting Live Trading...")
    print(f"   Mode: Paper Trading (Recommended)")
    print(f"   Initial Capital: ${capital:,.2f}")
    print(f"   Symbols: {symbols}")

    # Import and run (import here to avoid loading if not needed)
    try:
        from trading_engine_v5 import TradingEngineV5
        engine = TradingEngineV5(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
            initial_capital=capital,
            news_api_key=news_api_key,
            twitter_bearer_token=twitter_token
        )

        engine.run_live_trading(symbols=symbols)
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_simulation():
    """Run simulation mode"""
    print("\n" + "=" * 70)
    print("🎮 SIMULATION MODE")
    print("=" * 70)

    symbols = get_watchlist()
    capital = get_initial_capital()

    print("\n🎮 Starting Simulation...")
    print(f"   Mode: Simulation with Real/Sample Data")
    print(f"   Initial Capital: ${capital:,.2f}")
    print(f"   Symbols: {symbols}")

    try:
        from trading_engine_v5 import TradingEngineV5
        engine = TradingEngineV5(
            api_key=None,
            secret_key=None,
            paper=True,
            initial_capital=capital
        )

        engine.run_live_trading(symbols=symbols, check_interval=300)
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_backtest():
    """Run backtest mode"""
    print("\n" + "=" * 70)
    print("🔬 BACKTEST MODE")
    print("=" * 70)

    symbols = get_watchlist()

    print("\nEnter date range (YYYY-MM-DD) or press Enter for default (1 year):")
    start_date = input("Start date: ").strip() or None
    end_date = input("End date: ").strip() or None

    capital = get_initial_capital()

    print("\n🔬 Running Backtest...")

    try:
        from backtester import BacktestEngine
        backtester = BacktestEngine(initial_capital=capital)
        results = backtester.run_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )

        if results:
            print("\n✅ Backtest completed!")
            export_choice = input("Export results? (y/n): ").strip().lower()
            if export_choice == 'y':
                backtester.export_backtest_results(results)
    except Exception as e:
        print(f"❌ Error: {e}")

def run_dl_training():
    """Run deep learning training mode"""
    print("\n" + "=" * 70)
    print("🧠 DEEP LEARNING TRAINING MODE")
    print("=" * 70)

    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not available. Please install: pip install torch")
        return

    symbols = get_watchlist()

    print("\nTraining Parameters:")
    epochs = input("Number of epochs (default 50): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 50

    print(f"\n🧠 Training deep learning models on {len(symbols)} symbols...")
    print(f"   Epochs: {epochs}")
    print(f"   This may take several minutes...")

    try:
        from deep_learning import DeepLearningPredictor
        from data_fetcher import SimplifiedDataFetcher
        from indicators import AdvancedIndicators

        dl_predictor = DeepLearningPredictor()
        data_fetcher = SimplifiedDataFetcher()

        for symbol in symbols:
            print(f"\nTraining on {symbol}...")
            data = data_fetcher.fetch_data(symbol, period="2y")

            if len(data) < 100:
                print(f"  ⚠️ Insufficient data for {symbol}")
                continue

            data_with_indicators = AdvancedIndicators.calculate_all_indicators(data)
            dl_predictor.train_models(data_with_indicators, epochs=epochs)

        print("\n✅ Training completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_options_analysis():
    """Run options strategy analysis"""
    print("\n" + "=" * 70)
    print("📊 OPTIONS STRATEGY ANALYSIS")
    print("=" * 70)

    if not SCIPY_AVAILABLE:
        print("❌ SciPy not available for options pricing. Install: pip install scipy")
        return

    symbol = input("\nEnter symbol (e.g., AAPL): ").strip().upper()

    if not symbol:
        print("❌ Invalid symbol")
        return

    print(f"\n📊 Analyzing options strategies for {symbol}...")

    try:
        from options_trading import OptionsAnalyzer
        from data_fetcher import SimplifiedDataFetcher
        from indicators import AdvancedIndicators

        data_fetcher = SimplifiedDataFetcher()
        options_analyzer = OptionsAnalyzer()

        data = data_fetcher.fetch_data(symbol, period="3mo")
        data_with_indicators = AdvancedIndicators.calculate_all_indicators(data)

        trend = AdvancedIndicators.get_trend_direction(data_with_indicators)

        print(f"\nMarket Trend: {trend}")

        # Map trend to outlook
        outlook_map = {
            'BULLISH': 'BULLISH',
            'BEARISH': 'BEARISH',
            'SIDEWAYS': 'NEUTRAL'
        }
        outlook = outlook_map.get(trend, 'NEUTRAL')

        signal = options_analyzer.generate_options_signal(symbol, data_with_indicators, outlook)

        if signal:
            print(f"\nRecommended Strategy: {signal.strategy_name}")
            print(f"Confidence: {signal.confidence:.1%}")
            print(f"Max Risk: ${signal.max_risk:,.2f}")
            print(f"Max Profit: ${signal.max_profit:,.2f}")
            print(f"Breakeven Points: {[f'${bp:.2f}' for bp in signal.breakeven_points]}")
            print(f"\nLegs:")
            for i, leg in enumerate(signal.legs, 1):
                print(f"  {i}. {leg.action.value.upper()} {leg.option_type.value.upper()} @ ${leg.strike:.2f} (Premium: ${leg.premium:.2f})")
        else:
            print("\n⚠️ No suitable options strategy at this time")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def get_watchlist():
    """Get symbol watchlist from user"""
    print("\n" + "=" * 70)
    print("SELECT WATCHLIST:")
    print("=" * 70)
    print("1. Tech Stocks (AAPL, MSFT, GOOGL, TSLA, NVDA)")
    print("2. Blue Chips (AAPL, MSFT, JNJ, PG, KO)")
    print("3. Growth Stocks (TSLA, NVDA, AMD, CRM, NFLX)")
    print("4. Crypto (BTC-USD, ETH-USD, SOL-USD)")
    print("5. Custom (Enter your own)")
    print("=" * 70)

    watchlist_choice = input("Select watchlist (1-5): ").strip()

    if watchlist_choice == "1":
        return WATCHLISTS['tech']
    elif watchlist_choice == "2":
        return WATCHLISTS['blue_chips']
    elif watchlist_choice == "3":
        return WATCHLISTS['growth']
    elif watchlist_choice == "4":
        return WATCHLISTS.get('crypto', ['BTC-USD', 'ETH-USD'])
    elif watchlist_choice == "5":
        symbols_input = input("Enter symbols (comma-separated, e.g., AAPL,MSFT): ").strip()
        return [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    else:
        return WATCHLISTS['default']

def get_initial_capital():
    """Get initial capital from user"""
    capital_input = input("\nInitial capital (default $100,000): ").strip()
    try:
        return float(capital_input.replace(',', '').replace('$', ''))
    except (ValueError, AttributeError):
        return 100000

def show_help():
    """Show comprehensive help information"""
    help_text = """
=" * 70
🚀 ULTIMATE TRADING ALGORITHM V5.0 - HELP & DOCUMENTATION
=" * 70

WHAT'S NEW IN VERSION 5:

1. 🧠 DEEP LEARNING
   - LSTM Neural Networks for price prediction
   - Transformer models for sequence analysis
   - Ensemble predictions combining multiple models
   - Auto-training on historical data

2. ⏱️ MULTI-TIMEFRAME ANALYSIS
   - Simultaneous analysis across 1m, 5m, 15m, 1h, 1d timeframes
   - Trend confluence detection
   - Support/resistance with multi-timeframe confirmation
   - Higher timeframe context for better decision making

3. 📊 OPTIONS TRADING
   - Black-Scholes pricing model
   - Greeks calculation (Delta, Gamma, Theta, Vega)
   - Multiple strategies: Calls, Puts, Spreads, Iron Condor
   - Risk/reward analysis for each strategy

4. 💬 SENTIMENT ANALYSIS
   - News sentiment from major financial sources
   - Social media sentiment (Twitter)
   - Combined sentiment scoring
   - Weighted by source quality and engagement

TRADING MODES:

1. LIVE TRADING
   - Connects to Alpaca broker API
   - Paper trading (recommended) or live trading
   - Real-time market data
   - Automated order execution

2. SIMULATION
   - Test strategies without API
   - Uses real or sample market data
   - Full strategy testing
   - Performance tracking

3. BACKTEST
   - Test on historical data
   - Comprehensive performance metrics
   - Strategy comparison
   - Export detailed reports

4. DEEP LEARNING TRAINING
   - Train LSTM and Transformer models
   - Custom epoch configuration
   - Model validation
   - Save trained models

5. OPTIONS ANALYSIS
   - Analyze options strategies
   - Calculate optimal strikes
   - Risk/reward visualization
   - Strategy recommendations

STRATEGIES:

1. Enhanced Momentum (V5)
   - Traditional technical + DL + Sentiment
   - Multi-factor confirmation
   - Adaptive confidence scoring

2. Multi-Timeframe
   - Cross-timeframe trend alignment
   - Confluence zone detection
   - Higher timeframe filtering

3. Sentiment-Driven
   - News + Social media analysis
   - Technical confirmation
   - Volume weighted sentiment

4. Hybrid (Recommended)
   - Combines ALL V5 features
   - Weighted ensemble approach
   - Highest accuracy potential

5. Options Strategies
   - Bull/Bear spreads
   - Iron Condor
   - Long Calls/Puts
   - Risk-defined strategies

DEPENDENCIES:

Core (Required):
  - pandas, numpy: Data handling
  - yfinance: Market data
  - scikit-learn: Traditional ML

V5 Features (Optional):
  - torch: Deep learning (LSTM/Transformer)
  - scipy: Options pricing
  - nltk, textblob: Sentiment analysis
  - newsapi-python: News data
  - tweepy: Twitter data
  - alpaca-trade-api: Live trading

INSTALLATION:

1. Install core dependencies:
   pip install pandas numpy yfinance scikit-learn

2. Install V5 features:
   pip install torch transformers scipy nltk textblob
   pip install newsapi-python tweepy alpaca-trade-api

3. Or use the built-in installer (Option 6 in main menu)

GETTING STARTED:

1. Start with Simulation Mode (#2) to test
2. Try Backtest Mode (#3) to validate strategies
3. Train deep learning models (#4) for better predictions
4. Move to Paper Trading (#1) when confident
5. Monitor performance and adjust

API KEYS (Optional but Recommended):

- Alpaca: For live/paper trading
  Get at: https://alpaca.markets

- News API: For news sentiment
  Get at: https://newsapi.org

- Twitter: For social sentiment
  Get at: https://developer.twitter.com

RISK WARNING:

- Past performance does not guarantee future results
- All trading involves risk of loss
- Start with paper trading
- Use proper position sizing
- Never risk more than you can afford to lose

FOR MORE INFO:

- GitHub: Check VERSION_5_GUIDE.md
- Documentation: See README.md
- Issues: Report bugs on GitHub

=" * 70
    """
    print(help_text)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
