"""Main entry point for the Ultimate Trading Algorithm."""

import sys
from datetime import datetime

from config import (
    print_dependency_status, install_dependencies, WATCHLISTS,
    YF_AVAILABLE, SKLEARN_AVAILABLE, TALIB_AVAILABLE, 
    SCIPY_AVAILABLE, ALPACA_AVAILABLE
)
from trading_engine import TradingEngine
from backtester import BacktestEngine

def main():
    """Main function to run the Ultimate Trading Algorithm"""
    print("🚀 Ultimate Advanced Trading Algorithm v2.0")
    print("=" * 50)
    
    # Print dependency status
    print_dependency_status()
    print()
    
    # Check if we need to install dependencies
    if not any([YF_AVAILABLE, SKLEARN_AVAILABLE, TALIB_AVAILABLE, SCIPY_AVAILABLE, ALPACA_AVAILABLE]):
        print("⚠️  Most dependencies are missing. The algorithm will run with simplified features.")
        install_choice = input("Would you like to install missing dependencies? (y/n): ").strip().lower()
        if install_choice == 'y':
            install_dependencies()
            return
    
    # Main menu
    print("Select Mode:")
    print("1. Live Trading with Alpaca")
    print("2. Simulation Mode")
    print("3. Backtest Mode")
    print("4. Install Dependencies")
    print("5. Exit")
    
    choice = input("\nSelect mode (1-5): ").strip()
    
    if choice == "4":
        install_dependencies()
        return
    elif choice == "5":
        print("👋 Goodbye!")
        return
    
    # Get API credentials if needed for live trading
    api_key = None
    secret_key = None
    
    if choice == "1":
        print("\n🔑 Enter Alpaca API Credentials:")
        api_key = input("API Key: ").strip()
        secret_key = input("Secret Key: ").strip()
        
        if not api_key or not secret_key:
            print("❌ Invalid credentials. Switching to simulation mode.")
            choice = "2"
    
    # Get symbol selection
    print("\nSelect Watchlist:")
    print("1. Tech Stocks (AAPL, MSFT, GOOGL, TSLA, NVDA)")
    print("2. Blue Chips (AAPL, MSFT, JNJ, PG, KO)")
    print("3. Growth Stocks (TSLA, NVDA, AMD, CRM, NFLX)")
    print("4. Custom (enter your own)")
    
    watchlist_choice = input("Select watchlist (1-4): ").strip()
    
    if watchlist_choice == "1":
        watchlist = WATCHLISTS['tech']
    elif watchlist_choice == "2":
        watchlist = WATCHLISTS['blue_chips']
    elif watchlist_choice == "3":
        watchlist = WATCHLISTS['growth']
    elif watchlist_choice == "4":
        symbols_input = input("Enter symbols separated by commas (e.g., AAPL,MSFT,GOOGL): ").strip()
        watchlist = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
        if not watchlist:
            watchlist = WATCHLISTS['default']
    else:
        watchlist = WATCHLISTS['default']
    
    print(f"\n📋 Selected symbols: {watchlist}")
    
    try:
        if choice in ["1", "2"]:
            # Live/Simulation Trading
            run_live_trading(watchlist, api_key, secret_key, choice == "1")
        elif choice == "3":
            # Backtest Mode
            run_backtest(watchlist)
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import logging
        logging.error(f"Main error: {e}", exc_info=True)

def run_live_trading(symbols, api_key=None, secret_key=None, is_live=False):
    """Run live or simulation trading"""
    print(f"\n🎯 Starting {'Live' if is_live else 'Simulation'} Trading")
    
    # Get initial capital
    capital_input = input("Initial capital (default: $100,000): ").strip()
    try:
        initial_capital = float(capital_input.replace(',', ''))
    except (ValueError, AttributeError):
        initial_capital = 100000
    
    # Get check interval
    interval_input = input("Check interval in seconds (default: 300): ").strip()
    try:
        check_interval = int(interval_input)
    except (ValueError, TypeError):
        check_interval = 300
    
    # Initialize trading engine
    engine = TradingEngine(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,  # Always use paper trading for safety
        initial_capital=initial_capital
    )
    
    print(f"\n🚀 Trading Engine Initialized")
    print(f"   Mode: {'Live (Paper)' if is_live else 'Simulation'}")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Symbols: {symbols}")
    print(f"   Check Interval: {check_interval}s")
    
    # Start trading
    try:
        engine.run_live_trading(
            symbols=symbols,
            check_interval=check_interval
        )
    except KeyboardInterrupt:
        print("\n🛑 Stopping trading engine...")
        engine.stop_trading()

def run_backtest(symbols):
    """Run backtest mode"""
    print("\n🔬 Starting Backtest Mode")
    
    # Get date range
    print("Enter date range (YYYY-MM-DD format) or press Enter for default (1 year)")
    start_date = input("Start date (optional): ").strip()
    end_date = input("End date (optional): ").strip()
    
    if not start_date:
        start_date = None
        end_date = None
    elif start_date and not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get initial capital
    capital_input = input("Initial capital (default: $100,000): ").strip()
    try:
        initial_capital = float(capital_input.replace(',', ''))
    except (ValueError, AttributeError):
        initial_capital = 100000
    
    # Get commission rate
    commission_input = input("Commission per trade (default: $0): ").strip()
    try:
        commission = float(commission_input.replace(',', ''))
    except (ValueError, AttributeError):
        commission = 0.0
    
    print(f"\n🔬 Backtest Configuration:")
    print(f"   Symbols: {symbols}")
    print(f"   Date Range: {start_date or 'Auto'} to {end_date or 'Auto'}")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Commission: ${commission:.2f} per trade")
    
    # Initialize and run backtest
    backtester = BacktestEngine(initial_capital=initial_capital)
    
    results = backtester.run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        commission=commission
    )
    
    # Export results
    if results:
        export_choice = input("\nExport results to file? (y/n): ").strip().lower()
        if export_choice == 'y':
            backtester.export_backtest_results(results)

def interactive_demo():
    """Run an interactive demo of the system"""
    print("\n🎮 Interactive Demo Mode")
    print("This will run a quick simulation with sample data")
    
    # Use default watchlist
    symbols = WATCHLISTS['tech'][:3]  # Use first 3 tech stocks
    
    # Create trading engine
    engine = TradingEngine(
        initial_capital=50000,  # Smaller amount for demo
        api_key=None,
        secret_key=None,
        paper=True
    )
    
    print(f"\n📊 Demo Configuration:")
    print(f"   Symbols: {symbols}")
    print(f"   Initial Capital: $50,000")
    print(f"   Mode: Simulation with sample data")
    
    input("Press Enter to start demo...")
    
    try:
        # Run for a few iterations only
        for i in range(3):
            print(f"\n--- Demo Iteration {i+1} ---")
            results = engine.run_single_iteration(symbols)
            
            status = engine.get_status()
            print(f"Portfolio Value: ${status['portfolio_value']:,.2f}")
            print(f"Positions: {status['positions_count']}")
            print(f"Signals Generated: {results['signals_generated']}")
            print(f"Signals Executed: {results['signals_executed']}")
            
            import time
            time.sleep(2)  # Brief pause between iterations
        
        print("\n✅ Demo completed!")
        
        # Show final status
        final_status = engine.get_status()
        print(f"\nFinal Results:")
        print(f"   Portfolio Value: ${final_status['portfolio_value']:,.2f}")
        print(f"   Total Return: {final_status['total_return']*100:.2f}%")
        print(f"   Total Trades: {final_status['total_trades']}")
        
    except Exception as e:
        print(f"Demo error: {e}")

def show_help():
    """Show help information"""
    help_text = """
🚀 Ultimate Trading Algorithm Help

MODES:
1. Live Trading - Connect to Alpaca for real trading (paper mode)
2. Simulation - Run with simulated orders and sample/real data
3. Backtest - Test strategies on historical data

FEATURES:
- Multiple technical analysis strategies
- Machine learning price prediction
- Advanced risk management
- Portfolio optimization
- Comprehensive performance tracking

DEPENDENCIES:
- yfinance: Real market data (optional)
- scikit-learn: Machine learning features (optional)  
- TA-Lib: Advanced technical indicators (optional)
- scipy: Statistical functions (optional)
- alpaca-trade-api: Live trading (optional)

The system works with minimal dependencies using sample data and
simplified indicators when full packages aren't available.

GETTING STARTED:
1. Run the main script
2. Choose your mode (simulation recommended for beginners)
3. Select symbols to trade
4. Configure parameters
5. Monitor performance

For live trading, you'll need Alpaca API credentials.
Paper trading is recommended for testing strategies.
    """
    print(help_text)

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--help', '-h', 'help']:
            show_help()
            sys.exit(0)
        elif arg == '--demo':
            interactive_demo()
            sys.exit(0)
        elif arg == '--version':
            print("Ultimate Trading Algorithm v2.0")
            sys.exit(0)
    
    # Run main application
    main()