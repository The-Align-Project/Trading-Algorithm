#!/usr/bin/env python3
"""
Automated run script for live trading mode.
Designed for deployment on DigitalOcean or other cloud platforms.
"""

import argparse
import sys
import os
from config import setup_logging, print_dependency_status, get_market_status_message
from trading_engine import TradingEngine

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Trading Algorithm in Live Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  ALPACA_API_KEY        Alpaca API Key (required)
  ALPACA_SECRET_KEY     Alpaca Secret Key (required)
  WATCHLIST             Comma-separated symbols (default: AAPL,MSFT,GOOGL,TSLA,NVDA)
  INITIAL_CAPITAL       Starting capital (default: 100000)
  CHECK_INTERVAL        Check interval in seconds (default: 300)
  PAPER_TRADING         Use paper trading (default: true)

Example:
  python3 run_live.py --api-key YOUR_KEY --secret-key YOUR_SECRET --symbols AAPL,MSFT,GOOGL
        """
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.environ.get('ALPACA_API_KEY'),
        help='Alpaca API Key (or set ALPACA_API_KEY env var)'
    )
    
    parser.add_argument(
        '--secret-key',
        type=str,
        default=os.environ.get('ALPACA_SECRET_KEY'),
        help='Alpaca Secret Key (or set ALPACA_SECRET_KEY env var)'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        default=os.environ.get('WATCHLIST', 'AAPL,MSFT,GOOGL,TSLA,NVDA'),
        help='Comma-separated list of symbols to trade (default: AAPL,MSFT,GOOGL,TSLA,NVDA)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=float(os.environ.get('INITIAL_CAPITAL', '100000')),
        help='Initial capital amount (default: 100000)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=int(os.environ.get('CHECK_INTERVAL', '300')),
        help='Check interval in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--paper',
        type=str,
        default=os.environ.get('PAPER_TRADING', 'true'),
        help='Use paper trading: true/false (default: true)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Trading Algorithm v2.0'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for automated live trading."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    print("=" * 60)
    print("🚀 Ultimate Advanced Trading Algorithm v2.0 - LIVE MODE")
    print("=" * 60)
    print()
    
    # Validate API credentials
    if not args.api_key or not args.secret_key:
        print("❌ ERROR: Alpaca API credentials are required!")
        print()
        print("Set environment variables:")
        print("  export ALPACA_API_KEY='your_api_key'")
        print("  export ALPACA_SECRET_KEY='your_secret_key'")
        print()
        print("Or use command line arguments:")
        print("  --api-key YOUR_KEY --secret-key YOUR_SECRET")
        sys.exit(1)
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    if not symbols:
        print("❌ ERROR: No valid symbols provided!")
        sys.exit(1)
    
    # Convert paper trading string to boolean
    paper_trading = args.paper.lower() in ('true', 'yes', '1', 'y')
    
    # Print configuration
    print("📋 Configuration:")
    print(f"   Mode: {'Paper Trading' if paper_trading else 'LIVE TRADING'}")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Initial Capital: ${args.capital:,.2f}")
    print(f"   Check Interval: {args.interval}s")
    print()
    
    # Print dependency status
    print_dependency_status()
    print()
    
    # Print market status
    print(get_market_status_message())
    print()
    
    # Warning for live trading
    if not paper_trading:
        print("⚠️  WARNING: LIVE TRADING MODE ENABLED!")
        print("⚠️  Real money will be used for trades!")
        print("⚠️  Ensure you understand the risks before proceeding.")
        print()
    
    # Initialize trading engine
    try:
        engine = TradingEngine(
            api_key=args.api_key,
            secret_key=args.secret_key,
            paper=paper_trading,
            initial_capital=args.capital
        )
        
        print("✅ Trading Engine Initialized Successfully")
        print()
        print("🎯 Starting Live Trading Loop...")
        print("   Press Ctrl+C to stop")
        print("=" * 60)
        print()
        
        # Start trading
        engine.run_live_trading(
            symbols=symbols,
            check_interval=args.interval
        )
        
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("🛑 Trading stopped by user (Ctrl+C)")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Fatal Error: {e}")
        print("=" * 60)
        import logging
        logging.error(f"Fatal error in live trading: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        try:
            engine.stop_trading()
            print("✅ Trading engine stopped gracefully")
        except:
            pass

if __name__ == "__main__":
    main()
