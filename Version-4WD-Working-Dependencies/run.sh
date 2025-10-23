#!/bin/bash
# All-in-one run script for DigitalOcean deployment
# This script starts the trading algorithm in live mode with environment variables

set -e  # Exit on error

echo "🚀 Starting Trading Algorithm in Live Mode..."
echo "================================================"

# Check for required environment variables
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
    echo "❌ ERROR: Required environment variables not set!"
    echo "Please set: ALPACA_API_KEY, ALPACA_SECRET_KEY"
    exit 1
fi

# Set default values for optional environment variables
WATCHLIST=${WATCHLIST:-"AAPL,MSFT,GOOGL,TSLA,NVDA"}
INITIAL_CAPITAL=${INITIAL_CAPITAL:-100000}
CHECK_INTERVAL=${CHECK_INTERVAL:-300}

echo "📋 Configuration:"
echo "   Watchlist: $WATCHLIST"
echo "   Initial Capital: \$$INITIAL_CAPITAL"
echo "   Check Interval: ${CHECK_INTERVAL}s"
echo "   Paper Trading: ${PAPER_TRADING:-true}"
echo ""

# Run the Python script with environment variables
python3 run_live.py \
    --api-key "$ALPACA_API_KEY" \
    --secret-key "$ALPACA_SECRET_KEY" \
    --symbols "$WATCHLIST" \
    --capital "$INITIAL_CAPITAL" \
    --interval "$CHECK_INTERVAL" \
    --paper "${PAPER_TRADING:-true}"
