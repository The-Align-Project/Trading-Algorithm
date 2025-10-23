#!/bin/bash
# Script to enable aggressive testing mode for faster trade execution

echo "🔥 Enabling AGGRESSIVE TESTING MODE..."
echo ""
echo "⚠️  WARNING: This mode is for TESTING ONLY!"
echo "   - Lower confidence threshold (30% vs 60%)"
echo "   - More trades per iteration (5 vs 2)"
echo "   - More relaxed signal requirements"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "❌ Cancelled"
    exit 1
fi

# Update config.py
sed -i '' 's/MIN_CONFIDENCE = 0.6  # Require 60% confidence for trades/MIN_CONFIDENCE = 0.3  # TESTING: Lowered for faster trades/' config.py
sed -i '' 's/MAX_EXECUTIONS_PER_ITERATION = 2  # Max 2 trades per iteration to prevent overtrading/MAX_EXECUTIONS_PER_ITERATION = 5  # TESTING: Increased for more action/' config.py

# Update strategies.py
sed -i '' 's/if buy_score >= 3:  # Need 3 out of 4 indicators bullish/if buy_score >= 2:  # TESTING: Lowered threshold/' strategies.py
sed -i '' 's/elif sell_score >= 2:  # Need 2 out of 3 indicators bearish/elif sell_score >= 1:  # TESTING: Lowered threshold/' strategies.py

echo ""
echo "✅ AGGRESSIVE MODE ENABLED!"
echo ""
echo "Changes made:"
echo "  - MIN_CONFIDENCE: 0.6 → 0.3 (30% confidence)"
echo "  - MAX_EXECUTIONS_PER_ITERATION: 2 → 5 trades per cycle"
echo "  - Buy threshold: 3 → 2 indicators (more relaxed)"
echo "  - Sell threshold: 2 → 1 indicator (more relaxed)"
echo ""
echo "⚠️  Remember to run './restore_settings.sh' when done testing!"
echo ""
