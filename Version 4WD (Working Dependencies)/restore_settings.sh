#!/bin/bash
# Script to restore original conservative trading settings

echo "Restoring original conservative settings..."

# Restore config.py
sed -i '' 's/MIN_CONFIDENCE = 0.3  # TEMPORARY: Lowered for testing (was 0.6)/MIN_CONFIDENCE = 0.6/' config.py
sed -i '' 's/MAX_EXECUTIONS_PER_ITERATION = 5  # TEMPORARY: Increased for testing (was 2)/MAX_EXECUTIONS_PER_ITERATION = 2/' config.py

# Restore strategies.py
sed -i '' 's/# TEMPORARY: More aggressive for testing (was >= 3)//' strategies.py
sed -i '' 's/if buy_score >= 2:/if buy_score >= 3:/' strategies.py
sed -i '' 's/# TEMPORARY: More aggressive for testing (was >= 2)//' strategies.py
sed -i '' 's/elif sell_score >= 1:/elif sell_score >= 2:/' strategies.py

echo "✅ Original settings restored!"
echo ""
echo "Changes made:"
echo "  - MIN_CONFIDENCE: 0.3 → 0.6"
echo "  - MAX_EXECUTIONS_PER_ITERATION: 5 → 2"
echo "  - Buy threshold: 2 → 3"
echo "  - Sell threshold: 1 → 2"
