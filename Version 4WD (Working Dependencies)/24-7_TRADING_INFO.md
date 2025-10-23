# 24/7 Trading Mode

## Overview
The trading algorithm now runs continuously 24/7 but **only trades when the US stock market is open**.

## How It Works

### Market Hours (US Eastern Time)
- **Trading Hours**: 9:30 AM - 4:00 PM EST/EDT
- **Monday - Friday only** (closed weekends)

### Behavior

#### When Market is OPEN 🟢
- Algorithm runs normally
- Analyzes symbols every check interval (default: 5 minutes)
- Executes trades when signals are found
- Updates positions and monitors portfolio

#### When Market is CLOSED 🔴
- **Algorithm stays running** (doesn't exit)
- Displays market status messages:
  - "Market CLOSED (Pre-market)" - Before 9:30 AM
  - "Market CLOSED (After-hours)" - After 4:00 PM
  - "Market CLOSED (Weekend)" - Saturday/Sunday
- Shows countdown to market open
- Sleeps and checks periodically
- **No trades executed** until market opens

### Smart Sleep Logic
- If market opens soon (within check interval): Waits precisely until market opens
- If market is far from opening: Sleeps for check interval, then checks again
- Automatically resumes trading when market opens

## Example Output

### Market Closed
```
==================================================
🔴 Market CLOSED (Pre-market) - Opens in 0h 32m
💤 Sleeping for 300s (Market opens in ~0h 32m)
```

### Market Opening
```
==================================================
🟢 Market is OPEN - Trading active
Trading Iteration #1 - 2025-10-22 09:30:15
📊 Fetched data for 5/5 symbols
```

### Market Hours Detected Automatically
The algorithm automatically detects:
- Current day (weekday vs weekend)
- Current time in US Eastern timezone
- Whether market is open or closed
- Exact time until next market open

## Benefits

✅ **Run and forget** - Start it anytime, runs 24/7
✅ **No wasted resources** - Doesn't trade during closed hours
✅ **Auto-resume** - Automatically starts trading when market opens
✅ **Timezone aware** - Works regardless of your local timezone
✅ **Weekend safe** - Won't try to trade on weekends

## Configuration

The market hours checking uses:
- `pytz` library for timezone handling
- US/Eastern timezone for market hours
- Standard NYSE/NASDAQ trading hours

## Starting the Bot

Simply start as normal:
```bash
python main.py
# Select: 1 (Live Trading)
# Enter credentials
# Select watchlist and settings
```

The bot will:
1. Start immediately
2. Check if market is open
3. If closed: Wait and display status
4. If open: Start trading
5. Continue running 24/7

## Stopping the Bot

Press `Ctrl+C` at any time to stop gracefully.
The bot will:
- Cancel open orders
- Generate performance report
- Clean up resources
- Exit cleanly

## Notes

- The algorithm respects **regular market hours only** (no pre-market or after-hours trading)
- This ensures liquidity and better execution prices
- Weekend and holiday closures are detected automatically
- You can leave it running indefinitely - it will handle market open/close cycles

## US Market Holidays

The current implementation does not account for US market holidays (e.g., Thanksgiving, Christmas). On these days, the algorithm will wait until the next regular trading day.

Common US market holidays:
- New Year's Day
- Martin Luther King Jr. Day
- Presidents' Day
- Good Friday
- Memorial Day
- Independence Day (July 4th)
- Labor Day
- Thanksgiving
- Christmas

On these days, the algorithm will show "Market CLOSED" and wait until the next trading day.
