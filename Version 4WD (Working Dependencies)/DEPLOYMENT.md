# Deployment Guide - DigitalOcean & Docker

This guide covers deploying the trading algorithm on DigitalOcean App Platform and using Docker.

## 🚀 Quick Start - One Command Run

### Local Docker Deployment

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env with your API credentials
nano .env

# 3. Run with Docker Compose
docker-compose up -d

# 4. View logs
docker-compose logs -f trading-bot
```

### Direct Python Run

```bash
# Set environment variables
export ALPACA_API_KEY='your_api_key'
export ALPACA_SECRET_KEY='your_secret_key'
export WATCHLIST='AAPL,MSFT,GOOGL,TSLA,NVDA'
export INITIAL_CAPITAL='100000'
export CHECK_INTERVAL='300'
export PAPER_TRADING='true'

# Run the script
bash run.sh
```

## ☁️ DigitalOcean App Platform Deployment

### Prerequisites

1. DigitalOcean account
2. GitHub repository connected to DigitalOcean
3. Alpaca API credentials

### Step 1: Prepare Repository

```bash
# Ensure all deployment files are in your repo
git add .do/app.yaml run.sh run_live.py Dockerfile
git commit -m "Add deployment configuration"
git push origin main
```

### Step 2: Create App on DigitalOcean

1. **Via Dashboard:**
   - Go to [DigitalOcean App Platform](https://cloud.digitalocean.com/apps)
   - Click "Create App"
   - Connect your GitHub repository
   - Select branch: `main`
   - DigitalOcean will detect the `.do/app.yaml` configuration

2. **Via doctl CLI:**
   ```bash
   # Install doctl
   brew install doctl  # macOS
   
   # Authenticate
   doctl auth init
   
   # Create app from spec
   doctl apps create --spec .do/app.yaml
   ```

### Step 3: Set Environment Variables

In the DigitalOcean dashboard:

1. Go to your app → Settings → App-Level Environment Variables
2. Add the following **SECRET** variables:
   - `ALPACA_API_KEY` = Your Alpaca API Key
   - `ALPACA_SECRET_KEY` = Your Alpaca Secret Key

3. Add the following regular variables (optional):
   - `WATCHLIST` = `AAPL,MSFT,GOOGL,TSLA,NVDA`
   - `INITIAL_CAPITAL` = `100000`
   - `CHECK_INTERVAL` = `300`
   - `PAPER_TRADING` = `true`

### Step 4: Deploy

```bash
# Manual deployment trigger
doctl apps create-deployment <app-id>

# Or push to trigger auto-deployment
git push origin main
```

### Step 5: Monitor

```bash
# View logs
doctl apps logs <app-id> --follow

# Check status
doctl apps get <app-id>
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t trading-algorithm .

# Run with environment variables
docker run -d \
  --name trading-bot \
  --restart unless-stopped \
  -e ALPACA_API_KEY='your_key' \
  -e ALPACA_SECRET_KEY='your_secret' \
  -e WATCHLIST='AAPL,MSFT,GOOGL,TSLA,NVDA' \
  -e INITIAL_CAPITAL='100000' \
  -e CHECK_INTERVAL='300' \
  -e PAPER_TRADING='true' \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/results:/app/results \
  trading-algorithm

# View logs
docker logs -f trading-bot

# Stop
docker stop trading-bot
```

### Using Docker Compose (Recommended)

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ALPACA_API_KEY` | ✅ Yes | - | Alpaca API Key |
| `ALPACA_SECRET_KEY` | ✅ Yes | - | Alpaca Secret Key |
| `WATCHLIST` | No | AAPL,MSFT,GOOGL,TSLA,NVDA | Comma-separated symbols |
| `INITIAL_CAPITAL` | No | 100000 | Starting capital amount |
| `CHECK_INTERVAL` | No | 300 | Check interval in seconds |
| `PAPER_TRADING` | No | true | Use paper trading (true/false) |

### Command Line Arguments

```bash
python3 run_live.py \
  --api-key YOUR_KEY \
  --secret-key YOUR_SECRET \
  --symbols AAPL,MSFT,GOOGL \
  --capital 100000 \
  --interval 300 \
  --paper true
```

## 📊 Monitoring & Logs

### View Logs

**Docker:**
```bash
docker-compose logs -f trading-bot
```

**DigitalOcean:**
```bash
doctl apps logs <app-id> --follow
```

**Local Files:**
```bash
tail -f logs/trading_bot_*.log
```

### Performance Reports

Results are saved in the `results/live/` directory:
- `performance_report_TIMESTAMP.txt` - Detailed performance metrics
- Trades are logged in real-time to console and log files

### Health Checks

The application logs status every iteration:
- Portfolio value
- Active positions
- Signals generated/executed
- Win rate and performance metrics

## 🛑 Stopping the Bot

### Docker
```bash
docker-compose down
# or
docker stop trading-bot
```

### DigitalOcean
```bash
# Pause app
doctl apps update <app-id> --spec .do/app.yaml

# Or delete app
doctl apps delete <app-id>
```

### Manual
Press `Ctrl+C` if running in foreground

## ⚠️ Important Notes

### Paper Trading vs Live Trading

**Always start with Paper Trading:**
```bash
PAPER_TRADING=true  # Safe - No real money
```

**For Live Trading (Real Money):**
```bash
PAPER_TRADING=false  # DANGER - Uses real money!
```

### Security Best Practices

1. **Never commit API keys to Git**
   - Use `.env` files (already in `.gitignore`)
   - Use DigitalOcean's secret environment variables

2. **Restrict API permissions**
   - Use Alpaca paper trading keys for testing
   - Limit live trading key permissions

3. **Monitor regularly**
   - Set up alerts for unusual activity
   - Check logs daily
   - Review performance reports

### Resource Requirements

**Minimum:**
- CPU: 1 core
- RAM: 512MB
- Storage: 1GB

**Recommended:**
- CPU: 2 cores
- RAM: 1GB
- Storage: 5GB (for logs and historical data)

### Cost Estimation

**DigitalOcean App Platform:**
- Basic (512MB RAM): $5/month
- Professional (1GB RAM): $12/month

**Alternatives:**
- AWS EC2 t2.micro: ~$8-10/month
- Google Cloud Run: Pay per use
- Heroku: $7/month (basic)

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Rebuild with dependencies
docker-compose build --no-cache
```

**2. API Connection Failed**
```bash
# Check credentials
echo $ALPACA_API_KEY
echo $ALPACA_SECRET_KEY

# Test API connection
python3 -c "from alpaca_trade_api import REST; api = REST('KEY', 'SECRET', paper=True); print(api.get_account())"
```

**3. Container Keeps Restarting**
```bash
# Check logs
docker-compose logs trading-bot

# Run in foreground for debugging
docker-compose up
```

**4. No Trades Being Executed**
- This is normal! The algorithm waits for high-confidence signals
- Market must be open for live data
- Check confidence threshold in `config.py`

### Debug Mode

```bash
# Run with verbose output
docker-compose logs -f trading-bot

# Or set debug logging
# Edit config.py: logging.basicConfig(level=logging.DEBUG)
```

## 📚 Additional Resources

- [DigitalOcean App Platform Docs](https://docs.digitalocean.com/products/app-platform/)
- [Docker Documentation](https://docs.docker.com/)
- [Alpaca API Docs](https://alpaca.markets/docs/)
- Main README: [README.md](README.md)
- Quick Start: [QUICK_START.md](QUICK_START.md)

## 🔄 Updating the App

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose up -d --build

# Or on DigitalOcean (auto-deploys on push)
git push origin main
```

## 💡 Tips for Production

1. **Start small**: Begin with paper trading and small capital
2. **Monitor actively**: Check logs and reports daily
3. **Set limits**: Use the risk management settings in `config.py`
4. **Backup data**: Regularly backup logs and results directories
5. **Test updates**: Always test changes in paper trading first
6. **Stay informed**: Monitor market conditions and algorithm performance

---

**Ready to deploy?** Start with paper trading on Docker locally, then move to DigitalOcean when comfortable! 🚀
