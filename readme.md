# Start bot
python bot.py

# Start/Stop
curl -X POST http://localhost:8000/start
curl -X POST http://localhost:8000/stop

# Status (shows last trade, equity, current instrument/granularity)
curl http://localhost:8000/status

# Switch timeframe to M15
curl -X POST http://localhost:8000/config -H "Content-Type: application/json" -d '{"granularity":"M15"}'

# Switch back to M5
curl -X POST http://localhost:8000/config -H "Content-Type: application/json" -d '{"granularity":"M5"}'

# Change instrument
curl -X POST http://localhost:8000/config -H "Content-Type: application/json" -d '{"instrument":"GBP_USD"}'

# Last 100 trades (JSON)
curl http://localhost:8000/trades

# All trades as CSV (download)
curl http://localhost:8000/trades.csv -o trades.csv

# Recent equity snapshots
curl http://localhost:8000/equity

# Activate environment
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate

.venv\Scripts\deactivate




