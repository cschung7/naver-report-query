#!/bin/bash
# SmartQuery Multi-Process Startup Script for Railway
# Starts 4 backend Flask services then the Gateway on $PORT

set -e

APP_DIR="${APP_DIR:-/app}"
GATEWAY_PORT="${PORT:-8000}"

echo "=== SmartQuery Starting ==="
echo "App directory: $APP_DIR"
echo "Gateway port: $GATEWAY_PORT"

# Start backend services in background
echo "Starting FirmAnalysis on :8080..."
cd "$APP_DIR/firm" && python api/simple_api.py &
FIRM_PID=$!

echo "Starting InvestmentStrategy on :8001..."
cd "$APP_DIR/invest" && python api/simple_api.py &
INVEST_PID=$!

echo "Starting Industry on :8002..."
cd "$APP_DIR/industry" && python api/simple_api.py &
INDUSTRY_PID=$!

echo "Starting EconAnalysis on :8003..."
cd "$APP_DIR/econ" && python api/simple_api.py &
ECON_PID=$!

# Wait for backends to start
echo "Waiting for backends to initialize..."
sleep 15

# Check backend health
for port in 8080 8001 8002 8003; do
    if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "  Backend on :$port is healthy"
    else
        echo "  WARNING: Backend on :$port not responding yet"
    fi
done

# Start Gateway (foreground - Railway watches this process)
echo "Starting Gateway on :$GATEWAY_PORT..."
cd "$APP_DIR/gateway" && PORT=$GATEWAY_PORT python api/gateway.py

# If gateway exits, kill all backends
echo "Gateway exited, shutting down..."
kill $FIRM_PID $INVEST_PID $INDUSTRY_PID $ECON_PID 2>/dev/null
wait
