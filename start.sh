#!/bin/bash
# SmartQuery Multi-Process Startup Script for Railway
# Starts 4 backend Flask services then the Gateway on $PORT

APP_DIR="${APP_DIR:-/app}"
GATEWAY_PORT="${PORT:-8000}"

echo "=== SmartQuery Starting ==="
echo "App directory: $APP_DIR"
echo "Gateway port: $GATEWAY_PORT"
ls -la "$APP_DIR/" 2>/dev/null | head -15

# Start backend services in background using subshells
echo "Starting FirmAnalysis on :8080..."
(cd "$APP_DIR/firm" && python api/simple_api.py) &
FIRM_PID=$!

echo "Starting InvestmentStrategy on :8001..."
(cd "$APP_DIR/invest" && python api/simple_api.py) &
INVEST_PID=$!

echo "Starting Industry on :8002..."
(cd "$APP_DIR/industry" && python api/simple_api.py) &
INDUSTRY_PID=$!

echo "Starting EconAnalysis on :8003..."
(cd "$APP_DIR/econ" && python api/simple_api.py) &
ECON_PID=$!

# Wait for backends to start
echo "Waiting for backends to initialize..."
sleep 10

# Check backend health (non-fatal)
for port in 8080 8001 8002 8003; do
    if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "  Backend on :$port is healthy"
    else
        echo "  WARNING: Backend on :$port not responding yet"
    fi
done

# Start Gateway (foreground - Railway watches this process)
echo "Starting Gateway on :$GATEWAY_PORT..."
cd "$APP_DIR/gateway" && PORT=$GATEWAY_PORT exec python api/gateway.py
