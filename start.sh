#!/bin/bash
# SmartQuery Startup Script for Railway
# Runs FirmAnalysis service directly on $PORT (includes all core features)

APP_DIR="${APP_DIR:-/app}"

echo "=== SmartQuery Starting ==="
echo "App directory: $APP_DIR"
echo "PORT: ${PORT:-8080}"

cd "$APP_DIR/firm" && exec python api/simple_api.py
