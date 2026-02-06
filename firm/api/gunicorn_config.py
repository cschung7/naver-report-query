"""
Gunicorn configuration for SmartQuery API

Usage: gunicorn -c api/gunicorn_config.py api.simple_api:app
"""
import os

# Server socket
bind = "0.0.0.0:8000"

# Worker processes
# Use 1 worker with threads to share the ChromaDB client
workers = 1
threads = 4
worker_class = "gthread"
timeout = 600  # 10 minutes for first request (model loading)

# Logging
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = "info"

# Pre-load the app before forking workers
# This loads SmartQuery once, then workers share the memory
preload_app = True


def on_starting(server):
    """Called just before the master process is initialized."""
    print("=" * 60, flush=True)
    print("SmartQuery API Server (Gunicorn)", flush=True)
    print("=" * 60, flush=True)
    print("Pre-loading SmartQuery (~2 minutes)...", flush=True)


def when_ready(server):
    """Called just after the server is started."""
    print("Server is ready. Listening on: http://0.0.0.0:8000", flush=True)


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    print(f"Worker {worker.pid} spawned", flush=True)
