#!/bin/bash
# Local development script - runs frontend and backend concurrently

set -e

echo "🚀 Starting local development servers..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Copying .env.example to .env..."
    cp .env.example .env
    echo "   Please edit .env with your credentials"
fi

# Function to run frontend
run_frontend() {
    echo "📦 Starting Docusaurus frontend..."
    cd docs
    npm install
    npm start
}

# Function to run backend
run_backend() {
    echo "🐍 Starting FastAPI backend..."
    cd backend
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8000
}

# Run both in background
run_frontend &
FRONTEND_PID=$!

if command -v python3 &> /dev/null; then
    run_backend &
    BACKEND_PID=$!
else
    echo "⚠️  Python3 not found - backend not started"
    BACKEND_PID=""
fi

# Trap Ctrl+C to kill both processes
trap "echo '🛑 Stopping servers...'; kill $FRONTEND_PID $BACKEND_PID 2>/dev/null; exit" INT

echo "✅ Development servers running:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

# Wait for both processes
wait
