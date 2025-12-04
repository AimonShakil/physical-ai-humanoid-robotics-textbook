#!/bin/bash
# Deployment script for Physical AI Textbook

set -e  # Exit on error

echo "🚀 Starting deployment process..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "   Please copy .env.example to .env and configure your credentials"
    exit 1
fi

echo "✓ Environment file found"

# Build frontend
echo "📦 Building frontend..."
cd docs
npm ci
npm run build
cd ..
echo "✓ Frontend built successfully"

# Install backend dependencies (if Python available)
if command -v python3 &> /dev/null; then
    echo "🐍 Installing backend dependencies..."
    cd backend
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install -r requirements.txt
    cd ..
    echo "✓ Backend dependencies installed"
else
    echo "⚠️  Python3 not found - skipping backend setup"
fi

# Deploy to Railway (if Railway CLI installed)
if command -v railway &> /dev/null; then
    echo "🚂 Deploying to Railway..."
    railway up
    echo "✓ Deployed to Railway successfully"
else
    echo "⚠️  Railway CLI not found - skipping Railway deployment"
    echo "   Install with: npm install -g @railway/cli"
fi

echo "✅ Deployment complete!"
