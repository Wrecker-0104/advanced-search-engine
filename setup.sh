#!/bin/bash

echo ""
echo "================================================"
echo "   Advanced Search Engine - Quick Setup"
echo "================================================"
echo ""

echo "[1/4] Setting up Backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/4] Setting up Frontend..."
cd ../frontend

echo "Installing Node.js dependencies..."
npm install

echo ""
echo "[3/4] Configuration..."
cd ../backend

if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cat > .env << EOF
# Backend Environment Variables
DATABASE_URL=sqlite:///./search_engine.db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-super-secret-key-change-this-in-production
API_KEY_REQUIRED=true
VALID_API_KEYS=["sk-search-engine-2025-demo-key-123456"]
YAHOO_SEARCH_API_KEY=your-searchapi-io-key-here
USE_REAL_SEARCH=true
EOF
    echo ".env file created! Please edit it with your API keys."
fi

echo ""
echo "[4/4] Setup Complete!"
echo ""
echo "================================================"
echo "   🚀 Your Search Engine is Ready!"
echo "================================================"
echo ""
echo "To start your search engine:"
echo ""
echo "1. Start Backend (Terminal 1):"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python simple_main.py"
echo ""
echo "2. Start Frontend (Terminal 2):"
echo "   cd frontend"
echo "   npm start"
echo ""
echo "3. Open your browser:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8002"
echo "   API Docs: http://localhost:8002/docs"
echo ""
echo "================================================"
echo "   🔑 Demo API Key Available:"
echo "   sk-search-engine-2025-demo-key-123456"
echo "================================================"
echo ""
echo "For Yahoo Search API setup, visit:"
echo "https://www.searchapi.io/"
echo ""
echo "Happy Searching! 🔍"
echo ""
