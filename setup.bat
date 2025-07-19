@echo off
echo.
echo ================================================
echo   Advanced Search Engine - Quick Setup
echo ================================================
echo.

echo [1/4] Setting up Backend...
cd backend
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo [2/4] Setting up Frontend...
cd ..\frontend

echo Installing Node.js dependencies...
npm install

echo.
echo [3/4] Configuration...
cd ..\backend
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env 2>nul || (
        echo # Backend Environment Variables > .env
        echo DATABASE_URL=sqlite:///./search_engine.db >> .env
        echo REDIS_URL=redis://localhost:6379 >> .env
        echo SECRET_KEY=your-super-secret-key-change-this-in-production >> .env
        echo API_KEY_REQUIRED=true >> .env
        echo VALID_API_KEYS=["sk-search-engine-2025-demo-key-123456"] >> .env
        echo YAHOO_SEARCH_API_KEY=your-searchapi-io-key-here >> .env
        echo USE_REAL_SEARCH=true >> .env
    )
    echo .env file created! Please edit it with your API keys.
)

echo.
echo [4/4] Setup Complete!
echo.
echo ================================================
echo   🚀 Your Search Engine is Ready!
echo ================================================
echo.
echo To start your search engine:
echo.
echo 1. Start Backend (Terminal 1):
echo    cd backend
echo    venv\Scripts\activate
echo    python simple_main.py
echo.
echo 2. Start Frontend (Terminal 2):
echo    cd frontend  
echo    npm start
echo.
echo 3. Open your browser:
echo    Frontend: http://localhost:3000
echo    Backend API: http://localhost:8002
echo    API Docs: http://localhost:8002/docs
echo.
echo ================================================
echo   🔑 Demo API Key Available:
echo   sk-search-engine-2025-demo-key-123456
echo ================================================
echo.
echo For Yahoo Search API setup, visit:
echo https://www.searchapi.io/
echo.
echo Happy Searching! 🔍
echo.
pause
