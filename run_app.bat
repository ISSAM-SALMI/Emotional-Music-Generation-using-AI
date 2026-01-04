@echo off
echo ==========================================
echo      Starting AI Music Generator
echo ==========================================

echo.
echo [1/2] Starting Backend Server...
start "Backend Server" cmd /k "python web_app/backend/main.py"

echo.
echo [2/2] Starting Frontend...
cd web_app\frontend
start "Frontend" cmd /k "npm run dev"

echo.
echo ==========================================
echo      App is running!
echo      Backend: http://localhost:8000
echo      Frontend: http://localhost:5173
echo ==========================================
pause
