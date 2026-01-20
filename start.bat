@echo off
echo Starting OptiExacta - AI-Powered People Analytics Platform
echo.
echo Starting Backend Server (FastAPI)...
start "OptiExacta Backend" cmd /k "python web_app.py"
timeout /t 3 /nobreak > nul

echo Starting Frontend Server (React + Vite)...
start "OptiExacta Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo OptiExacta is starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit...
pause > nul
