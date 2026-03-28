@echo off
echo ==========================================
echo Starting Video Anomaly Detection System
echo ==========================================

start cmd /k "cd backend && python main.py"
start cmd /k "cd frontend && npm run dev"

echo System handles are opening in new windows.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo ==========================================
pause
