@echo off
echo ==========================================
echo Setting up Lazada Crawler Environment...
echo ==========================================

cd /d "%~dp0"

echo Installing requirements...
pip install -r requirements.txt

echo.
echo ==========================================
echo Starting Lazada Crawler App...
echo ==========================================
echo Please make sure you have Chrome installed.
echo.

streamlit run app.py
pause
