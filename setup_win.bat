@echo off
REM Setup script for Windows to prepare the backend for deployment

echo Setting up Python virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Environment setup complete.
echo To start the backend server, run: uvicorn main:app --reload --port 8001

REM Keep the window open
pause
