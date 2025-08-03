@echo off
chcp 65001 >nul
echo MenZ-ReazonSpeech Real-time Speech Recognition
echo ===============================================

REM Activate virtual environment
if not exist "venv" (
    echo Virtual environment not found
    echo Please run setup.bat to setup the environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Start real-time recognition
echo Starting real-time speech recognition...
echo Press Ctrl+C to exit
echo.
python -m reazon_speech.main %* 