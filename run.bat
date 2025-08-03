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
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate virtual environment
    echo Please check if setup.bat completed successfully
    echo.
    pause
    exit /b 1
)

echo Checking Python environment...
python --version
if errorlevel 1 (
    echo.
    echo ERROR: Python not found in virtual environment
    echo Please run setup.bat again
    echo.
    pause
    exit /b 1
)

echo Checking reazon_speech module...
python -c "import reazon_speech; print('Module loaded successfully')" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: reazon_speech module not found or dependencies missing
    echo.
    echo Please run setup.bat to install dependencies
    echo.
    pause
    exit /b 1
)

REM Start real-time recognition
echo.
echo Starting real-time speech recognition...
echo Press Ctrl+C to exit
echo.
python -m reazon_speech.main %*
if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    echo Check the error messages above
    echo.
    echo Please run setup.bat to fix dependencies
    echo.
    pause
    exit /b 1
)

echo.
echo Application ended normally
pause 