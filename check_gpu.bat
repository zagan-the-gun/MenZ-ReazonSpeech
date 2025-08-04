@echo off
:: MenZ-ReazonSpeech GPU Environment Check
setlocal

title MenZ-ReazonSpeech GPU Check

echo.
echo ==========================================
echo MenZ-ReazonSpeech GPU Environment Check
echo ==========================================
echo.

:: Check virtual environment
if not exist "venv" (
    echo [WARNING] Virtual environment not found.
    echo Running basic check only.
    echo For complete check, run setup.bat first.
    echo.
    
    where python >nul 2>nul
    if %errorlevel% == 0 (
        echo [INFO] Running environment check...
        python check_gpu.py
    ) else (
        echo [ERROR] Python not found.
        echo Please install Python and try again.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Virtual environment found.
    echo [INFO] Activating virtual environment...
    
    call venv\Scripts\activate.bat
    
    echo [INFO] Running GPU environment check...
    python check_gpu.py
    
    echo.
    echo [INFO] Deactivating virtual environment
    call deactivate
)

echo.
echo ==========================================
echo [SUCCESS] Check Complete
echo ==========================================
echo.
pause 