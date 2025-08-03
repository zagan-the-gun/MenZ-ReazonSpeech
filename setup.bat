@echo off
echo MenZ-ReazonSpeech Setup Script (Windows)
echo ==========================================
echo.

:CHECK_PYTHON
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    echo Install Python 3.12+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python version:
python --version

echo.
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo.
    echo ERROR: Failed to create virtual environment
    echo Check if Python is properly installed
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate virtual environment
    echo.
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo.
    echo WARNING: Failed to upgrade pip
    echo Continuing with current version...
    echo.
)

echo.
echo GPU support? (y/n)
set /p GPU="Choice: "

if /i "%GPU%"=="y" (
    echo Installing PyTorch with CUDA...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo CUDA failed, using CPU version
        pip install torch torchvision torchaudio
    )
) else (
    echo Installing CPU version...
    pip install torch torchvision torchaudio
)

echo Preparing build environment...
pip install cython wheel setuptools-scm
if errorlevel 1 (
    echo WARNING: Failed to install build dependencies
)

echo Installing dependencies...
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Installation failed
    echo Retrying with dependency fixes...
    echo.
    pip install --no-cache-dir --no-deps "nemo_toolkit[asr]>=1.21.0"
    if errorlevel 1 (
        echo.
        echo ERROR: Installation failed
        echo Please check the error messages above
        echo.
        pause
        exit /b 1
    )
)

echo Verifying Silero VAD installation...
python -c "import torch; model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True); print('Silero VAD verified successfully')"
if errorlevel 1 (
    echo WARNING: Silero VAD verification failed
    echo Attempting to fix Silero VAD...
    pip uninstall -y silero-vad
    pip install --no-cache-dir silero-vad
    python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True, force_reload=True)"
)

:SUCCESS
echo.
echo ===============================================
echo Setup completed successfully!
echo ===============================================
echo.
echo Usage:
echo   run.bat                    # Basic execution
echo   run.bat --show-level       # Show audio level
echo   run.bat --verbose          # Verbose output
echo   run.bat --info             # Device info
echo.
echo Direct execution:
echo   python -m reazon_speech.main
echo   python -m reazon_speech.cli audio.wav
echo ===============================================
pause
goto END

:END