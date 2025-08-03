@echo off
chcp 65001 >nul
echo MenZ-ReazonSpeech Setup Script (Windows)
echo ==========================================

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed
    echo Please install Python 3.12 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Checking Python version...
python --version

REM Check if Python 3.12 or higher
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
for /f "tokens=1 delims=." %%i in ("%PYTHON_VERSION%") do set PYTHON_MAJOR=%%i
for /f "tokens=2 delims=." %%i in ("%PYTHON_VERSION%") do set PYTHON_MINOR=%%i

if %PYTHON_MAJOR% LSS 3 (
    echo ERROR: Python 3.12 or higher is required
    pause
    exit /b 1
)

if %PYTHON_MAJOR% EQU 3 (
    if %PYTHON_MINOR% LSS 12 (
        echo ERROR: Python 3.12 or higher is required
        pause
        exit /b 1
    )
)

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip, setuptools, wheel
echo Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel

REM Ask user for GPU support
echo.
echo Do you want to install GPU (CUDA) support? (y/n)
set /p GPU_SUPPORT="Enter choice: "

if /i "%GPU_SUPPORT%"=="y" (
    echo Installing GPU version with CUDA support...
    echo Installing PyTorch with CUDA...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo WARNING: Failed to install CUDA version, falling back to CPU
        goto CPU_INSTALL
    )
    
    echo Installing other dependencies...
    pip install numpy scipy librosa soundfile omegaconf hydra-core pytorch-lightning webrtcvad pydub sounddevice websockets tqdm requests huggingface_hub numba pytest black flake8
    
    echo Installing NeMo with GPU support...
    pip install "nemo_toolkit[asr]>=1.21.0"
    if errorlevel 1 (
        echo WARNING: Failed to install NeMo with GPU support
        echo Trying CPU version...
        pip install "nemo_toolkit[asr]>=1.21.0" --extra-index-url https://download.pytorch.org/whl/cpu
    )
) else (
    :CPU_INSTALL
    echo Installing CPU version...
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    
    echo Installing NeMo...
    pip install "nemo_toolkit[asr]>=1.21.0"
    if errorlevel 1 (
        echo WARNING: Failed to install NeMo
        echo Trying CPU version...
        pip install "nemo_toolkit[asr]>=1.21.0" --extra-index-url https://download.pytorch.org/whl/cpu
    )
)

echo.
echo Setup completed successfully!
echo.
echo Usage:
echo Basic execution (with microphone selection):
echo   run.bat
echo.
echo Execution with options:
echo   run.bat --show-level    # Show audio level
echo   run.bat --verbose       # Verbose output
echo   run.bat --info          # Show device info
echo.
echo Direct execution:
echo   python -m reazon_speech.main
echo   python -m reazon_speech.cli audio.wav
echo.
pause 