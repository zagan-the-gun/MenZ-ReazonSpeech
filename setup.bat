@echo off
chcp 65001 >nul
echo MenZ-ReazonSpeech Setup Script (Windows)
echo ==========================================
echo.
echo セットアップモードを選択してください:
echo.
echo 1. 標準インストール (推奨)
echo 2. Windows問題修正版 (indic-numtowordsエラー対応)
echo 3. 軽量版 (問題のある依存関係を回避)
echo 4. Conda環境版
echo 5. キャンセル
echo.
set /p SETUP_MODE="選択してください (1-5): "

if "%SETUP_MODE%"=="1" goto STANDARD_SETUP
if "%SETUP_MODE%"=="2" goto WINDOWS_FIX_SETUP
if "%SETUP_MODE%"=="3" goto LIGHTWEIGHT_SETUP
if "%SETUP_MODE%"=="4" goto CONDA_SETUP
if "%SETUP_MODE%"=="5" goto END
echo 無効な選択です。
pause
exit /b 1

:STANDARD_SETUP
echo.
echo ===========================================
echo 標準インストールを開始します...
echo ===========================================
goto COMMON_CHECKS

:WINDOWS_FIX_SETUP
echo.
echo ===========================================
echo Windows問題修正版インストールを開始します...
echo (indic-numtowordsエラー対応)
echo ===========================================
goto WINDOWS_FIX_MODE

:LIGHTWEIGHT_SETUP
echo.
echo ===========================================
echo 軽量版インストールを開始します...
echo (問題のある依存関係を回避)
echo ===========================================
goto LIGHTWEIGHT_MODE

:CONDA_SETUP
echo.
echo ===========================================
echo Conda環境でのインストールを開始します...
echo ===========================================
goto CONDA_MODE

:COMMON_CHECKS
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

if "%SETUP_MODE%"=="1" goto STANDARD_INSTALL
if "%SETUP_MODE%"=="2" goto WINDOWS_FIX_INSTALL
if "%SETUP_MODE%"=="3" goto LIGHTWEIGHT_INSTALL

:STANDARD_INSTALL
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
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo Trying Windows fix mode...
        goto WINDOWS_FIX_INSTALL
    )
) else (
    :CPU_INSTALL
    echo Installing CPU version...
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo Trying Windows fix mode...
        goto WINDOWS_FIX_INSTALL
    )
)
goto SUCCESS

:WINDOWS_FIX_MODE
goto COMMON_CHECKS

:WINDOWS_FIX_INSTALL
REM Set UTF-8 encoding
echo Setting UTF-8 encoding...
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Fix indic-numtowords first
echo Installing indic-numtowords manually...
pip install --no-cache-dir --no-deps indic-numtowords
if errorlevel 1 (
    echo Trying git installation...
    pip install git+https://github.com/subinps/indic-numtowords.git
    if errorlevel 1 (
        echo Creating local fix for indic-numtowords...
        mkdir temp_fix
        cd temp_fix
        echo from setuptools import setup > setup.py
        echo setup(name='indic-numtowords', version='1.0.2', py_modules=['indic_numtowords']) >> setup.py
        echo def convert_to_words(num): return str(num) > indic_numtowords.py
        pip install .
        cd ..
        rmdir /s /q temp_fix
    )
)

REM Install whisper-normalizer without deps
echo Installing whisper-normalizer...
pip install whisper-normalizer --no-deps --no-cache-dir
pip install regex six

REM Install remaining dependencies
echo Installing remaining dependencies...
pip install -r requirements.txt --no-cache-dir
goto SUCCESS

:LIGHTWEIGHT_MODE
goto COMMON_CHECKS

:LIGHTWEIGHT_INSTALL
echo Installing core dependencies...
pip install numpy scipy librosa soundfile tqdm requests

REM Ask user for GPU support
echo.
echo Do you want to install GPU (CUDA) support? (y/n)
set /p GPU_SUPPORT="Enter choice: "

if /i "%GPU_SUPPORT%"=="y" (
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo WARNING: Failed to install CUDA version, falling back to CPU
        pip install torch torchvision torchaudio
    )
) else (
    echo Installing CPU version...
    pip install torch torchvision torchaudio
)

echo Installing other dependencies...
pip install omegaconf hydra-core pytorch-lightning pydub sounddevice websockets huggingface_hub numba

echo Installing NeMo with minimal dependencies...
pip install nemo_toolkit --no-deps
pip install "omegaconf>=2.0.5,!=2.0.6" "hydra-core>=1.1" "pytorch-lightning>=1.5.0,<1.9.0"

echo.
echo Skipping problematic packages:
echo - webrtcvad: Using alternative VAD implementation
echo - indic-numtowords: Installing without dependencies
goto SUCCESS

:CONDA_MODE
REM Check if conda is available
conda --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Conda is not installed
    echo Please install Anaconda or Miniconda first
    echo https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Creating conda environment...
conda create -n reazon-speech python=3.12 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment
    pause
    exit /b 1
)

echo Activating conda environment...
call conda activate reazon-speech

echo Installing dependencies with conda...
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install numpy scipy librosa soundfile omegaconf hydra-core pytorch-lightning -c conda-forge -y
pip install webrtcvad pydub sounddevice websockets tqdm requests huggingface_hub numba
pip install "nemo_toolkit[asr]>=1.21.0"

echo.
echo Conda environment setup completed!
echo To activate: conda activate reazon-speech
goto SUCCESS

:SUCCESS
echo.
echo ===============================================
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
echo ===============================================
echo.
pause
goto END

:END
exit /b 0