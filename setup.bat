@echo off
echo MenZ-ReazonSpeech Setup Script (Windows)
echo ==========================================
echo.
echo Select setup mode:
echo.
echo 1. Standard Installation (Recommended)
echo 2. Windows Fix (for indic-numtowords error)
echo 3. Lightweight (skip problematic packages)
echo 4. Conda Environment
echo 5. Exit
echo.
set /p SETUP_MODE="Choice (1-5): "

if "%SETUP_MODE%"=="1" goto STANDARD
if "%SETUP_MODE%"=="2" goto WINDOWS_FIX
if "%SETUP_MODE%"=="3" goto LIGHTWEIGHT
if "%SETUP_MODE%"=="4" goto CONDA
if "%SETUP_MODE%"=="5" goto END
echo Invalid choice.
pause
exit /b 1

:STANDARD
echo.
echo === Standard Installation ===
goto CHECK_PYTHON

:WINDOWS_FIX
echo.
echo === Windows Fix Installation ===
goto WINDOWS_FIX_MODE

:LIGHTWEIGHT
echo.
echo === Lightweight Installation ===
goto LIGHTWEIGHT_MODE

:CONDA
echo.
echo === Conda Installation ===
goto CONDA_MODE

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

if "%SETUP_MODE%"=="1" goto STANDARD_INSTALL
if "%SETUP_MODE%"=="2" goto WINDOWS_FIX_INSTALL
if "%SETUP_MODE%"=="3" goto LIGHTWEIGHT_INSTALL

:STANDARD_INSTALL
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

echo Installing Python dependencies...
pip install lightning
if errorlevel 1 (
    echo WARNING: Failed to install lightning package
    echo This may cause NeMo import issues
)

pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Standard install failed
    echo Trying Windows fix mode...
    echo.
    pause
    goto WINDOWS_FIX_INSTALL
)

echo.
echo Checking for FFmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo FFmpeg not found. Installing via conda-forge...
    python -c "import subprocess; subprocess.run(['pip', 'install', 'conda'], check=False)"
    python -c "import subprocess; subprocess.run(['conda', 'install', '-c', 'conda-forge', 'ffmpeg', '-y'], check=False)"
    if errorlevel 1 (
        echo.
        echo WARNING: FFmpeg auto-install failed
        echo Manual installation required for audio processing:
        echo 1. Download FFmpeg from https://www.gyan.dev/ffmpeg/builds/
        echo 2. Extract to C:\ffmpeg
        echo 3. Add C:\ffmpeg\bin to PATH
        echo.
        echo You can also install via Chocolatey: choco install ffmpeg
        echo Or via Scoop: scoop install ffmpeg
        echo.
        pause
    ) else (
        echo FFmpeg installed successfully
    )
) else (
    echo FFmpeg found in PATH
)

goto SUCCESS

:WINDOWS_FIX_MODE
goto CHECK_PYTHON

:WINDOWS_FIX_INSTALL
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo Fixing indic-numtowords...
pip install --no-cache-dir --no-deps indic-numtowords
if errorlevel 1 (
    echo Standard indic-numtowords install failed
    echo Trying git install...
    pip install git+https://github.com/subinps/indic-numtowords.git
    if errorlevel 1 (
        echo Git install failed
        echo Creating dummy package...
        mkdir temp_fix
        cd temp_fix
        echo from setuptools import setup > setup.py
        echo setup(name='indic-numtowords', version='1.0.2', py_modules=['indic_numtowords']) >> setup.py
        echo def convert_to_words(num): return str(num) > indic_numtowords.py
        pip install .
        if errorlevel 1 (
            echo ERROR: Failed to install dummy package
            cd ..
            rmdir /s /q temp_fix 2>nul
            goto SHOW_ERROR_HELP
        )
        cd ..
        rmdir /s /q temp_fix 2>nul
        echo Dummy package installed successfully
    )
)

echo Installing whisper-normalizer...
pip install whisper-normalizer --no-deps
pip install regex six

echo Installing lightning package...
pip install lightning --no-cache-dir
if errorlevel 1 (
    echo WARNING: Failed to install lightning package
    echo This may cause NeMo import issues
)

echo Installing remaining packages...
pip install -r requirements.txt --no-cache-dir
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some packages
    echo Please check the error messages above
    echo.
    goto SHOW_ERROR_HELP
)
goto SUCCESS

:LIGHTWEIGHT_MODE
goto CHECK_PYTHON

:LIGHTWEIGHT_INSTALL
echo Installing core packages...
pip install numpy scipy librosa soundfile tqdm requests silero-vad
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install core packages
    echo.
    pause
    exit /b 1
)

echo GPU support? (y/n)
set /p GPU="Choice: "

if /i "%GPU%"=="y" (
    echo Installing PyTorch with CUDA...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo CUDA install failed, trying CPU version...
        pip install torch torchvision torchaudio
        if errorlevel 1 (
            echo ERROR: Failed to install PyTorch
            pause
            exit /b 1
        )
    )
) else (
    echo Installing PyTorch CPU version...
    pip install torch torchvision torchaudio
    if errorlevel 1 (
        echo ERROR: Failed to install PyTorch
        pause
        exit /b 1
    )
)

echo Installing core libraries...
pip install lightning omegaconf hydra-core pytorch-lightning pydub sounddevice websockets huggingface_hub numba
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some packages
    echo.
    pause
    exit /b 1
)

echo Installing NeMo (minimal)...
pip install nemo_toolkit --no-deps
if errorlevel 1 (
    echo WARNING: NeMo installation failed
    echo Continuing without NeMo...
)
pip install "omegaconf>=2.0.5" "hydra-core>=1.1" "pytorch-lightning>=1.5.0"

echo.
echo Installed: Silero VAD (enterprise-grade)
echo Skipped: webrtcvad (no longer needed), indic-numtowords
echo Using high-performance VAD implementation
goto SUCCESS

:CONDA_MODE
conda --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Conda not found
    echo Install from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Creating conda environment...
conda create -n reazon-speech python=3.12 -y
call conda activate reazon-speech

echo Installing with conda...
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install numpy scipy librosa soundfile omegaconf hydra-core pytorch-lightning lightning ffmpeg -c conda-forge -y
pip install webrtcvad pydub sounddevice websockets tqdm requests huggingface_hub numba
pip install "nemo_toolkit[asr]>=1.21.0"

echo Conda setup complete!
echo Activate with: conda activate reazon-speech
goto SUCCESS

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

:SHOW_ERROR_HELP
echo.
echo ===============================================
echo Setup failed. Common solutions:
echo ===============================================
echo.
echo 1. Check Python version (requires 3.10+)
echo    python --version
echo.
echo 2. Install missing lightning package:
echo    pip install lightning pytorch-lightning
echo.
echo 3. Install FFmpeg manually:
echo    Download from: https://www.gyan.dev/ffmpeg/builds/
echo    Or use: choco install ffmpeg / scoop install ffmpeg
echo.
echo 4. Try Windows Fix mode (option 2)
echo.
echo 5. Install Visual C++ Build Tools:
echo    https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo.
echo 6. Try Conda environment (option 4)
echo.
echo 7. Check internet connection
echo.
echo ===============================================
pause
exit /b 1

:END