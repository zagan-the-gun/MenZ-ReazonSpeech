@echo off
echo MenZ-ReazonSpeech セットアップスクリプト (Windows)
echo ================================================

REM Pythonのバージョンチェック
python --version >nul 2>&1
if errorlevel 1 (
    echo エラー: Pythonがインストールされていません
    echo https://www.python.org/downloads/ からPython 3.12以上をインストールしてください
    pause
    exit /b 1
)

echo Pythonのバージョンを確認中...
python --version

REM Python 3.12以上かチェック
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
for /f "tokens=1 delims=." %%i in ("%PYTHON_VERSION%") do set PYTHON_MAJOR=%%i
for /f "tokens=2 delims=." %%i in ("%PYTHON_VERSION%") do set PYTHON_MINOR=%%i

if %PYTHON_MAJOR% LSS 3 (
    echo エラー: Python 3.12以上が必要です
    pause
    exit /b 1
)

if %PYTHON_MAJOR% EQU 3 (
    if %PYTHON_MINOR% LSS 12 (
        echo エラー: Python 3.12以上が必要です
        pause
        exit /b 1
    )
)

REM 仮想環境の作成
echo.
echo 仮想環境を作成中...
python -m venv venv
if errorlevel 1 (
    echo エラー: 仮想環境の作成に失敗しました
    pause
    exit /b 1
)

REM 仮想環境のアクティベート
echo 仮想環境をアクティベート中...
call venv\Scripts\activate.bat

REM pip、setuptools、wheelのアップグレード
echo pip、setuptools、wheelをアップグレード中...
python -m pip install --upgrade pip setuptools wheel

REM 依存関係のインストール
echo.
echo 依存関係をインストール中...
pip install -r requirements.txt
if errorlevel 1 (
    echo エラー: 依存関係のインストールに失敗しました
    pause
    exit /b 1
)

REM NeMoのインストール
echo.
echo NeMoをインストール中...
pip install "nemo_toolkit[asr]>=1.21.0"
if errorlevel 1 (
    echo 警告: NeMoのインストールに失敗しました
    echo CPU版を試行中...
    pip install "nemo_toolkit[asr]>=1.21.0" --extra-index-url https://download.pytorch.org/whl/cpu
)

echo.
echo セットアップが完了しました！
echo.
echo 使用方法:
echo 基本的な実行（マイク選択あり）:
echo   run.bat
echo.
echo オプション付き実行:
echo   run.bat --show-level    # 音声レベル表示付き
echo   run.bat --verbose       # 詳細出力
echo   run.bat --info          # デバイス情報を表示
echo.
echo 直接実行:
echo   python -m reazon_speech.main
echo   python -m reazon_speech.cli audio.wav
echo.
pause 