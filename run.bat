@echo off
echo MenZ-ReazonSpeech リアルタイム音声認識
echo ========================================

REM 仮想環境のアクティベート
if not exist "venv" (
    echo 仮想環境が見つかりません
    echo setup.batを実行してセットアップしてください
    pause
    exit /b 1
)

echo 仮想環境をアクティベート中...
call venv\Scripts\activate.bat

REM リアルタイム認識を開始
echo リアルタイム音声認識を開始します...
echo Ctrl+C で終了
echo.
python -m reazon_speech.main %* 