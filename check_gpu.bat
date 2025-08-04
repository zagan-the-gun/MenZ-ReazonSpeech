@echo off
:: MenZ-ReazonSpeech GPU環境チェック Windows実行スクリプト
::
:: このスクリプトはWindows環境でGPU環境をチェックします。
::

title MenZ-ReazonSpeech GPU環境チェック

echo.
echo ==========================================
echo 🔍 MenZ-ReazonSpeech GPU環境チェック
echo ==========================================
echo.

:: 仮想環境の確認とアクティベート
if not exist "venv" (
    echo ⚠️ 仮想環境が見つかりません。
    echo 基本的なチェックのみ実行します。
    echo 完全なチェックには setup.bat を実行してセットアップを完了してください。
    echo.
    
    :: 仮想環境なしで実行
    where python >nul 2>nul
    if %errorlevel% == 0 (
        echo Pythonで環境チェックを実行中...
        python check_gpu.py
    ) else (
        echo ❌ Pythonが見つかりません。
        echo Pythonをインストールしてから再実行してください。
        pause
        exit /b 1
    )
) else (
    echo ✅ 仮想環境が見つかりました。
    echo 仮想環境をアクティベートしています...
    
    :: 仮想環境のアクティベート
    call venv\Scripts\activate.bat
    
    echo GPU環境チェックを実行中...
    python check_gpu.py
    
    echo.
    echo 🔧 仮想環境を非アクティベート
    call deactivate
)

echo.
echo ==========================================
echo ✅ GPU環境チェック完了
echo ==========================================
echo.
pause 