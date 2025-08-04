#!/bin/bash
# MenZ-ReazonSpeech GPU環境チェック Unix実行スクリプト
#
# このスクリプトはUnix系システム（macOS/Linux）でGPU環境をチェックします。
#

echo "=========================================="
echo "🔍 MenZ-ReazonSpeech GPU環境チェック"
echo "=========================================="
echo

# 仮想環境の確認とアクティベート
if [ ! -d "venv" ]; then
    echo "⚠️ 仮想環境が見つかりません。"
    echo "基本的なチェックのみ実行します。"
    echo "完全なチェックには setup.sh を実行してセットアップを完了してください。"
    echo
    
    # 仮想環境なしで実行
    if command -v python3 &> /dev/null; then
        echo "Python3で環境チェックを実行中..."
        python3 check_gpu.py
    elif command -v python &> /dev/null; then
        echo "Pythonで環境チェックを実行中..."
        python check_gpu.py
    else
        echo "❌ Pythonが見つかりません。"
        echo "Pythonをインストールしてから再実行してください。"
        exit 1
    fi
else
    echo "✅ 仮想環境が見つかりました。"
    echo "仮想環境をアクティベートしています..."
    
    # 仮想環境のアクティベート
    source venv/bin/activate
    
    echo "GPU環境チェックを実行中..."
    python check_gpu.py
    
    echo
    echo "🔧 仮想環境を非アクティベート"
    deactivate
fi

echo
echo "=========================================="
echo "✅ GPU環境チェック完了"
echo "=========================================="
echo 