#!/bin/bash

echo "MenZ-ReazonSpeech リアルタイム音声認識"
echo "========================================"

# 仮想環境のアクティベート
if [ ! -d "venv" ]; then
    echo "仮想環境が見つかりません"
    echo "setup.shを実行してセットアップしてください"
    exit 1
fi

echo "仮想環境をアクティベート中..."
source venv/bin/activate

# リアルタイム認識を開始
echo "リアルタイム音声認識を開始します..."
echo "Ctrl+C で終了"
echo ""
python -m reazon_speech.main "$@" 