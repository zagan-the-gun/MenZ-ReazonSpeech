#!/bin/bash

echo "MenZ-ReazonSpeech セットアップスクリプト (macOS/Linux)"
echo "================================================"

# Pythonのバージョンチェック
if ! command -v python3 &> /dev/null; then
    echo "エラー: Python3がインストールされていません"
    echo "https://www.python.org/downloads/ からPython 3.10以上をインストールしてください"
    exit 1
fi

echo "Pythonのバージョンを確認中..."
python3 --version

# Python 3.12以上かチェック
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ]; then
    echo "エラー: Python 3.10以上が必要です"
    exit 1
fi

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; then
    echo "エラー: Python 3.10以上が必要です"
    exit 1
fi

# 仮想環境の作成
echo ""
echo "仮想環境を作成中..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "エラー: 仮想環境の作成に失敗しました"
    exit 1
fi

# 仮想環境のアクティベート
echo "仮想環境をアクティベート中..."
source venv/bin/activate

# pip、setuptools、wheelのアップグレード
echo "pip、setuptools、wheelをアップグレード中..."
python -m pip install --upgrade pip setuptools wheel

# ビルド環境の準備
echo "ビルド環境を準備中..."
pip install cython wheel setuptools-scm

# 依存関係のインストール
echo ""
echo "依存関係をインストール中..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "エラー: 依存関係のインストールに失敗しました"
    exit 1
fi

# NeMoのインストール
echo ""
echo "NeMoをインストール中..."
pip install "nemo_toolkit[asr]>=1.21.0"
if [ $? -ne 0 ]; then
    echo "警告: NeMoのインストールに失敗しました"
    echo "CPU版を試行中..."
    pip install "nemo_toolkit[asr]>=1.21.0" --extra-index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "セットアップが完了しました！"
echo ""
echo "使用方法:"
echo "基本的な実行（マイク選択あり）:"
echo "  ./run.sh"
echo ""
echo "オプション付き実行:"
echo "  ./run.sh --show-level    # 音声レベル表示付き"
echo "  ./run.sh --verbose       # 詳細出力"
echo "  ./run.sh --info          # デバイス情報を表示"
echo ""
echo "直接実行:"
echo "  python -m reazon_speech.main"
echo "  python -m reazon_speech.cli audio.wav"
echo "" 