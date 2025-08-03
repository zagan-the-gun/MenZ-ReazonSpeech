# MenZ-ReazonSpeech

NeMoを使用してReazonSpeechを実行するための環境です。WindowsとMacの両方で動作します。

## 概要

このプロジェクトは、NVIDIA NeMoフレームワークを使用してReazonSpeechモデルを実行するための環境を提供します。参考プロジェクトとして以下のリポジトリを参考にしています：

- [YukariWhisper](https://github.com/tyapa0/YukariWhisper)
- [MenZ-translation](https://github.com/zagan-the-gun/MenZ-translation)

## 新機能：WebSocket字幕送信

ゆかりねっとやゆかコネNEOなどのアプリケーションにリアルタイムで音声認識結果を送信できます。

## 必要な環境

- Python 3.12（推奨）
- CUDA対応GPU（推奨）

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/your-username/MenZ-ReazonSpeech.git
cd MenZ-ReazonSpeech
```

### 2. 仮想環境の作成

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 依存関係のインストール

```bash
# まずpip、setuptools、wheelをアップグレード
pip install --upgrade pip setuptools wheel

# 依存関係のインストール
pip install -r requirements.txt
```

### 4. NeMoのインストール

```bash
# CUDA 11.8の場合
pip install "nemo_toolkit[asr]>=1.21.0"

# CPUのみの場合
pip install "nemo_toolkit[asr]>=1.21.0" --extra-index-url https://download.pytorch.org/whl/cpu
```

## 音声ライブラリについて

このプロジェクトでは、リアルタイム音声入出力に**sounddevice**を使用しています。sounddeviceは以下の利点があります：

- **簡単なインストール**: PortAudioなどの追加ライブラリのインストールが不要
- **NumPyとの統合**: NumPy配列で直接音声データを扱える
- **現代的なAPI**: より直感的で使いやすいインターフェース
- **クロスプラットフォーム**: Windows、macOS、Linuxで動作

## 設定ファイル

`config.ini`ファイルで詳細な設定が可能です。

### 設定ファイルの作成

```bash
python main.py --create-config
```

## WebSocket字幕送信の使用方法

### 基本的な使用方法

WebSocket機能は設定ファイル（`config.ini`）で制御します。設定後、通常通り起動するだけです：

```bash
# 設定ファイルの通りに起動（WebSocket設定も含む）
python main.py
```

### 設定ファイルでの設定

`config.ini`を編集してWebSocket設定を永続化できます：

```ini
[websocket]
enabled = true
port = 50000
host = localhost
text_type = 0
```

### 対応形式

- **text_type = 0**: ゆかりねっと形式（プレーンテキスト）
- **text_type = 1**: ゆかコネNEO形式（JSON）

### ゆかりねっとでの設定

1. ゆかりねっとを起動
2. 「設定」→「音声認識エンジン」→「サードパーティ製の音声認識エンジンを使用する」にチェック
3. 「認識結果待ち受けポート」を50000に設定（または指定したポート）
4. 「音声認識を開始」をクリック

## 実行例

```bash
# 詳細表示で実行（WebSocket設定はconfig.iniで制御）
python main.py --verbose

# 出力例：
# === MenZ-ReazonSpeech リアルタイム音声認識 ===
# サンプルレート: 16000
# VAD閾値: 0.3
# デバイス: cuda
# WebSocket送信: 有効 (localhost:50000)
# 送信形式: ゆかりねっと
# Ctrl+C で終了
```

### 従来の使い方

```bash
# デフォルト設定ファイルを作成
python -m reazon_speech.main --create-config

# カスタム設定ファイルを作成
python -m reazon_speech.main --create-config --config my_config.ini
```

### 設定項目

#### [model]
- `name`: 使用するモデル名
- `cache_dir`: モデルキャッシュディレクトリ

#### [audio]
- `sample_rate`: サンプルレート
- `chunk_length_s`: チャンク長（秒）
- `stride_length_s`: ストライド長（秒）

#### [inference]
- `batch_size`: バッチサイズ
- `device`: 使用デバイス（auto/cpu/cuda）

#### [vad]
- `threshold`: VAD閾値
- `min_speech_duration_ms`: 最小音声持続時間（ミリ秒）
- `max_speech_duration_s`: 最大音声持続時間（秒）

#### [speech_detection]
- `threshold`: 音声検出閾値
- `silence_frames`: 無音フレーム数
- `min_duration`: 最小音声持続時間（秒）
- `max_duration`: 最大音声持続時間（秒）

#### [output]
- `format`: 出力フォーマット（text/json/srt）
- `language`: 言語

#### [debug]
- `enabled`: デバッグ有効/無効
- `show_level`: 音声レベル表示有効/無効

### 設定ファイルの使用例

```bash
# デフォルト設定ファイルを使用
./run.sh

# カスタム設定ファイルを使用
python -m reazon_speech.main --config my_config.ini

# 設定ファイルとコマンドライン引数を併用
python -m reazon_speech.main --config my_config.ini --verbose
```

## 使用方法

### リアルタイム音声認識（推奨）

```bash
# 基本的なリアルタイム認識
python -m reazon_speech.main

# 特定のマイクデバイスを指定
python -m reazon_speech.main --device 0

# 詳細出力モード
python -m reazon_speech.main --verbose

# デバイス情報を表示
python -m reazon_speech.main --info

# カスタム設定
python -m reazon_speech.main --sample-rate 16000 --vad-threshold 0.5
```

### 音声ファイルの認識

```python
from reazon_speech import ReazonSpeechModel

# モデルの初期化
model = ReazonSpeechModel()

# 音声ファイルの認識
result = model.transcribe("path/to/audio.wav")
print(result)
```

### バッチ処理

```python
from reazon_speech import ReazonSpeechModel

model = ReazonSpeechModel()

# 複数の音声ファイルを処理
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = model.transcribe_batch(audio_files)

for audio_file, result in zip(audio_files, results):
    print(f"{audio_file}: {result}")
```

## ディレクトリ構造

```
MenZ-ReazonSpeech/
├── README.md
├── requirements.txt
├── setup.py
├── reazon_speech/
│   ├── __init__.py
│   ├── model.py
│   ├── utils.py
│   └── config.py
├── scripts/
│   ├── download_model.py
│   └── transcribe.py
├── tests/
│   └── test_model.py
└── examples/
    └── basic_usage.py
```

## トラブルシューティング

### CUDA関連のエラー

CUDAが利用できない場合、CPU版のPyTorchをインストールしてください：

```bash
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### メモリ不足エラー

大きな音声ファイルを処理する際にメモリ不足が発生する場合：

1. 音声ファイルを小さなチャンクに分割
2. バッチサイズを小さくする
3. GPUメモリをクリアする

### 音声デバイスの問題

音声デバイスが認識されない場合：

```bash
# 利用可能な音声デバイスを確認
python -c "import sounddevice as sd; print(sd.query_devices())"
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

プルリクエストやイシューの報告を歓迎します。 