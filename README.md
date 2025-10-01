# MenZ-ReazonSpeech

NeMoを使用してReazonSpeechを実行するための環境です。WindowsとMacの両方で動作します。

## 概要

このプロジェクトは、NVIDIA NeMoフレームワークを使用してReazonSpeechモデルを実行するための環境を提供します。参考プロジェクトとして以下のリポジトリを参考にしています：

- [YukariWhisper](https://github.com/tyapa0/YukariWhisper)
- [MenZ-translation](https://github.com/zagan-the-gun/MenZ-translation)

## 新機能：WebSocket字幕送信

ゆかりねっとやゆかコネNEOなどのアプリケーションにリアルタイムで音声認識結果を送信できます。

## 新機能：WebSocket音声受信（Unity→Python）

Unity から 16kHz/モノラルの PCM16LE を WebSocket 経由で本ツールに送ることで、同一PC上で低遅延 ASR を実現します。

### プロトコル仕様（簡易）
- **URL**: `ws://localhost:60001`
- **サブプロトコル**: `pcm16.v1`
- **フレーム**: 1 WebSocketバイナリ = 1 チャンク
  - 形式: PCM16LE（16kHz / モノ）
  - 推奨チャンク: 20–30ms（例: 16kHzなら 320–480 サンプル → 640–960 バイト）
- **圧縮**: なし（permessage-deflate 無効推奨）
- **任意テキスト制御（JSON）**:
  - 送信: `{ "type": "hello", "format":"pcm16", "sample_rate":16000, "channels":1, "chunk_samples":480 }`
  - 応答: `{ "type": "ok", "version": "pcm16.v1", "sample_rate":16000, "channels":1 }`

#### 制御メッセージ: flush
- 用途: クライアントが「発話終了」と判断したタイミングで即時確定させる
- 送信（テキストJSON）:
  ```json
  {"type":"flush"}
  ```
- 挙動:
  - 受信時点までにバッファされた音声をその場で確定し、ASRを実行します
  - 末尾パディング（post_speech_padding）は付与しません
  - 最小長チェックは行いません（極短でも確定）
  - バッファが空の場合は何もしません
- 注意:
  - flushを送らない場合は、無音継続が `pause_threshold` を超えた時点で自動的に確定します（従来動作）

### 設定（config.ini）
```ini
[audio_ws]
enabled = false      ; 受信を有効にする場合は true
host = localhost     ; 同一PC運用のため localhost を推奨
port = 60001         ; Unity は ws://localhost:60001 に接続
```

### 起動方法（受信サーバ）
```bash
python -m reazon_speech.main --audio-ws
# 必要に応じて上書き
# python -m reazon_speech.main --audio-ws --audio-ws-port 60001 --audio-ws-host localhost
```

### Unity側の送信要件
- サンプルレート 16kHz、モノラル、PCM16LE
- 1フレーム=1チャンクでバイナリ送信（20–30ms推奨）
- サブプロトコル `pcm16.v1` を指定

備考: WebSocket字幕送信（認識結果の外部配信）とは別機能です。音声は `[audio_ws]`、テキストは `[websocket]` で設定します。

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

## GPU選択機能

複数のGPUがある環境で、最適なGPUを自動選択または手動指定できます。

### GPU情報の確認

```bash
# 詳細なGPU情報を表示
python -m reazon_speech.main --info

# GPU環境チェック（包括的）
python check_gpu.py

# Windows用バッチファイル
check_gpu.bat

# Unix系システム用シェルスクリプト
./check_gpu.sh
```

### 設定ファイルでのGPU選択

`config.ini`の`[inference]`セクションで設定：

```ini
[inference]
device = cuda
gpu_id = auto        # 自動選択（メモリ使用量が最も少ないGPU）
# gpu_id = 0           # GPU 0を指定
# gpu_id = 1           # GPU 1を指定
```

### コマンドラインでのGPU選択

```bash
# 特定のGPUを指定
python -m reazon_speech.main --device cuda --gpu-id 1

# GPU IDを指定
python -m reazon_speech.main --gpu-id auto
python -m reazon_speech.main --gpu-id 0
```

### GPU選択方法の詳細

- **auto**: 自動選択（デフォルト）
  - メモリ使用量が最も少ないGPUを選択
  - 複数GPU環境で推奨

- **0, 1, 2...**: 特定のGPU ID
  - 指定したGPU番号を直接使用
  - 確実に特定のGPUを使用したい場合

## GPU環境チェック

環境の準備状況を包括的に確認できます。

### チェック内容

- **Python環境**: バージョン、プラットフォーム
- **PyTorch環境**: バージョン、CUDA/MPS対応状況
- **NeMo環境**: 音声認識フレームワーク
- **音声処理ライブラリ**: sounddevice、librosa、numpy
- **システムリソース**: CPU、メモリ、ディスク容量
- **ReazonSpeechモデル**: モデルアクセステスト
- **設定ファイル**: 設定の検証
- **総合テスト**: 音声認識エンジンの初期化テスト

### 実行方法

```bash
# Python直接実行
python check_gpu.py

# Windows用バッチファイル
check_gpu.bat

# Unix系システム用シェルスクリプト
./check_gpu.sh
```

### 出力例

```
🔍 MenZ-ReazonSpeech 環境チェック
==================================================

🐍 Python環境:
  バージョン: 3.12.0
  プラットフォーム: macOS-14.0

🔥 PyTorch環境:
  PyTorch: 2.1.0
  CUDA利用可能: False
  MPS（Apple Silicon）: 利用可能

🎯 NeMo環境:
  NeMo: 1.21.0

🎵 音声処理ライブラリ:
  sounddevice: 0.4.6
  librosa: 0.10.1
  numpy: 1.24.3

💻 システムリソース:
  CPU: 8コア (使用率: 15.2%)
  メモリ: 16.0GB (使用率: 45.3%)
  利用可能ディスク容量: 256.5GB

🎤 ReazonSpeechモデルアクセステスト:
  モデル: reazon-research/reazonspeech-nemo-v2
  ✅ トークナイザー読み込み成功
  ✅ モデル読み込み成功

⚙️ 設定ファイルチェック:
  設定されたデバイス: mps
  設定されたGPU ID: auto
  モデル名: reazon-research/reazonspeech-nemo-v2

🧪 総合動作テスト:
  ✅ 設定ファイル作成成功
  ✅ 音声認識エンジン初期化成功
  ✅ 総合テスト成功

==================================================
✅ 環境チェック完了
==================================================
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
text_type = 2
speaker = zagan
```

### 対応形式

- **text_type = 0**: ゆかりねっと形式（プレーンテキスト）
- **text_type = 1**: ゆかコネNEO形式（JSON）
- **text_type = 2**: MCP形式（Model Context Protocol - JSON-RPC 2.0準拠）

### MCP形式について

MCP（Model Context Protocol）は、JSON-RPC 2.0に準拠した標準的なメッセージングプロトコルです。

**送信される形式:**
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subtitle",
  "params": {
    "text": "認識されたテキスト",
    "speaker": "zagan",
    "type": "subtitle",
    "language": "ja"
  }
}
```

**設定項目:**
- `speaker`: 話者識別子（例: zagan, menz, speaker_1）

**注意:** `language`は日本語専用モデルのため`"ja"`に固定されています。

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

# 特定のGPUを使用
python -m reazon_speech.main --device cuda --gpu-id 1

# GPU IDを指定
python -m reazon_speech.main --gpu-id auto

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

### Windowsバッチファイルの文字エンコーディング問題

Windowsで`check_gpu.bat`を実行した際に文字化けが発生する場合：

```bash
# 直接Pythonで実行
python check_gpu.py
```

バッチファイルはASCII文字のみを使用し、文字エンコーディング問題を回避するよう修正されています。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

プルリクエストやイシューの報告を歓迎します。 