#!/usr/bin/env python3
"""
MenZ-ReazonSpeech メイン実行ファイル
リアルタイム音声認識システム
"""

import sys
import argparse
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from reazon_speech.realtime import RealtimeTranscriber
from reazon_speech.config import ModelConfig
from reazon_speech.utils import get_device_info


def print_result(text: str):
    """認識結果を表示するコールバック"""
    print(f"認識結果: {text}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="MenZ-ReazonSpeech リアルタイム音声認識システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的なリアルタイム認識
  python main.py

  # 特定のマイクデバイスを指定
  python main.py --device 0

  # 詳細出力モード
  python main.py --verbose

  # デバイス情報を表示（利用可能なGPU一覧も表示）
  python main.py --info

  # GPU設定
  python main.py --device-type auto        # 自動選択（推奨）
  python main.py --device-type cuda        # デフォルトGPU使用
  python main.py --device-type cuda:0      # GPU 0を指定
  python main.py --device-type cuda:1      # GPU 1を指定
  python main.py --device-type cpu         # CPU使用

  # GPU番号指定（別の方法）
  python main.py --device-type cuda --gpu-id 1

  # カスタム設定
  python main.py --sample-rate 16000 --vad-threshold 0.5
  
  # WebSocket字幕送信
  python main.py --websocket --websocket-port 50000
  python main.py --websocket --text-type 1  # ゆかコネNEO形式
        """
    )
    
    parser.add_argument(
        "--device",
        type=int,
        help="マイクデバイス番号"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細出力"
    )
    
    parser.add_argument(
        "--show-level",
        action="store_true",
        help="音声レベルを表示"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="デバイス情報を表示"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        help="サンプルレート（設定ファイルから読み込み）"
    )
    
    parser.add_argument(
        "--max-speech-duration",
        type=float,
        default=float("inf"),
        help="最大音声持続時間（秒、デフォルト: 無制限）"
    )
    
    parser.add_argument(
        "--device-type",
        default="auto",
        help="使用デバイス（例: auto, cpu, cuda, cuda:0, cuda:1）"
    )
    
    parser.add_argument(
        "--gpu-id",
        type=int,
        help="使用するGPU番号（0, 1, 2など）。--device-type cudaと組み合わせて使用"
    )
    
    parser.add_argument(
        "--config",
        default="config.ini",
        help="設定ファイルのパス（デフォルト: config.ini）"
    )
    
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="デフォルト設定ファイルを作成"
    )
    
    # WebSocketテキスト送信（既存）
    parser.add_argument(
        "--websocket",
        action="store_true",
        help="WebSocket字幕送信を有効にする"
    )
    
    parser.add_argument(
        "--websocket-port",
        type=int,
        help="WebSocket送信先ポート（デフォルト: 50000）"
    )
    
    parser.add_argument(
        "--websocket-host",
        type=str,
        help="WebSocket送信先ホスト（デフォルト: localhost）"
    )
    
    parser.add_argument(
        "--text-type",
        type=int,
        choices=[0, 1],
        help="送信形式（0: ゆかりねっと, 1: ゆかコネNEO）"
    )
    
    # WebSocket音声受信（新規）
    parser.add_argument(
        "--audio-ws",
        action="store_true",
        help="WebSocketで音声を受信してASRする"
    )
    parser.add_argument(
        "--audio-ws-host",
        type=str,
        help="音声受信WSのバインドホスト（デフォルト: 0.0.0.0）"
    )
    parser.add_argument(
        "--audio-ws-port",
        type=int,
        help="音声受信WSのポート（デフォルト: 60060）"
    )

    args = parser.parse_args()
    
    # デフォルト設定ファイルの作成
    if args.create_config:
        config = ModelConfig()
        config.save_ini(args.config)
        print(f"設定ファイルを作成しました: {args.config}")
        return
    
    # デバイス情報の表示
    if args.info:
        from .utils import print_gpu_info
        print_gpu_info()
        return
    
    # 設定の読み込み
    try:
        config = ModelConfig.from_ini(args.config)
        print(f"設定ファイルを読み込みました: {args.config}")
    
    except Exception as e:
        print(f"設定ファイルの読み込みに失敗しました: {e}")
        print("デフォルト設定を使用します")
        config = ModelConfig()
    
    # コマンドライン引数で設定を上書き
    if args.device_type != "auto":
        config.device = args.device_type
        # GPU番号が指定されている場合
        if args.gpu_id is not None and args.device_type == "cuda":
            config.device = f"cuda:{args.gpu_id}"
    elif config.device == "auto":
        # autoが指定された場合、自動的に適切なデバイスを選択
        import torch
        if torch.cuda.is_available():
            config.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config.device = "mps"
        else:
            config.device = "cpu"
        print(f"自動デバイス選択: {config.device}")
    if args.sample_rate is not None:
        config.sample_rate = args.sample_rate

    if args.max_speech_duration != float("inf"):
        config.max_speech_duration = args.max_speech_duration
    if args.verbose:
        config.debug = True
    if args.show_level:
        config.show_level = True
    
    # WebSocketテキスト送信設定の上書き
    if args.websocket:
        config.websocket_enabled = True
    if args.websocket_port is not None:
        config.websocket_port = args.websocket_port
    if args.websocket_host is not None:
        config.websocket_host = args.websocket_host
    if args.text_type is not None:
        config.text_type = args.text_type
    # WebSocket音声受信設定の上書き
    if args.audio_ws:
        config.audio_ws_enabled = True
    if args.audio_ws_host is not None:
        config.audio_ws_host = args.audio_ws_host
    if args.audio_ws_port is not None:
        config.audio_ws_port = args.audio_ws_port
    
    if args.verbose:
        print("=== MenZ-ReazonSpeech リアルタイム音声認識 ===")
        print(f"サンプルレート: {config.sample_rate}")
        print(f"VAD閾値: {config.silero_threshold}")
        print(f"デバイス: {config.device}")
        
        if config.websocket_enabled:
            print(f"WebSocket送信: 有効 ({config.websocket_host}:{config.websocket_port})")
            format_name = "ゆかりねっと" if config.text_type == 0 else "ゆかコネNEO"
            print(f"送信形式: {format_name}")
        else:
            print("WebSocket送信: 無効")
        
        print("Ctrl+C で終了")
        print("=" * 50)
    
    try:
        # リアルタイム認識開始
        if getattr(config, 'audio_ws_enabled', False):
            from reazon_speech.ws_audio_server import WebSocketAudioServer
            with RealtimeTranscriber(config, callback=print_result, show_level=args.show_level) as transcriber:
                # WebSocket音声受信サーバ起動
                server = WebSocketAudioServer(config, transcriber)
                server.start()
                # 認識テキストの既存WebSocket送信も有効化
                if transcriber.websocket_sender:
                    transcriber.websocket_sender.start()
                # RealtimeTranscriberからも直接broadcastできるように参照を持たせる
                setattr(transcriber, 'ws_audio_server', server)
                print("WebSocket音声受信モード。Unityから接続してください。Ctrl+Cで終了。")
                try:
                    while True:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    pass
                finally:
                    if transcriber.websocket_sender:
                        transcriber.websocket_sender.stop()
                    server.stop()
        else:
            with RealtimeTranscriber(config, callback=print_result, show_level=args.show_level) as transcriber:
                transcriber.start_recording(args.device)
    
    except KeyboardInterrupt:
        print("\n終了しました")
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 