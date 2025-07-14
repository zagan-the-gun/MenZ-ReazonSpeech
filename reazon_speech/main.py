#!/usr/bin/env python3
"""
MenZ-ReazonSpeech メイン実行ファイル
リアルタイム音声認識システム
"""

import sys
import argparse
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

  # デバイス情報を表示
  python main.py --info

  # カスタム設定
  python main.py --sample-rate 16000 --vad-threshold 0.5
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
        "--vad-threshold",
        type=float,
        help="VAD閾値（設定ファイルから読み込み）"
    )
    
    parser.add_argument(
        "--max-speech-duration",
        type=float,
        default=float("inf"),
        help="最大音声持続時間（秒、デフォルト: 無制限）"
    )
    
    parser.add_argument(
        "--device-type",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="使用デバイス（デフォルト: auto）"
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
    
    args = parser.parse_args()
    
    # デフォルト設定ファイルの作成
    if args.create_config:
        config = ModelConfig()
        config.save_ini(args.config)
        print(f"設定ファイルを作成しました: {args.config}")
        return
    
    # デバイス情報の表示
    if args.info:
        device_info = get_device_info()
        print("=== デバイス情報 ===")
        for key, value in device_info.items():
            print(f"{key}: {value}")
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
    if args.vad_threshold is not None:
        config.vad_threshold = args.vad_threshold
    if args.max_speech_duration != float("inf"):
        config.max_speech_duration = args.max_speech_duration
    if args.verbose:
        config.debug = True
    if args.show_level:
        config.show_level = True
    
    if args.verbose:
        print("=== MenZ-ReazonSpeech リアルタイム音声認識 ===")
        print(f"サンプルレート: {config.sample_rate}")
        print(f"VAD閾値: {config.vad_threshold}")
        print(f"デバイス: {config.device}")
        print("Ctrl+C で終了")
        print("=" * 50)
    
    try:
        # リアルタイム認識開始
        with RealtimeTranscriber(config, callback=print_result, show_level=args.show_level) as transcriber:
            transcriber.start_recording(args.device)
    
    except KeyboardInterrupt:
        print("\n終了しました")
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 