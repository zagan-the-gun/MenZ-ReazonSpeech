"""
コマンドラインインターフェース
"""

import argparse
import sys
from pathlib import Path
from typing import List

from .model import ReazonSpeechModel
from .config import ModelConfig
from .utils import get_device_info


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="ReazonSpeechを使用した音声認識ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一ファイルの文字起こし
  reazon-speech audio.wav

  # 出力ファイルを指定
  reazon-speech audio.wav -o output.txt

  # JSON形式で出力
  reazon-speech audio.wav -f json -o output.json

  # SRT形式で出力
  reazon-speech audio.wav -f srt -o output.srt

  # 複数ファイルの処理
  reazon-speech audio1.wav audio2.wav audio3.wav -d output_dir

  # デバイス情報の表示
  reazon-speech --info

  # 特定のGPUを使用
  reazon-speech audio.wav --device cuda --gpu-id 1

  # GPU IDを指定
  reazon-speech audio.wav --gpu-id auto
        """
    )
    
    parser.add_argument(
        "audio_files",
        nargs="*",
        help="音声ファイルのパス"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="出力ファイルのパス"
    )
    
    parser.add_argument(
        "-d", "--output-dir",
        help="出力ディレクトリ（複数ファイル用）"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["text", "json", "srt"],
        default="text",
        help="出力フォーマット（デフォルト: text）"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="使用デバイス（デフォルト: auto）"
    )
    
    parser.add_argument(
        "--gpu-id",
        type=int,
        help="使用するGPU番号（0, 1, 2など）。--device cudaと組み合わせて使用"
    )
    
    parser.add_argument(
        "--gpu-id",
        type=str,
        help="使用するGPU ID（0, 1, 2... または auto）"
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
        "--info",
        action="store_true",
        help="デバイス情報を表示"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細な出力"
    )
    
    args = parser.parse_args()
    
    # デバイス情報の表示
    if args.info:
        device_info = get_device_info()
        print("=== デバイス情報 ===")
        for key, value in device_info.items():
            print(f"{key}: {value}")
        return
    
    # 音声ファイルが指定されていない場合
    if not args.audio_files:
        parser.print_help()
        return
    
    # 設定の作成
    config = ModelConfig()
    
    # コマンドライン引数で設定を上書き
    if args.device != "auto":
        config.device = args.device
        # GPU番号が指定されている場合
        if args.gpu_id is not None and args.device == "cuda":
            config.device = f"cuda:{args.gpu_id}"
    
    if args.gpu_id is not None:
        config.gpu_id = args.gpu_id
    
    if args.sample_rate is not None:
        config.sample_rate = args.sample_rate
    if args.vad_threshold is not None:
        config.vad_threshold = args.vad_threshold
    if args.max_speech_duration != float("inf"):
        config.max_duration = args.max_speech_duration
    if args.format != "text":
        config.output_format = args.format
    
    try:
        # モデルの初期化
        if args.verbose:
            print("モデルを初期化中...")
        
        model = ReazonSpeechModel(config)
        
        if args.verbose:
            model_info = model.get_model_info()
            print(f"モデル: {model_info['model_name']}")
            print(f"デバイス: {model_info['device']}")
            print(f"サンプルレート: {model_info['sample_rate']}")
        
        # 単一ファイルの処理
        if len(args.audio_files) == 1:
            audio_file = args.audio_files[0]
            
            if args.verbose:
                print(f"音声ファイルを処理中: {audio_file}")
            
            result = model.transcribe(
                audio_file,
                output_path=args.output,
                output_format=args.format
            )
            
            if not args.output:
                print(result)
        
        # 複数ファイルの処理
        else:
            if not args.output_dir:
                print("エラー: 複数ファイルの処理には --output-dir が必要です")
                sys.exit(1)
            
            if args.verbose:
                print(f"出力ディレクトリ: {args.output_dir}")
            
            results = model.transcribe_batch(
                args.audio_files,
                output_dir=args.output_dir
            )
            
            if args.verbose:
                print(f"{len(results)}個のファイルを処理しました")
    
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 