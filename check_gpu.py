#!/usr/bin/env python3
"""
GPU環境チェックスクリプト
MenZ-ReazonSpeechで利用可能なGPU環境を確認します。
"""

import sys
import platform
import torch
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from reazon_speech.config import ModelConfig
from reazon_speech.utils import get_device_info


def check_python_environment():
    """Python環境チェック"""
    print("🐍 Python環境:")
    print(f" バージョン: {sys.version}")
    print(f" プラットフォーム: {platform.platform()}")
    print()


def check_pytorch():
    """PyTorch環境チェック"""
    print("🔥 PyTorch環境:")
    try:
        print(f" PyTorch: {torch.__version__}")
        print(f" CUDA利用可能: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f" CUDAバージョン: {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            print(f" GPUデバイス数: {gpu_count}")
            print()
            
            print(" 利用可能なGPU:")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_properties = torch.cuda.get_device_properties(i)
                gpu_memory = gpu_properties.total_memory / (1024**3)
                compute_capability = f"{gpu_properties.major}.{gpu_properties.minor}"
                
                # メモリ使用状況
                try:
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    free = gpu_memory - cached
                    print(f" GPU {i}: {gpu_name}")
                    print(f" メモリ: {gpu_memory:.1f}GB (使用中: {cached:.1f}GB, 空き: {free:.1f}GB)")
                    print(f" Compute Capability: {compute_capability}")
                    print(f" 設定例: device = cuda, gpu_id = {i}")
                except Exception:
                    print(f" GPU {i}: {gpu_name}")
                    print(f" メモリ: {gpu_memory:.1f}GB")
                    print(f" Compute Capability: {compute_capability}")
                    print(f" 設定例: device = cuda, gpu_id = {i}")
                print()
            
            if gpu_count > 1:
                print(" 🔍 複数GPUが検出されました！")
                print(" 設定ファイルでGPU IDを指定することで特定のGPUを使用できます:")
                print(" config.ini の [inference] セクションで")
                print(" device = cuda")
                print(" gpu_id = 0 # 使用したいGPUのID（0から始まる）")
                print()
        
        # Apple Silicon (MPS) チェック
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(" MPS（Apple Silicon）: 利用可能")
            print()
            
    except ImportError:
        print(" ❌ PyTorchがインストールされていません")
        print(" pip install torch でインストールしてください")
        print()


def check_nemo():
    """NeMo環境チェック"""
    print("🎯 NeMo環境:")
    try:
        import nemo
        print(f" NeMo: {nemo.__version__}")
        print()
    except ImportError:
        print(" ❌ NeMoがインストールされていません")
        print(" pip install nemo_toolkit[asr] でインストールしてください")
        print()


def check_audio_libraries():
    """音声処理ライブラリチェック"""
    print("🎵 音声処理ライブラリ:")
    
    # sounddevice
    try:
        import sounddevice as sd
        print(f" sounddevice: {sd.__version__}")
    except ImportError:
        print(" ❌ sounddeviceがインストールされていません")
        print(" pip install sounddevice でインストールしてください")
    
    # librosa
    try:
        import librosa
        print(f" librosa: {librosa.__version__}")
    except ImportError:
        print(" ❌ librosaがインストールされていません")
        print(" pip install librosa でインストールしてください")
    
    # numpy
    try:
        import numpy as np
        print(f" numpy: {np.__version__}")
    except ImportError:
        print(" ❌ numpyがインストールされていません")
        print(" pip install numpy でインストールしてください")
    
    print()


def check_system_resources():
    """システムリソースチェック"""
    print("💻 システムリソース:")
    try:
        import psutil
        
        # CPU情報
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f" CPU: {cpu_count}コア (使用率: {cpu_percent}%)")
        
        # メモリ情報
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_percent = memory.percent
        print(f" メモリ: {memory_gb:.1f}GB (使用率: {memory_percent}%)")
        
        if memory_gb < 4:
            print(" ⚠️ メモリが4GB未満です。動作が制限される可能性があります。")
        elif memory_gb < 8:
            print(" ⚠️ メモリが8GB未満です。大きなモデルで問題が生じる可能性があります。")
        
        # ディスク容量
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        print(f" 利用可能ディスク容量: {disk_gb:.1f}GB")
        
        if disk_gb < 5:
            print(" ⚠️ ディスク容量が不足しています。モデルダウンロードに必要です。")
        
        print()
        
    except ImportError:
        print(" psutilがインストールされていません（オプション）")
        print(" pip install psutil でより詳細な情報を確認できます")
        print()


def check_reazonspeech_model():
    """ReazonSpeechモデルアクセステスト"""
    print("🎤 ReazonSpeechモデルアクセステスト:")
    try:
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf
        
        model_name = "reazon-research/reazonspeech-nemo-v2"
        print(f" モデル: {model_name}")
        
        # NeMoモデルの読み込みテスト
        print(" NeMoモデル読み込み中...")
        print(" (この処理には時間がかかる場合があります...)")
        
        # NeMoモデルのダウンロードと読み込み（RNN-Tモデル）
        try:
            model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
            model_type = "RNN-T"
        except:
            # フォールバック: CTCモデル
            model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name)
            model_type = "CTC"
        
        print(f" ✅ NeMoモデル読み込み成功 ({model_type})")
        
        # モデル情報の表示
        if hasattr(model, '_cfg') and hasattr(model._cfg, 'sample_rate'):
            print(f" サンプルレート: {model._cfg.sample_rate} Hz")
        
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'vocabulary'):
            vocab_size = len(model.decoder.vocabulary) if hasattr(model.decoder.vocabulary, '__len__') else "不明"
            print(f" 語彙サイズ: {vocab_size}")
        elif hasattr(model, 'tokenizer'):
            vocab_size = model.tokenizer.vocab_size if hasattr(model.tokenizer, 'vocab_size') else "不明"
            print(f" 語彙サイズ: {vocab_size}")
        
        print()
        
    except ImportError as e:
        print(f" ❌ NeMoインポートエラー: {e}")
        print(" NeMoがインストールされていません")
        print(" pip install nemo_toolkit[asr] でインストールしてください")
        print()
    except Exception as e:
        print(f" ❌ エラー: {e}")
        print(" ネットワーク接続またはモデルダウンロードに問題があります")
        print(" 初回実行時はモデルダウンロードに時間がかかります")
        print()


def check_configuration():
    """設定ファイルチェック"""
    print("⚙️ 設定ファイルチェック:")
    try:
        config = ModelConfig()
        print(f" 設定されたデバイス: {config.device}")
        print(f" 設定されたGPU ID: {config.gpu_id}")
        print(f" モデル名: {config.model_name}")
        print()
        
        # 実際に使用されるデバイスをテスト
        print("音声認識エンジン設定確認:")
        print(f" 設定されたデバイス: {config.device}")
        
        # デバイス情報の表示
        if config.device.startswith('cuda'):
            import torch
            if torch.cuda.is_available():
                if ':' in config.device:
                    gpu_id = int(config.device.split(':')[1])
                else:
                    gpu_id = torch.cuda.current_device()
                
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                
                try:
                    allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    print(f" GPU詳細: {gpu_name} (ID: {gpu_id})")
                    print(f" GPU メモリ: {gpu_memory:.1f}GB (現在使用中: {allocated:.1f}GB)")
                except Exception:
                    print(f" GPU詳細: {gpu_name} (ID: {gpu_id}, メモリ: {gpu_memory:.1f}GB)")
            else:
                print(" ⚠️ CUDAが利用できません")
        elif config.device == "mps":
            print(" Apple Silicon GPU (MPS) を使用")
        else:
            print(" CPUを使用")
        
        print()
        
    except Exception as e:
        print(f"設定確認エラー: {e}")
        print()


def run_comprehensive_test():
    """総合テスト"""
    print("🧪 総合動作テスト:")
    try:
        print(" 設定ファイル作成テスト...")
        config = ModelConfig()
        print(" ✅ 設定ファイル作成成功")
        
        print(" デバイス情報取得テスト...")
        device_info = get_device_info()
        print(f" 設定されたデバイス: {config.device}")
        
        # 使用中のGPU詳細情報を表示（設定ベース）
        if config.device.startswith('cuda'):
            import torch
            if torch.cuda.is_available():
                if ':' in config.device:
                    gpu_id = int(config.device.split(':')[1])
                else:
                    gpu_id = torch.cuda.current_device()
                
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                print(f" GPU詳細: {gpu_name} (ID: {gpu_id}, メモリ: {gpu_memory:.1f}GB)")
            else:
                print(" ⚠️ CUDAが利用できません")
        elif config.device == "mps":
            print(" Apple Silicon GPU (MPS) を使用")
        else:
            print(" CPUを使用")
        
        print(" ✅ 総合テスト成功")
        print()
        
    except Exception as e:
        print(f" ❌ 総合テストエラー: {e}")
        print(" 詳細な診断には各個別テストの結果を確認してください")
        print()


def main():
    """メイン処理"""
    print("="*50)
    print("🔍 MenZ-ReazonSpeech 環境チェック")
    print("="*50)
    print()
    
    # 基本環境チェック
    check_python_environment()
    check_pytorch()
    check_nemo()
    check_audio_libraries()
    
    # システムリソースチェック
    check_system_resources()
    
    # ReazonSpeechモデルチェック
    check_reazonspeech_model()
    
    # 設定ファイルチェック
    check_configuration()
    
    # 総合テスト
    run_comprehensive_test()
    
    print("="*50)
    print("✅ 環境チェック完了")
    print("="*50)
    print()
    print("推奨事項:")
    print("- GPU利用時は十分なメモリ（8GB以上）を確保してください")
    print("- 初回実行時はモデルダウンロードに時間がかかります")
    print("- 安定した動作にはネットワーク接続が必要です")
    print("- 音声認識には適切なマイク設定が必要です")


if __name__ == "__main__":
    main() 