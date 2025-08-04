"""
ユーティリティモジュール
"""

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from typing import List, Optional, Tuple, Union
from pathlib import Path
import requests
from tqdm import tqdm
from .vad_silero import SileroVAD
import re


class AudioProcessor:
    """音声処理クラス"""
    
    def __init__(self, sample_rate: int = 16000, vad_level: int = 2, config=None):
        self.sample_rate = sample_rate
        # Silero VAD（エンタープライズグレード）
        threshold_map = {0: 0.3, 1: 0.4, 2: 0.5, 3: 0.6}
        threshold = threshold_map.get(vad_level, 0.5)
        
        # configがある場合は設定を使用
        if config:
            vad_threshold = getattr(config, 'silero_threshold', threshold)
            min_speech_ms = getattr(config, 'min_speech_duration_ms', 30)
            min_silence_ms = getattr(config, 'min_silence_duration_ms', 100)
        else:
            vad_threshold = threshold
            min_speech_ms = 30
            min_silence_ms = 100
            
        self.vad = SileroVAD(
            threshold=vad_threshold, 
            sampling_rate=sample_rate,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms
        )
        print(f"Using Silero VAD (threshold={vad_threshold}, min_speech={min_speech_ms}ms, min_silence={min_silence_ms}ms)")
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """音声ファイルを読み込み"""
        try:
            # 音声ファイルを読み込み
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            raise ValueError(f"音声ファイルの読み込みに失敗しました: {e}")
    
    def save_audio(self, audio: np.ndarray, file_path: str) -> None:
        """音声ファイルを保存"""
        sf.write(file_path, audio, self.sample_rate)
    
    def resample_audio(self, audio: np.ndarray, target_sr: int) -> np.ndarray:
        """音声のリサンプリング"""
        if len(audio.shape) == 1:
            audio = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_sr)
        else:
            audio = librosa.resample(audio.T, orig_sr=self.sample_rate, target_sr=target_sr).T
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """音声の正規化"""
        if len(audio) == 0:
            return audio
        return audio / np.max(np.abs(audio))
    
    def split_audio(self, audio: np.ndarray, chunk_length: int, stride: int) -> List[np.ndarray]:
        """音声をチャンクに分割"""
        chunks = []
        for i in range(0, len(audio), stride):
            chunk = audio[i:i + chunk_length]
            if len(chunk) == chunk_length:
                chunks.append(chunk)
        return chunks
    
    def detect_speech_segments(self, audio: np.ndarray, 
                              max_speech_duration_s: float = float("inf")) -> List[Tuple[int, int]]:
        """音声セグメントの検出（VAD処理用）"""
        # 16kHzに変換（WebRTC VADは16kHzのみ対応）
        if self.sample_rate != 16000:
            audio_16k = self.resample_audio(audio, 16000)
        else:
            audio_16k = audio
        
        # PCM形式に変換
        audio_int16 = (audio_16k * 32767).astype(np.int16)
        
        # フレームサイズ（30ms）
        frame_size = int(0.03 * 16000)
        speech_segments = []
        
        for i in range(0, len(audio_int16) - frame_size, frame_size):
            frame = audio_int16[i:i + frame_size]
            is_speech = self.vad.is_speech(frame.tobytes(), 16000)
            
            if is_speech:
                start_time = i / 16000
                # 次の非音声フレームを探す
                j = i + frame_size
                while j < len(audio_int16) - frame_size:
                    frame = audio_int16[j:j + frame_size]
                    if not self.vad.is_speech(frame.tobytes(), 16000):
                        break
                    j += frame_size
                
                end_time = j / 16000
                duration = end_time - start_time
                
                # 最大持続時間のチェック（VAD処理は短い雑音除去が重要）
                if duration <= max_speech_duration_s:
                    speech_segments.append((start_time, end_time))
        
        return speech_segments
    
    def convert_audio_format(self, input_path: str, output_path: str, 
                           target_format: str = "wav") -> None:
        """音声フォーマットの変換"""
        try:
            # libROSA + soundfileでFFmpegレス変換
            audio, sr = librosa.load(input_path, sr=self.sample_rate)
            sf.write(output_path, audio, self.sample_rate, format=target_format.upper())
        except Exception as e:
            # フォールバック: pydub使用（FFmpegが必要）
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(input_path)
                audio.export(output_path, format=target_format)
            except Exception as pydub_error:
                raise ValueError(f"音声変換に失敗: librosa error: {e}, pydub error: {pydub_error}")


class ModelDownloader:
    """モデルダウンローダー"""
    
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def download_model(self, model_name: str, force_download: bool = False) -> str:
        """モデルのダウンロード"""
        model_path = self.cache_dir / model_name.replace("/", "_")
        
        if model_path.exists() and not force_download:
            return str(model_path)
        
        # Hugging Face Hubからダウンロード
        try:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=self.cache_dir,
                local_dir=model_path
            )
            return model_path
        except Exception as e:
            raise RuntimeError(f"モデルのダウンロードに失敗しました: {e}")
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """モデルパスの取得"""
        model_path = self.cache_dir / model_name.replace("/", "_")
        return str(model_path) if model_path.exists() else None
    
    def list_models(self) -> List[str]:
        """キャッシュされたモデルの一覧"""
        models = []
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                models.append(item.name)
        return models
    
    def remove_model(self, model_name: str) -> bool:
        """モデルの削除"""
        model_path = self.cache_dir / model_name.replace("/", "_")
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)
            return True
        return False


def get_device_info() -> dict:
    """デバイス情報の取得"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu"
    }
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpus = []
        
        for i in range(gpu_count):
            torch.cuda.set_device(i)
            gpu_props = torch.cuda.get_device_properties(i)
            memory_used = torch.cuda.memory_allocated(i)
            memory_free = torch.cuda.memory_reserved(i) - memory_used
            
            # multiprocessor_countは古いPyTorchでは属性名が異なる場合がある
            multiprocessor_count = getattr(gpu_props, 'multiprocessor_count', 
                                         getattr(gpu_props, 'multi_processor_count', 0))
            
            gpu_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": gpu_props.total_memory,
                "memory_total_gb": round(gpu_props.total_memory / (1024**3), 2),
                "memory_used_mb": round(memory_used / (1024**2), 1),
                "memory_free_mb": round(memory_free / (1024**2), 1),
                "memory_usage_percent": round((memory_used / gpu_props.total_memory) * 100, 1),
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "multiprocessor_count": multiprocessor_count,
                "performance_score": gpu_props.major * 10 + gpu_props.minor + multiprocessor_count * 0.1
            }
            gpus.append(gpu_info)
        
        info.update({
            "device": "cuda",
            "cuda_version": torch.version.cuda,
            "gpu_count": gpu_count,
            "current_gpu": torch.cuda.current_device(),
            "gpus": gpus
        })
    
    # MPS (Apple Silicon) サポートチェック
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info["mps_available"] = True
    else:
        info["mps_available"] = False
    
    return info


def print_gpu_info():
    """GPU情報を見やすく表示"""
    info = get_device_info()
    
    print("=== デバイス情報 ===")
    print(f"CUDA利用可能: {info['cuda_available']}")
    print(f"MPS利用可能: {info['mps_available']}")
    
    if info['cuda_available']:
        print(f"CUDAバージョン: {info['cuda_version']}")
        print(f"GPU数: {info['gpu_count']}")
        print(f"現在のGPU: {info['current_gpu']}")
        print()
        
        # GPUをメモリ使用量でソート
        gpus_sorted = sorted(info['gpus'], key=lambda x: x['memory_usage_percent'])
        
        for gpu in gpus_sorted:
            status = "推奨" if gpu['memory_usage_percent'] < 50 else "使用中" if gpu['memory_usage_percent'] < 80 else "高負荷"
            print(f"GPU {gpu['id']}: {gpu['name']} [{status}]")
            print(f"  メモリ: {gpu['memory_total_gb']} GB (使用中: {gpu['memory_used_mb']} MB, 空き: {gpu['memory_free_mb']} MB, 使用率: {gpu['memory_usage_percent']}%)")
            print(f"  コンピュート能力: {gpu['compute_capability']}")
            print(f"  マルチプロセッサ数: {gpu['multiprocessor_count']}")
            print(f"  性能スコア: {gpu['performance_score']:.1f}")
            print()
        
        print("=== GPU選択オプション ===")
        print("設定ファイル (config.ini) の [inference] セクションで設定:")
        print("  device = cuda")
        print("  gpu_id = auto        # 自動選択（メモリ使用量が最も少ないGPU）")
        print("  gpu_id = 0           # GPU 0を指定")
        print("  gpu_id = 1           # GPU 1を指定")
        print()
        print("コマンドライン引数:")
        print("  --device cuda        # CUDA使用（自動選択）")
        print("  --device cuda --gpu-id 1  # GPU 1を指定")
        print("  --gpu-id auto        # 自動選択")
        print("  --gpu-id 0           # GPU 0を指定")
        print("  --device cpu         # CPU使用")
        print()
        print("推奨設定:")
        if len(info['gpus']) > 1:
            best_memory_gpu = min(info['gpus'], key=lambda x: x['memory_usage_percent'])
            print(f"  メモリ効率重視: gpu_id = {best_memory_gpu['id']} (使用率: {best_memory_gpu['memory_usage_percent']}%)")
        else:
            print("  単一GPU環境のため、gpu_id = autoが最適です")
    else:
        print("GPUは利用できません")
        print("使用例:")
        print("  --device-type cpu         # CPU使用")
    
    print("=" * 40)



def filter_text(text: str, min_length: int = 3, exclude_whitespace_only: bool = True) -> Optional[str]:
    """基本的なテキストフィルタリング（品質向上用）
    
    Args:
        text: フィルタリング対象のテキスト
        min_length: 最小文字数（これ未満は除外）
        exclude_whitespace_only: 空白のみの文字列を除外するかどうか
        
    Returns:
        フィルタリング後のテキスト（除外される場合はNone）
    """
    if not text:
        return None
    
    # 空白のみのチェック
    if exclude_whitespace_only and not text.strip():
        return None
    
    # 最小文字数のチェック（空白を除いた文字数）
    if len(text.strip()) < min_length:
        return None
    
    return text


def format_time(seconds: float) -> str:
    """時間のフォーマット"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    else:
        return f"{minutes:02d}:{secs:02d},{millisecs:03d}" 