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
import webrtcvad
from pydub import AudioSegment
import re


class AudioProcessor:
    """音声処理クラス"""
    
    def __init__(self, sample_rate: int = 16000, vad_level: int = 2):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_level)
    
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
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format=target_format)


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
        info.update({
            "device": "cuda",
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "current_gpu": torch.cuda.current_device(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory
        })
    
    return info



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