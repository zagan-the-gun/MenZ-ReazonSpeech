"""
Silero VAD実装
エンタープライズグレードの音声アクティビティ検出
"""

import numpy as np
import torch
from typing import Optional, Union
import warnings

try:
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    SILERO_AVAILABLE = True
except ImportError:
    try:
        # フォールバック: PyTorch Hubから直接ロード
        import torch
        SILERO_AVAILABLE = True
    except ImportError:
        SILERO_AVAILABLE = False


class SileroVAD:
    """Silero VAD実装（YukariWhisper方式）"""
    
    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000, 
                 min_speech_duration_ms: int = 30, min_silence_duration_ms: int = 100,
                 device: str = "cpu", gpu_id: int = 0):
        """
        Args:
            threshold: VAD閾値（0.0-1.0）
            sampling_rate: サンプリングレート（8000または16000）
            min_speech_duration_ms: 最小音声長さ（ミリ秒）
            min_silence_duration_ms: 最小無音長さ（ミリ秒）
            device: 使用デバイス（"cpu", "cuda", "mps"）
            gpu_id: GPU番号（cudaの場合のみ有効）
        """
        if not SILERO_AVAILABLE:
            raise ImportError(
                "Silero VAD is not available. Install with: pip install silero-vad"
            )
        
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.base_device = device
        self.gpu_id = gpu_id
        self.model = None
        self.utils = None
        
        # 実際のデバイス文字列を構築
        if device == "cuda":
            self.device = f"cuda:{gpu_id}"
        else:
            self.device = device
        
        # サンプリングレート検証
        if sampling_rate not in [8000, 16000]:
            warnings.warn(
                f"Silero VAD is optimized for 8kHz and 16kHz, got {sampling_rate}Hz"
            )
        
        # CPUスレッド数を最適化
        torch.set_num_threads(1)
        
        self._load_model()
    
    def _load_model(self):
        """Silero VADモデルをロード"""
        try:
            # 方法1: silero-vadパッケージから
            try:
                self.model = load_silero_vad()
                self.model = self.model.to(self.device)
                print(f"Silero VAD loaded from silero-vad package (device: {self.device})")
                return
            except (ImportError, AttributeError, FileNotFoundError, RuntimeError) as e:
                print(f"silero-vad package failed: {e}")
            
            # 方法2: PyTorch Hubから
            try:
                self.model, self.utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
                self.model = self.model.to(self.device)
                print(f"Silero VAD loaded from PyTorch Hub (device: {self.device})")
                return
            except Exception as e:
                print(f"PyTorch Hub failed: {e}")
            
            # 方法3: 強制再ダウンロード
            print("Forcing Silero VAD download...")
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=True,
                trust_repo=True
            )
            self.model = self.model.to(self.device)
            print(f"Silero VAD loaded with force reload (device: {self.device})")
            
        except Exception as e:
            print(f"All VAD methods failed: {e}")
            raise ImportError(
                "Silero VAD is required but could not be loaded. "
                "Please run setup.bat again or install manually: "
                "pip install silero-vad && python -c \"import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)\""
            )
    
    def is_speech(self, audio_bytes: bytes, sample_rate: int) -> bool:
        """
        音声データが音声かどうかを判定
        
        Args:
            audio_bytes: PCM音声データ（16bit）
            sample_rate: サンプリングレート
            
        Returns:
            音声と判定された場合True
        """
        # バイトデータをnumpy配列に変換
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
        if len(audio_data) == 0:
            return False
        
        # Silero VADが必須
        if self.model is None:
            raise RuntimeError("Silero VAD model is not loaded. Please run setup.bat again.")
        
        # サンプリングレート調整（必要に応じて）
        if sample_rate != self.sampling_rate:
            # 簡易リサンプリング（より正確にはresampleを使用）
            import scipy.signal
            audio_data = scipy.signal.resample(
                audio_data, 
                int(len(audio_data) * self.sampling_rate / sample_rate)
            )
        
        # PyTorchテンソルに変換
        wav_tensor = torch.from_numpy(audio_data).float()
        
        # モデルと同じデバイスにテンソルを移動
        wav_tensor = wav_tensor.to(self.device)
        
        try:
            if self.utils:
                # PyTorch Hub版の場合
                speech_timestamps = self.utils[0](  # get_speech_timestamps
                    wav_tensor, 
                    self.model, 
                    sampling_rate=self.sampling_rate,
                    threshold=self.threshold,
                    min_speech_duration_ms=self.min_speech_duration_ms,
                    min_silence_duration_ms=self.min_silence_duration_ms
                )
            else:
                # silero-vadパッケージ版の場合
                speech_timestamps = get_speech_timestamps(
                    wav_tensor, 
                    self.model, 
                    sampling_rate=self.sampling_rate,
                    threshold=self.threshold,
                    min_speech_duration_ms=self.min_speech_duration_ms,
                    min_silence_duration_ms=self.min_silence_duration_ms
                )
            
            # 音声区間が検出されたかどうか
            return len(speech_timestamps) > 0
            
        except Exception as e:
            print(f"Silero VAD error: {e}")
            raise RuntimeError(f"Silero VAD processing failed: {e}. Please check your installation.")
    
    def get_speech_confidence(self, audio_bytes: bytes, sample_rate: int) -> float:
        """
        音声の信頼度を取得（0.0-1.0）
        
        Args:
            audio_bytes: PCM音声データ
            sample_rate: サンプリングレート
            
        Returns:
            音声信頼度（0.0-1.0）
        """
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
        if len(audio_data) == 0:
            return 0.0
        
        # サンプリングレート調整
        if sample_rate != self.sampling_rate:
            import scipy.signal
            audio_data = scipy.signal.resample(
                audio_data, 
                int(len(audio_data) * self.sampling_rate / sample_rate)
            )
        
        wav_tensor = torch.from_numpy(audio_data).float()
        
        # モデルと同じデバイスにテンソルを移動
        wav_tensor = wav_tensor.to(self.device)
        
        try:
            # モデルの直接出力を取得
            with torch.no_grad():
                speech_prob = self.model(wav_tensor.unsqueeze(0)).squeeze().item()
            return max(0.0, min(1.0, speech_prob))
        except Exception:
            return 0.5  # エラー時は中間値


# SileroVADWrapperは不要（完全移行のため削除）


# Silero VAD専用ファクトリ関数
def Vad(aggressiveness: int = 2):
    """Silero VAD専用ファクトリ関数"""
    threshold_map = {0: 0.3, 1: 0.4, 2: 0.5, 3: 0.6}
    threshold = threshold_map.get(aggressiveness, 0.5)
    return SileroVAD(threshold=threshold)


# YukariWhisper風の設定クラス
class VADConfig:
    """VAD設定クラス"""
    
    def __init__(self, 
                 vad_threshold: float = 0.5,
                 pause_threshold: int = 5,  # 0.1秒単位
                 phrase_threshold: int = 10,  # 0.1秒単位
                 sampling_rate: int = 16000):
        self.vad_threshold = vad_threshold
        self.pause_threshold = pause_threshold
        self.phrase_threshold = phrase_threshold
        self.sampling_rate = sampling_rate