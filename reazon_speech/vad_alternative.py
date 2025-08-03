"""
WebRTC VADの代替実装
ビルドツールが不要なPython実装
"""

import numpy as np
from typing import Optional


class SimpleVAD:
    """シンプルなVAD実装（webrtcvadの代替）"""
    
    def __init__(self, aggressiveness: int = 2):
        """
        Args:
            aggressiveness: VADの積極性（0-3、高いほど音声検出が厳格）
        """
        self.aggressiveness = aggressiveness
        # 積極性に応じた閾値設定
        self.energy_threshold = {
            0: 0.001,   # 非常に敏感
            1: 0.005,   # 敏感
            2: 0.01,    # 標準
            3: 0.02     # 厳格
        }.get(aggressiveness, 0.01)
        
        self.zero_crossing_threshold = {
            0: 0.1,
            1: 0.15,
            2: 0.2,
            3: 0.25
        }.get(aggressiveness, 0.2)
    
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
        
        # エネルギー計算
        energy = np.mean(audio_data ** 2)
        
        # ゼロ交差率計算
        zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0) / len(audio_data)
        
        # スペクトル重心計算（簡易版）
        spectral_centroid = self._compute_spectral_centroid(audio_data, sample_rate)
        
        # 音声判定
        is_energy_high = energy > self.energy_threshold
        is_zcr_appropriate = zero_crossings < self.zero_crossing_threshold
        is_spectral_good = spectral_centroid > 200  # 人間の音声の基本周波数範囲
        
        return is_energy_high and is_zcr_appropriate and is_spectral_good
    
    def _compute_spectral_centroid(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """スペクトル重心の簡易計算"""
        try:
            # FFT計算
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            
            # 周波数軸
            freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
            
            # スペクトル重心
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                return centroid
            else:
                return 0.0
        except:
            return 0.0


class WebRTCVADWrapper:
    """WebRTC VADのラッパークラス（代替実装との互換性用）"""
    
    def __init__(self, aggressiveness: int = 2):
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self.use_native = True
            print("WebRTC VADを使用します")
        except ImportError:
            self.vad = SimpleVAD(aggressiveness)
            self.use_native = False
            print("代替VAD実装を使用します（WebRTC VADが利用できません）")
    
    def is_speech(self, audio_bytes: bytes, sample_rate: int) -> bool:
        """音声判定"""
        return self.vad.is_speech(audio_bytes, sample_rate)


# webrtcvadの代替としてエクスポート
def Vad(aggressiveness: int = 2):
    """WebRTC VADまたは代替実装を返す"""
    return WebRTCVADWrapper(aggressiveness)