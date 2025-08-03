"""
設定管理モジュール
"""

from dataclasses import dataclass
from typing import Optional, List
import os
import configparser
from pathlib import Path


@dataclass
class ModelConfig:
    """ReazonSpeechモデルの設定"""
    
    # モデル設定
    model_name: str = "reazon-research/reazonspeech-nemo-v2"
    cache_dir: str = "./models"
    
    # 音声処理設定
    sample_rate: int = 16000
    chunk_length_s: int = 30
    stride_length_s: int = 1
    chunk_size: int = 1024
    frame_duration_ms: int = 30
    vad_level: int = 2
    
    # 推論設定
    batch_size: int = 1
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # 音声検出設定
    vad_threshold: float = 0.3  # VAD閾値
    min_audio_level: float = 0.01  # 音声レベル最小閾値
    pause_threshold: int = 8    # 無音継続時間（0.1秒単位）
    phrase_threshold: int = 5   # 最小音声継続時間（0.1秒単位）
    max_duration: float = 30.0  # 最大音声継続時間（秒）
    
    # 出力設定
    output_format: str = "text"  # "text", "json", "srt"
    language: str = "ja"
    
    # WebSocket設定
    websocket_enabled: bool = False   # WebSocket送信の有効/無効
    websocket_port: int = 50000       # 送信先ポート（ゆかりねっと用）
    websocket_host: str = "localhost" # 送信先ホスト
    text_type: int = 0                # 送信形式（0: ゆかりねっと, 1: ゆかコネNEO）
    
    # 基本的なフィルタリング設定（品質向上用）
    min_length: int = 3               # 最小文字数フィルタ（ノイズ除去）
    exclude_whitespace_only: bool = True  # 空白のみの結果を除外
    

    # デバッグ設定
    show_debug: bool = True  # デバッグ情報表示（音声レベル、プログレスバー、デバッグ情報）
    show_transcription: bool = True  # 翻訳テキスト表示
    
    def __post_init__(self):
        """初期化後の処理"""
        # キャッシュディレクトリの作成
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # デバイスの自動選択
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """辞書から設定を作成"""
        return cls(**config_dict)
    
    @classmethod
    def from_ini(cls, config_path: str = "config.ini") -> "ModelConfig":
        """INIファイルから設定を読み込み"""
        config = cls()
        
        if os.path.exists(config_path):
            parser = configparser.ConfigParser()
            parser.read(config_path, encoding='utf-8')
            
            # モデル設定
            if 'model' in parser:
                config.model_name = parser.get('model', 'name', fallback=config.model_name)
                config.cache_dir = parser.get('model', 'cache_dir', fallback=config.cache_dir)
            
            # 音声処理設定
            if 'audio' in parser:
                config.sample_rate = parser.getint('audio', 'sample_rate', fallback=config.sample_rate)
                config.chunk_length_s = parser.getint('audio', 'chunk_length_s', fallback=config.chunk_length_s)
                config.stride_length_s = parser.getint('audio', 'stride_length_s', fallback=config.stride_length_s)
                config.chunk_size = parser.getint('audio', 'chunk_size', fallback=config.chunk_size)
                config.frame_duration_ms = parser.getint('audio', 'frame_duration_ms', fallback=config.frame_duration_ms)
                config.vad_level = parser.getint('audio', 'vad_level', fallback=config.vad_level)
            
            # 推論設定
            if 'inference' in parser:
                config.batch_size = parser.getint('inference', 'batch_size', fallback=config.batch_size)
                config.device = parser.get('inference', 'device', fallback=config.device)
            
            # 音声検出設定
            if 'recognizer' in parser:
                config.vad_threshold = parser.getfloat('recognizer', 'vad_threshold', fallback=config.vad_threshold)
                config.min_audio_level = parser.getfloat('recognizer', 'min_audio_level', fallback=config.min_audio_level)
                config.pause_threshold = parser.getint('recognizer', 'pause_threshold', fallback=config.pause_threshold)
                config.phrase_threshold = parser.getint('recognizer', 'phrase_threshold', fallback=config.phrase_threshold)
                config.max_duration = parser.getfloat('recognizer', 'max_duration', fallback=config.max_duration)
            # 出力設定
            if 'output' in parser:
                config.output_format = parser.get('output', 'format', fallback=config.output_format)
                config.language = parser.get('output', 'language', fallback=config.language)
            
            # WebSocket設定
            if 'websocket' in parser:
                config.websocket_enabled = parser.getboolean('websocket', 'enabled', fallback=config.websocket_enabled)
                config.websocket_port = parser.getint('websocket', 'port', fallback=config.websocket_port)
                config.websocket_host = parser.get('websocket', 'host', fallback=config.websocket_host)
                config.text_type = parser.getint('websocket', 'text_type', fallback=config.text_type)
            
            # 基本的なフィルタリング設定
            if 'filtering' in parser:
                config.min_length = parser.getint('filtering', 'min_length', fallback=config.min_length)
                config.exclude_whitespace_only = parser.getboolean('filtering', 'exclude_whitespace_only', fallback=config.exclude_whitespace_only)
            

            # デバッグ設定
            if 'debug' in parser:
                config.show_debug = parser.getboolean('debug', 'show_debug', fallback=config.show_debug)
                config.show_transcription = parser.getboolean('debug', 'show_transcription', fallback=config.show_transcription)
        
        return config
    
    def save_ini(self, config_path: str = "config.ini") -> None:
        """設定をINIファイルに保存"""
        parser = configparser.ConfigParser()
        
        # モデル設定
        parser['model'] = {
            'name': self.model_name,
            'cache_dir': self.cache_dir
        }
        
        # 音声処理設定
        parser['audio'] = {
            'sample_rate': str(self.sample_rate),
            'chunk_length_s': str(self.chunk_length_s),
            'stride_length_s': str(self.stride_length_s),
            'chunk_size': str(self.chunk_size),
            'frame_duration_ms': str(self.frame_duration_ms),
            'vad_level': str(self.vad_level)
        }
        
        # 推論設定
        parser['inference'] = {
            'batch_size': str(self.batch_size),
            'device': self.device
        }
        
        # 音声検出設定
        parser['recognizer'] = {
            'vad_threshold': str(self.vad_threshold),
            'min_audio_level': str(self.min_audio_level),
            'pause_threshold': str(self.pause_threshold),
            'phrase_threshold': str(self.phrase_threshold),
            'max_duration': str(self.max_duration)
        }
        
        # 出力設定
        parser['output'] = {
            'format': self.output_format,
            'language': self.language
        }
        
        # WebSocket設定
        parser['websocket'] = {
            'enabled': str(self.websocket_enabled),
            'port': str(self.websocket_port),
            'host': self.websocket_host,
            'text_type': str(self.text_type)
        }
        
        # 基本的なフィルタリング設定
        parser['filtering'] = {
            'min_length': str(self.min_length),
            'exclude_whitespace_only': str(self.exclude_whitespace_only)
        }
        

        # デバッグ設定
        parser['debug'] = {
            'show_debug': str(self.show_debug),
            'show_transcription': str(self.show_transcription)
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            parser.write(f)
    
    def to_dict(self) -> dict:
        """設定を辞書に変換"""
        return {
            "model_name": self.model_name,
            "cache_dir": self.cache_dir,
            "sample_rate": self.sample_rate,
            "chunk_length_s": self.chunk_length_s,
            "stride_length_s": self.stride_length_s,
            "chunk_size": self.chunk_size,
            "frame_duration_ms": self.frame_duration_ms,
            "vad_level": self.vad_level,
            "batch_size": self.batch_size,
            "device": self.device,
            "vad_threshold": self.vad_threshold,
            "min_audio_level": self.min_audio_level,
            "pause_threshold": self.pause_threshold,
            "phrase_threshold": self.phrase_threshold,
            "max_duration": self.max_duration,
            "output_format": self.output_format,
            "language": self.language,
            "websocket_enabled": self.websocket_enabled,
            "websocket_port": self.websocket_port,
            "websocket_host": self.websocket_host,
            "text_type": self.text_type,
            "min_length": self.min_length,
            "exclude_whitespace_only": self.exclude_whitespace_only,
            "show_debug": self.show_debug,
            "show_transcription": self.show_transcription,
        }


# デフォルト設定
DEFAULT_CONFIG = ModelConfig() 