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
    chunk_size: int = 1024
    frame_duration_ms: int = 30
    vad_level: int = 2
    
    # 推論設定
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    gpu_id: str = "auto"  # "auto", "0", "1", "2", etc.
    # FP16モード（CUDA時のみ有効）
    float16: bool = False
    
    # Silero VAD設定（エンタープライズグレード）
    silero_threshold: float = 0.5           # Silero VAD閾値（0.0-1.0）
    min_speech_duration_ms: int = 10        # 最小音声長さ（ミリ秒）
    min_silence_duration_ms: int = 100      # 最小無音長さ（ミリ秒）
    pause_threshold: int = 5                # 無音継続時間（0.1秒単位）YukariWhisper互換
    phrase_threshold: int = 2               # 最小音声継続時間（0.1秒単位）YukariWhisper互換
    max_duration: float = 30.0              # 最大音声継続時間（秒）
    min_audio_level: float = 0.00005        # 音声レベル最小閾値（後方互換性）
    pre_speech_padding_ms: int = 300        # 発話開始前に付与する先頭パディング（ミリ秒）
    post_speech_padding_ms: int = 150       # 発話終了時に付与する末尾パディング（ミリ秒）
    

    
    # WebSocket設定
    websocket_enabled: bool = False   # WebSocket送信の有効/無効
    websocket_host: str = "localhost" # 送信先ホスト
    websocket_port: int = 50001       # 送信先ポート
    text_type: int = 0                # 送信形式（0: ゆかりねっと, 1: ゆかコネNEO）
    websocket_subtitle: str = ""      # 送信するJSONに含めるsubtitle識別子
    
    # 音声受信(WebSocket)設定
    audio_ws_enabled: bool = False    # 音声をWebSocketで受信する
    audio_ws_host: str = "localhost"    # 受信サーバのバインドホスト
    audio_ws_port: int = 60001        # 受信サーバのポート
    
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
        
        # デバイスの自動選択と検証
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # GPU IDの処理
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                print(f"警告: CUDAが利用できません。CPUに変更します。")
                self.device = "cpu"
            else:
                # GPU IDの検証と設定
                if self.gpu_id == "auto":
                    # 自動選択: メモリ使用量が最も少ないGPU
                    selected_gpu = self._select_best_gpu()
                    if selected_gpu is not None:
                        self.device = f"cuda:{selected_gpu}"
                        print(f"GPU自動選択: {self.device} ({torch.cuda.get_device_name(selected_gpu)})")
                else:
                    # 特定のGPU IDを指定
                    try:
                        gpu_id = int(self.gpu_id)
                        if gpu_id >= torch.cuda.device_count():
                            print(f"警告: GPU {gpu_id}は存在しません。GPU 0に変更します。")
                            self.device = "cuda:0"
                        else:
                            self.device = f"cuda:{gpu_id}"
                    except (ValueError, TypeError):
                        print(f"警告: 無効なGPU ID '{self.gpu_id}'。GPU 0に変更します。")
                        self.device = "cuda:0"
    
    def _select_best_gpu(self) -> Optional[int]:
        """最適なGPUを選択（メモリ使用量が最も少ないGPU）"""
        import torch
        
        if not torch.cuda.is_available():
            return None
        
        gpu_count = torch.cuda.device_count()
        if gpu_count <= 1:
            return 0
        
        # メモリ使用量が最も少ないGPUを選択
        best_gpu = 0
        min_memory = float('inf')
        
        for i in range(gpu_count):
            torch.cuda.set_device(i)
            memory_used = torch.cuda.memory_allocated(i)
            if memory_used < min_memory:
                min_memory = memory_used
                best_gpu = i
        
        return best_gpu
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """辞書から設定を作成"""
        return cls(**config_dict)
    
    @classmethod
    def from_ini(cls, config_path: str = "config.ini") -> "ModelConfig":
        """INIファイルから設定を読み込み"""
        if not os.path.exists(config_path):
            return cls()
        
        parser = configparser.ConfigParser()
        parser.read(config_path, encoding='utf-8')
        
        # デフォルト値を取得して、設定ファイルで上書き
        defaults = cls()
        config_dict = {}
        
        # 設定マッピング定義（MenZ-FuguMT風にシンプル化）
        config_mapping = {
            'model': {
                'name': ('model_name', str),
                'cache_dir': ('cache_dir', str),
            },
            'audio': {
                'sample_rate': ('sample_rate', int),
                'chunk_size': ('chunk_size', int),
                'frame_duration_ms': ('frame_duration_ms', int),
                'vad_level': ('vad_level', int),
            },
            'inference': {
                'device': ('device', str),
                'gpu_id': ('gpu_id', str),
                'float16': ('float16', bool),
            },
            'recognizer': {
                'min_audio_level': ('min_audio_level', float),
                'pause_threshold': ('pause_threshold', int),
                'phrase_threshold': ('phrase_threshold', int),
                'max_duration': ('max_duration', float),
                'pre_speech_padding_ms': ('pre_speech_padding_ms', int),
                'post_speech_padding_ms': ('post_speech_padding_ms', int),
            },
            'silero_vad': {
                'threshold': ('silero_threshold', float),
                'min_speech_duration_ms': ('min_speech_duration_ms', int),
                'min_silence_duration_ms': ('min_silence_duration_ms', int),
            },

            'websocket': {
                'enabled': ('websocket_enabled', bool),
                'port': ('websocket_port', int),
                'host': ('websocket_host', str),
                'text_type': ('text_type', int),
                'subtitle': ('websocket_subtitle', str),
            },
            'audio_ws': {
                'enabled': ('audio_ws_enabled', bool),
                'host': ('audio_ws_host', str),
                'port': ('audio_ws_port', int),
            },
            'filtering': {
                'min_length': ('min_length', int),
                'exclude_whitespace_only': ('exclude_whitespace_only', bool),
            },
            'debug': {
                'show_debug': ('show_debug', bool),
                'show_transcription': ('show_transcription', bool),
            }
        }
        
        # 設定ファイルから値を読み込み
        for section_name, section_config in config_mapping.items():
            if section_name in parser:
                section = parser[section_name]
                for key, (attr_name, value_type) in section_config.items():
                    if key in section:
                        default_value = getattr(defaults, attr_name)
                        try:
                            if value_type == bool:
                                config_dict[attr_name] = section.getboolean(key, fallback=default_value)
                            elif value_type == int:
                                config_dict[attr_name] = section.getint(key, fallback=default_value)
                            elif value_type == float:
                                config_dict[attr_name] = section.getfloat(key, fallback=default_value)
                            else:
                                config_dict[attr_name] = section.get(key, fallback=default_value)
                        except ValueError:
                            config_dict[attr_name] = default_value
        
        return cls(**config_dict)
    
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
            'chunk_size': str(self.chunk_size),
            'frame_duration_ms': str(self.frame_duration_ms),
            'vad_level': str(self.vad_level)
        }
        
        # 推論設定
        parser['inference'] = {
            'device': self.device,
            'gpu_id': self.gpu_id,
            'float16': str(self.float16)
        }
        
        # 音声検出設定
        parser['recognizer'] = {
            'min_audio_level': str(self.min_audio_level),
            'pause_threshold': str(self.pause_threshold),
            'phrase_threshold': str(self.phrase_threshold),
            'max_duration': str(self.max_duration),
            'pre_speech_padding_ms': str(self.pre_speech_padding_ms),
            'post_speech_padding_ms': str(self.post_speech_padding_ms),
        }
        
        # Silero VAD設定
        parser['silero_vad'] = {
            'threshold': str(self.silero_threshold),
            'min_speech_duration_ms': str(self.min_speech_duration_ms),
            'min_silence_duration_ms': str(self.min_silence_duration_ms),
            'pause_threshold': str(self.pause_threshold),
            'phrase_threshold': str(self.phrase_threshold)
        }
        

        
        # WebSocket設定
        parser['websocket'] = {
            'enabled': str(self.websocket_enabled),
            'port': str(self.websocket_port),
            'host': self.websocket_host,
            'text_type': str(self.text_type),
            'subtitle': self.websocket_subtitle
        }
        
        # 音声受信(WebSocket)設定
        parser['audio_ws'] = {
            'enabled': str(self.audio_ws_enabled),
            'host': self.audio_ws_host,
            'port': str(self.audio_ws_port)
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
            "chunk_size": self.chunk_size,
            "frame_duration_ms": self.frame_duration_ms,
            "vad_level": self.vad_level,
            "device": self.device,
            "gpu_id": self.gpu_id,
            "float16": self.float16,
            "silero_threshold": self.silero_threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "pause_threshold": self.pause_threshold,
            "phrase_threshold": self.phrase_threshold,
            "max_duration": self.max_duration,
            "min_audio_level": self.min_audio_level,
            "pre_speech_padding_ms": self.pre_speech_padding_ms,
            "post_speech_padding_ms": self.post_speech_padding_ms,
            "websocket_enabled": self.websocket_enabled,
            "websocket_port": self.websocket_port,
            "websocket_host": self.websocket_host,
            "text_type": self.text_type,
            "audio_ws_enabled": self.audio_ws_enabled,
            "audio_ws_host": self.audio_ws_host,
            "audio_ws_port": self.audio_ws_port,
            "min_length": self.min_length,
            "exclude_whitespace_only": self.exclude_whitespace_only,
            "show_debug": self.show_debug,
            "show_transcription": self.show_transcription,
        }
