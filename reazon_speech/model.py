"""
ReazonSpeechモデルクラス
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from .config import ModelConfig, DEFAULT_CONFIG
from .utils import ModelDownloader, get_device_info
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyRNNTInfer

class ReazonSpeechModel:
    """ReazonSpeechモデルクラス"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Args:
            config: モデル設定
        """
        self.config = config or DEFAULT_CONFIG
        self.model_downloader = ModelDownloader(self.config.cache_dir)
        
        # モデルの初期化
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """モデルの初期化"""
        try:
            # プログレスバーの制御
            if not self.config.show_debug:
                import os
                os.environ['TQDM_DISABLE'] = '1'
                # tqdmを無効化
                import tqdm
                tqdm.tqdm = lambda *args, **kwargs: None
            
            # モデルのダウンロード
            model_path = self.model_downloader.download_model(self.config.model_name)
            
            # RNN-Tモデルの読み込み
            
            # モデルファイルの検索
            model_files = list(Path(model_path).glob("*.nemo"))
            if not model_files:
                raise RuntimeError(f"NeMoモデルファイルが見つかりません: {model_path}")
            
            model_file = model_files[0]
            
            # モデルの読み込み
            self.model = EncDecRNNTBPEModel.restore_from(str(model_file))
            self.model = self.model.to(self.config.device)
            self.model.eval()
            
            # デコーダの構成
            if self.model.decoding is None:
                print("デコーダ設定が見つかりません。手動で構成します...")
                
                # setup_test_dataで構成を試行
                self.model.setup_test_data(cfg=None)
                
                # それでもdecodingがNoneの場合は手動で構成
                if self.model.decoding is None:
                    self.model.decoding = GreedyRNNTInfer(
                        decoder_model=self.model.decoder,
                        joint_model=self.model.joint,
                        blank_index=self.model.tokenizer.blank_id
                    )
                    print("Greedy Decoderを手動で構成しました")
                else:
                    print("setup_test_dataでデコーダが構成されました")
            
            print(f"RNN-Tモデルを読み込みました: {model_file}")
            print(f"デバイス: {self.config.device}")
            
        except Exception as e:
            raise RuntimeError(f"モデルの初期化に失敗しました: {e}")
    
    def transcribe_audio_segment(self, audio_segment: np.ndarray) -> str:
        """
        音声セグメントの文字起こし
        
        Args:
            audio_segment: 音声セグメント（numpy配列）
            
        Returns:
            文字起こし結果
        """
        try:
            # プログレスバーの制御
            if not self.config.show_debug:
                import os
                os.environ['TQDM_DISABLE'] = '1'
                # tqdmを無効化
                import tqdm
                tqdm.tqdm = lambda *args, **kwargs: None
            # 音声テンソル化
            audio_tensor = torch.tensor(audio_segment, dtype=torch.float32).unsqueeze(0).to(self.config.device)
            length_tensor = torch.tensor([len(audio_segment)], dtype=torch.int32).to(self.config.device)

            # モデルを評価モードに設定
            self.model.eval()
            
            # 推論
            with torch.no_grad():
                # 方法1: NeMoのtranscribeメソッドを使用（キーワード引数で）
                if hasattr(self.model, 'transcribe') and callable(self.model.transcribe):
                    try:
                        # キーワード引数として渡す
                        result = self.model.transcribe(audio=audio_segment)
                        if result and len(result) > 0:
                            # 結果がHypothesisオブジェクトの場合はtext属性を取得
                            if hasattr(result[0], 'text'):
                                return result[0].text
                            # 結果が文字列の場合はそのまま返す
                            elif isinstance(result[0], str):
                                return result[0]
                            # その他の場合は文字列に変換
                            else:
                                return str(result[0])
                    except Exception as e:
                        if self.config.show_debug:
                            print(f"transcribeメソッドでエラー: {e}")
                
                # 方法2: エンコーダー+デコーダーを手動で使用（キーワード引数で）
                try:
                    # キーワード引数として渡す
                    encoded, encoded_len = self.model.encoder(
                        audio_signal=audio_tensor, 
                        length=length_tensor
                    )
                    
                    # デコーダーを使用してデコード
                    if hasattr(self.model, 'decoding') and self.model.decoding is not None:
                        # greedy_decodeメソッドを使用
                        if hasattr(self.model.decoding, 'greedy_decode'):
                            decoded = self.model.decoding.greedy_decode(
                                encoder_output=encoded, 
                                encoder_lengths=encoded_len
                            )
                            if decoded and len(decoded) > 0:
                                return decoded[0] if isinstance(decoded[0], str) else str(decoded[0])
                        
                        # 別のデコード方法を試す
                        elif hasattr(self.model.decoding, 'decode'):
                            decoded = self.model.decoding.decode(
                                encoder_output=encoded, 
                                encoder_lengths=encoded_len
                            )
                            if decoded and len(decoded) > 0:
                                return decoded[0] if isinstance(decoded[0], str) else str(decoded[0])
                    
                except Exception as e:
                    if self.config.show_debug:
                        print(f"手動デコードでエラー: {e}")
                
                # 方法3: 音声ファイルを一時的に保存して処理
                try:
                    import tempfile
                    import soundfile as sf
                    
                    # 一時ファイルに音声を保存
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        sf.write(temp_file.name, audio_segment, self.config.sample_rate)
                        
                        # ファイルパスを使って文字起こし
                        result = self.model.transcribe(paths2audio_files=[temp_file.name])
                        
                        # ファイルを削除
                        import os
                        os.unlink(temp_file.name)
                        
                        if result and len(result) > 0:
                            if hasattr(result[0], 'text'):
                                return result[0].text
                            elif isinstance(result[0], str):
                                return result[0]
                            else:
                                return str(result[0])
                                
                except Exception as e:
                    if self.config.show_debug:
                        print(f"ファイルベース文字起こしでエラー: {e}")
                
                if self.config.show_debug:
                    print("すべての文字起こし方法が失敗しました")
                return ""

        except Exception as e:
            print(f"RNN-Tデコードエラー: {e}")
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報の取得"""
        device_info = get_device_info()
        
        return {
            "model_name": self.config.model_name,
            "device": self.config.device,
            "sample_rate": self.config.sample_rate,
            "device_info": device_info,
            "config": self.config.to_dict()
        } 