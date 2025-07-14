"""
リアルタイム音声認識モジュール
"""

import numpy as np
import threading
import time
import queue
from typing import Optional, Callable, List
import webrtcvad
import collections

from .model import ReazonSpeechModel
from .config import ModelConfig
from .utils import AudioProcessor


class RealtimeTranscriber:
    """リアルタイム音声認識クラス"""
    
    def __init__(self, config: Optional[ModelConfig] = None, 
                 callback: Optional[Callable[[str], None]] = None,
                 show_level: bool = False):
        """
        Args:
            config: モデル設定
            callback: 認識結果を受け取るコールバック関数
            show_level: 音声レベル表示の有効/無効
        """
        self.config = config or ModelConfig()
        self.callback = callback
        self.show_level = show_level
        self.model = ReazonSpeechModel(self.config)
        self.audio_processor = AudioProcessor(self.config.sample_rate, self.config.vad_level)
        
        # 音声入力設定
        self.chunk_size = self.config.chunk_size
        self.channels = 1
        self.rate = self.config.sample_rate
        
        # 状態管理
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # VAD設定
        self.vad = webrtcvad.Vad(self.config.vad_level)
        self.frame_duration = self.config.frame_duration_ms  # ms
        self.frame_size = int(self.rate * self.frame_duration / 1000)
        
        # 音声バッファ（max_durationに合わせる）
        # 推奨設定（25秒）
        self.audio_buffer = collections.deque(maxlen=int(self.rate * self.config.max_duration))
        
        # カウンタベースの音声検出
        self.speech_counter = 0
        self.silence_counter = 0
        self.audio_data_list = []
        
        # sounddeviceの遅延インポート
        self.sd = None
        
        # デバッグ用の時間管理
        self.start_time = None
        
    def _get_sounddevice(self):
        """sounddeviceの遅延インポート"""
        if self.sd is None:
            import sounddevice as sd
            self.sd = sd
        return self.sd
        
    def list_microphones(self) -> List[dict]:
        """利用可能なマイクの一覧を取得"""
        try:
            sd = self._get_sounddevice()
            devices = sd.query_devices()
            microphones = []
            
            for i, device in enumerate(devices):
                # max_input_channelsを使用
                if device.get('max_input_channels', 0) > 0:
                    microphones.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device.get('max_input_channels', 1),
                        'sample_rate': int(device.get('default_samplerate', 16000))
                    })
            return microphones
        except Exception as e:
            print(f"マイク一覧の取得に失敗: {e}")
            return []
    
    def select_microphone(self) -> Optional[int]:
        """マイク選択"""
        microphones = self.list_microphones()
        
        if not microphones:
            print("利用可能なマイクが見つかりません")
            return None
        
        print("利用可能なマイク:")
        for mic in microphones:
            print(f"  {mic['index']}: {mic['name']} (チャンネル: {mic['channels']}, サンプルレート: {mic['sample_rate']})")
        
        # デフォルトデバイスを表示
        try:
            sd = self._get_sounddevice()
            default_device = sd.query_devices(kind='input')
            print(f"\nデフォルトデバイス: {default_device['index']} - {default_device['name']}")
        except Exception as e:
            print(f"デフォルトデバイスの取得に失敗: {e}")
        
        # ユーザーにマイク選択を促す
        while True:
            try:
                choice = input("\nマイクの番号を選択してください (Enterでデフォルト): ").strip()
                
                if not choice:  # Enterキーの場合
                    try:
                        default_device = sd.query_devices(kind='input')
                        selected_mic = next((mic for mic in microphones if mic['index'] == default_device['index']), None)
                        if selected_mic:
                            print(f"選択されたマイク: {selected_mic['name']}")
                            return default_device['index']
                        else:
                            print("デフォルトデバイスが見つかりません")
                            continue
                    except:
                        print("デフォルトデバイスが見つかりません")
                        continue
                
                choice = int(choice)
                selected_mic = next((mic for mic in microphones if mic['index'] == choice), None)
                if selected_mic:
                    print(f"選択されたマイク: {selected_mic['name']}")
                    return choice
                else:
                    print("無効な選択です。有効な番号を入力してください。")
            except ValueError:
                print("無効な入力です。数字を入力してください。")
            except KeyboardInterrupt:
                print("\nキャンセルされました")
                return None
    
    def audio_callback(self, indata, frames, time, status):
        """音声入力コールバック"""
        if self.is_recording:
            # 音声データを取得
            audio_data = indata[:, 0]  # 最初のチャンネルのみ使用
            
            # VADで音声検出（リアルタイム処理）
            if len(audio_data) >= self.frame_size:
                frame = audio_data[:self.frame_size]
                frame_int16 = (frame * 32767).astype(np.int16)
                is_speech = self.vad.is_speech(frame_int16.tobytes(), self.rate)
                
                # 音声レベルを計算
                level = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                
                # 音声レベル表示
                if hasattr(self, 'show_level') and self.show_level and self.config.show_debug:
                    level_bar = "█" * min(int(level * 20), 20)
                    print(f"\r音声レベル: {level_bar:<20} {level:6.3f}", end="", flush=True)
                
                if is_speech:
                    # 音声を検出
                    self.speech_counter += 1
                    self.silence_counter = 0
                    self.audio_data_list.append(audio_data.flatten())
                    
                    if self.config.show_debug:
                        print(f"\r音声検出: レベル={level:.3f}, VAD={is_speech}, カウンタ={self.speech_counter}", end="", flush=True)
                else:
                    # 無音を検出
                    self.silence_counter += 1
                    self.audio_data_list.append(audio_data.flatten())
                    
                    if self.config.show_debug:
                        print(f"\r無音検出: レベル={level:.3f}, VAD={is_speech}, カウンタ={self.silence_counter}", end="", flush=True)
                    
                    # 無音継続でセグメント終了
                    # ReazonSpeech（RNN-T）はWhisperの25秒制限がないため、無音継続時間のみで分割
                    if self.silence_counter > self.config.pause_threshold:
                        
                        # 最小音声継続時間のチェック
                        if self.speech_counter >= self.config.phrase_threshold:
                            if self.config.show_debug:
                                print(f"\n音声セグメント終了: 音声カウンタ={self.speech_counter}, 無音カウンタ={self.silence_counter}, フレーム数={len(self.audio_data_list)}")
                            
                            # 音声セグメントを処理（別スレッドで実行）
                            audio_segment = np.concatenate(self.audio_data_list)
                            self._process_audio_segment_async(audio_segment)
                        
                        # リセット
                        self.speech_counter = 0
                        self.silence_counter = 0
                        self.audio_data_list = []
    
    def process_audio(self):
        """音声処理スレッド（現在は不要）"""
        # 音声処理はaudio_callback内でリアルタイム実行
        # このメソッドは後方互換性のために残す
        pass
    
    def _process_audio_segment_async(self, audio_segment):
        """音声セグメントの非同期処理"""
        if len(audio_segment) == 0:
            return
        
        # 別スレッドで音声認識を実行
        threading.Thread(target=self._process_audio_segment, args=(audio_segment,), daemon=True).start()
    
    def _process_audio_segment(self, audio_segment):
        """音声セグメントの処理"""
        if len(audio_segment) == 0:
            return
            
        duration = len(audio_segment) / self.rate
        
        if self.config.show_debug:
            print(f"音声セグメント処理開始: 持続時間={duration:.2f}s, サンプル数={len(audio_segment)}")
        
        # 最小持続時間のチェック（phrase_thresholdを秒に変換）
        min_duration = self.config.phrase_threshold * 0.1  # 0.1秒単位を秒に変換
        
        if duration >= min_duration:
            if self.config.show_debug:
                print(f"持続時間OK ({duration:.2f}s >= {min_duration:.2f}s) - 文字起こし開始")
            
            try:
                # 処理開始時間を記録
                processing_start_time = time.time()
                
                result = self.model.transcribe_audio_segment(audio_segment)
                if result and result.strip():
                    # 文章途中の「。」を「、」に変換し、末尾の句読点を削除
                    processed_result = result.strip()
                    # 文章途中の「。」を「、」に変換（末尾以外）
                    if len(processed_result) > 1:
                        # 最後の文字以外の「。」を「、」に変換
                        processed_result = processed_result[:-1].replace('。', '、') + processed_result[-1]
                    # 末尾の句読点を削除
                    processed_result = processed_result.rstrip('。')
                    
                    # 翻訳テキスト表示
                    if self.config.show_transcription:
                        # 処理時間を計算
                        processing_time = time.time() - processing_start_time
                        char_count = len(processed_result)
                        print(f"speech[{processing_time:.4f}]({char_count}){processed_result}")
                    
                    if self.callback:
                        self.callback(processed_result)
                else:
                    if self.config.show_debug:
                        print("文字起こし結果が空です")
            except Exception as e:
                print(f"文字起こしエラー: {e}")
        else:
            if self.config.show_debug:
                print(f"持続時間が短すぎます: {duration:.2f}s < {min_duration:.2f}s")
    
    def start_recording(self, device_index: Optional[int] = None):
        """録音開始"""
        if self.is_recording:
            print("既に録音中です")
            return
        
        # デバイス選択
        if device_index is None:
            device_index = self.select_microphone()
            if device_index is None:
                return
        
        # 時間管理の初期化
        self.start_time = time.time()
        
        # ストリーム開始
        try:
            sd = self._get_sounddevice()
            self.stream = sd.InputStream(
                device=device_index,
                channels=self.channels,
                samplerate=self.rate,
                blocksize=self.chunk_size,
                callback=self.audio_callback,
                dtype=np.float32
            )
            
            self.is_recording = True
            print(f"録音開始 (デバイス: {device_index})")
            print("音声認識を開始しました。話してください...")
            
            # 音声処理スレッド開始
            self.process_thread = threading.Thread(target=self.process_audio)
            self.process_thread.start()
            
            # ストリーム開始
            with self.stream:
                try:
                    while self.is_recording:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    self.stop_recording()
                    
        except Exception as e:
            print(f"録音開始エラー: {e}")
            self.stop_recording()
    
    def stop_recording(self):
        """録音停止"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        
        print("録音停止")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_recording()


def print_result(text: str):
    """認識結果を表示するコールバック"""
    print(f"認識結果: {text}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="リアルタイム音声認識")
    parser.add_argument("--device", type=int, help="マイクデバイス番号")
    parser.add_argument("--verbose", action="store_true", help="詳細出力")
    
    args = parser.parse_args()
    
    # 設定
    config = ModelConfig()
    if args.verbose:
        config.vad_threshold = 0.3  # より敏感に
    
    # リアルタイム認識開始
    with RealtimeTranscriber(config, callback=print_result) as transcriber:
        transcriber.start_recording(args.device)


if __name__ == "__main__":
    main() 