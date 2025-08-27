"""
WebSocket音声受信サーバ

Unityからの16kHz/モノラル/PCM16LEバイナリフレームを受け取り、
RealtimeTranscriber にフレームを流し込む。
"""

import asyncio
import json
import threading
from typing import Optional

import numpy as np
import websockets

from .config import ModelConfig
from .realtime import RealtimeTranscriber


class WebSocketAudioServer:
    """WebSocketでPCM16LE音声を受信して転送するサーバ"""

    def __init__(self, config: ModelConfig, transcriber: RealtimeTranscriber):
        self.config = config
        self.transcriber = transcriber
        self.host = getattr(config, 'audio_ws_host', '0.0.0.0')
        self.port = int(getattr(config, 'audio_ws_port', 60060))
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()
        self._clients: set[websockets.WebSocketServerProtocol] = set()

    async def _handle_client(self, websocket):
        """クライアント1接続のハンドラ"""
        # 初期メッセージ（任意）を受け取ってもよい
        self._clients.add(websocket)
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # PCM16LE -> float32[-1,1]
                    if len(message) == 0:
                        continue
                    pcm16 = np.frombuffer(message, dtype=np.int16)
                    audio_f32 = (pcm16.astype(np.float32) / 32767.0)
                    # モノラル前提（Unity送信がモノラル）
                    try:
                        self.transcriber.ingest_audio_block(audio_f32)
                    except Exception as e:
                        # 例外は接続を落とさない
                        print(f"ingest_audio_block error: {e}")
                else:
                    # 仕様通り: JSONテキストのみ受理
                    try:
                        data = json.loads(message)
                    except Exception:
                        continue
                    msg_type = data.get('type')
                    if msg_type == 'hello':
                        if getattr(self.config, 'show_debug', False):
                            print("[WS] hello received")
                        await websocket.send(json.dumps({
                            'type': 'ok',
                            'version': 'pcm16.v1',
                            'sample_rate': self.config.sample_rate,
                            'channels': 1,
                        }))
                        if getattr(self.config, 'show_debug', False):
                            print("[WS] ok sent")
                    elif msg_type == 'eos':
                        # ストリーム終了通知（必要なら処理）
                        pass
                    elif msg_type == 'flush':
                        # 収集中の音声を即時確定
                        try:
                            self.transcriber.finalize_from_flush()
                            if getattr(self.config, 'show_debug', False):
                                print("[WS] flush requested -> segment forced")
                        except Exception as e:
                            if getattr(self.config, 'show_debug', False):
                                print(f"[WS] flush error: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            try:
                self._clients.discard(websocket)
            except Exception:
                pass

    async def _run_server(self):
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            subprotocols=['pcm16.v1'],
            max_size=None,
            compression=None,
        ):
            # 起動通知
            print(f"WebSocketAudioServer listening on ws://{self.host}:{self.port} (pcm16.v1)")
            # 停止イベント待ち
            while not self._stop_event.is_set():
                await asyncio.sleep(0.1)

    async def _broadcast_text_async(self, text: str):
        """接続中の全クライアントへテキストを送信（イベントループ内）"""
        if not self._clients:
            return
        to_remove = []
        for ws in list(self._clients):
            try:
                await ws.send(text)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            try:
                self._clients.discard(ws)
            except Exception:
                pass

    def broadcast_text(self, text: str):
        """他スレッドから呼び出して、全クライアントへテキスト送信"""
        if not text:
            return
        if not self._loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast_text_async(text), self._loop)
        except Exception as e:
            if getattr(self.config, 'show_debug', False):
                print(f"[WS] broadcast error: {e}")

    def start(self):
        """サーバ開始（バックグラウンド）"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._run_server())
            finally:
                pending = asyncio.all_tasks(loop=self._loop)
                for task in pending:
                    task.cancel()
                try:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
                self._loop.close()

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    def stop(self):
        """サーバ停止"""
        self._stop_event.set()
        if self._loop:
            try:
                asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self._loop).result(timeout=1)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

