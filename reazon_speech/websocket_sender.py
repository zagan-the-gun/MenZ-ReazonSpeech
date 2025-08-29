#!/usr/bin/env python3
"""
シンプルなWebSocket送信機能
テストで成功した方法と同じ実装
"""
import asyncio
import websockets
import threading
import json
import time
from .config import ModelConfig

class WebSocketSender:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.websocket = None
        self.is_connected = False
        self.loop = None
        self.thread = None
        self.stop_event = None

    def start(self):
        """WebSocket送信を開始"""
        if not self.config.websocket_enabled:
            return
            
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run_websocket_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """WebSocket送信を停止"""
        if self.stop_event:
            self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def _run_websocket_loop(self):
        """asyncioループを別スレッドで実行"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._websocket_manager())
        except Exception as e:
            if self.config.show_debug:
                print(f"WebSocketループエラー: {e}")
        finally:
            if self.loop:
                self.loop.close()

    async def _websocket_manager(self):
        """WebSocket接続を管理"""
        uri = f"ws://{self.config.websocket_host}:{self.config.websocket_port}"
        
        while not self.stop_event.is_set():
            try:
                # 接続試行
                websocket = await asyncio.wait_for(
                    websockets.connect(uri, ping_interval=20, ping_timeout=10),
                    timeout=5.0
                )
                
                async with websocket:
                    self.websocket = websocket
                    self.is_connected = True
                    
                    # 接続完了メッセージを送信
                    message = self._format_message("MenZ-ReazonSpeech接続完了")
                    await self.websocket.send(message)
                    
                    # 接続を維持（閉塞を検知したら抜ける）
                    while not self.stop_event.is_set():
                        try:
                            # 1秒だけ閉塞を待ってタイムアウトしたら継続
                            await asyncio.wait_for(self.websocket.wait_closed(), timeout=1.0)
                            # 閉塞した
                            break
                        except asyncio.TimeoutError:
                            # まだ開いている
                            continue
                        
            except (asyncio.TimeoutError, ConnectionRefusedError, Exception) as e:
                if self.config.show_debug:
                    if isinstance(e, asyncio.TimeoutError):
                        print("WebSocket接続タイムアウト")
                    elif isinstance(e, ConnectionRefusedError):
                        print("WebSocket接続拒否（サーバーが起動していない可能性）")
                    else:
                        print(f"WebSocket接続エラー: {e}")
                
            # 接続失敗時は状態をリセット
            self.is_connected = False
            self.websocket = None
            
            if not self.stop_event.is_set():
                await asyncio.sleep(3)

    def send_text(self, text: str):
        """テキストを送信"""
        if not self.config.websocket_enabled:
            return
            
        if not text.strip():
            return
        
        # 接続が切れている場合、再接続を待つ
        if not self.is_connected or not self.websocket:
            if self.config.show_debug:
                print("WebSocket接続が切れています。再接続を待機中...")
            
            # 最大5秒間、再接続を待つ
            max_wait_time = 5.0
            wait_interval = 0.1
            waited_time = 0.0
            
            while waited_time < max_wait_time and (not self.is_connected or not self.websocket):
                time.sleep(wait_interval)
                waited_time += wait_interval
            
            # まだ接続できていない場合は送信をスキップ
            if not self.is_connected or not self.websocket:
                if self.config.show_debug:
                    print("WebSocket再接続に失敗しました。メッセージ送信をスキップします。")
                return
            else:
                if self.config.show_debug:
                    print("WebSocket再接続が完了しました。メッセージを送信します。")
        
        try:
            message = self._format_message(text)
            
            if self.loop and not self.loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(
                    self._send_message_simple(message), 
                    self.loop
                )
                future.result(timeout=2.0)
                
        except Exception as e:
            if self.config.show_debug:
                print(f"WebSocket送信エラー: {e}")
                import traceback
                traceback.print_exc()
            
            # 接続エラー時は状態を更新して再接続を試行
            self.is_connected = False
            self.websocket = None
            
            if self.config.show_debug:
                print("WebSocket接続が切れました。再接続を待機中...")
            
            # 最大5秒間、再接続を待つ
            max_wait_time = 5.0
            wait_interval = 0.1
            waited_time = 0.0
            
            while waited_time < max_wait_time and (not self.is_connected or not self.websocket):
                time.sleep(wait_interval)
                waited_time += wait_interval
            
            # 再接続が成功した場合、再度送信を試行
            if self.is_connected and self.websocket:
                if self.config.show_debug:
                    print("WebSocket再接続が完了しました。メッセージを再送信します。")
                try:
                    message = self._format_message(text)
                    if self.loop and not self.loop.is_closed():
                        future = asyncio.run_coroutine_threadsafe(
                            self._send_message_simple(message), 
                            self.loop
                        )
                        future.result(timeout=2.0)
                except Exception as retry_e:
                    if self.config.show_debug:
                        print(f"WebSocket再送信でもエラー: {retry_e}")
            else:
                if self.config.show_debug:
                    print("WebSocket再接続に失敗しました。メッセージ送信をスキップします。")

    async def _send_message_simple(self, message: str):
        """メッセージ送信"""
        if self.websocket and self.is_connected:
            await self.websocket.send(message)

    def _format_message(self, text: str) -> str:
        """メッセージを形式に応じてフォーマット"""
        if self.config.text_type == 0:
            # ゆかりねっと形式（プレーンテキスト）
            return text
        elif self.config.text_type == 1:
            # ゆかコネNEO形式（JSON）
            payload = {"text": text}
            if getattr(self.config, 'websocket_subtitle', None):
                payload["subtitle"] = self.config.websocket_subtitle
            return json.dumps(payload, ensure_ascii=False)
        else:
            return text