#!/usr/bin/env python3
"""
MenZ-ReazonSpeech メイン実行ファイル
MCPクライアントとしてzagaroidに接続し、音声認識リクエストを処理します。
"""

import sys
import asyncio
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from reazon_speech.config import ModelConfig
from reazon_speech.model import ReazonSpeechModel
from reazon_speech.mcp_client import MCPClient
from reazon_speech.jsonrpc_handler import JSONRPCHandler


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """メイン関数"""
    try:
        # 設定の読み込み
        config_path = "config.ini"
        try:
            config = ModelConfig.from_ini(config_path)
            logger.info(f"設定ファイルを読み込みました: {config_path}")
        except Exception as e:
            logger.warning(f"設定ファイルの読み込みに失敗: {e}")
            logger.info("デフォルト設定を使用します")
            config = ModelConfig()
        
        # デバイスの自動選択
        if config.device == "auto":
            import torch
            if torch.cuda.is_available():
                config.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                config.device = "mps"
            else:
                config.device = "cpu"
            logger.info(f"自動デバイス選択: {config.device}")
        
        # モデルのロード
        logger.info(f"音声認識モデルをロード中: {config.model_name}")
        logger.info(f"デバイス: {config.device}")
        model = ReazonSpeechModel(config)
        logger.info("モデルのロードが完了しました")
        
        # JSON-RPCハンドラーの初期化
        jsonrpc_handler = JSONRPCHandler(model, config)
        
        # MCPクライアントの初期化
        mcp_client = MCPClient(config, model, jsonrpc_handler)
        
        # シャットダウンイベントの設定
        shutdown_event = asyncio.Event()
        
        def signal_handler():
            logger.info("シャットダウンシグナルを受信しました")
            shutdown_event.set()
        
        # シグナルハンドラーの登録
        import signal
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
        
        try:
            # クライアント開始
            await mcp_client.start_client(shutdown_event)
        finally:
            # シグナルハンドラーの解除
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)
        
    except KeyboardInterrupt:
        logger.info("終了しました")
    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
