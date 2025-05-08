# lambda_functions/preprocessor/lambda_preprocessor_function.py
import json
import os
from aws_lambda_powertools import Logger
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ロガーの初期化
logger = Logger()

# 環境変数から設定を読み込む（デフォルト値付き）
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 100))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 20))

def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """テキストを指定されたサイズのチャンクに分割する"""
    if not text:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    return text_splitter.split_text(text)

def create_response(status_code, body):
    """Lambda関数のレスポンスを生成する"""
    return {
        "statusCode": status_code,
        "body": json.dumps(body)  # 元のJSONシリアライズ方法を維持
    }

@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    """Lambda関数のメインハンドラー"""
    try:
        # イベントからメッセージを取得
        message = event.get("message")
        if not message:
            logger.warning("メッセージが送信されていません")
            return create_response(400, {"error": "メッセージが送信されていません"})

        logger.info(f"受信メッセージ長: {len(message)}文字")            
        # テキスト分割処理
        chunks = split_text(message)
        
        if not chunks:
            logger.warning("テキスト分割後のチャンクがありません")
            return create_response(200, {"processed_message": ""})
            
        logger.info(f"分割後のチャンク数: {len(chunks)}")        
        # 元のコードと同じく、最初のチャンクのみを返す
        return create_response(200, {"processed_message": chunks[0]})

    except Exception as e:
        logger.exception(f"テキスト処理中にエラーが発生しました: {e}")
        return create_response(500, {
            "error": "サーバーエラーが発生しました", 
            "details": str(e)
        })