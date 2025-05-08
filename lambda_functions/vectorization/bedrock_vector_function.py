import json
import os
from aws_lambda_powertools import Logger
from langchain_community.embeddings import BedrockEmbeddings
import boto3

# ロガーの初期化
logger = Logger()

# Bedrockの設定
BEDROCK_REGION_NAME = os.environ.get("BEDROCK_REGION_NAME", "ap-northeast-1")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    """
    クエリテキストをBedrockモデルを使用してベクトル化
    """
    try:
        # クエリテキストの取得
        query_text = event.get("query_text")
        
        if not query_text:
            logger.warning("クエリテキストが送信されていません")
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "クエリテキストが送信されていません"
                }, ensure_ascii=False)
            }
        
        # Bedrockクライアントの作成
        bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION_NAME)
        
        # LangChain BedrockEmbeddingsでテキストをベクトル化
        embeddings = BedrockEmbeddings(
            model_id=EMBEDDING_MODEL_ID,
            region_name=BEDROCK_REGION_NAME,
            client=bedrock_client
        )
        
        # テキストのベクトル化を実行
        query_vector = embeddings.embed_query(query_text)
        
        logger.info(f"ベクトル化完了: ベクトル次元数: {len(query_vector)}")       
        # ベクトル化結果とモデル情報をレスポンスとして返却
        return {
            "statusCode": 200,
            "body": json.dumps({
                "query_vector": query_vector,
                "model_id": EMBEDDING_MODEL_ID
            }, ensure_ascii=False)
        }

    except Exception as e:
        logger.exception(f"Bedrockベクトル化エラー: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Bedrockベクトル化でエラーが発生しました",
                "details": str(e)
            }, ensure_ascii=False)
        }