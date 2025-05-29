# lambda_functions/document_search/bedrock_kb_search_function.py
import json
import os
from aws_lambda_powertools import Logger
import boto3

# ロガーの初期化
logger = Logger()

# 環境変数から設定を取得
AWS_REGION = os.environ.get("AWS_REGION")
BEDROCK_KB_ID = os.environ.get("BEDROCK_KB_ID")

@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    """検索クエリを受け取り、Amazon Bedrock Knowledge Baseを使用して関連ドキュメントを検索する"""
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
        
        if not BEDROCK_KB_ID:
            logger.error("BEDROCK_KB_ID環境変数が設定されていません")
            return {
                "statusCode": 500,
                "body": json.dumps({
                    "error": "BEDROCK_KB_ID環境変数が設定されていません"
                }, ensure_ascii=False)
            }
            
        logger.info(f"検索クエリ: '{query_text}'、Bedrock Knowledge Base ID: {BEDROCK_KB_ID}")
        
        # Bedrock Knowledge Baseクライアントの初期化
        bedrock_kb_client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
        
        # Bedrock Knowledge Base APIを呼び出し
        response = bedrock_kb_client.retrieve(
            knowledgeBaseId=BEDROCK_KB_ID,
            retrievalQuery={
                "text": query_text
            },
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 5
                }
            }
        )
        
        # レスポンスから関連ドキュメントを抽出
        retrieved_results = response.get("retrievalResults", [])
        
        # 関連ドキュメントの情報を整形
        document_contents = []
        for result in retrieved_results:
            content = result.get("content", {}).get("text", "")
            metadata = result.get("metadata", {})
            
            # メタデータからタイトルを取得
            title = metadata.get("title", "不明なドキュメント")

            document_contents.append({
                "content": content,
                "metadata": {
                    "title": title
                }
            })
        
        logger.info(f"検索結果件数: {len(document_contents)}件")
        # 検索結果の関連ドキュメントの内容とメタデータを返却
        return {
            "statusCode": 200,
            "body": json.dumps({
                "related_documents": document_contents
            }, ensure_ascii=False)
        }
        
    except Exception as e:
        logger.exception(f"Bedrock Knowledge Base検索エラー: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Bedrock Knowledge Base検索でエラーが発生しました",
                "details": str(e)
            }, ensure_ascii=False)
        }