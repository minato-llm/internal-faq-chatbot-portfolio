# lambda_functions/document_search/kendra_search_function.py
import json
import os
from aws_lambda_powertools import Logger
from langchain_community.retrievers import AmazonKendraRetriever
import boto3

# ロガーの初期化
logger = Logger()

# 環境変数からKendraインデックスIDを取得
KENDRA_INDEX_ID = os.environ.get("KENDRA_INDEX_ID")
# AWSリージョン
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-1")

@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    """検索クエリを受け取り、Amazon Kendraを使用して関連ドキュメントを検索する"""
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
        
        if not KENDRA_INDEX_ID:
            logger.error("KENDRA_INDEX_ID環境変数が設定されていません")
            return {
                "statusCode": 500,
                "body": json.dumps({
                    "error": "KENDRA_INDEX_ID環境変数が設定されていません"
                }, ensure_ascii=False)
            }
            
        logger.info(f"検索クエリ: '{query_text}'、KendraインデックスID: {KENDRA_INDEX_ID}")        
        # Kendraクライアントの初期化
        kendra_client = boto3.client("kendra", region_name=AWS_REGION)
        
        # LangChain AmazonKendraRetriever でドキュメント検索
        kendra_retriever = AmazonKendraRetriever(
            index_id=KENDRA_INDEX_ID,
            client=kendra_client,
            top_k=5,  # 最も関連性の高い最大5件を取得
            attribute_filter={
                'EqualsTo': {
                    'Key': '_language_code',
                    'Value': {
                        'StringValue': 'ja'
                    }
                }
            }
        )
        
        # 関連ドキュメントの取得
        related_documents = kendra_retriever.get_relevant_documents(query_text)
        
        # ドキュメント内容を抽出して整形
        document_contents = []
        for doc in related_documents:
            # メタデータを含む場合は必要に応じて追加
            document_contents.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        logger.info(f"検索結果件数: {len(document_contents)}件")       
        # ドキュメント検索で取得した関連ドキュメントの内容とメタデータを返却
        return {
            "statusCode": 200,
            "body": json.dumps({
                "related_documents": document_contents
            }, ensure_ascii=False)
        }
        
    except Exception as e:
        logger.exception(f"Kendra検索エラー: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Kendra検索でエラーが発生しました",
                "details": str(e)
            }, ensure_ascii=False)
        }