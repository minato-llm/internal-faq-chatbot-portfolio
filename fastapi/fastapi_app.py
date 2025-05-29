# fastapi/fastapi_app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import boto3
import asyncio
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
import logging
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 環境変数の初期化
AWS_REGION = os.environ.get("AWS_REGION")
TEXT_PREPROCESSOR_LAMBDA_FUNCTION_NAME = os.environ.get("TEXT_PREPROCESSOR_LAMBDA_FUNCTION_NAME")
BEDROCK_VECTOR_LAMBDA_FUNCTION_NAME = os.environ.get("BEDROCK_VECTOR_LAMBDA_FUNCTION_NAME")
BEDROCK_KB_SEARCH_LAMBDA_FUNCTION_NAME = os.environ.get("BEDROCK_KB_SEARCH_LAMBDA_FUNCTION_NAME")

# AWS Lambdaクライアントの初期化
lambda_client = boto3.client("lambda", region_name=AWS_REGION)

# LangChain Bedrock LLMの初期化
bedrock_llm = ChatBedrock(
    model_id=os.environ.get("BEDROCK_ID"),
    region_name=AWS_REGION,
    provider=os.environ.get("BEDROCK_PROVIDER")
)

@app.post("/chat")
async def chat_endpoint(request: Request):
    """社内FAQチャットボットのエンドポイント"""
    try:
        
        request_body = await request.json()
        user_message = request_body.get("message")
        session_id = request_body.get("session_id")
        messages_history = request_body.get("messages_history", [])

        if not user_message:
            return JSONResponse({"error": "メッセージが送信されていません"}, status_code=400)
        
        logger.info("LLMの回答生成を開始します")
        # AWS Lambda (前処理) 関数を呼び出す
        lambda_response_preprocess = await call_lambda_function(TEXT_PREPROCESSOR_LAMBDA_FUNCTION_NAME, {"message": user_message})
        processed_message = lambda_response_preprocess.get("processed_message")
      
        # AWS Lambda (ベクトル化) 関数を呼び出す
        lambda_response_vectorize = await call_lambda_function(BEDROCK_VECTOR_LAMBDA_FUNCTION_NAME, {"query_text": processed_message})
        query_vector = lambda_response_vectorize.get("query_vector")

        # AWS Lambda (ナレッジ検索) 関数を呼び出す
        lambda_response_bedrock_kb = await call_lambda_function(BEDROCK_KB_SEARCH_LAMBDA_FUNCTION_NAME, {"query_text": processed_message})
        related_documents = lambda_response_bedrock_kb.get("related_documents", [])
        
        # 関連ドキュメントの情報を抽出
        document_info = []
        context_texts = []
        unique_titles = set()

        # 関連ドキュメントが存在する場合、各ドキュメントからタイトルを抽出し、
        # RAG用のコンテキストとして内容を保存
        if related_documents:
            for doc in related_documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                # メタデータから必要な情報を抽出
                title = metadata.get("title", "不明なドキュメント")
                
                # 一意のドキュメントタイトルのみをUI表示用に追加
                if title not in unique_titles:
                    unique_titles.add(title)
                    # ドキュメントタイトル情報を保存
                    document_info.append({
                        "title": title,
                    })              
                # RAG用にコンテキストを追加
                context_texts.append(content)
            
        # 会話履歴を含むRAGプロンプトを作成
        rag_prompt = create_rag_prompt(processed_message, context_texts, messages_history)
        
        # LangChain Bedrock LLMで回答生成
        messages = [HumanMessage(content=rag_prompt)]
        ai_response = bedrock_llm.invoke(messages)
        response_text = ai_response.content
        logger.info("LLMからの回答生成が完了しました")

        # 生成した回答と関連ドキュメント情報を返却
        final_response = {
            "response": response_text,
            "related_documents": document_info
        }
        return JSONResponse(final_response)

    except json.JSONDecodeError:
        logger.error("無効なJSON形式です")
        return JSONResponse({"error": "無効なJSON形式です"}, status_code=400)
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        return JSONResponse({"error": "サーバーエラーが発生しました"}, status_code=500)


async def call_lambda_function(function_name: str, payload: dict):
    """AWS Lambda関数を非同期で呼び出す"""
    try:
        # 現在のイベントループを取得
        loop = asyncio.get_event_loop()
        
        # boto3は同期APIのため、非同期コンテキストで実行するために別スレッドで実行
        # これによりFastAPIのイベントループがブロックされるのを防ぐ
        response = await loop.run_in_executor(
            None,
            lambda: lambda_client.invoke(
                FunctionName=function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload)
            )
        )

        # レスポンスのペイロードをバイトに変換
        payload_bytes = await loop.run_in_executor(
            None,
            lambda: response["Payload"].read()
        )
        
        # バイトを文字列に変換
        payload_str = payload_bytes.decode("utf-8")    
        # 文字列をJSONに変換
        payload_json = json.loads(payload_str)
        
        if response["StatusCode"] == 200:
            logger.info(f"Lambda関数 '{function_name}' の呼び出しが成功しました")
            return json.loads(payload_json["body"])
        else:
            logger.error(f"Lambda関数エラー: {payload_json}")
            return {"error": "Lambda関数の実行に失敗しました", "details": payload_json}

    except Exception as e:
        logger.error(f"Lambda関数の呼び出し中に例外が発生: {e}")
        return {"error": "Lambda関数の呼び出しに失敗しました", "details": str(e)}


def format_conversation_history(messages_history=None, max_messages=60) -> str:
    """会話履歴を整形して文字列として返す"""
    
    if not messages_history:
        return ""
        
    conversation_text = ""
    # 最新の数件の会話のみを使用（トークン制限を考慮）
    recent_messages = messages_history[-max_messages:]
    for message in recent_messages:
        if message["role"] == "user":
            conversation_text += f"ユーザー: {message['content']}\n"
        elif message["role"] == "assistant":
            conversation_text += f"アシスタント: {message['content']}\n"
    
    return f"過去の会話:\n{conversation_text}" if conversation_text else ""

def create_rag_prompt(query: str, documents: list[str], messages_history=None) -> str:
    """社内FAQチャットボット用のRAGプロンプトを作成する"""
    # 関連ドキュメントをコンテキストとして結合
    context = "\n".join(documents)
    # 会話履歴の呼び出し
    past_conversation = format_conversation_history(messages_history)
    
    prompt = f"""
    社内FAQチャットボットです。関連ドキュメントを参考にして、質問に答えてください。
    
    関連ドキュメント:
    {context}
    
    {past_conversation}
    
    質問:
    {query}
    
    回答:
    """
    return prompt