# fastapi/fastapi_app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import boto3
import asyncio
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
import logging

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# AWS Lambdaクライアントの初期化
lambda_client = boto3.client("lambda", region_name="ap-northeast-1")
TEXT_PREPROCESSOR_LAMBDA_FUNCTION_NAME = "lambda-preprocessor-lambda"
BEDROCK_VECTOR_LAMBDA_FUNCTION_NAME = "bedrock-vector-lambda"
KENDRA_SEARCH_LAMBDA_FUNCTION_NAME = "kendra-search-lambda"

# LangChain Bedrock LLMの初期化
bedrock_llm = ChatBedrock(
    model_id="arn:aws:bedrock:ap-northeast-1:545009855730:inference-profile/apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="ap-northeast-1",
    provider="anthropic"
)

@app.post("/chat")
async def chat_endpoint(request: Request):
    """社内FAQチャットボットのエンドポイント"""
    try:
        request_body = await request.json()
        user_message = request_body.get("message")

        if not user_message:
            return JSONResponse({"error": "メッセージが送信されていません"}, status_code=400)
        
        logger.info("LLMの回答生成を開始します")
        # AWS Lambda (前処理) 関数を呼び出す
        lambda_response_preprocess = await call_lambda_function(TEXT_PREPROCESSOR_LAMBDA_FUNCTION_NAME, {"message": user_message})
        processed_message = lambda_response_preprocess.get("processed_message")
      
        # AWS Lambda (ベクトル化) 関数を呼び出す
        lambda_response_vectorize = await call_lambda_function(BEDROCK_VECTOR_LAMBDA_FUNCTION_NAME, {"query_text": processed_message})
        query_vector = lambda_response_vectorize.get("query_vector")

        # AWS Lambda (Kendra検索) 関数を呼び出す
        lambda_response_kendra = await call_lambda_function(KENDRA_SEARCH_LAMBDA_FUNCTION_NAME, {"query_text": processed_message})
        related_documents = lambda_response_kendra.get("related_documents", [])
        
        # RAGプロンプトを作成
        context_texts = []
        for doc in related_documents:
            content = doc.get("content", "")
            context_texts.append(content)
        rag_prompt = create_rag_prompt(processed_message, context_texts)

        # LangChain Bedrock LLMで回答生成
        messages = [HumanMessage(content=rag_prompt)]
        ai_response = bedrock_llm.invoke(messages)
        response_text = ai_response.content
        logger.info("LLMからの回答生成が完了しました")

        # 生成した回答を返却
        final_response = {"response": response_text}
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


def create_rag_prompt(query: str, documents: list[str]) -> str:
    """社内FAQチャットボット用のRAGプロンプトを作成する"""
    context = "\n".join(documents)
    prompt = f"""
    社内FAQチャットボットです。関連ドキュメントを参考にして、質問に答えてください。
    もし関連ドキュメントに答えがない場合は、「関連ドキュメントに回答がありませんでした。」と回答してください。

    関連ドキュメント:
    {context}

    質問:
    {query}

    回答:
    """
    return prompt