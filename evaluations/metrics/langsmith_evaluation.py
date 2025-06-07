import os
from langsmith import Client
from langsmith.run_trees import RunTree
from langchain_core.tracers.schemas import Run
from langchain_core.messages import HumanMessage
from langchain_aws import ChatBedrock
import boto3
import json

# LangSmith設定
client = Client()
project_name = os.environ.get("LANGCHAIN_PROJECT", "internal_faq_chatbot_evaluation")

# Bedrockの設定
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
BEDROCK_ID = os.environ.get("BEDROCK_ID")
BEDROCK_PROVIDER = os.environ.get("BEDROCK_PROVIDER")

# テスト用データの読み込み
with open("test_questions.json", "r", encoding="utf-8") as f:
    test_questions = json.load(f)

# FastAPIアプリケーションのエンドポイント
API_ENDPOINT = "http://localhost:8000/chat"

# Bedrock LLMの初期化
bedrock_llm = ChatBedrock(
    model_id=BEDROCK_ID,
    region_name=AWS_REGION,
    provider=BEDROCK_PROVIDER
)

# LangSmithでトレースするAPIリクエスト関数
def trace_api_request(question, expected_answer=None):
    run_tree = RunTree(
        name="faq_chatbot_evaluation",
        project_name=project_name,
        inputs={"question": question, "expected_answer": expected_answer}
    )
    
    try:
        # APIリクエストの実行コード（requests等を使用）
        import requests
        
        response = requests.post(
            API_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json={"message": question, "session_id": None, "messages_history": []}
        )
        
        result = response.json()
        
        # 結果を記録
        run_tree.end(
            outputs={"response": result.get("response"), "related_documents": result.get("related_documents")}
        )
        
        return result
    except Exception as e:
        run_tree.end(error=str(e))
        raise e

# テスト実行
for test_case in test_questions:
    trace_api_request(test_case["question"], test_case.get("expected_answer"))

print(f"評価が完了しました。LangSmith UIで結果を確認してください: https://smith.langchain.com/projects/{os.environ.get('LANGCHAIN_PROJECT')}") 