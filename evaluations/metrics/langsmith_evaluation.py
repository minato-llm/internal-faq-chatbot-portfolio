# evaluations/metrics/langsmith_evaluation.py
import os
import sys
import json
import requests
import time
from pathlib import Path
from langsmith import Client
from langsmith.run_trees import RunTree
from langchain_core.tracers.schemas import Run
from langchain_core.messages import HumanMessage
from langchain_aws import ChatBedrock
import boto3
from botocore.config import Config

# LangSmith設定
client = Client()
project_name = os.environ.get("LANGCHAIN_PROJECT", "internal_faq_chatbot_evaluation")

# Bedrockのレート制限対策: リトライ設定
boto3_config = Config(
    retries={
        'max_attempts': 10,  # 最大リトライ回数
        'mode': 'adaptive'   # 適応モードでリトライ間隔を自動調整
    }
)

# Bedrockの設定
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
BEDROCK_ID = os.environ.get("BEDROCK_ID")
BEDROCK_PROVIDER = os.environ.get("BEDROCK_PROVIDER")

# FastAPIアプリケーションのエンドポイント
API_ENDPOINT = "http://localhost:8000/chat"

# Bedrock LLMの初期化
bedrock_llm = ChatBedrock(
    model_id=BEDROCK_ID,
    region_name=AWS_REGION,
    provider=BEDROCK_PROVIDER,
    config=boto3_config
)

def run_langsmith_evaluation():
    """LangSmith評価を実行してトレースを記録"""
    
    try:
        # テスト用データの読み込み
        test_questions_path = Path(__file__).parent.parent / "data" / "test_questions.json"
        
        if not test_questions_path.exists():
            print(f"テストデータファイルが見つかりません: {test_questions_path}")
            return
            
        with open(test_questions_path, "r", encoding="utf-8") as f:
            test_questions = json.load(f)
        
        print(f"{len(test_questions)}件の質問でLangSmith評価を開始します...")
        
        # テスト実行
        for i, test_case in enumerate(test_questions, 1):
            print(f"質問 {i}/{len(test_questions)}: {test_case['question'][:50]}...")
            
            # エラーハンドリング: 3回までリトライ
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    trace_api_request(test_case["question"], project_name, API_ENDPOINT)
                    break  # 成功したらループを抜ける
                except Exception as e:
                    if attempt < max_retries - 1:
                        retry_wait = 15 * (attempt + 1)  # 15秒、30秒、45秒と待機時間を増やす
                        print(f"エラー発生: {e}")
                        print(f"{retry_wait}秒後に再試行します（{attempt+1}/{max_retries}）...")
                        time.sleep(retry_wait)
                    else:
                        print(f"最大リトライ回数に達しました: {e}")
            
            # 次のリクエスト前に30秒待機（レート制限対策）
            if i < len(test_questions):
                time.sleep(30)
        
        print(f"評価が完了しました。")
        print(f"LangSmith UIで結果を確認: https://smith.langchain.com/projects/{project_name}")
        
    except Exception as e:
        print(f"評価実行中にエラー: {e}")

def trace_api_request(question, project_name, api_endpoint):
    """LangSmithでトレースするAPIリクエスト関数"""
    run_tree = RunTree(
        name="faq_chatbot_evaluation",
        project_name=project_name,
        inputs={"question": question}
    )
    
    try:
        response = requests.post(
            api_endpoint,
            headers={"Content-Type": "application/json"},
            json={"message": question, "session_id": None, "messages_history": []},
            timeout=30
        )
        
        response.raise_for_status()  # HTTPエラーをチェック
        result = response.json()
        
        # 結果を記録
        run_tree.end(
            outputs={
                "response": result.get("response"), 
                "related_documents": result.get("related_documents")
            }
        )
        
        return result
        
    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {e}"
        run_tree.end(error=error_msg)
        print(f"APIリクエストエラー: {error_msg}")
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        run_tree.end(error=error_msg)
        print(f"予期しないエラー: {error_msg}")

if __name__ == "__main__":
    run_langsmith_evaluation() 