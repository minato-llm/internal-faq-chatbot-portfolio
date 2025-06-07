import json
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
)
from ragas import evaluate
from datasets import Dataset
from langchain_aws import BedrockEmbeddings
import boto3
import os
import nltk
from langchain_aws import ChatBedrock
import pandas as pd
from ragas.run_config import RunConfig
from botocore.config import Config
import requests
import time
from ragas.llms import LangchainLLMWrapper

# 仮想環境内のNLTKデータパスを追加
venv_nltk_data = os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'nltk_data')
nltk.data.path.append(venv_nltk_data)

# AWS設定
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
BEDROCK_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
BEDROCK_ID = os.environ.get("BEDROCK_ID")
BEDROCK_PROVIDER = os.environ.get("BEDROCK_PROVIDER")

# テスト用データの準備 - 質問と模範回答のソースとして使用
test_data = [
    {
        "question": "年次有給休暇は入社してからどれくらいで付与されますか？",
    },
    {
        "question": "休日に勤務した場合の代休制度について教えてください",
    },
    {
        "question": "遅刻や欠勤をする場合、どのように連絡すればよいですか？",
    },
    {
        "question": "遅刻や欠勤をする場合、どのように連絡すればよいですか？",
    },
    {
        "question": "休暇を取得する際の申請方法を教えてください",
    }
]

# AWS設定 - より保守的なタイムアウトとリトライ設定
config = Config(
    connect_timeout=120,  # 接続タイムアウト
    read_timeout=900,     # 読み取りタイムアウト
    retries={
        "max_attempts": 6,  # リトライ回数
        "mode": "adaptive",
        "total_max_attempts": 10
    }
)

# Bedrockクライアントの初期化
bedrock_client = boto3.client(
    "bedrock-runtime", 
    region_name=AWS_REGION,
    config=config
)

# LLM用のクライアントを別途作成
bedrock_runtime_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=config
)

# Bedrock LLMの初期化
bedrock_llm = ChatBedrock(
    model_id=BEDROCK_ID,
    region_name=AWS_REGION,
    provider=BEDROCK_PROVIDER,
    client=bedrock_runtime_client,
    # 日本語statement抽出のための設定追加
    model_kwargs={
        "temperature": 0.1,  # より確定的な出力
        "top_p": 0.9,
        "max_tokens": 2048
    }
)

# Bedrockエンベディングの設定
embeddings = BedrockEmbeddings(
    model_id=BEDROCK_MODEL_ID,
    region_name=AWS_REGION,
    client=bedrock_client
)

# 実行設定：安定性重視
run_config = RunConfig(
    timeout=600,
    max_workers=1  # スロットリング回避
)

# 安全にfloat値に変換する関数
def safe_float_conversion(value):
    """安全にfloat値に変換する関数"""
    if value is None:
        return None
    
    # 配列の場合は平均値を取る
    if hasattr(value, '__iter__') and not isinstance(value, str):
        try:
            # numpy配列やリストの場合
            import numpy as np
            arr = np.array(value)
            # NaN値を除外して平均値を計算
            arr_clean = arr[~np.isnan(arr)]
            return float(arr_clean.mean()) if len(arr_clean) > 0 else None
        except:
            return None
    
    # スカラー値の場合
    try:
        if pd.isna(value):
            return None
        return float(value)
    except:
        return None

# EvaluationResultオブジェクトから値を安全に取得する関数
def get_metric_value(results, metric_name):
    """EvaluationResultオブジェクトから指定されたメトリック値を取得"""
    try:
        # 辞書形式でのアクセスを試行
        if hasattr(results, '__getitem__'):
            return results[metric_name]
        # 属性アクセスを試行
        elif hasattr(results, metric_name):
            return getattr(results, metric_name)
        # to_pandas()でDataFrameに変換してからアクセス
        elif hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            if metric_name in df.columns:
                return df[metric_name].iloc[0] if len(df) > 0 else None
        return None
    except Exception as e:
        print(f"メトリック '{metric_name}' の取得中にエラー: {e}")
        return None

def extract_contexts(response_json):
    """レスポンスからコンテキストを抽出"""
    related_docs = response_json.get("related_documents", [])
    contexts = []
    
    for doc in related_docs:
        if isinstance(doc, dict):
            if "content" in doc and doc["content"]:
                content = doc["content"].strip()
                if content:
                    contexts.append(content)
            elif "title" in doc and doc["title"]:
                title = doc["title"].strip()
                if title:
                    contexts.append(title)
        elif isinstance(doc, str) and doc.strip():
            contexts.append(doc.strip())
    
    return contexts

def evaluate_model_answers():
    """RAGシステムの回答を評価"""

    # 1. テストデータから質問と模範回答を抽出
    test_questions = [item["question"] for item in test_data]
    
    # 2. 実際のRAGシステムに質問を送信して回答を取得
    actual_answers = []
    contexts_list = []
    
    for i, question in enumerate(test_questions):
        print(f"\n質問 {i+1}/{len(test_questions)}: {question}")
        
        # 最初の質問以外は15秒待機
        if i > 0:
            wait_time = 15
            print(f"スロットリング回避のため、{wait_time}秒待機します...")
            time.sleep(wait_time)
        
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={"message": question},
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                if answer:
                    actual_answers.append(answer)
                    print(f"✓ 回答取得成功: {answer[:50]}...")
                    
                    # コンテキスト抽出
                    contexts = extract_contexts(result)
                    
                    # 空のコンテキストの場合はフォールバック
                    if not contexts:
                        contexts = [f"関連する質問内容: {question}"]
                    
                    contexts_list.append(contexts)
                    print(f"  最終コンテキスト数: {len(contexts)}")
                else:
                    print("✗ 空の回答のため除外")
                    actual_answers.append("無効な回答")
                    contexts_list.append(["無効なコンテキスト"])
            else:
                print(f"✗ HTTPエラー: {response.status_code}")
                actual_answers.append("HTTPエラー")
                contexts_list.append(["HTTPエラー"])
                
        except Exception as e:
            print(f"✗ 例外エラー: {e}")
            actual_answers.append("例外エラー")
            contexts_list.append(["例外エラー"])
    
    # 3. 評価データセットを作成
    dataset_items = []
    for q, actual, contexts in zip(test_questions, actual_answers, contexts_list):
        dataset_items.append({
            "user_input": q,
            "response": actual,
            "retrieved_contexts": contexts
        })
    
    # 4. データ検証
    valid_items = []
    for item in dataset_items:
        # 最低限の条件のみチェック
        if (item.get("response", "").strip() and 
            item.get("retrieved_contexts") and 
            len(item.get("retrieved_contexts", [])) > 0 and
            len(item.get("response", "").strip()) > 5):  # 最小文字数チェック追加
            valid_items.append(item)
    
    if not valid_items:
        return {
            "error": "すべてのデータが無効です",
            "total_questions": len(test_questions),
            "valid_responses": 0
        }
    
    print(f"\n評価に使用するデータ数: {len(valid_items)}")
    
    # 5. データセットを作成
    eval_dataset = Dataset.from_list(valid_items)
    
    # 6. 評価実行
    try:
        # Faithfulness専用の設定
        llm_wrapper = LangchainLLMWrapper(bedrock_llm)
        
        # Faithfulness メトリックを個別設定
        faithfulness_metric = Faithfulness(
            llm=llm_wrapper
        )
        
        # 評価実行
        results = evaluate(
            dataset=eval_dataset,
            metrics=[
                faithfulness_metric,
                AnswerRelevancy(),
            ],
            embeddings=embeddings,
            llm=bedrock_llm,
            run_config=run_config
        )
        
        print("✓ 評価完了")
        
        # 結果の抽出
        serializable_results = {
            "faithfulness": safe_float_conversion(get_metric_value(results, "faithfulness")),
            "answer_relevancy": safe_float_conversion(get_metric_value(results, "answer_relevancy")),
            "details": [],
            "summary": {
                "total_samples": len(eval_dataset),
                "evaluation_metrics": ["faithfulness", "answer_relevancy"]
            }
        }
        
        # 詳細データ
        df = results.to_pandas()
        for i in range(len(eval_dataset)):
            row_data = {
                "question": eval_dataset[i]["user_input"],
                "answer": eval_dataset[i]["response"][:200] + "..." if len(eval_dataset[i]["response"]) > 200 else eval_dataset[i]["response"],
                "contexts_count": len(eval_dataset[i]["retrieved_contexts"]),
                "faithfulness": safe_float_conversion(df.iloc[i].get("faithfulness")),
                "answer_relevancy": safe_float_conversion(df.iloc[i].get("answer_relevancy"))
            }
            serializable_results["details"].append(row_data)
        
        # 結果の保存
        with open("ragas_evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        return serializable_results
        
    except Exception as eval_error:
        print(f"✗ 評価中にエラー: {eval_error}")
        return {
            "error": str(eval_error),
            "total_questions": len(test_questions),
            "valid_responses": len(valid_items)
        }

# メイン処理
if __name__ == "__main__":
    print("実際のRAGシステムを評価開始...")
    results = evaluate_model_answers()
    print("\n最終結果:")
    for metric, value in results.items():
        if metric not in ["details", "summary"]:
            print(f"{metric}: {value}") 