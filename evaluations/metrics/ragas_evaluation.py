# evaluations/metrics/ragas_evaluation.py
import json
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
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
import random
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
    # 会社概要に関する質問
    {
        "question": "本社の所在地を教えてください。",
        "reference": "本社は〒100-0001 東京都千代田区架空町1-1-1 架空ビル10階に所在しています。"
    },
    {
        "question": "当社の代表取締役社長は誰ですか？",
        "reference": "田中陽子が代表取締役社長兼CEOを務めています。"
    },
    {
        "question": "当社の事業年度はいつからいつまでですか？",
        "reference": "当社の事業年度は毎年4月1日から翌年3月31日までです。"
    },
    {
        "question": "当社の従業員数は何名ですか？",
        "reference": "当社の従業員数は連結で550名、単独で500名です（2024年5月20日現在）。"
    },
    {
        "question": "当社の主な製品について教えてください。",
        "reference": "AIチャットボットソリューション「Karaku-Chat」、業務効率化AIシステム「Karaku-Assist」、開発プラットフォーム「Karaku-Cloud」、診断サービス「Karaku-Guard」などがあります。"
    },
    
    # 給与計算規則に関する質問
    {
        "question": "残業手当の計算方法を教えてください。",
        "reference": "残業手当は法定労働時間を超える労働に対し、基本給の1時間当たりの賃金額の1.25倍（月60時間を超える時間外労働については1.5倍）が支給されます。"
    },
    {
        "question": "給与の支給日はいつですか？",
        "reference": "給与の支給日は原則として毎月25日です。ただし、支給日が金融機関休業日の場合は、前営業日となります。"
    },
    {
        "question": "賞与の支給時期はいつですか？",
        "reference": "賞与は原則として年2回、夏季（6月）および冬季（12月）に支給されます。"
    },
    {
        "question": "賞与の算定期間はいつからいつまでですか？",
        "reference": "夏季賞与の算定期間は前年12月1日から当年5月31日まで、冬季賞与の算定期間は当年6月1日から11月30日までです。"
    },
    {
        "question": "深夜労働の手当はどのように計算されますか？",
        "reference": "深夜手当は午後10時から午前5時までの労働に対し、基本給の1時間当たりの賃金額の0.25倍が支給されます。"
    },

    # 勤怠管理マニュアルに関する質問
    {
        "question": "勤怠管理マニュアルはどのような従業員に適用されますか？",
        "reference": "このマニュアルは、会社に勤務するすべての従業員（正社員、契約社員、嘱託社員、パートタイマー、アルバイトを含む）に適用されます。"
    },
    {
        "question": "年次有給休暇はいつ付与されますか？",
        "reference": "入社6ヶ月経過後に10日付与されます。その後、勤続年数1年ごとに1日～2日ずつ加算され、最大で年20日付与されます。"
    },
    {
        "question": "夏季休暇と年末年始休暇の期間はいつですか？",
        "reference": "夏季休暇は原則として毎年8月13日から8月15日までの3日間、年末年始休暇は原則として毎年12月29日から1月3日までの6日間とします。"
    },
    {
        "question": "休暇を取得する際の申請方法を教えてください。",
        "reference": "従業員は休暇を取得する場合は、原則として休暇取得日の3日前までにシステムまたは所定の休暇申請書により申請し、所属部署長の承認を得なければなりません。当日の急な休暇申請は原則として認められません。"
    },
    {
        "question": "勤怠管理システムが利用できない場合はどうすればよいですか？",
        "reference": "システムを利用できない場合、または会社が認めた場合は、タイムカードによる勤怠管理を行うことができます。タイムカードは出勤時および退勤時に従業員本人が打刻するものとします。"
    },
    
    # =============================================================================
    # 特殊パターンの質問
    # =============================================================================

    # 特殊文字を含む質問
    {
        "question": "当社の主な取＠先の銀行を教えて？さい。",
        "reference": "主な取引先の銀行は、星空銀行架空支店と大森信用金庫の本社営業部です。"
    },
    # 誤字脱字を含む質問
    {
        "question": "当社の発行済株総はいらですか？",
        "reference": "当社は100万株を発行しています。"
    },
    # 複数のドキュメントを参照する質問
    {
        "question": "今年の夏季休暇と夏季賞与について教えて下さい。",
        "reference": "夏季休暇は原則として毎年8月13日から8月15日までの3日間、夏季賞与は6月に支給されます。"
    },
     # 複数のドキュメントの関連性を問う質問
    {
        "question": "給与計算規則と勤怠管理マニュアルの関係性を教えて下さい。",
        "reference": "給与計算規則と勤怠管理マニュアルは密接に関連しています。勤怠管理マニュアルの第25条では遅刻・早退・欠勤等の勤怠状況が給与計算に反映されると明記されており、時間外労働・休日労働に関する規定も両規則間で連動しています。"
    },
    #関連文書に明示的に記載がない質問
    {
        "question": "プログラミングの勉強方法について教えて下さい。",
        "reference": "申し訳ありませんが、関連ドキュメントが提供されていないため、プログラミングの勉強方法について正確な回答はできかねます。"
    }
]

# AWS設定 - より保守的なタイムアウトとリトライ設定
config = Config(
    connect_timeout=120,  # 接続タイムアウト
    read_timeout=900,     # 読み取りタイムアウト
    retries={
        "max_attempts": 10,  # リトライ回数
        "mode": "adaptive",
        "total_max_attempts": 15
    }
)

# Bedrockクライアントの初期化
bedrock_client = boto3.client(
    "bedrock-runtime", 
    region_name=AWS_REGION,
    config=config
)

# LLM用のクライアントの初期化
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
    # Bedrock LLMの生成パラメータ設定
    model_kwargs={
        "temperature": 0.1, 
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
    timeout=900,  # タイムアウト
    max_workers=1  # スロットリング回避
)

def evaluate_model_answers():
    """RAGシステムの回答を評価"""

    # 1. テストデータから質問と模範回答を抽出
    test_questions = [item["question"] for item in test_data]
    
    # 2. 実際のRAGシステムに質問を送信して回答を取得
    actual_answers = []
    contexts_list = []
    
    for i, question in enumerate(test_questions):
        print(f"\n質問 {i+1}/{len(test_questions)}: {question}")
        
        # スロットリング回避のため、最初の質問以外は30秒待機
        if i > 0:
            # 時間間隔を長くしてThrottling対策
            wait_time = 30 + random.uniform(0, 10)  # 30〜40秒のランダムな待機時間
            time.sleep(wait_time)
        
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={"message": question},
                headers={"Content-Type": "application/json"},
                timeout=200
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
        # テストデータから対応するreference(模範回答)を取得
        reference = next((item["reference"] for item in test_data if item["question"] == q), "")
        
        dataset_items.append({
            "user_input": q,
            "response": actual,
            "retrieved_contexts": contexts,
            "reference": reference
        })
    
    # 4. データ検証
    valid_items = []
    for item in dataset_items:
        # 最低限の条件のみチェック
        if (item.get("response", "").strip() and 
            item.get("retrieved_contexts") and 
            len(item.get("retrieved_contexts", [])) > 0 and
            len(item.get("response", "").strip()) > 5):  # 最小文字数チェック
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
                ContextPrecision(),
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
            "context_precision": safe_float_conversion(get_metric_value(results, "context_precision")),
            "details": [],
            "summary": {
                "total_samples": len(eval_dataset),
                "evaluation_metrics": ["faithfulness", "answer_relevancy", "context_precision"]
            }
        }
        
        # 詳細データ
        df = results.to_pandas()
        for i in range(len(eval_dataset)):
            row_data = {
                "question": eval_dataset[i]["user_input"],
                "answer": eval_dataset[i]["response"][:200] + "..." if len(eval_dataset[i]["response"]) > 200 else eval_dataset[i]["response"],
                "faithfulness": safe_float_conversion(df.iloc[i].get("faithfulness")),
                "answer_relevancy": safe_float_conversion(df.iloc[i].get("answer_relevancy")),
                "context_precision": safe_float_conversion(df.iloc[i].get("context_precision")),
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

# メイン処理
if __name__ == "__main__":
    print("実際のRAGシステムを評価開始...")
    results = evaluate_model_answers()
    print("\n最終結果:")
    for metric, value in results.items():
        if metric not in ["details", "summary"]:
            print(f"{metric}: {value}") 