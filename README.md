# RAGを用いた社内FAQチャットボット (LangChain, AWS Bedrock, FastAPI)

## プロジェクト概要

本プロジェクトは、従来のシステムエンジニアとして培ってきた知識と経験に加え、  
LLMエンジニアとしてのキャリアチェンジを目指すための学習成果を示すポートフォリオとして開発しました。  
LangChainによるLLMとの連携、AWS BedrockでのLLMの利用、そしてFastAPIを用いたAPIサービス構築を組み合わせています。  
LLM＋RAGを使用し、社内ドキュメント (会社概要、給与計算規則、勤怠管理マニュアル) を参照して質問応答を行います。

## 技術スタック

-   **言語：** Python 3.11.6
-   **フレームワーク：**
    -   LangChain 0.2.17
    -   FastAPI 0.100.0
-   **ライブラリ：**
    -   pandas 2.1.2
    -   numpy 1.25.2
    -   Uvicorn 0.23.2
    -   LangChain-Community 0.2.19
    -   LangChain-AWS 0.1.18
    -   LangChain-Core 0.2.43
    -   AWS Lambda Powertools 2.36.0
    -   Boto3 1.34.131
    -   Python-dotenv 1.0.0
-   **インフラ：** AWS (Lambda, S3, Bedrock Knowledge Base, Bedrock, IAM)
-   **エディタ：** Cursor
-   **ノートブック：** Google Colab
-   **コード管理：** Git, GitHub
-   **チャットUI：** Streamlit 1.27.2
-   **埋め込みモデル：** Titan Text Embeddings v2
-   **ベクトルDB：** Amazon OpenSearch Serverless
-   **言語モデル：** Claude 3.5 Sonnet 20241022 v2
-   **LLM評価：** Rages, LangSmith

## ディレクトリ構成
```
├── internal_faq_chatbot/
│   ├── documents/           # 関連ドキュメント
│   │   ├── 会社概要.pdf
│   │   ├── 給与計算規則.pdf
│   │   └── 勤怠管理マニュアル.pdf
│   ├── fastapi/             # チャットの送信・受信
│   │   └── fastapi_app.py
│   ├── lambda_functions/    # Lambda関数
│   │   ├── document_search/ # ドキュメント検索
│   │   │   ├── bedrock_kb_search_function.py
│   │   │   └── requirements.txt
│   │   ├── preprocessor/    # チャットの前処理
│   │   │   ├── lambda_preprocessor_function.py
│   │   │   └── requirements.txt
│   │   └── vectorization/   # チャットのベクトル化
│   │       ├── bedrock_vector_function.py
│   │       └── requirements.txt
│   ├── streamlit/           # 画面UI
│   │   └── streamlit_app.py
│   ├── .gitignore           # Git管理から除外
│   └── requirements.txt     # プロジェクト全体の依存関係
```

## 動作手順

1.  **(チャットの送信・受信)**
    -   チャットボットがユーザーのメッセージを受信し、リクエストをAWS Lambdaに転送。
2.  **(AWS Lambdaによる質問の前処理)**
    -   受信したユーザーメッセージはAWS Lambda関数で処理され、自然言語処理（NLP）を用いて前処理を実施。
3.  **(AWS Bedrockによるベクトル化)**
    -   前処理されたテキストデータはAWS Bedrockエンドポイントに送信され、数値ベクトルに変換される。
4.  **(Bedrock Knowledge Baseを使用した関連ドキュメントの検索)**
    -   ※事前にS3に保存された社内ドキュメントのインデックス作成を行う。  
        前処理済みのテキストを基にAWS LambdaからBedrock Knowledge Baseにクエリを送信。  
        Bedrock Knowledge Baseは作成済みのインデックスを参照してテキストベースで検索を行い、質問に関連する情報を取得。
5.  **(チャット文と関連ドキュメントを元に、生成された回答をAWS Bedrockで処理)**
    -  取得した関連ドキュメントとユーザーの質問を組み合わせて、AWS Bedrock上のAIモデルに入力。  
       これらの情報を基に適切な回答を生成。
6.  **(生成した回答内容の受信)**
    -   AWS Bedrockで生成された回答は、AWS Lambdaを経由してチャットボットを通じてユーザーに回答を返信。

## AWSアーキテクチャー図

(ここにAWSアーキテクチャー図を挿入)
<!-- ![AWSアーキテクチャー図](./path/to/your/architecture_diagram.png) -->

## デモ

(ここにデモのスクリーンショット挿入)
<!-- ![デモ画像1](./path/to/your/demo_image1.png) -->
<!-- ![デモ画像2](./path/to/your/demo_image2.png) -->

## 評価
1.  **(LLM・RAGの性能評価)**
    -   作成したLLM・RAGシステムを数百件のテストデータ(質問と想定回答の組み合わせ)を用意してRages、LangSmithで性能評価する。  
    (ここに性能評価結果 (正解率など)を記載)
2.  **(LLM・RAGの精度改善)**
    -   Rages 、LangSmithで性能評価した結果を元に分析してLLM・RAGの精度改善を実施。再テストして前回と比べて精度が上がっているか検証する。  
    (ここに性能改善方法を記載)

<!-- 例：
- 正解率：XX%
- Hallucination発生率：YY%
-->

## ポートフォリオ作成によって得た経験、スキル

*   RAGシステムの開発・評価・改善の一連サイクルの実践経験
*   Anthropicが提供するClaude APIの利用経験
*   LangChainのLLMフレームワークを用いた開発経験
*   FastAPIを用いたWeb APIの開発経験
*   AWS (Lambda、S3、Bedrock Knowledge Base、Bedrock、IAM) でのシステム構築経験
*   Rages、LangSmithを利用したLLMの性能評価経験
*   プロンプト、RAGの精度改善経験

## 改善点・今後の展望
(ここに改善点や今後の展望を記載)
