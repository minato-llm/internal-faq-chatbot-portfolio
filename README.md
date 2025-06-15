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
    -   Uvicorn 0.23.2
    -   LangChain-Community 0.2.19
    -   LangChain-AWS 0.1.18
    -   LangChain-Core 0.2.43
    -   AWS Lambda Powertools 2.36.0
    -   Boto3 1.34.131
    -   Python-dotenv 1.0.0
    -   ragas 0.2.0
    -   langsmith 0.1.112
    -   datasets 2.15.0
    -   nltk 3.8.1
    -   pypdf 3.15.1
-   **インフラ：** AWS (Lambda, S3, Bedrock, IAM)
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
1.  **(RAG性能評価)**
    -   作成したRAGシステムをテストデータ(質問と想定回答の組み合わせ)を用意してRages、LangSmithで性能評価する。
    ### Ragas評価結果 
    - **忠実性（Faithfulness）**: 85.3%  
    - **回答関連性（Answer Relevancy）**: 46.7%  
    - **コンテキスト精度（Context Precision）**: 85.0%

    ### LangSmith評価結果
    - **実行回数**: 20回
    - **エラー率**: 0%
    - **レイテンシー**:
      - P50: 3.10秒
      - P99: 9.02秒
    - **トークン数**: 26,623トークン
    　![image](https://github.com/user-attachments/assets/34ef59bf-7ec9-4619-9d52-28d254737fdf)
    　![image](https://github.com/user-attachments/assets/9520815d-4243-47b5-b464-5f49cd1086b8)


    ### 性能考察
    **Ragas評価分析**:
    - **忠実性（85.3%）**: 関連ドキュメントの情報に忠実な回答ができており、ハルシネーションが少ない
    - **回答関連性（46.7%）**: 質問に対する回答の関連性が低く、改善の余地が大きい
    - **コンテキスト精度（85.0%）**: 適切なコンテキスト選択ができている

    **LangSmith評価分析**:
    - **レイテンシ**: P50は3.10秒と許容範囲だが、P99が9.02秒と大幅に遅延
    - **トークン効率**: 26,623トークンと多量のトークンを消費している
    - **安定性**: エラー率0%と安定した動作を実現

    **課題と改善策**:
    1.  **回答関連性の低さ**: プロンプトエンジニアリングによる質問理解力の向上
    2.  **レイテンシの不安定さ**: Lambda関数の見直しとチャンキング改善
    3.  **トークン消費量**: 関連ドキュメントの効率的な選択とチャンキング戦略の見直し

    画像から確認できるように、関連ドキュメントが適切にチャンキングされておらず、大きなテキストブロックとして処理されているため、質問に直接関係ない情報も回答に含まれています。

    
2.  **(LLM・RAGの精度改善)**
    -   Rages 、LangSmithで性能評価した結果を元に分析してプロンプトエンジニアリング・RAGの精度改善を実施。再テストして前回と比べて精度が上がっているか検証する。  
    プロンプトエンジニアリングの改善
　　![image](https://github.com/user-attachments/assets/ffcb6d48-1951-4f6f-ac4a-6111b38ed157)







　　

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
