import os
import boto3
import asyncio
from langchain_aws import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import OpenSearchVectorSearch 
from pypdf import PdfReader
from io import BytesIO
import hashlib
import logging
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth
import json

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境変数の初期化
AWS_REGION = os.environ.get("AWS_REGION")

def lambda_handler(event, context):
    """
    S3からPDFドキュメントを処理し、OpenSearchに保存するパイプライン
    1. S3からPDFファイルを取得
    2. PDFからテキストを抽出
    3. テキストをチャンキング
    4. チャンクをベクトル化
    5. ベクトルをOpenSearchに保存
    """
    try:
        # S3クライアントの初期化
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # S3バケット名とプレフィックスを環境変数から取得
        s3_bucket = os.environ.get("PDF_S3_BUCKET_NAME")
        s3_prefix = os.environ.get("PDF_S3_PREFIX", "")
        
        # OpenSearch設定を環境変数から取得
        opensearch_endpoint = os.environ.get("OPENSEARCH_ENDPOINT")
        opensearch_index = os.environ.get("OPENSEARCH_INDEX", "pdf_documents")
        
        # 埋め込みモデルの初期化
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=AWS_REGION
        )
        
        # PDFファイルのリストを取得
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
        
        all_chunks = []
        
        # 各PDFファイルを処理
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.pdf'):
                # PDFファイルをS3から取得
                pdf_obj = s3_client.get_object(Bucket=s3_bucket, Key=obj['Key'])
                
                # PDFからテキストを抽出
                pdf_content = pdf_obj['Body'].read()
                pdf_reader = PdfReader(BytesIO(pdf_content))
                
                pdf_text = ""
                metadata = {
                    "source": f"s3://{s3_bucket}/{obj['Key']}",
                    "title": obj['Key'].split('/')[-1].replace('.pdf', '')
                }
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        pdf_text += text + "\n"
                        
                # テキストをチャンキング（階層的チャンキングを適用）
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", "。", "、", " ", ""],
                    length_function=len
                )
                
                chunks = text_splitter.create_documents(
                    texts=[pdf_text],
                    metadatas=[{**metadata, "page": "all"}]
                )
                
                all_chunks.extend(chunks)
        
        # 重複チャンクの除去（内容のハッシュ値に基づく）
        unique_chunks = {}
        for chunk in all_chunks:
            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
            if content_hash not in unique_chunks:
                unique_chunks[content_hash] = chunk
        
        unique_chunk_list = list(unique_chunks.values())

        session = boto3.Session(region_name=AWS_REGION)
        credentials = session.get_credentials()

	    # OpenSearch Serverless('aoss')用のSigV4認証情報を作成
        auth = AWSV4SignerAuth(credentials, AWS_REGION, 'aoss') 

        # OpenSearchベクトルストアの初期化
        vector_store = OpenSearchVectorSearch(
            index_name=opensearch_index,
            embedding_function=embeddings,
            opensearch_url=opensearch_endpoint,
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )        

        # チャンクをベクトル化してOpenSearchに保存
        if unique_chunk_list:
            logger.info(f"Adding {len(unique_chunk_list)} chunks to OpenSearch.")
            vector_store.add_documents(unique_chunk_list, bulk_size=500)
        
        logger.info(f"Successfully processed {len(unique_chunk_list)} chunks.")
        # 成功時のレスポンスをAPI Gatewayプロキシ統合形式で返す
        return { 
            "statusCode": 200,
            "body": json.dumps({ "status": "success", "chunks_processed": len(unique_chunk_list) })
        }
        
    except Exception as e:
        logger.error(f"S3 PDF処理中にエラーが発生しました: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({ "status": "error", "message": str(e) })
        }