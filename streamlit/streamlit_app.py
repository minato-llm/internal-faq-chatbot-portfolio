# streamlit/streamlit_app.py
import streamlit as st
import requests
import json

st.title("社内FAQチャットボット")

# セッション状態を初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# セッションIDを保存するためのセッション状態を初期化
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# 過去のメッセージを表示
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(message["content"])
    elif message["role"] == "documents":
        with st.chat_message("assistant", avatar="📄"):
            st.write("以下の関連ドキュメントが見つかりました:")
            for doc in message["content"]:
                # 関連ドキュメントのタイトル表示
                st.markdown(f"* {doc['title']}")

# ユーザー入力
user_message = st.chat_input("質問を入力してください")

if user_message:
    # ユーザーメッセージを表示
    st.chat_message("user").write(user_message)
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    # APIリクエスト
    api_url = "http://localhost:8000/chat"  # FastAPIアプリケーションのURL
    headers = {"Content-Type": "application/json"}
    data = {
        "message": user_message,
        "session_id": st.session_state.session_id,  
        "messages_history": st.session_state.messages
    }

    try:
        # リクエストを送信
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        # レスポンスのステータスコードを確認
        response.raise_for_status()
        # レスポンスのJSONデータを取得
        response_data = response.json()
        bot_response = response_data.get("response", "応答がありません")
        related_documents = response_data.get("related_documents", [])
        
        # セッションIDを保存（初回のみ）
        if not st.session_state.session_id:
            st.session_state.session_id = response_data.get("session_id")
        
        # 関連ドキュメント表示の処理
        with st.chat_message("assistant", avatar="📄"):
            if related_documents:
                st.write("以下の関連ドキュメントが見つかりました:")
                for doc in related_documents:
                    # 関連ドキュメントのタイトル表示
                    title = doc['title']
                    st.markdown(f"* **{title}**")
            else:
                st.write("関連ドキュメントが見つかりませんでした")
        
        # ドキュメント情報をセッションに保存（既存コードを維持）
        if related_documents:
            st.session_state.messages.append({"role": "documents", "content": related_documents})
        else:
            st.session_state.messages.append({"role": "documents", "content": []})
        
        # ボットの回答を表示
        with st.chat_message("assistant"):
            st.write(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

    except requests.exceptions.RequestException as e:
        st.error(f"APIリクエストエラー: {e}")
    except json.JSONDecodeError:
        st.error("API応答のJSONデコードエラー")
    except Exception as e:
        st.error(f"予期せぬエラー: {e}")