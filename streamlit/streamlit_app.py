# streamlit/streamlit_app.py
import streamlit as st
import requests
import json

st.title("社内FAQチャットボット")

user_message = st.text_input("質問を入力してください")

if st.button("送信"):
    if user_message:
        api_url = "http://localhost:8000/chat" # FastAPIアプリケーションのURL
        headers = {'Content-Type': 'application/json'}
        data = {'message': user_message}

        try:
            # リクエストを送信
            response = requests.post(api_url, headers=headers, data=json.dumps(data))
            # レスポンスのステータスコードを確認
            response.raise_for_status()
            # レスポンスのJSONデータを取得
            bot_response = response.json().get("response", "応答がありません")
            st.text_area("回答:", value=bot_response, height=200)

        except requests.exceptions.RequestException as e:
            st.error(f"APIリクエストエラー: {e}")
        except json.JSONDecodeError:
            st.error("API応答のJSONデコードエラー")
        except Exception as e:
            st.error(f"予期せぬエラー: {e}")
    else:
        st.warning("質問を入力してください")