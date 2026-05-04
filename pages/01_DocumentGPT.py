
import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("DocumentGPT")


# 세션에 대화내용 확인 > 없으면 []
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# 메세지 전송 함수
def send_message(message, role, save=True):
        with st.chat_message(role):
            st.write(message)
        if save: 
            st.session_state["messages"].append({"message": message, "role": role})
            


# 메세지 저장한거 그리기
for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)


# input에 메세지 입력
message =st.chat_input("Enter a message")

# 메세지 입력 후 저장 > send_message 호출 > sessions 에 저장
if message:
    send_message(message, "human")
    time.sleep(1)
    send_message(f"You said: {message}", "ai")