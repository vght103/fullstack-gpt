
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)



@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()

    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")


    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )


    loader = UnstructuredFileLoader(file_path)

  
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()


    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)


    retriever = vectorstore.as_retriever()

    return retriever


# 메세지 전송
def send_message(message,role, save=True):
    # send_message 시작될때 with 실행하고 곧바로 close > chat message role 전달 , message 그리기
    with st.chat_message(role):
        st.markdown(message)
    
    # 저장하는거면 session에 추가
    if save:
        st.session_state["messages"].append({"message":message, "role":role})


# 챗 히스토리 화면에 보여주기
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


st.title("DocumentGPT")


with st.sidebar:
    file = st.file_uploader("업로드 하세요 .txt .pdf", type=["pdf", "docx", "txt"])


if file:
    retriever = embed_file(file)
    send_message("질문하셔도 됩니다.","ai", save=False)
    paint_history()
    message = st.chat_input("무엇이든 물어보든가")    

# 메세지 보내기
    if message:
        send_message(message, "human")

# 파일이 없을 시 세션 초기화(대화 초기화)
else:
    # session 초기화
    st.session_state["messages"] = []
