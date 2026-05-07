from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.faiss import FAISS
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()])


# st.cache_data 이 함수는 한번만 실행하며, 다른 파일을 넣지 않는 이상 재실행하지 않고 캐싱된걸 다시 준다.
# 같은 파일, 다른 파일을 업로드한것을 알아채서 동작한다.
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


def save_message(message, role, save=True):
    st.session_state["messages"].append({"message": message, "role": role})


# 메세지 전송
def send_message(message, role, save=True):
    # send_message 시작될때 with 실행하고 곧바로 close > chat message role 전달 , message 그리기
    with st.chat_message(role):
        st.markdown(message)

    # 저장하는거면 session에 추가
    if save:
        save_message(message, role)


# 챗 히스토리 화면에 보여주기
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            content 안에서만 대답하기. 다른 추리로 말하지 않기
            Context: {context}
        """,
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")


with st.sidebar:
    file = st.file_uploader("업로드 하세요 .txt .pdf", type=["pdf", "docx", "txt"])


if file:
    retriever = embed_file(file)
    send_message("질문하셔도 됩니다.", "ai", save=False)
    paint_history()
    message = st.chat_input("무엇이든 물어보든가")

    # 메세지 보내기
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)

else:
    st.session_state["messages"] = []
