
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from langchain.retrievers import WikipediaRetriever

st.set_page_config(
    page_title="QuizeGPT",
    page_icon="❓",
)

st.title("QuizeGPT")


llm = ChatOpenAI(
  temperature=0.1,
  model="gpt-3.5-turbo-1106"
)

@st.cache_data(show_spinner="loading file...")
def split_file(file):
    file_content = file.read()

    file_path = f"./.cache/quize_files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)
    return docs

with st.sidebar:
  docs = None

  select = st.selectbox("선택하세요.",
    (
    "File","Wikipedia"
    ),
  )

  if select == "File":
    file = st.file_uploader("업로드 하세요. .docx, .txt, .pdf 등", type=["pdf","txt","docx"])

    if file:
      docs = split_file(file)
      
  else:
    topic = st.text_input("위키피디아 검색...")
    if(topic):
      retriever = WikipediaRetriever(top_k_results=3)
      with st.status("searching..."):
        docs = retriever.get_relevant_documents(topic)
        

if not docs:
  st.markdown(
    """
      문서를 추가 또는 입력
    """
  )
else:
  st.write(docs)
