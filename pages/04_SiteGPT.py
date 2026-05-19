from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.title("SiteGPT")


html2text_transformer = Html2TextTransformer()

st.markdown(
    """
  Welcome!

  Use this chatbot to ask questions to an AI about the content of a website!
"""
)


with st.sidebar:
    url = st.text_input("URL 작성", placeholder="https://naver.com")


if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(transformed)
