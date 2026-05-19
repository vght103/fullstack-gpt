from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser, output_parser
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
import json

st.set_page_config(
    page_title="QuizeGPT",
    page_icon="❓",
)

st.title("QuizeGPT")


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = (
            text.replace("```", "")
            .replace("json", "")
            .replace(", ]", "]")
            .replace(", }", "}")
        )
        return json.loads(text)


output_parser = JsonOutputParser()


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a helpful assistant that is role playing as a teacher.

                Answer Korean.

                Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

                Each question should have 4 answers, three of them must be incorrect and one should be correct.

                Use (o) to signal the correct answer.

                Question examples:
                    
                Question: What is the color of the ocean?
                Answers: Red|Yellow|Green|Blue(o)

                Question: What is the capital or Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut

                Question: When was Avatar released?
                Answers: 2007|2001|2009(o)|1998

                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o)|Painter|Actor|Model

                Your turn!

                Context: {context}
             """,
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm


formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a powerful formatting algorithm.

                You format exam questions into JSON format.
                Answers with (o) are the correct ones.

                Example Input:
                Question: What is the color of the ocean?
                Answers: Red|Yellow|Green|Blue(o)

                Question: What is the capital or Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut

                Question: When was Avatar released?
                Answers: 2007|2001|2009(o)|1998

                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o)|Painter|Actor|Model


                Example Output:

                ```json
                {{ "questions": [
                {{
                "question": "What is the color of the ocean?",
                "answers": [
                {{
                "answer": "Red",
                "correct": false
                }},
                {{
                "answer": "Yellow",
                "correct": false
                }},
                {{
                "answer": "Green",
                "correct": false
                }},
                {{
                "answer": "Blue",
                "correct": true
                }},
                ]
                }},
                {{
                "question": "What is the capital or Georgia?",
                "answers": [
                {{
                "answer": "Baku",
                "correct": false
                }},
                {{
                "answer": "Tbilisi",
                "correct": true
                }},
                {{
                "answer": "Manila",
                "correct": false
                }},
                {{
                "answer": "Beirut",
                "correct": false
                }},
                ]
                }},
                {{
                "question": "When was Avatar released?",
                "answers": [
                {{
                "answer": "2007",
                "correct": false
                }},
                {{
                "answer": "2001",
                "correct": false
                }},
                {{
                "answer": "2009",
                "correct": true
                }},
                {{
                "answer": "1998",
                "correct": false
                }},
                ]
                }},
                {{
                "question": "Who was Julius Caesar?",
                "answers": [
                {{
                "answer": "A Roman Emperor",
                "correct": true
                }},
                {{ 
                "answer": "Painter",
                "correct": false
                }},
                {{
                "answer": "Actor",
                "correct": false
                }},
                {{
                "answer": "Model",
                "correct": false
                }},
                ]
                }}
                ]
                }}
                ```
                Your turn!
                Questions: {context}
            """,
        )
    ]
)

formatting_chain = formatting_prompt | llm


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


@st.cache_data(show_spinner="퀴즈 생성중...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="퀴즈 생성중...")
def wiki_search(term):
    retriever = WikipediaRetriever(
        top_k_results=3,
    )
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None

    select = st.selectbox(
        "선택하세요.",
        ("File", "Wikipedia"),
    )

    if select == "File":
        file = st.file_uploader(
            "업로드 하세요. .docx, .txt, .pdf 등", type=["pdf", "txt", "docx"]
        )

        if file:
            docs = split_file(file)

    else:
        topic = st.text_input("위키피디아 검색...")
        if topic:
            wiki_search(topic)


if not docs:
    st.markdown(
        """
      문서를 추가 또는 입력
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    st.write(response)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.wirte(question["question"])
            value = st.radio(
                "옵션선택",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )

            if {"answer": value, "correct": True} in question["answers"]:
                st.success("정답!")
            elif value is not None:
                st.error("탈락")
        button = st.form_submit_button()
