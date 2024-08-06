import os
import streamlit as st
from dotenv import load_dotenv

from streamlit_option_menu import option_menu

from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from langchain_community.graphs import Neo4jGraph
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

import utils.contants as constants
from utils.helpers import stream_data, KnowledgeGraphBuilder, create_knowledge_graph_embeddings
from utils.helpers import load_and_split_webpage, generate_embeddings, generate_graph, load_and_split_documents

load_dotenv()

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
GROQ_AI_KEY = os.environ.get("GROQ_AI_KEY")
GROQ_AI_MODEL = os.environ.get("GROQ_AI_MODEL")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
BOWHEAD_OPEN_AI_KEY = os.environ.get("BOWHEAD_OPEN_AI_KEY")
BOWHEAD_GOOGLE_AI_KEY = os.environ.get("BOWHEAD_GOOGLE_AI_KEY")
BOWHEAD_ANTHROPIC_AI_KEY = os.environ.get("BOWHEAD_ANTHROPIC_AI_KEY")

merger_langfuse_handler = CallbackHandler(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
    host="https://us.cloud.langfuse.com",
    trace_name="merger"
)

avatars = {"human": constants.USER, "ai": constants.ASSISTANT}
assistant_responses = []
google_msgs = StreamlitChatMessageHistory(key="google_messages")
rag_doc_msgs = StreamlitChatMessageHistory(key="rag_doc_messages")
rag_web_msgs = StreamlitChatMessageHistory(key="rag_web_messages")
kg_doc_msgs = StreamlitChatMessageHistory(key="rag_kg_messages")

st.set_page_config(layout='wide')
with st.sidebar:
    selected = option_menu(
        menu_title="Multimate RAG",
        options=["🕸️ Knowledge Graph", "📑 Documents", "🌐 Website"],
        default_index=0,
    )

if selected == "🕸️ Knowledge Graph":
    kg_chat, kg_visuals = st.columns(2)

    neo4j_graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    knowledge_graph_built = False

    def on_kg_doc_upload():
        rag_doc_msgs.clear()
        if source_docs:
            st.session_state.kg = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=BOWHEAD_OPEN_AI_KEY),
                chain_type='stuff',
                retriever=create_knowledge_graph_embeddings(
                    load_and_split_documents(files=source_docs)).as_retriever(),
            )


    kg_rag_content = kg_chat.container(height=600)
    source_docs = kg_rag_content.file_uploader(
        key="knowledge_graph",
        label="Upload Documents", type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        on_change=on_kg_doc_upload,
    )
    kg_content = kg_visuals.container(height=600)

    kg_prompt = st.chat_input(placeholder="Generate a knowledge graph.", key="kg_prompt")

    if kg_prompt:
        if not source_docs:
            kg_rag_content.warning("Please upload a file!")
        else:
            kg_doc_msgs.add_user_message(kg_prompt)

    with kg_rag_content:
        for idx, msg in enumerate(kg_doc_msgs.messages):
            with st.chat_message(avatars[msg.type]):
                st.write(msg.content)

    if source_docs:
        if not knowledge_graph_built:
            kg_builder = KnowledgeGraphBuilder(
                NEO4J_URI,
                NEO4J_USERNAME,
                NEO4J_PASSWORD,
                BOWHEAD_ANTHROPIC_AI_KEY
            )
            kg_builder.build_knowledge_graph(source_docs)
            with kg_content:
                generate_graph(kg_builder.all_nodes, kg_builder.all_relationships)

            knowledge_graph_built = True

    if kg_prompt:
        try:
            if 'kg' in st.session_state:
                result = st.session_state.kg.invoke(kg_prompt)["result"]
                kg_doc_msgs.add_ai_message(result)
                kg_rag_content.chat_message(constants.ASSISTANT).write_stream(stream_data(result))
            else:
                kg_rag_content.warning("Please upload a file!")
        except Exception as e:
            print(e)

if selected == "📑 Documents":

    def on_doc_upload():
        rag_doc_msgs.clear()
        if source_docs:
            st.session_state.qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=BOWHEAD_OPEN_AI_KEY),
                chain_type='stuff',
                retriever=generate_embeddings(load_and_split_documents(files=source_docs)).as_retriever(),
            )


    source_docs = st.file_uploader(
        label="Upload Documents", type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        on_change=on_doc_upload,
    )
    doc_rag_content = st.container(height=400, border=False)
    rag_prompt = st.chat_input(placeholder="Summarize this document.", key="doc_prompt")

    if rag_prompt:
        if not source_docs:
            doc_rag_content.warning("Please upload a file!")
        else:
            rag_doc_msgs.add_user_message(rag_prompt)

    with doc_rag_content:
        for idx, msg in enumerate(rag_doc_msgs.messages):
            with st.chat_message(avatars[msg.type]):
                st.write(msg.content)

    if source_docs:
        st.session_state.qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=BOWHEAD_OPEN_AI_KEY),
            chain_type='stuff',
            retriever=generate_embeddings(load_and_split_documents(files=source_docs)).as_retriever(
                search_kwargs={"k": 10}),
        )

    if rag_prompt:
        try:
            if 'qa' in st.session_state:
                result = st.session_state.qa.invoke(rag_prompt)["result"]
                rag_doc_msgs.add_ai_message(result)
                doc_rag_content.chat_message(constants.ASSISTANT).write_stream(stream_data(result))
            else:
                doc_rag_content.warning("Please upload a file!")
        except Exception as e:
            print(e)

if selected == "🌐 Website":

    def on_url_submit():
        rag_web_msgs.clear()
        if website_url:
            st.session_state.web_qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=BOWHEAD_OPEN_AI_KEY),
                chain_type='stuff',
                retriever=generate_embeddings(load_and_split_webpage(website_url)).as_retriever()
            )


    website_url = st.container().text_input(label="Website URL", value="", on_change=on_url_submit)
    rag_web_content = st.container(height=550, border=False)
    rag_web_prompt = st.chat_input(placeholder="Summarize this website.", key="web_prompt")

    if rag_web_prompt:
        if not website_url:
            rag_web_content.warning("Please input a valid url!")
        else:
            rag_web_msgs.add_user_message(rag_web_prompt)

    with rag_web_content:
        for idx, msg in enumerate(rag_web_msgs.messages):
            with st.chat_message(avatars[msg.type]):
                st.write(msg.content)

    if website_url:
        st.session_state.web_qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=BOWHEAD_OPEN_AI_KEY),
            chain_type='stuff',
            retriever=generate_embeddings(load_and_split_webpage(website_url)).as_retriever()
        )

    if rag_web_prompt:
        try:
            if 'web_qa' in st.session_state:
                response = st.session_state.web_qa.invoke(rag_web_prompt)["result"]
                rag_web_msgs.add_ai_message(response)
                rag_web_content.chat_message(constants.ASSISTANT).write_stream(stream_data(response))
            else:
                rag_web_content.warning("Please input a valid url!")
        except Exception as e:
            print(e)
