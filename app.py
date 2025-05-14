import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
embedding_api_key = os.getenv("OPENAI_API_KEY")

# Page config
st.set_page_config(page_title="Sutra RAG Chat", page_icon="📚", layout="wide")

# Language options
languages = [
    "English", "Hindi", "Gujarati", "Bengali", "Tamil", 
    "Telugu", "Kannada", "Malayalam", "Punjabi", "Marathi", 
    "Urdu", "Assamese", "Odia", "Sanskrit", "Korean", 
    "Japanese", "Arabic", "French", "German", "Spanish", 
    "Portuguese", "Russian", "Chinese", "Vietnamese", "Thai", 
    "Indonesian", "Turkish", "Polish", "Ukrainian", "Dutch", 
    "Italian", "Greek", "Hebrew", "Persian"
]

# Callback handler for streaming
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# Sidebar UI
st.sidebar.image("https://framerusercontent.com/images/3Ca34Pogzn9I3a7uTsNSlfs9Bdk.png", use_container_width=True)
st.sidebar.title("Settings")

sutra_api_key = st.sidebar.text_input("Enter your Sutra API Key", type="password")
selected_language = st.sidebar.selectbox("Select language for responses:", languages)

st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

# Functions to get models
def get_streaming_chat_model(callback_handler=None):
    return ChatOpenAI(
        api_key=sutra_api_key,
        base_url="https://api.two.ai/v2",
        model="sutra-v2",
        temperature=0.7,
        streaming=True,
        callbacks=[callback_handler] if callback_handler else None
    )

def get_chat_model():
    return ChatOpenAI(
        api_key=sutra_api_key,
        base_url="https://api.two.ai/v2",
        model="sutra-v2",
        temperature=0.7
    )

# Session state init
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Process documents
def process_documents(uploaded_files, chunk_size=1000, chunk_overlap=100):
    documents = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploaded_files:
        temp_path = os.path.join(temp_dir.name, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=embedding_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=get_chat_model(),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# App title
st.markdown(
    '<h1><img src="https://framerusercontent.com/images/9vH8BcjXKRcC5OrSfkohhSyDgX0.png" width="60"/> Sutra Document Chatbot 📚</h1>',
    unsafe_allow_html=True
)

# Process documents button
if uploaded_files and st.sidebar.button("Process Documents"):
    if not sutra_api_key:
        st.sidebar.error("Please enter your Sutra API key before proceeding.")
    else:
        with st.spinner("Processing documents..."):
            st.session_state.conversation = process_documents(uploaded_files)
            st.session_state.documents_processed = True
            st.success(f"{len(uploaded_files)} documents processed!")

st.sidebar.markdown(f"Responses will be in: **{selected_language}**")
st.sidebar.divider()

# Main chat area
if not st.session_state.documents_processed:
    st.info("Please upload documents and click 'Process Documents' in the sidebar to start chatting.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        try:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                stream_handler = StreamHandler(response_placeholder)
                chat = get_streaming_chat_model(stream_handler)

                rag_response = st.session_state.conversation.invoke(user_input)
                context = rag_response["answer"]

                system_message = f"""
                You are a helpful assistant that answers questions about documents. 
                Use the following context to answer the question.

                CONTEXT:
                {context}

                Please respond in {selected_language}.
                """

                messages = [HumanMessage(content=f"{system_message}\n\nQuestion: {user_input}")]
                response = chat.invoke(messages)
                answer = response.content

                st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error: {str(e)}")
            if "API key" in str(e):
                st.error("Please ensure your Sutra API key is valid.")
