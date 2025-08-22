import os
import streamlit as st

# ----------------------
# LangChain / Ollama Imports
# ----------------------
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# ----------------------
# CSS for dark theme
# ----------------------
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FFFFFF; }
.stChatInput input { background-color: #1E1E1E !important; color: #FFFFFF !important; border: 1px solid #3A3A3A !important; }
.stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { background-color: #1E1E1E !important; border: 1px solid #3A3A3A !important; color: #E0E0E0 !important; border-radius: 10px; padding: 15px; margin: 10px 0; }
.stChatMessage[data-testid="stChatMessage"]:nth-child(even) { background-color: #2A2A2A !important; border: 1px solid #404040 !important; color: #F0F0F0 !important; border-radius: 10px; padding: 15px; margin: 10px 0; }
.stChatMessage .avatar { background-color: #00FFAA !important; color: #000000 !important; }
.stChatMessage p, .stChatMessage div { color: #FFFFFF !important; }
.stFileUploader { background-color: #1E1E1E; border: 1px solid #3A3A3A; border-radius: 5px; padding: 15px; }
h1, h2, h3 { color: #00FFAA !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------
# Sidebar: Mode & Model
# ----------------------
st.sidebar.title("🛠 Assistant Config")
mode = st.sidebar.radio("Select Mode", ["Coding Assistant", "PDF Assistant"])
selected_model = st.sidebar.selectbox("Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:3b"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)

# ----------------------
# Initialize LLM & Embeddings
# ----------------------
llm_engine = ChatOllama(model=selected_model, base_url="http://localhost:11434", temperature=temperature)
embedding_model = OllamaEmbeddings(model=selected_model)
document_vector_db = InMemoryVectorStore(embedding_model)
language_model = OllamaLLM(model=selected_model)

# ----------------------
# Coding Assistant Setup
# ----------------------
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

def generate_ai_response(prompt_chain):
    messages = prompt_chain.format_prompt().to_messages()  # fixed
    response = llm_engine(messages)
    return response.content

# ----------------------
# PDF Assistant Setup
# ----------------------
PDF_STORAGE_PATH = 'document_store/pdfs/'
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

def save_uploaded_file(uploaded_file):
    path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
    return path

def load_pdf_documents(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def chunk_documents(raw_documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return splitter.split_documents(raw_documents)

def index_documents(document_chunks):
    document_vector_db.add_documents(document_chunks)

def find_related_documents(query):
    return document_vector_db.similarity_search(query, k=3)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = conversation_prompt.format_prompt(
        user_query=user_query, document_context=context_text  # fixed
    ).to_messages()
    response = language_model(formatted_prompt)
    return response.content

# ----------------------
# Streamlit UI
# ----------------------
st.title("🧠 Multi-Assistant AI")

if mode == "Coding Assistant":
    st.subheader("💻 Coding Assistant")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.message_log:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_query = st.chat_input("Type your coding question here...")
    if user_query:
        st.session_state.message_log.append({"role": "user", "content": user_query})
        with st.spinner("🧠 Processing..."):
            prompt_chain = build_prompt_chain()
            ai_response = generate_ai_response(prompt_chain)
        st.session_state.message_log.append({"role": "ai", "content": ai_response})
        st.experimental_rerun()

elif mode == "PDF Assistant":
    st.subheader("📘 PDF Document Mode")
    uploaded_pdf = st.file_uploader("Upload Research Document (PDF)", type="pdf")
    if uploaded_pdf:
        saved_path = save_uploaded_file(uploaded_pdf)
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
        st.success("✅ Document processed successfully! Ask your questions below.")

        user_input = st.chat_input("Enter your question about the document...")
        if user_input:
            with st.chat_message("user"): st.write(user_input)
            with st.spinner("Analyzing document..."):
                relevant_docs = find_related_documents(user_input)
                ai_response = generate_answer(user_input, relevant_docs)
            with st.chat_message("assistant", avatar="🤖"): st.write(ai_response)