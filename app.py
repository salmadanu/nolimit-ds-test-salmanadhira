import streamlit as st
from rag_utils import answer_query
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("📚 RAG Chatbot")

# Load or initialize vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local("faiss_index", 
                                   embeddings,
                                   allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_vectorstore()

# Debug mode toggle
debug_mode = st.checkbox("Debug Mode", value=False)

# User input
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching..."):
        answer = answer_query(vectorstore, query, debug=debug_mode)
    st.markdown("### 💬 Answer")
    st.write(answer)
