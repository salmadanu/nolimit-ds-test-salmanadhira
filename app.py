import streamlit as st
import pickle
from rag_utils import answer_query
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------- Load Vectorstore ------------------- #
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )
vectorstore = load_vectorstore()

# ------------------- UI ------------------- #
st.title("RAG Chatbot on Computational Media Analysis")
st.subheader("Ask questions regarding computational framing, fake news, propaganda detection, etc. using computational methods")

query = st.text_input("Enter question:")
if query:
    with st.spinner("Searching..."):
        answer = answer_query(query, vectorstore)
    st.markdown("### 💬 Answer")
    st.write(answer)


    # st.badge("Source", icon="🔍", color="blue")
    # for s in sources:
    #     st.markdown(f"- {s.metadata.get('source', 'Unknown')}")

 
