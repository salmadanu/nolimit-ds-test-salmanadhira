import streamlit as st
import pickle
from rag_utils import answer_query

# ------------------- Load Vectorstore ------------------- #
@st.cache_resource
def load_vectorstore():
    with open("vectorstore/faiss_store.pkl", "rb") as f:
        return pickle.load(f)

vectorstore = load_vectorstore()

# ------------------- UI ------------------- #
st.title("RAG Chatbot on Computational Media Analysis")
st.subheader("Ask questions regarding computational framing, fake news, propaganda detection, etc. using computational methods")

query = st.text_input("Enter question:")
if query:
    with st.spinner("Searching..."):
        answer, sources = answer_query(query, vectorstore)
    st.badge("Answer", icon="💬", color="green")
    st.write(answer)

    # st.badge("Source", icon="🔍", color="blue")
    # for s in sources:
    #     st.markdown(f"- {s.metadata.get('source', 'Unknown')}")

 
