import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint

hf_token = st.secrets["api_keys"]["huggingface"]

def answer_query(vectorstore, query):
    llm = HuggingFaceEndpoint(repo_id ="google/flan-t5-base", 
                              huggingfacehub_api_token=hf_token, 
                              temperature= 0.3,
                              max_new_tokens=512)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    answer = qa({"query": query})
    return answer