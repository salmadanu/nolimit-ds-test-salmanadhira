import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint

hf_token = st.secrets["api_keys"]["huggingface"]

def answer_query(vectorstore, query):
    llm = HuggingFaceEndpoint(repo_id ="meta-llama/Meta-Llama-3-8B-Instruct", 
                               huggingfacehub_api_token=hf_token, 
                              temperature= 0.6)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    answer = qa({"query": query})
    return answer