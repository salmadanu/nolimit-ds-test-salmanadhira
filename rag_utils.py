import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint

hf_token = st.secrets["api_keys"]["huggingface"]

def answer_query(vectorstore, query, debug=False):
    # Initialize LLM via endpoint
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
        huggingfacehub_api_token=hf_token,
        temperature=0.3,
        max_new_tokens=512
    )

    # Build RAG chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=debug
    )

    if debug:
        docs = vectorstore.similarity_search(query, k=3)
        for i, doc in enumerate(docs):
            st.write(f"Doc {i+1}:", doc.page_content[:300], "...")

    # Generate answer
    answer = qa({"query": query})
    return answer
