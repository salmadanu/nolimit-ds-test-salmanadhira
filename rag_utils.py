import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatHuggingFaceEndpoint


hf_token = st.secrets["api_keys"]["huggingface"]

def answer_query(vectorstore, query, debug=False):
    llm = ChatHuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        huggingfacehub_api_token=hf_token,
        temperature=0.3,
        max_new_tokens=512
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff" 
    )

    if debug:
        st.write("Vectorstore type:", type(vectorstore))
        docs = vectorstore.as_retriever().get_relevant_documents(query)
        st.write("Retrieved docs (top 3):")
        for i, doc in enumerate(docs[:3]):
            st.write(f"Doc {i+1}:", doc.page_content[:300], "...")  # show first 300 chars

    answer = qa({"query": query})
    return answer
