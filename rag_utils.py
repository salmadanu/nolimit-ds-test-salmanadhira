import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint


hf_token = st.secrets["api_keys"]["huggingface"]

def answer_query(vectorstore, query, debug=False):
    llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={
        "temperature": 0.3,
        "max_new_tokens": 512
    },
    huggingfacehub_api_token=st.secrets["api_keys"]["huggingface"]
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
