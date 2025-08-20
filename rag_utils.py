from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st

hf_token = st.secrets["api_keys"]["huggingface"]
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large", 
    huggingfacehub_api_token=hf_token,
    temperature=0.3,
    max_new_tokens=512
)

template = """
You are an academic assistant summarizing research papers. 
Answer the QUESTION clearly using the CONTEXT.
- Write in complete, well-formed sentences (fix broken or fragmented text).
- If methods or models are mentioned, list and EXPLAIN their purpose. 
- Cite authors or papers if available in the context.
- If the answer is not in the context, reply: "The context does not provide enough information."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

def answer_query(query:str, vectorstore, k=5):
    results = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    filled_prompt = prompt.format(context=context, question=query)
    answer = llm.invoke(filled_prompt)
    return answer.strip()[0].upper() + answer.strip()[1:]
