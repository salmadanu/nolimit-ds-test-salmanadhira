import gradio as gr
from langchain import PromptTemplate, LLMChain
from langchain_huggingface import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Load embeddings & FAISS vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=-1
)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

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
qa_chain = LLMChain(llm=llm, prompt=prompt)

def answer_query_gradio(query, k=10):
    results = vectorstore.similarity_search(query, k=k)
    context = "\n".join([res.page_content for res in results[:3]])
    return qa_chain.run({"context": context, "question": query})


iface = gr.Interface(
    fn=answer_query_gradio,  # function to run
    inputs=gr.Textbox(lines=2, placeholder="Ask a question..."),  # user input
    outputs=gr.Textbox(),  # answer output
    title="RAG Chatbot",
    description="Ask questions about your documents using the Flan-T5 RAG model."
)

iface.launch()

