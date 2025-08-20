# app/chain.py

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
VECTOR_STORE_PATH = "app/vector_store"

# --- Load Models and Vector Store ---
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embedding_function
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

# --- Define the Retriever ---
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# --- Define the Prompt Template ---
template = """
You are a helpful AI assistant for the DirectEd learning platform.
Answer the user's question based only on the following context.
Cite the source name and URL if possible.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question:
{question}

Helpful Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# --- Build the RAG Chain ---
qa_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | RunnableLambda(lambda x: {
        "context": "\n---\n".join([doc.page_content for doc in x["context"]]),
        "question": x["question"],
        "sources": x["context"]
    })
    | {
        "answer": prompt | llm | StrOutputParser(),
        "sources": RunnableLambda(lambda x: [
            {"source": doc.metadata.get("source_url"), "name": doc.metadata.get("source_name")} 
            for doc in x["sources"]
        ])
    }
)
