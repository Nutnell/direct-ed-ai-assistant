# app/chain.py

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


from pydantic import BaseModel, Field
from typing import List

# Define the Input and Output Schemas
class ChatInput(BaseModel):
    
    input: str = Field(
        ...,
        description="The user's question to the AI assistant.",
        examples=["What is LLMOps?"]
    )

class Source(BaseModel):
    source: str = Field(..., description="The URL of the source document.")
    name: str = Field(..., description="The name of the source (e.g., 'DirectEd Curriculum').")

class ChatOutput(BaseModel):
    answer: str = Field(..., description="The AI-generated answer to the question.")
    sources: List[Source] = Field(..., description="A list of source documents used for the answer.")


VECTOR_STORE_PATH = "app/vector_store"

# Load Models and Vector Store
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embedding_function
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Define the Prompt Template
template = """
You are a helpful AI assistant for the DirectEd learning platform.
Answer the user's question based only on the following context.

Context:
{context}

Question:
{question}

Helpful Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Build the RAG Chain

# map the 'input' from Pydantic model to the 'question' used in the chain
_qa_chain = (
    {
        "context": lambda x: retriever.invoke(x["input"]),
        "question": lambda x: x["input"]
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

# Attach the corrected schema to the chain
qa_chain = _qa_chain.with_types(input_type=ChatInput, output_type=ChatOutput)