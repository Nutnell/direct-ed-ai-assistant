# main.py

import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ------------------- Setup -------------------
load_dotenv()
# LangSmith configuration is now automatically picked up from .env
print("✅ Environment variables loaded.")

# --- Configuration ---
VECTOR_STORE_PATH = "app/vector_store"

# --- Models & Vector Store ---
print("Loading open-source embedding model 'all-MiniLM-L6-v2'...")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embedding_function
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
print("✅ Setup complete. Vector store and LLM are ready.")


# ------------------- Core AI Logic (with Source Retrieval) -------------------

# 1. Retriever Component (no change)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 2. Prompt Template (no change)
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

# 3. New Q&A Chain with Source Handling
# This chain is now designed to return a dictionary containing both the answer and the sources.
qa_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | RunnableLambda(lambda x: {
        "context": "\n---\n".join([doc.page_content for doc in x["context"]]),
        "question": x["question"],
        "sources": x["context"] # Pass the original documents through
    })
    | {
        "answer": prompt | llm | StrOutputParser(),
        "sources": RunnableLambda(lambda x: x["sources"]) # Keep the sources
    }
)

print("✅ RAG chain with source retrieval created successfully.")

# Helper function to format and print the output nicely
def print_response(response: dict):
    print("\n--- Answer ---")
    print(response.get("answer"))
    print("\n--- Sources ---")
    if response.get("sources"):
        for i, source_doc in enumerate(response["sources"]):
            # Extract metadata
            source_name = source_doc.metadata.get('source_name', 'N/A')
            source_url = source_doc.metadata.get('source_url', 'N/A')
            print(f"{i+1}. {source_name}: {source_url}")
    print("---------------")

# ------------------- Main Execution (for testing) -------------------

if __name__ == "__main__":
    print("\n--- AI Assistant Ready ---")
    print("Enter your question or type 'exit' to quit.")

    while True:
        user_question = input("\nQuestion: ")
        if user_question.lower() == 'exit':
            break
        
        # Invoke the chain and get the dictionary response
        response = qa_chain.invoke(user_question)
        print_response(response)