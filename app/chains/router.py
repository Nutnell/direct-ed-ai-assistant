# app/chains/router.py

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.schemas.api_models import ChatInput, ChatOutput
from app.prompts.templates import rag_prompt, quiz_generator_prompt

# --- Models & Vector Store Setup ---
VECTOR_STORE_PATH = "app/vector_store"
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embedding_function
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# --- Helper Functions ---
def format_docs(docs):
    return "\n---\n".join(doc.page_content for doc in docs)

def get_sources_from_docs(docs):
    return [
        {"source": doc.metadata.get("source_url"), "name": doc.metadata.get("source_name")}
        for doc in docs
    ]

# === THE FIX IS HERE ===
# We create a new chain that explicitly extracts the 'input' string for the retriever.
retrieval_chain = (
    RunnableLambda(lambda x: x["input"]) # Extract the 'input' string from the dictionary
    | retriever
)

# --- 1. Tutoring RAG Chain (AdaptiveConversationChain) ---
rag_chain = (
    {
        "context": retrieval_chain, # Use the new retrieval_chain
        "question": RunnableLambda(lambda x: x["input"]) # Pass the input string
    }
    | RunnableLambda(lambda x: {
        "context": format_docs(x["context"]),
        "question": x["question"],
        "sources": get_sources_from_docs(x["context"])
    })
    | {
        "answer": rag_prompt | llm | StrOutputParser(),
        "sources": RunnableLambda(lambda x: x["sources"])
    }
)

# --- 2. Quiz Generation Chain (ContentGenerator) ---
quiz_chain = (
    {
        "context": retrieval_chain, # Use the new retrieval_chain
        "question": RunnableLambda(lambda x: x["input"]) # Pass the input string
    }
    | RunnableLambda(lambda x: {
        "context": format_docs(x["context"]),
        "sources": get_sources_from_docs(x["context"])
    })
    | {
        "answer": quiz_generator_prompt | llm | StrOutputParser(),
        "sources": RunnableLambda(lambda x: x["sources"])
    }
)

# --- 3. The Router ---
router_chain = RunnableBranch(
    (lambda x: x["request_type"] == "quiz_generation", quiz_chain),
    rag_chain,
).with_types(input_type=ChatInput, output_type=ChatOutput)