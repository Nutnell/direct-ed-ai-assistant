# app/chains/router.py

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableConfig,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import get_buffer_string
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Import schemas and prompts
from app.schemas.api_models import ChatInput, ChatOutput
from app.prompts.templates import rag_prompt, quiz_generator_prompt, condense_question_prompt

# --- Models & Vector Store Setup ---
VECTOR_STORE_PATH = "app/vector_store"
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embedding_function
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# --- In-Memory Chat History Store ---
memories = {}

def get_memory_for_session(session_id: str) -> ChatMessageHistory:
    """Return or create a ChatMessageHistory for the given session."""
    if session_id not in memories:
        memories[session_id] = ChatMessageHistory()
    return memories[session_id]

# --- Helper Functions ---
def format_docs(docs):
    return "\n---\n".join(doc.page_content for doc in docs)

def get_sources_from_docs(docs):
    return [
        {"source": doc.metadata.get("source_url"), "name": doc.metadata.get("source_name")}
        for doc in docs
    ]

def get_chat_history(_input, config: RunnableConfig):
    """Fetch chat history for the given session (RunnableLambda-compatible)."""
    session_id = config["configurable"].get("session_id")
    if not session_id:
        raise ValueError("Session ID not found in config.")
    return get_memory_for_session(session_id).messages

# --- Chain Definitions ---

# 1. Conversational RAG Chain
conversational_rag_chain = (
    RunnablePassthrough.assign(
        standalone_question=(
            RunnableLambda(lambda x: {
                "question": x["input"],
                "chat_history": get_buffer_string(x["chat_history"])
            })
            | condense_question_prompt
            | llm
            | StrOutputParser()
        )
    )
    | RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["standalone_question"]),
    )
    | RunnableParallel(
        answer=(
            RunnableLambda(lambda x: {
                "context": format_docs(x["context"]),
                "question": x["input"],
            })
            | rag_prompt
            | llm
            | StrOutputParser()
        ),
        sources=RunnableLambda(lambda x: get_sources_from_docs(x["context"])),
    )
)

# 2. Stateless Quiz Generation Chain
quiz_chain = (
    RunnableParallel(
        context=RunnableLambda(lambda x: retriever.invoke(x["input"])),
    )
    | RunnableParallel(
        answer=(
            RunnableLambda(lambda x: {"context": format_docs(x["context"])})
            | quiz_generator_prompt | llm | StrOutputParser()
        ),
        sources=RunnableLambda(lambda x: get_sources_from_docs(x["context"])),
    )
)

# 3. The Router
router = RunnableBranch(
    (lambda x: x.get("request_type") == "quiz_generation", quiz_chain),
    conversational_rag_chain,
)

# 4. Final Chain with Message History Management
router_chain_with_history = RunnableWithMessageHistory(
    router,
    get_memory_for_session,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
).with_types(
    input_type=ChatInput, output_type=ChatOutput
)
