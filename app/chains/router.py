# app/chains/router.py

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.llms.custom import CustomChatModel
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
custom_llm = CustomChatModel(api_url="https://nutnell-directed-ai.hf.space/generate")
openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
llm = custom_llm.with_fallbacks([openai_llm])
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

# --- Chain Definitions with Error Handling ---

def run_conversational_rag(input_dict: dict):
    """Encapsulates the conversational RAG logic with error handling."""
    try:
        # chain logic
        chain = (
            RunnablePassthrough.assign(
                standalone_question=(
                    RunnableLambda(lambda x: {
                        "question": x["input"],
                        "chat_history": get_buffer_string(x["chat_history"])
                    })
                    | condense_question_prompt | llm | StrOutputParser()
                )
            )
            | RunnablePassthrough.assign(
                context=lambda x: retriever.invoke(x["standalone_question"]),
            )
        )

        retrieved = chain.invoke(input_dict)

        if not retrieved.get("context"):
            return {"answer": "I'm sorry, I couldn't find any relevant information for your question.", "sources": []}

        answer_chain = (
            RunnableParallel(
                answer=(
                    RunnableLambda(lambda x: {
                        "context": format_docs(x["context"]),
                        "question": x["input"],
                    })
                    | rag_prompt | llm | StrOutputParser()
                ),
                sources=RunnableLambda(lambda x: get_sources_from_docs(x["context"])),
            )
        )
        return answer_chain.invoke(retrieved)

    except Exception as e:
        print(f"Error in conversational_rag_chain: {e}")
        return {"answer": "An unexpected error occurred. Please try again later.", "sources": []}


def run_quiz_generation(input_dict: dict):
    """Encapsulates the quiz generation logic with error handling."""
    try:
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
        return quiz_chain.invoke(input_dict)
    except Exception as e:
        print(f"Error in run_quiz_generation: {e}")
        return {"answer": "An unexpected error occurred while generating the quiz.", "sources": []}


# The Router run_educational_assistant()
run_educational_assistant = RunnableBranch(
    (lambda x: x.get("request_type") == "quiz_generation", run_quiz_generation),
    run_conversational_rag,
)

# Chain with Message History Management

educational_assistant_chain = RunnableWithMessageHistory(
    run_educational_assistant,
    get_memory_for_session,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
).with_types(
    input_type=ChatInput, output_type=ChatOutput
)