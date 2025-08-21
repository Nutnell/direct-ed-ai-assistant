# app/chains/router.py

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from app.schemas.api_models import ChatInput, ChatOutput
from app.prompts.templates import rag_prompt, quiz_generator_prompt


VECTOR_STORE_PATH = "app/vector_store"
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embedding_function
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Load the Fine-Tuned Llama 3 Model
BASE_MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
ADAPTER_PATH = "fine_tuning/results/llama-3-8b-instruct-direct-ed"

print("Loading fine-tuned Llama 3 model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

if os.path.exists(ADAPTER_PATH):
    # Merge the LoRA adapter with the base model
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()
    print("✅ Successfully loaded fine-tuned Llama 3 model adapter.")
else:
    # Fallback to the base model
    model = base_model
    print("⚠️ Fine-tuned adapter not found. Falling back to the base Llama 3 model.")

# Hugging Face pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512
)
# Wrap the pipeline to be used as a LangChain LLM
ft_llm = HuggingFacePipeline(pipeline=pipe)


memories = {}
def get_memory_for_session(session_id: str) -> ConversationBufferMemory:
    if session_id not in memories:
        memories[session_id] = ConversationBufferMemory(return_messages=False)
    return memories[session_id]

# Helper Functions
def format_docs(docs):
    return "\n---\n".join(doc.page_content for doc in docs)

def get_sources_from_docs(docs):
    return [
        {"source": doc.metadata.get("source_url"), "name": doc.metadata.get("source_name")}
        for doc in docs
    ]

# Memory-enabled Conversational RAG function
def conversational_rag_chain(input_data: dict, config: dict) -> dict:
    session_id = config["configurable"].get("session_id")
    if not session_id:
        raise ValueError("Session ID must be provided for conversational requests.")
    
    memory = get_memory_for_session(session_id)
    
    conversation = ConversationChain(llm=ft_llm, memory=memory)
    
    question = input_data["input"]
    docs = retriever.invoke(question)
    
    if not docs:
        return {"answer": "I'm sorry, I couldn't find any relevant information for your question.", "sources": []}

    formatted_context = format_docs(docs)
    sources = get_sources_from_docs(docs)
    

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant for the DirectEd learning platform. Answer the user's question based only on the provided context and conversation history."},
        {"role": "user", "content": f"Context:\n{formatted_context}\n\nConversation History:\n{memory.buffer}\n\nQuestion: {question}"},
    ]
    prompt_with_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    response_text = conversation.predict(input=prompt_with_context)

    if "assistant" in response_text:
        response_text = response_text.split("assistant")[1].strip()

    return {"answer": response_text, "sources": sources}

# Stateless Quiz Generation Chain
quiz_chain = (
    RunnableParallel(
        context=RunnableLambda(lambda x: retriever.invoke(x["input"])),
    )
    | RunnableParallel(
        answer=(
            RunnableLambda(lambda x: {"context": format_docs(x["context"])})
            | quiz_generator_prompt | ft_llm | StrOutputParser()
        ),
        sources=RunnableLambda(lambda x: get_sources_from_docs(x["context"])),
    )
)

# The Router
router = RunnableBranch(
    (lambda x, config: x.get("request_type") == "quiz_generation", quiz_chain),
    conversational_rag_chain,
)

# Final Chain for the API
educational_assistant_chain = router.with_types(
    input_type=ChatInput, output_type=ChatOutput
)