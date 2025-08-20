# main.py

from dotenv import load_dotenv

# Updated imports for new LangChain packages
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# We need to import a Chat Model, which is the "brain" of our AI
# Let's use OpenAI for now as it's straightforward to set up
from langchain_openai import ChatOpenAI

# ------------------- Setup -------------------
load_dotenv()
print("✅ Environment variables loaded.")

# --- Configuration ---
VECTOR_STORE_PATH = "app/vector_store"

# --- Models & Vector Store ---
print("Loading open-source embedding model 'all-MiniLM-L6-v2'...")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the existing vector store
vector_store = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embedding_function
)

# Initialize the LLM we'll use for answering questions
# Make sure your OPENAI_API_KEY is set in the .env file
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

print("✅ Setup complete. Vector store and LLM are ready.")

# ------------------- Core AI Logic -------------------

# 1. The EducationalRetriever Component
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 2. The Prompt Template
template = """
You are a helpful AI assistant for the DirectEd learning platform.
Answer the user's question based only on the following context.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question:
{question}

Helpful Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. The Q&A Chain
qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ Q&A chain created successfully.")

# ------------------- Main Execution (for testing) -------------------

if __name__ == "__main__":
    print("\n--- AI Assistant Ready ---")
    print("Enter your question or type 'exit' to quit.")

    while True:
        user_question = input("\nQuestion: ")
        if user_question.lower() == 'exit':
            break

        answer = qa_chain.invoke(user_question)

        print("\nAnswer:")
        print(answer)
